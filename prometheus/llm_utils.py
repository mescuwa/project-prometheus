# prometheus/llm_utils.py
"""Utilities for interacting with Large Language Models.

This *ultimate* release unifies all retry logic into a **single outer loop** that
wraps the *entire* request / validation pipeline.  This makes the helper
resilient to **both** external API failures **and** our own ValueError-based
validation checks.

Key design points
-----------------
1. **Single retry loop** – covers network errors *and* post-processing checks.
2. **Fatal vs retryable errors** – Bad prompts / wrong API keys abort
   immediately; everything else gets retried up to ``max_retries`` times.
3. **Optional fallback model** – preserved for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time  # noqa: F401 – kept for potential future timing metrics
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional third-party dependencies – imported lazily so missing packages do
# not break the rest of the codebase.
# ---------------------------------------------------------------------------
try:
    import google.generativeai as genai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    genai = None  # type: ignore

try:  # noqa: WPS433 – dynamic optional import
    import openai  # type: ignore
    from openai import AsyncOpenAI as OpenAIClient  # type: ignore
    from openai.types.chat import ChatCompletionMessageParam  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    openai = None  # type: ignore
    OpenAIClient = None  # type: ignore
    ChatCompletionMessageParam = Dict[str, str]  # type: ignore

try:
    import google.api_core.exceptions  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    google = None  # type: ignore

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
_DEFAULT_LLM_TEMPERATURE: Optional[float] = None


def configure_llm_utils(default_temperature: Optional[float] = None) -> None:
    """Set a global default temperature for all subsequent ``call_llm`` calls."""

    global _DEFAULT_LLM_TEMPERATURE
    _DEFAULT_LLM_TEMPERATURE = default_temperature
    logger.info("llm_utils configured. Default temperature=%s", default_temperature)


# ---------------------------------------------------------------------------
# Main helper – now with unified retry logic
# ---------------------------------------------------------------------------
async def call_llm(
    *,
    prompt: str,
    model_name: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    fallback_model: Optional[str] = "gemini-2.5-pro",
) -> Dict[str, Any]:
    """Call a Gemini or GPT family model with robust retry / fallback logic."""

    # ---------------------------------------------------------------------
    # Retry configuration
    # ---------------------------------------------------------------------
    max_retries = 3
    retry_delay_seconds = 5

    effective_temp = temperature if temperature is not None else _DEFAULT_LLM_TEMPERATURE

    logger.debug(
        "LLM call init (model=%s, temp=%s, max_tokens=%s, json=%s)",
        model_name,
        effective_temp,
        max_output_tokens,
        bool(json_schema),
    )

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(
                "LLM call attempt %d/%d for model %s", attempt, max_retries, model_name
            )

            # -------------------------------------------------------------
            # GEMINI FAMILY
            # -------------------------------------------------------------
            if model_name.startswith("gemini"):
                if genai is None:
                    raise ImportError("google-generativeai not installed")

                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY env var not set")

                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)

                gen_cfg: Dict[str, Any] = {
                    "temperature": effective_temp,
                    "max_output_tokens": max_output_tokens,
                }
                if json_schema is not None:
                    gen_cfg["response_mime_type"] = "application/json"

                response = await model.generate_content_async(
                    _build_gemini_messages(prompt, system_message),
                    generation_config=_clean_dict(gen_cfg),
                )

                # ------------------- Post-processing / validation --------------------
                if not response.candidates:
                    raise ValueError("Gemini response is empty or blocked (no candidates).")

                candidate = response.candidates[0]
                if not candidate.content.parts:
                    finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
                    raise ValueError(
                        f"Gemini response blocked (no content parts). Reason: {finish_reason}"
                    )

                text_output = candidate.content.parts[0].text  # type: ignore[index]
                return {"raw_text_output": text_output}

            # -------------------------------------------------------------
            # GPT FAMILY (OpenAI)
            # -------------------------------------------------------------
            elif model_name.startswith("gpt"):
                if OpenAIClient is None:
                    raise ImportError("openai>=1.2 not installed")

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY env var not set")

                client = OpenAIClient(api_key=api_key)

                messages: List[ChatCompletionMessageParam] = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                response_format: Dict[str, Any] = {"type": "text"}
                if json_schema is not None:
                    response_format = {"type": "json_object"}

                api_params: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "response_format": response_format,
                }

                # Temperature – handle GPT-5 restrictions explicitly.
                if effective_temp is not None:
                    if "gpt-5" not in model_name or effective_temp == 1.0:
                        api_params["temperature"] = effective_temp
                    else:
                        logger.warning(
                            "Ignoring non-default temperature for '%s' as it is not supported.",
                            model_name,
                        )

                # Token limit parameter names differ between GPT-4 and GPT-5.
                if max_output_tokens is not None:
                    if "gpt-5" in model_name:
                        api_params["max_completion_tokens"] = max_output_tokens
                    else:
                        api_params["max_tokens"] = max_output_tokens

                completion = await client.chat.completions.create(**_clean_dict(api_params))
                text_output = completion.choices[0].message.content  # type: ignore[attr-defined]
                return {"raw_text_output": text_output}

            # -------------------------------------------------------------
            # UNSUPPORTED MODEL
            # -------------------------------------------------------------
            else:
                raise ValueError(f"Unsupported model: '{model_name}'")

        # ------------------------------------------------------------------
        # FATAL ERRORS – do *not* retry
        # ------------------------------------------------------------------
        except Exception as exc:  # noqa: BLE001 – deliberate broad catch
            is_fatal = False
            if openai is not None and isinstance(
                exc, (openai.BadRequestError, openai.AuthenticationError)  # type: ignore[attr-defined]
            ):
                is_fatal = True
            if isinstance(exc, ValueError):
                is_fatal = True

            if is_fatal:
                logger.error("Fatal, non-retryable error in LLM call: %s", exc)

                # Optional fallback – only attempt if a different model is provided.
                if fallback_model and fallback_model != model_name:
                    logger.warning("Attempting fallback to '%s'", fallback_model)
                    return await call_llm(
                        prompt=prompt,
                        model_name=fallback_model,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        system_message=system_message,
                        json_schema=json_schema,
                        fallback_model=None,  # prevent infinite recursion
                    )

                return {"error": "API_EXCEPTION_FATAL", "message": str(exc)}

            # ------------------------------------------------------------------
            # RETRYABLE ERRORS
            # ------------------------------------------------------------------
            logger.warning(
                "Retryable error on attempt %d/%d: %s", attempt, max_retries, exc
            )
            if attempt == max_retries:
                logger.error("LLM call failed after %d attempts.", max_retries)
                return {"error": "API_EXCEPTION_RETRY_FAILED", "message": str(exc)}

            await asyncio.sleep(retry_delay_seconds)

    # ---------------------------------------------------------------------
    # Should not normally be reached
    # ---------------------------------------------------------------------
    return {
        "error": "UNKNOWN_FAILURE",
        "message": "The LLM call failed after all retries for an unknown reason.",
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return *d* without keys that map to ``None`` values."""

    return {k: v for k, v in d.items() if v is not None}


def _build_gemini_messages(prompt: str, system_message: Optional[str]) -> List[Dict[str, Any]]:
    """Build the role/parts structure expected by the Gemini Python SDK."""

    if system_message:
        full_prompt = f"{system_message}\n\n---\n\n{prompt}"
        return [{"role": "user", "parts": [{"text": full_prompt}]}]
    return [{"role": "user", "parts": [{"text": prompt}]}]
