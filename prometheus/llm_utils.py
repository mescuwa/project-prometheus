import asyncio
import logging
import os
import time  # Added for retry sleep
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    genai = None  # Stub when the package isn't installed

try:
    from openai import AsyncOpenAI as OpenAIClient  # OpenAI>=1.2 has async client
    from openai.types.chat import ChatCompletionMessageParam  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAIClient = None  # type: ignore
    ChatCompletionMessageParam = Dict[str, str]  # fallback

# NEW import for handling Gemini internal errors that should trigger retries
try:
    import google.api_core.exceptions  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    google = None


_DEFAULT_LLM_TEMPERATURE: Optional[float] = None


def configure_llm_utils(default_temperature: Optional[float] = None) -> None:
    """Configure global defaults for the LLM helpers."""

    global _DEFAULT_LLM_TEMPERATURE
    _DEFAULT_LLM_TEMPERATURE = default_temperature
    logger.info("llm_utils configured. Default temperature=%s", default_temperature)


# ---------------------------------------------------------------------------
# Unified LLM wrapper (Gemini & GPT)
# ---------------------------------------------------------------------------

async def call_gemini(
    *,
    prompt: str,
    model_name: str,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call either Gemini or GPT models with basic retry logic for transient errors."""

    # Retry configuration
    max_retries = 3
    retry_delay_seconds = 5

    effective_temp = temperature if temperature is not None else _DEFAULT_LLM_TEMPERATURE
    logger.debug(
        "LLM call (model=%s, temp=%s, system_message=%s, json_schema=%s)",
        model_name,
        effective_temp,
        bool(system_message),
        bool(json_schema),
    )

    # ---------------------------------------------------------------------
    # Gemini branch
    # ---------------------------------------------------------------------
    if model_name.startswith("gemini"):
        if genai is None:  # pragma: no cover
            return {"error": "MISSING_DEPENDENCY", "message": "google-generativeai not installed"}
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "API_KEY_MISSING", "message": "GEMINI_API_KEY env var not set"}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        gen_cfg: Dict[str, Any] = {
            "temperature": effective_temp,
            "max_output_tokens": max_output_tokens,
        }
        if json_schema is not None:
            gen_cfg["response_mime_type"] = "application/json"

        # inner function to generate content (async if available)
        async def _generate() -> Any:  # type: ignore[override]
            if hasattr(model, "async_generate_content"):
                return await model.async_generate_content(
                    _build_gemini_messages(prompt, system_message),
                    generation_config=_clean_dict(gen_cfg),
                )
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    _build_gemini_messages(prompt, system_message),
                    generation_config=_clean_dict(gen_cfg),
                ),
            )

        # Retry loop for transient 500 errors
        for attempt in range(max_retries):
            try:
                response = await _generate()
                candidate = response.candidates[0]
                text_output = candidate.content.parts[0].text  # type: ignore[attr-defined]
                return {"raw_text_output": text_output}
            except Exception as exc:  # Catch and inspect
                # If google exceptions module present and error is InternalServerError, retry
                if (
                    google is not None
                    and isinstance(exc, google.api_core.exceptions.InternalServerError)
                ):
                    logger.warning(
                        "Gemini 500 Internal Error (attempt %s/%s). Retrying in %ss…",
                        attempt + 1,
                        max_retries,
                        retry_delay_seconds,
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay_seconds)
                        continue
                # For any other error or after max retries, record and exit
                logger.exception("Gemini API error: %s", exc)
                return {"error": "API_EXCEPTION", "message": str(exc)}

    # ---------------------------------------------------------------------
    # GPT branch (unchanged)
    # ---------------------------------------------------------------------
    if model_name.startswith("gpt"):
        if OpenAIClient is None:  # pragma: no cover
            return {"error": "MISSING_DEPENDENCY", "message": "openai>=1.2 not installed"}

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "API_KEY_MISSING", "message": "OPENAI_API_KEY env var not set"}

        client = OpenAIClient(api_key=api_key)

        messages: List[ChatCompletionMessageParam] = []  # type: ignore[var-annotated]
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response_format: Dict[str, Any] = {"type": "text"}
        if json_schema is not None:
            response_format = {"type": "json_object"}

        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=effective_temp,
                max_tokens=max_output_tokens,
                response_format=response_format,  # type: ignore[arg-type]
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("OpenAI API error: %s", exc)
            return {"error": "API_EXCEPTION", "message": str(exc)}

        try:
            text_output = completion.choices[0].message.content  # type: ignore[index]
            return {"raw_text_output": text_output}
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to parse OpenAI response: %s", exc)
            return {"error": "PARSE_ERROR", "message": str(exc)}

    # ---------------------------------------------------------------------
    return {"error": "UNSUPPORTED_MODEL", "message": f"Model '{model_name}' is not supported by call_gemini."}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return *d* without keys mapping to ``None`` values."""
    return {k: v for k, v in d.items() if v is not None}


def _build_gemini_messages(prompt: str, system_message: Optional[str]) -> List[Dict[str, Any]]:
    """Gemini helper – build the message dict list."""

    if system_message:
        return [
            {"role": "system", "parts": [{"text": system_message}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ]
    return [{"role": "user", "parts": [{"text": prompt}]}] 