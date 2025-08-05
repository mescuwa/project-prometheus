# prometheus/agents/research_agent.py
"""ResearchAgent leverages OpenAI's Deep Research API to perform long-running,
expensive, tool-augmented research jobs. The implementation follows the
public documentation verbatim so that it will either work out-of-the-box or
fail fast with a clear error if the installed SDK does not yet expose the
required surface (e.g., `client.responses.create`)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from openai import OpenAI  # Synchronous client – long-running requests benefit from a blocking call

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Thin wrapper around the OpenAI Deep Research endpoint."""

    def __init__(self, config: dict):
        # Pull the sub-section so that downstream look-ups are tidy
        try:
            self.config = config["research_agent"]
        except KeyError as exc:
            raise ValueError("[research_agent] section missing from configuration.") from exc

        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # A generous timeout – deep research jobs are intentionally slow.
        self.client = OpenAI(api_key=api_key, timeout=self.config.get("timeout", 1800.0))
        logger.info("ResearchAgent initialised (Deep Research mode).")

    def conduct_research(self, research_prompt: str) -> Optional[str]:
        """Run the deep-research job and return the report text (or None on failure)."""

        if not self.config.get("enabled", False):
            logger.info("ResearchAgent is disabled in config – skipping live research.")
            return "Research agent is disabled in the configuration."

        model_name: str = self.config.get("model", "o4-mini-deep-research")
        max_calls: int | None = self.config.get("max_tool_calls")

        logger.info("Initiating Deep Research with model %s…", model_name)
        logger.debug("Prompt: %s", research_prompt)

        try:
            # Call signature mirrors the documentation exactly.
            response = self.client.responses.create(
                model=model_name,
                input=research_prompt,
                tools=[
                    {"type": "web_search_preview"},
                    # Optional but extremely useful for data analysis / extraction
                    {"type": "code_interpreter", "container": {"type": "auto"}},
                ],
                max_tool_calls=max_calls,
            )

            # The final answer is expected in `output_text` per docs
            report: str = response.output_text  # type: ignore[attr-defined]
            logger.info("Deep Research task completed successfully.")

            # Log the tool-call trajectory for debugging (if available)
            if hasattr(response, "output"):
                logger.debug("Tool call trajectory: %s", response.output)

            return report

        except AttributeError:
            logger.error(
                "CRITICAL: Installed 'openai' package does not expose the Deep Research "
                "endpoint (client.responses.create). Please install a compatible SDK "
                "version or request access from OpenAI."
            )
            return None
        except Exception as exc:  # Catch-all so a caller can continue gracefully
            logger.error("An unexpected error occurred during Deep Research: %s", exc, exc_info=True)
            return None
