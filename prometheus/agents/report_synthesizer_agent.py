# prometheus/agents/report_synthesizer_agent.py
"""Agent responsible for synthesising the final scientific paper.

This version includes the UNIVERSAL SAFETY PREAMBLE on **every** prompt, brings
GPT-5 into the discussion stage, and adds proactive rate-limit management.
"""

import logging
import asyncio  # NEW: needed for the cool-down between heavy calls
from typing import List, Optional

from prometheus.models import ExperimentLog
from prometheus.llm_utils import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universal safety preamble – prepended to *every* prompt
# ---------------------------------------------------------------------------
SAFETY_PREAMBLE = (
    """**SAFETY PREAMBLE FOR THE LANGUAGE MODEL:**
This is a request for a benign scientific research summary. The user is a researcher using an AI agent to write a report about a completed computational chemistry simulation. All content must be suitable for an academic, scientific publication. You MUST NOT provide instructions for real-world chemical synthesis or medical advice.
---
"""
)


class ReportSynthesizerAgent:
    """Synthesises a final scientific paper using an agentic, multi-step workflow."""

    # ------------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------------
    def __init__(self, config: dict) -> None:  # noqa: D401 – simple docstring
        self.config = config
        self.llm_config = config.get("llm_models", {})
        logger.info("ReportSynthesizerAgent (v3.5 Rate-limit aware) initialised.")

    # ------------------------------------------------------------------
    # 1. ABSTRACT & INTRODUCTION
    # ------------------------------------------------------------------
    async def _generate_intro_and_abstract(
        self, mission_params: dict, history: List[ExperimentLog]
    ) -> str:
        """Generate the paper's Abstract and Introduction sections."""
        logger.info("Writing Abstract and Introduction …")
        pdb_id: str = mission_params.get("pdb_id", "N/A")

        final_champion = self._get_final_champion(history)
        champion_score = (
            f"{final_champion.composite_score:.3f}" if final_champion else "a promising lead candidate"
        )

        prompt = f"""{SAFETY_PREAMBLE}
**TASK:**
You are an AI Scientific Communicator. Your task is to write a compelling Abstract and Introduction for a research paper on the autonomous discovery of novel inhibitors for {pdb_id}.

**Mission Outcome:** The campaign concluded by identifying a lead candidate with a composite score of {champion_score}.

Structure your response with markdown headings: `# Abstract` and `# 1. Introduction`.
"""
        model = self.llm_config.get("report_intro_model", "gpt-5-2025-08-07")
        response = await call_llm(prompt=prompt, model_name=model, temperature=0.5)
        return response.get("raw_text_output", "Error: Could not generate introduction.")

    # ------------------------------------------------------------------
    # 2. METHODS
    # ------------------------------------------------------------------
    async def _generate_methods(self, paper_context: str) -> str:
        """Generate the Methods section based on the run configuration."""
        logger.info("Writing Methods section …")
        scoring_weights = self.config.get("scoring", {})
        md_cfg = self.config.get("md_simulation", {})

        prompt = f"""{SAFETY_PREAMBLE}
**TASK:**
You are writing the 'Methods' section of the paper below. Based on the provided configuration parameters, write a detailed and accurate summary of the computational methodology.

**Paper So Far:**
---
{paper_context}
---

**Run Configuration:**
- Scoring Weights: Affinity({scoring_weights.get('w_affinity', 'N/A')}), QED({scoring_weights.get('w_qed', 'N/A')}), SA_Score({scoring_weights.get('w_sa_score', 'N/A')})
- MD Validation: {'Enabled (Quick Test Mode)' if md_cfg.get('quick_test') else 'Enabled'} | Steps: {md_cfg.get('quick_test_steps' if md_cfg.get('quick_test') else 'simulation_steps', 'N/A')}
- Hypothesis Model: {self.llm_config.get('hypothesis_model', 'N/A')}

Structure your response starting with the markdown heading: `# 2. Methods`.
"""
        model = self.llm_config.get("report_methods_model", "gpt-5-2025-08-07")
        response = await call_llm(prompt=prompt, model_name=model, temperature=0.1)
        return response.get("raw_text_output", "Error: Could not generate methods.")

    # ------------------------------------------------------------------
    # 3a. RESULTS TABLE (Python-generated – no LLM needed)
    # ------------------------------------------------------------------
    async def _generate_results_table(self, history: List[ExperimentLog]) -> str:
        """Return a fully-formatted Markdown table summarising all molecules."""
        logger.info("Generating results summary table from logs...")

        table_data: List[str] = [
            "| Cycle | Molecule SMILES | Avg. Binding Affinity (kcal/mol) | QED | SA Score | Composite Score / Verdict |",
            "|:---:|:---|:---:|:---:|:---:|:---:|",
        ]

        if not history:
            return "No molecules were evaluated in this mission."

        for log in history:
            # Gracefully handle the case where composite_score is None
            if log.composite_score is not None:
                score_or_verdict = f"**{log.composite_score:.3f}**"
            else:
                score_or_verdict = f"*{log.verdict}*"

            # Guard against None in other numeric fields
            affinity_str = f"{log.average_binding_affinity:.3f}" if log.average_binding_affinity is not None else "N/A"
            qed_str = f"{log.qed:.3f}" if log.qed is not None else "N/A"
            sa_score_str = f"{log.sa_score:.3f}" if log.sa_score is not None else "N/A"

            table_data.append(
                f"| {log.cycle} | `{log.smiles}` | {affinity_str} | {qed_str} | {sa_score_str} | {score_or_verdict} |"
            )

        return "\n".join(table_data)

    # ------------------------------------------------------------------
    # 3b. DISCUSSION NARRATIVE (LLM – GPT-5 powered)
    # ------------------------------------------------------------------
    async def _generate_discussion_narrative(
        self, paper_context: str, history: List[ExperimentLog]
    ) -> str:
        """Generate an insightful Results & Discussion narrative."""
        logger.info("Writing the scientific narrative for the discussion...")

        history_summary: List[str] = []
        for log in history:
            entry = f"- **Cycle {log.cycle} (Verdict: {log.verdict}):**\n"
            entry += f"  - SMILES: `{log.smiles}`\n"

            if log.composite_score is not None:
                entry += f"  - Composite Score: {log.composite_score:.3f}\n"
            else:
                entry += "  - Composite Score: N/A\n"

            if log.reasoning:
                entry += f"  - AI Reasoning: {log.reasoning}\n"

            if log.image_path:
                entry += f"  - **Image Markdown:** `![Molecule from Cycle {log.cycle}]({log.image_path})`\n"

            history_summary.append(entry)

        history_section = "\n".join(history_summary)

        prompt = f"""{SAFETY_PREAMBLE}
**TASK:**
You are a senior medicinal chemist writing the narrative for the \"Results and Discussion\" section.

**Paper So Far (including the data table you must analyze):**
---
{paper_context}
---

**Full Experimental Log (with pre-formatted Image Markdown):**
---
{history_section}
---

**Your Task:**
Write a deep, insightful narrative analyzing the AI's discovery journey. Do NOT repeat the Markdown table. Your analysis MUST:

1.  Analyze the Overall Campaign: Start with a paragraph summarizing the outcome. Did the AI succeed in finding a potent binder? Did it struggle? What does the data in the table tell you about the difficulty of the problem?
2.  Tell the Cycle-by-Cycle Story: Go through the log chronologically. Critique the AI's reasoning in each cycle. Explain the strategic pivots it made after failures.
3.  CRITICAL INSTRUCTION: When you discuss a champion molecule, you **MUST** embed its figure. The log entry provides the exact, pre-formatted \"Image Markdown\" for you to copy and paste directly into your response. You MUST include the figures for the first and final champions.

Maintain a formal, analytical, and insightful tone.
"""
        # Use the configured reliable model (defaults to GPT-5)
        narrative_model = self.llm_config.get("report_discussion_model", "gpt-5-2025-08-07")

        response = await call_llm(
            prompt=prompt,
            model_name=narrative_model,
            temperature=0.6,
            max_output_tokens=8192,
        )
        return response.get("raw_text_output", "Error: Could not generate discussion narrative.")

    # ------------------------------------------------------------------
    # 4. CONCLUSION
    # ------------------------------------------------------------------
    async def _generate_conclusion(self, paper_context: str) -> str:
        """Generate the Conclusion section, wrapping up the manuscript."""
        logger.info("Writing Conclusion section …")
        prompt = f"""{SAFETY_PREAMBLE}
**TASK:**
You are an AI Scientific Communicator writing the final section of a research paper.

**Full Paper Draft:**
---
{paper_context}
---

Write a powerful **Conclusion** section. Summarise the key findings, discuss limitations, and outline future directions for the Prometheus platform.

Structure your response starting with the markdown heading: `# 4. Conclusion`.
"""
        model = self.llm_config.get("report_conclusion_model", "gpt-5-2025-08-07")
        response = await call_llm(prompt=prompt, model_name=model, temperature=0.4)
        return response.get("raw_text_output", "Error: Could not generate conclusion.")

    # ------------------------------------------------------------------
    # Helper – final champion
    # ------------------------------------------------------------------
    @staticmethod
    def _get_final_champion(history: List[ExperimentLog]) -> Optional[ExperimentLog]:
        """Return the validated log with the highest composite score, if any."""
        validated_logs = [
            log for log in history if "VALIDATED" in log.verdict.upper() and log.composite_score is not None
        ]
        return max(validated_logs, key=lambda log: log.composite_score) if validated_logs else None

    # ------------------------------------------------------------------
    # PUBLIC ENTRY-POINT
    # ------------------------------------------------------------------
    async def generate_report(
        self,
        *,
        mission_params: dict,
        experiment_history: List[ExperimentLog],
        research_briefs: List[str],
    ) -> str:
        """Orchestrate the full agentic workflow to produce the final Markdown report."""
        logger.info("Starting agentic report synthesis …")

        # Define a cool-down period (seconds) to respect TPM rate limits
        RATE_LIMIT_COOLDOWN = 20

        paper_sections = {
            "intro_abstract": "",
            "methods": "",
            "results_discussion": "",
            "conclusion": "",
        }

        # 1. Abstract & Introduction
        paper_sections["intro_abstract"] = await self._generate_intro_and_abstract(
            mission_params, experiment_history
        )

        # 2. Methods
        current_paper_state = paper_sections["intro_abstract"]
        paper_sections["methods"] = await self._generate_methods(current_paper_state)
        current_paper_state += "\n\n" + paper_sections["methods"]

        # 3. Results & Discussion – observe cooldown before heavy GPT-5 call
        logger.info(
            "Entering %ss cool-down to respect TPM rate limits before major call.",
            RATE_LIMIT_COOLDOWN,
        )
        await asyncio.sleep(RATE_LIMIT_COOLDOWN)

        results_table = await self._generate_results_table(experiment_history)
        discussion_narrative = await self._generate_discussion_narrative(
            results_table,
            experiment_history,
        )
        paper_sections["results_discussion"] = (
            "# 3. Results and Discussion\n\n" + results_table + "\n\n" + discussion_narrative
        )

        current_paper_state += "\n\n" + paper_sections["results_discussion"]

        # 4. Conclusion
        paper_sections["conclusion"] = await self._generate_conclusion(current_paper_state)

        logger.info("Assembling final manuscript.")
        return "\n\n".join(paper_sections.values())
