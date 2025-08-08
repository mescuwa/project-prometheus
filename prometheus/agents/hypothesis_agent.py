"""Hypothesis Agent – Direct large-context strategy (no summarization).

Feeds raw literature context directly to a large-context LLM (Gemini 2.5 Pro)
to generate diverse, chemically valid candidate molecules in a single guarded
prompt.
"""

import sys
import logging
from pathlib import Path
import asyncio
import json
from rdkit import Chem, RDLogger
from rdkit.Chem import SanitizeMol, SanitizeFlags
from typing import Optional, List
from pydantic import ValidationError

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# --- Imports ---
import lancedb
from sentence_transformers import SentenceTransformer
from prometheus.models import Step, ExperimentLog, HypothesisBatch
from prometheus.llm_utils import call_llm

logger = logging.getLogger(__name__)


class HypothesisAgent:
    """Formulates scientific hypotheses by querying a knowledge base and using an LLM."""

    def __init__(self, config: dict):
        logger.info("Initializing HypothesisAgent…")
        self.config = config
        try:
            kb_config = config["knowledge_base"]
            self.db_path = project_root / kb_config["db_path"]
            self.table_name = kb_config["table_name"]
            self.embedding_model_name = kb_config["embedding_model_name"]

            logger.info("Loading embedding model: %s", self.embedding_model_name)
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            logger.info("Connecting to knowledge base…")
            self.db = lancedb.connect(self.db_path)
            self.table = self.db.open_table(self.table_name)
            logger.info("HypothesisAgent initialized successfully.")

            # Suppress noisy RDKit logging
            RDLogger.DisableLog("rdApp.*")
        except Exception as exc:  # pragma: no cover – init should succeed
            logger.error("Failed to initialize HypothesisAgent: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _search_knowledge_base(self, query: str, limit: int = 5) -> str:
        """Return concatenated literature snippets relevant to *query*. Increased limit for better context."""
        logger.info("Searching knowledge base with query: '%s'", query)
        query_vector = self.embedding_model.encode(query)
        try:
            results = self.table.search(query_vector).limit(limit).to_df()
            if results.empty:
                logger.warning("No relevant knowledge found for the query.")
                return "No relevant literature found."

            context = "\n\n---\n\n".join(
                f"Source: {row['source']}\n\nContent: {row['content']}" for _, row in results.iterrows()
            )
            logger.info("Found %d relevant knowledge snippets.", len(results))
            return context
        except Exception as exc:
            logger.error("Failed to search knowledge base: %s", exc, exc_info=True)
            return "Error searching knowledge base."

    # ---------------------------------------------------------------------
    # Multi-Objective Hypothesis Generation with Iterative Feedback Loop
    # ---------------------------------------------------------------------
    async def generate_hypothesis(
        self,
        current_best_log: Optional[ExperimentLog],
        knowledge_context: str,
        mission_params: dict,
        sim_box: dict,
        project_root: Path,
        history: list[dict],
        scoring_weights: dict,
        best_hits: Optional[List[ExperimentLog]] = None,
        failed_smiles: Optional[List[str]] = None,
        batch_size: int = 1,
        strategic_injection: str = ""
    ) -> Optional[List[Step]]:
        """
        Generates a BATCH of new hypotheses, using iterative feedback on retries
        to correct chemically invalid SMILES.
        """
        logger.info(
            "Generating new multi-objective hypotheses (Guarded single-prompt strategy)…"
        )

        base_prompt_sections = []  # We'll build the prompt piece by piece

        # --- Build the static parts of the prompt ---
        w_aff = scoring_weights["w_affinity"]
        w_qed = scoring_weights["w_qed"]
        w_sa = scoring_weights["w_sa_score"]

        history_summary_lines: list[str] = [
            ("Cycle {cycle} – SMILES: {smiles} – Score: {score:.3f}").format(
                cycle=h.get("cycle"),
                smiles=h.get("smiles_tested"),
                score=h.get("composite_score", 0.0),
            )
            for h in history[-5:]
        ]
        history_summary = "\n".join(history_summary_lines) if history_summary_lines else "No prior cycles."

        champion_section = (
            f"""
**Current Champion Molecule (for in silico optimization):**
- SMILES: `{current_best_log.smiles}`
- **Composite Score to Beat: {current_best_log.composite_score:.3f}**"""
            if current_best_log
            else """
**Current Champion Molecule:**
- None. This is the first discovery cycle. The goal is to establish the first viable lead candidate."""
        )

        best_hits_section = ""
        if best_hits:
            hits_lines = [f"- SMILES: `{hit.smiles}` | Composite Score: {hit.composite_score:.3f}" for hit in best_hits[:2]]
            best_hits_section = "\n**Best Hits from Prior Cycles:**\n" + "\n".join(hits_lines)

        negative_section = ""
        if failed_smiles:
            neg_lines = [f"- `{sm}`" for sm in set(failed_smiles) if sm]
            negative_section = (
                "\n**CRITICAL CONSTRAINT: Do NOT propose any of the following SMILES strings, "
                "as they have already been determined to be chemically invalid:**\n"
                + "\n".join(neg_lines)
            )

        consultant_section = ""
        if strategic_injection:
            consultant_section = f"""
**Expert Consultant's Strategic Blueprints for this Cycle:**
The following strategies, core scaffolds, and key fragments have been provided.
---
{strategic_injection}
---
**YOUR TASK IS MODIFIED:** Your primary goal is to **chemically connect the provided Core and Fragment SMILES strings** to create a single, valid molecule that implements the consultant's strategy. You may add small, simple linkers if necessary, but your main job is to correctly merge the two provided pieces.
"""

        target_name = mission_params.get('pdb_id', 'the protein target')

        # Assemble the base prompt
        base_prompt = f"""
**SAFETY PREAMBLE...** (omitted for brevity, but it's the same as before)
---
**SYSTEM PREAMBLE:**
You are an expert medicinal chemist AI. Your role is to propose hypothetical molecules for virtual evaluation.

**MISSION CONTEXT:**
The simulation's goal is to design novel molecular structures to discover a potent inhibitor for {target_name}, optimizing for a multi-objective composite score.

**Scoring Function:**
Composite Score = ({w_aff:.1f} × Binding Affinity) + ({w_qed:.1f} × QED) + ({w_sa:.1f} × SA_Score)

{champion_section}
{best_hits_section}
{consultant_section}
**Relevant Scientific Literature (for background context):**
{knowledge_context}
**Previous Simulation Cycles:**
{history_summary}
{negative_section}

**TASK:**
Based on all the provided data, propose a BATCH of {batch_size} diverse and chemically valid molecules for the next simulation cycle. Your goal is to find structures with a higher predicted composite score.
Respond ONLY with a single JSON object in the following format:
{{"candidates": [
    {{"reasoning": "...", "new_smiles": "..."}},
    {{"reasoning": "...", "new_smiles": "..."}}
]}}
"""

        # --- NEW ITERATIVE FEEDBACK LOOP ---
        max_retries = 10
        feedback_prompt_addition = ""
        for attempt in range(1, max_retries + 1):
            logger.info("LLM attempt %d/%d", attempt, max_retries)

            # Add feedback from previous failed attempts
            final_prompt = base_prompt + feedback_prompt_addition

            try:
                model_name = self.config.get("llm_models", {}).get("hypothesis_model", "gemini-2.5-pro")
                llm_resp = await call_llm(
                    prompt=final_prompt,
                    model_name=model_name,
                    temperature=0.7,
                    max_output_tokens=65536,
                    json_schema={"type": "json_object"}
                )

                if llm_resp.get("error"):
                    logger.error("LLM returned an error: %s", llm_resp["message"])
                    feedback_prompt_addition = "\n**CORRECTION:** Your previous attempt resulted in an API error. Please try again."
                    continue

                raw = llm_resp.get("raw_text_output", "")

                if "```" in raw:
                    raw = raw.split("```json")[-1] if "```json" in raw else raw.split("```", 1)[-1]
                    raw = raw.split("```")[0]

                try:
                    llm_output = HypothesisBatch.parse_raw(raw)
                except ValidationError as e:
                    logger.warning("LLM output failed Pydantic validation: %s. Retrying…", e)
                    feedback_prompt_addition = f"\n**CORRECTION:** Your previous JSON output was malformed: {e}. Please ensure you follow the JSON schema exactly."
                    continue

                candidates = llm_output.candidates
                if not candidates or len(candidates) != batch_size:
                    logger.warning("Expected %d candidates, got %d – retrying…", batch_size, len(candidates or []))
                    feedback_prompt_addition = f"\n**CORRECTION:** You were asked to provide a batch of {batch_size} candidates, but you provided {len(candidates or [])}. Please provide the correct number."
                    continue

                final_steps: List[Step] = []
                protein_abs = project_root / mission_params["protein_file"]
                is_batch_valid = True

                for idx, cand in enumerate(candidates):
                    reasoning = cand.reasoning.strip()
                    new_smiles = cand.new_smiles.strip()

                    mol = Chem.MolFromSmiles(new_smiles, sanitize=False)  # Load without sanitizing first
                    if not new_smiles or mol is None:
                        logger.warning("Invalid SMILES syntax in candidate %d: %s", idx, new_smiles)
                        feedback_prompt_addition = f"\n**CORRECTION:** Your proposed SMILES string '{new_smiles}' has a syntax error. Please propose a new, syntactically correct SMILES."
                        if new_smiles and failed_smiles is not None:
                            failed_smiles.append(new_smiles)
                        is_batch_valid = False
                        break

                    # Now, try to sanitize to check for chemical validity (e.g., valence errors)
                    try:
                        sanitize_result = SanitizeMol(mol, catchErrors=True)
                        if sanitize_result != SanitizeFlags.SANITIZE_NONE:
                            raise ValueError(f"Sanitization error: {sanitize_result.name}")
                    except Exception as e:
                        logger.warning("Invalid chemical structure in candidate %d: %s. Error: %s", idx, new_smiles, e)
                        feedback_prompt_addition = f"\n**CORRECTION:** Your proposed molecule '{new_smiles}' is chemically impossible (e.g., an atom has too many bonds). Please correct the structure to be valid."
                        if failed_smiles is not None:
                            failed_smiles.append(new_smiles)
                        is_batch_valid = False
                        break

                    final_steps.append(
                        Step(
                            id=0,
                            action=f"Test hypothesis: {reasoning}",
                            relevant_file_paths=[str(protein_abs)],
                            target_file=str(protein_abs),
                            simulation_parameters={"ligand_smiles": new_smiles, **sim_box},
                        )
                    )

                if not is_batch_valid:
                    continue  # This will trigger the next attempt in the loop with the new feedback

                return final_steps  # Success!

            except Exception as exc:
                logger.error("LLM call or parsing failed (attempt %d): %s", attempt, exc)
                feedback_prompt_addition = "\n**CORRECTION:** Your previous attempt resulted in a system error. Please try again."
                continue

        logger.error("Failed to generate a valid batch of hypotheses after multiple attempts.")
        return None
