# prometheus/agents/hypothesis_agent.py
import sys
import logging
from pathlib import Path
import asyncio
import json
from rdkit import Chem, RDLogger
from typing import Optional, List

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# --- Imports ---
import lancedb
from sentence_transformers import SentenceTransformer
from prometheus.models import Step, ExperimentLog
from prometheus.llm_utils import call_gemini

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
    def _search_knowledge_base(self, query: str, limit: int = 3) -> str:
        """Return concatenated literature snippets relevant to *query*."""
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
    # Multi-Objective Hypothesis Generation (Guarded, Single-Prompt)
    # ---------------------------------------------------------------------
    async def generate_hypothesis(
        self,
        current_best_log: ExperimentLog,
        knowledge_context: str,
        mission_params: dict,
        sim_box: dict,
        project_root: Path,
        history: list[dict],
        scoring_weights: dict,
        batch_size: int = 1,
    ) -> Optional[List[Step]]:
        """
        Generates a BATCH of new hypotheses using a single, context-rich but
        carefully "guarded" prompt to align with AI safety policies.
        """
        logger.info(
            "Generating new multi-objective hypotheses (Guarded single-prompt strategy)…"
        )

        w_aff = scoring_weights["w_affinity"]
        w_qed = scoring_weights["w_qed"]
        w_sa = scoring_weights["w_sa_score"]

        # ------------------------------------------------------------------
        # 1. Format experiment history for inclusion in the prompt
        # ------------------------------------------------------------------
        history_summary_lines: list[str] = []
        for h in history[-5:]:  # only the last 5 cycles to keep the prompt concise
            history_summary_lines.append(
                (
                    "Cycle {cycle} – SMILES: {smiles} – Score: {score:.3f}"
                ).format(
                    cycle=h.get("cycle"),
                    smiles=h.get("smiles_tested"),
                    score=h.get("composite_score", 0.0),
                )
            )
        history_summary = "\n".join(history_summary_lines) if history_summary_lines else "No prior cycles."

        # ------------------------------------------------------------------
        # 2. Build the guarded mega-prompt
        # ------------------------------------------------------------------
        prompt = f"""
**SYSTEM PREAMBLE:**
You are an expert medicinal chemist AI operating as part of a computational research simulation called Prometheus. Your role is to propose hypothetical molecules for *in silico* (virtual) evaluation. You must not provide medical advice or instructions for real-world chemical synthesis. All of your outputs are for a simulated, theoretical context.

**MISSION CONTEXT:**
The simulation's goal is to design novel molecular structures to optimize a multi-objective composite score based on predicted properties against the EGFR T790M protein target.

**Scoring Function:**
Composite Score = ({w_aff:.1f} × Binding Affinity) + ({w_qed:.1f} × QED) + ({w_sa:.1f} × SA_Score)

**Current Champion Molecule (for in silico optimization):**
- SMILES: `{current_best_log.smiles}`
- **Composite Score to Beat: {current_best_log.composite_score:.3f}**
- --- Score Breakdown ---
  - Binding Affinity: {current_best_log.average_binding_affinity:.3f}
  - QED: {current_best_log.qed:.3f}
  - SA Score: {current_best_log.sa_score:.3f}

**Relevant Scientific Literature (for background context):**
{knowledge_context}

**Previous Simulation Cycles:**
{history_summary}

**TASK:**
Based on all the provided data, propose a BATCH of {batch_size} diverse and chemically valid molecules for the next simulation cycle. Your goal is to find structures with a higher predicted composite score.

For each proposed molecule, provide your scientific reasoning and the canonical SMILES string.

Respond ONLY with a single JSON object in the following format:
{{"candidates": [
    {{"reasoning": "...", "new_smiles": "..."}},
    {{"reasoning": "...", "new_smiles": "..."}}
]}}
"""

        # ------------------------------------------------------------------
        # 3. Call the LLM with robust retries & validation
        # ------------------------------------------------------------------
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            logger.info("LLM attempt %d/%d", attempt, max_retries)
            try:
                llm_resp = await call_gemini(
                    prompt=prompt,
                    model_name="gemini-2.5-pro",
                    temperature=0.7,
                    max_output_tokens=65536,
                )

                # Handle explicit errors from the helper util
                if llm_resp.get("error"):
                    logger.error("LLM returned an error: %s", llm_resp["message"])
                    continue

                raw = llm_resp.get("raw_text_output", "")
                if "```" in raw:
                    # Strip markdown code-fences if the model adds them
                    raw = (
                        raw.split("```json")[-1]
                        if "```json" in raw
                        else raw.split("```", 1)[1]
                    )
                    raw = raw.split("```", 1)[0]

                parsed = json.loads(raw)
                candidates = parsed.get("candidates", [])
                if len(candidates) != batch_size:
                    logger.warning(
                        "Expected %d candidates, got %d – retrying…", batch_size, len(candidates)
                    )
                    continue

                final_steps: List[Step] = []
                protein_abs = project_root / mission_params["protein_file"]

                valid_batch = True
                for idx, cand in enumerate(candidates):
                    reasoning = cand.get("reasoning", "").strip()
                    new_smiles = cand.get("new_smiles", "").strip()

                    if not new_smiles or Chem.MolFromSmiles(new_smiles) is None:
                        logger.warning(
                            "Invalid SMILES in candidate %d: %s – retrying…", idx, new_smiles
                        )
                        valid_batch = False
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

                if not valid_batch:
                    logger.info("One or more candidates invalid, retrying…")
                    continue

                # All good!
                return final_steps

            except Exception as exc:
                logger.error("LLM call or parsing failed (attempt %d): %s", attempt, exc)
                continue

        logger.error(
            "Failed to generate a valid batch of hypotheses after multiple attempts."
        )
        return None
