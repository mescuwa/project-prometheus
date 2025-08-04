# prometheus/agents/hypothesis_agent.py
import sys
import logging
from pathlib import Path
import asyncio  # Added asyncio import for bridging async LLM call
import json  # Added json import for parsing LLM response
from rdkit import Chem  # Added for chemical SMILES validation

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# --- Imports ---
import lancedb
from sentence_transformers import SentenceTransformer

from prometheus.models import Step
from prometheus.llm_utils import call_gemini

logger = logging.getLogger(__name__)

# --- Constants ---
DB_PATH = project_root / "data" / "knowledge_base.lancedb"
TABLE_NAME = "erlotinib_research"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


class HypothesisAgent:
    """An agent that formulates scientific hypotheses by querying a knowledge base and using an LLM."""

    def __init__(self):
        logger.info("Initializing HypothesisAgent…")
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Connecting to knowledge base…")
            self.db = lancedb.connect(DB_PATH)
            self.table = self.db.open_table(TABLE_NAME)
            logger.info("HypothesisAgent initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize HypothesisAgent: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _search_knowledge_base(self, query: str, limit: int = 3) -> str:
        """Returns concatenated text snippets relevant to the query from the vector DB."""
        logger.info(f"Searching knowledge base with query: '{query}'")
        query_vector = self.embedding_model.encode(query)
        try:
            results = self.table.search(query_vector).limit(limit).to_df()
            if results.empty:
                logger.warning("No relevant knowledge found for the query.")
                return "No relevant literature found."

            context = "\n\n---\n\n".join(
                f"Source: {row['source']}\n\nContent: {row['content']}" for _, row in results.iterrows()
            )
            logger.info(f"Found {len(results)} relevant knowledge snippets.")
            return context
        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}", exc_info=True)
            return "Error searching knowledge base."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_hypothesis(self, baseline_result: dict, mission_params: dict, sim_box: dict, project_root: Path) -> Step | None:
        """Generate a new hypothesis based on baseline docking results."""
        logger.info("Generating new hypothesis based on baseline result…")

        query = (
            f"Erlotinib binding to EGFR active site resulted in a score of {baseline_result.get('binding_affinity_kcal_mol')} kcal/mol. "
            f"What are known resistance mechanisms or potential improvements for quinazoline inhibitors like Erlotinib, "
            f"specifically regarding the T790M mutation?"
        )
        knowledge_context = self._search_knowledge_base(query)

        prompt = f"""
You are an expert medicinal chemist AI. Your goal is to propose a novel modification to the drug Erlotinib to improve its binding affinity to EGFR, especially in the context of T790M resistance.

**Baseline Experiment Result:**
- Molecule: Erlotinib
- Canonical SMILES: {mission_params['ligand_smiles']}
- Predicted Binding Affinity to EGFR (1M17): {baseline_result.get('binding_affinity_kcal_mol')} kcal/mol

**Relevant Scientific Literature:**
{knowledge_context}

**Your Task:**
Based on the baseline result and the literature, propose a single, chemically valid modification to Erlotinib. Your goal is to improve the binding affinity (achieve a more negative score).

1. **Reasoning:** Briefly explain your proposed modification and why it might be effective.
2. **Output:** Provide the canonical SMILES string for your new, modified molecule.

Respond ONLY with a JSON object in the following format:
{{"reasoning": "...", "new_smiles": "..."}}
"""

        # Bridge sync/async by awaiting inside asyncio.run()
        logger.info("Calling LLM to generate novel hypothesis…")
        try:
            llm_response = asyncio.run(
                call_gemini(prompt=prompt, model_name="gemini-2.5-pro", temperature=0.5)
            )
        except Exception as e:
            logger.error(f"Async LLM call failed: {e}", exc_info=True)
            return None

        raw_output = ""
        if isinstance(llm_response, dict):
            raw_output = llm_response.get("raw_text_output", "")
        else:
            raw_output = str(llm_response)

        if not raw_output or ("error" in raw_output.lower() and len(raw_output) < 120):
            logger.error(f"LLM returned error/empty output: {raw_output}")
            return None

        # Handle possible markdown fencing
        if "```" in raw_output:
            try:
                raw_output = raw_output.split("```json")[-1]
                raw_output = raw_output.split("```", 1)[0]
            except Exception:
                pass

        try:
            parsed = json.loads(raw_output)
            reasoning = parsed["reasoning"]
            new_smiles = parsed["new_smiles"]

            # Validate that the proposed SMILES is chemically valid
            mol = Chem.MolFromSmiles(new_smiles)
            if mol is None:
                logger.error(f"LLM hallucinated an invalid SMILES string: {new_smiles}")
                return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            logger.error(f"Raw output: {raw_output}")
            return None

        logger.info(f"LLM reasoning: {reasoning}")
        logger.info(f"New SMILES proposed: {new_smiles}")

        protein_abs_path = project_root / mission_params["protein_file"]
        return Step(
            id=2,
            action=f"Test new hypothesis: {reasoning}",
            details=None,
            relevant_file_paths=[str(protein_abs_path)],
            target_file=str(protein_abs_path),
            simulation_parameters={
                "ligand_smiles": new_smiles,
                "center_x": sim_box["center_x"],
                "center_y": sim_box["center_y"],
                "center_z": sim_box["center_z"],
                "size_x": sim_box["size_x"],
                "size_y": sim_box["size_y"],
                "size_z": sim_box["size_z"],
            },
        ) 