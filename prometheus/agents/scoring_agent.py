# prometheus/agents/scoring_agent.py
"""Agent responsible for computing molecular property scores (QED, SA, LogP).

This agent contains a robust, dynamic import for RDKit's Synthetic
Accessibility (SA) score module, which can be in a non-standard location
depending on the installation method.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Robust SA Scorer Import
# -----------------------------------------------------------------------------
# RDKit's SA Score calculator often resides in the "Contrib" directory, which
# may not be on the default PYTHONPATH. We dynamically locate that directory and
# add it so the import succeeds on as many systems as possible.
try:
    # Locate RDKit's installation directory and its Contrib folder
    rdkit_base = os.path.dirname(rdkit.__file__)
    contrib_path = os.path.join(rdkit_base, "Contrib")
    sys.path.append(contrib_path)

    # Attempt the import once the path has been appended
    from SA_Score import sascorer  # type: ignore

    _calculate_sa_score = sascorer.calculateScore  # type: ignore[attr-defined]
    logger.info("Successfully imported SA Scorer from RDKit Contrib directory.")
except ImportError:
    logger.warning(
        "Could not import SA Scorer from RDKit Contrib. Falling back to a simple "
        "heavy-atom heuristic for Synthetic Accessibility. For accurate SA scores, "
        "ensure RDKit Contrib scripts are installed and accessible."
    )

    # Fallback heuristic: use heavy-atom count as a rough proxy
    def _calculate_sa_score(mol: Chem.Mol) -> float:  # type: ignore[override]
        heavy_atoms = mol.GetNumHeavyAtoms()
        # SA ≈ 1 (easy) .. 10 (hard). Penalise very large molecules.
        return min(10.0, max(1.0, 1.0 + (heavy_atoms - 30) * 0.2))

# -----------------------------------------------------------------------------
# Scoring Agent
# -----------------------------------------------------------------------------

class ScoringAgent:
    """Calculates drug-likeness and synthetic accessibility metrics."""

    def __init__(self, config: dict):
        logger.info("ScoringAgent initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_scores(self, smiles: str) -> Optional[Dict[str, float]]:
        """Return a dictionary with QED, SA score, and LogP for *smiles*.

        Returns ``None`` when the SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error("ScoringAgent: Invalid SMILES provided: %s", smiles)
            return None

        try:
            qed_val = float(QED.qed(mol))
            sa_val = float(_calculate_sa_score(mol))
            logp_val = float(Descriptors.MolLogP(mol))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "ScoringAgent: Error computing scores for %s: %s", smiles, exc, exc_info=True
            )
            return None

        logger.info(
            "Computed scores for %s…  QED=%.3f  SA=%.3f  LogP=%.3f",
            smiles[:30],
            qed_val,
            sa_val,
            logp_val,
        )
        return {"qed": qed_val, "sa_score": sa_val, "logp": logp_val}
