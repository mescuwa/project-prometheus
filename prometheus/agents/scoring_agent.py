# prometheus/agents/scoring_agent.py
"""Agent responsible for computing molecular property scores (QED, SA, LogP)
and interaction fingerprints.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List

import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import prolif
import MDAnalysis as mda

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Robust SA Scorer Import
# -----------------------------------------------------------------------------
try:
    rdkit_base = os.path.dirname(rdkit.__file__)
    contrib_path = os.path.join(rdkit_base, "Contrib")
    sys.path.append(contrib_path)

    from SA_Score import sascorer  # type: ignore

    _calculate_sa_score = sascorer.calculateScore  # type: ignore[attr-defined]
    logger.info("Successfully imported SA Scorer from RDKit Contrib directory.")
except ImportError:
    logger.warning(
        "Could not import SA Scorer from RDKit Contrib. Falling back to a heavy-atom heuristic."
    )

    def _calculate_sa_score(mol: Chem.Mol) -> float:  # type: ignore[override]
        heavy_atoms = mol.GetNumHeavyAtoms()
        return min(10.0, max(1.0, 1.0 + (heavy_atoms - 30) * 0.2))


# -----------------------------------------------------------------------------
# Scoring Agent
# -----------------------------------------------------------------------------
class ScoringAgent:
    """Calculates drug-likeness, synthetic accessibility, and interaction fingerprints."""

    def __init__(self, config: dict):
        logger.info("ScoringAgent initialised.")
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _calculate_ifp_score(
        self,
        protein_file: Path,
        docked_ligand_file: Path,
        key_interactions: List[str],
    ) -> int:
        """
        Calculate a simple score based on the presence of user-defined key interactions
        between *docked_ligand_file* and *protein_file*.

        The score equals the number of key interactions that are present in the
        interaction fingerprint (IFP) as computed by ProLIF. If any error occurs
        (e.g. file missing, parsing failure), the method returns 0 and logs a warning.
        """
        if not docked_ligand_file.exists():
            logger.warning(
                "Docked ligand file not found for IFP analysis: %s", docked_ligand_file
            )
            return 0

        try:
            # ---------------------------
            # Load molecules via robust IO
            # ---------------------------
            # 1) Ligand: PDBQT → RDKit
            lig_rdkit = None
            try:
                lig_rdkit = Chem.MolFromPDBQTFile(str(docked_ligand_file))  # type: ignore[attr-defined]
            except AttributeError:
                # Fallback: read entire file and parse as PDB block (may work if columns are standard)
                with docked_ligand_file.open("r", encoding="utf-8", errors="ignore") as fp_in:
                    pdb_block = fp_in.read()
                lig_rdkit = Chem.MolFromPDBBlock(pdb_block, removeHs=False)

            if lig_rdkit is None:
                logger.error("RDKit failed to load PDBQT file: %s", docked_ligand_file)
                return 0

            # 2) Protein: PDB → MDAnalysis Universe (protein selection only)
            prot_universe = mda.Universe(str(protein_file))
            protein_atoms = prot_universe.select_atoms("protein")

            # Convert to ProLIF Molecule objects
            lig_mol = prolif.Molecule.from_rdkit(lig_rdkit)
            prot_mol = prolif.Molecule.from_mda(protein_atoms)

            # ---------------------------
            # Compute interaction fingerprint
            # ---------------------------
            fp = prolif.Fingerprint()
            fp.run(lig_mol, prot_mol)

            ifp_dict = fp.to_dict()
            interactions = list(ifp_dict.values())[0] if ifp_dict else {}

            score = 0
            found_interactions: list[str] = []
            for key_interaction in key_interactions:
                try:
                    interaction_type, residue_id = key_interaction.split("-", 1)
                except ValueError:
                    logger.debug("Malformed key interaction string: %s", key_interaction)
                    continue

                # ProLIF residue identifiers look like "ASP12.A"; they include chain.
                for prolif_res, residue_interactions in interactions.items():
                    if residue_id in str(prolif_res) and interaction_type in residue_interactions:
                        score += 1
                        found_interactions.append(key_interaction)
                        break  # Avoid double-counting this interaction

            if score > 0:
                logger.info("IFP Score: %d. Found key interactions: %s", score, found_interactions)
            return score
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "Failed to calculate IFP score for %s: %s", docked_ligand_file.name, exc, exc_info=True
            )
            return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate_scores(
        self,
        smiles: str,
        protein_file: Path | None = None,
        docked_ligand_file: Path | None = None,
        key_interactions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, float]]:
        """Return a dictionary with QED, SA score, LogP, and IFP score.

        If *protein_file*, *docked_ligand_file*, or *key_interactions* are not provided,
        the IFP score defaults to 0. Returns ``None`` if *smiles* cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error("ScoringAgent: Invalid SMILES provided: %s", smiles)
            return None

        # Standard property scores
        try:
            qed_val = float(QED.qed(mol))
            sa_val = float(_calculate_sa_score(mol))
            logp_val = float(Descriptors.MolLogP(mol))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "ScoringAgent: Error computing property scores for %s: %s", smiles, exc
            )
            return None

        if (
            protein_file is not None
            and docked_ligand_file is not None
            and key_interactions is not None
        ):
            ifp_score = self._calculate_ifp_score(
                protein_file, docked_ligand_file, key_interactions
            )
        else:
            ifp_score = 0

        logger.info(
            "Computed scores for %s… QED=%.3f SA=%.3f LogP=%.3f IFP=%d",
            smiles,
            qed_val,
            sa_val,
            logp_val,
            ifp_score,
        )

        return {
            "qed": qed_val,
            "sa_score": sa_val,
            "logp": logp_val,
            "ifp_score": float(ifp_score),
        }