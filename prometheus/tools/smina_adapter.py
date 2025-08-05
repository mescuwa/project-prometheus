# prometheus/tools/smina_adapter.py
"""Adapter for running multiple Smina (AutoDock Vina fork) docking simulations
with averaging over repeated runs for more robust scores.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import shutil

logger = logging.getLogger(__name__)


class SminaAdapter:
    """Wrapper around the Smina command-line interface."""

    def __init__(self, config: dict):
        smina_config = config["vina_tool"]
        self.smina_path = smina_config["executable_path"]
        # Open Babel must be in PATH – we assume the conda env took care of that.
        self.babel_path = "obabel"

    # ---------------------------------------------------------------------
    # Generic command helper
    # ---------------------------------------------------------------------

    def _run_command(self, command: list[str], cwd: Path) -> tuple[bool, str, str]:
        """Run *command* in *cwd*.  Returns (success, stdout, stderr)."""
        try:
            logger.debug("Running command: %s", " ".join(command))
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd,
            )
            if proc.returncode == 0:
                return True, proc.stdout, proc.stderr
            logger.error("Command failed (rc=%s): %s", proc.returncode, " ".join(command))
            logger.debug("STDOUT: %s", proc.stdout)
            logger.debug("STDERR: %s", proc.stderr)
            return False, proc.stdout, proc.stderr
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Exception running command: %s", exc, exc_info=True)
            return False, "", str(exc)

    # ---------------------------------------------------------------------
    # Preparation helpers
    # ---------------------------------------------------------------------

    def _prepare_receptor(self, protein_pdb_file: Path, out_dir: Path) -> Path | None:
        output = out_dir / f"{protein_pdb_file.stem}.pdbqt"
        cmd = [
            self.babel_path,
            "-i",
            "pdb",
            str(protein_pdb_file),
            "-o",
            "pdbqt",
            "-O",
            str(output),
            "-xr",
        ]
        ok, _, err = self._run_command(cmd, out_dir)
        if not ok:
            logger.error("Failed to prepare receptor: %s", err)
            return None
        return output

    def _prepare_ligand(self, smiles: str, out_dir: Path) -> Path | None:
        output = out_dir / "ligand.pdbqt"
        cmd = [
            self.babel_path,
            "-i",
            "smi",
            f"-:{smiles}",
            "-o",
            "pdbqt",
            "-O",
            str(output),
            "--gen3d",
            "-p",
            "7.4",
        ]
        ok, _, err = self._run_command(cmd, out_dir)
        if not ok:
            logger.error("Failed to prepare ligand: %s", err)
            return None
        return output

    # ---------------------------------------------------------------------
    # Config writer & Smina execution
    # ---------------------------------------------------------------------

    def _write_config(self, receptor: Path, ligand: Path, center: dict, box: dict, out_dir: Path) -> Path:
        cfg_path = out_dir / "conf.txt"
        cfg_path.write_text(
            (
                f"receptor = {receptor.name}\n"
                f"ligand = {ligand.name}\n\n"
                f"center_x = {center['x']}\ncenter_y = {center['y']}\ncenter_z = {center['z']}\n\n"
                f"size_x = {box['x']}\nsize_y = {box['y']}\nsize_z = {box['z']}\n\n"
                "exhaustiveness = 16\n"
            )
        )
        return cfg_path

    def _run_smina(self, cfg: Path, out_dir: Path) -> dict:
        log_path = out_dir / "smina.log"
        out_struct = out_dir / "docked_ligand.pdbqt"
        cmd = [self.smina_path, "--config", cfg.name, "--log", log_path.name, "--out", out_struct.name]
        ok, _, err = self._run_command(cmd, out_dir)
        return {
            "status": "SUCCESS" if ok else "ERROR",
            "log": log_path,
            "out_struct": out_struct,
            "error": err if not ok else None,
        }

    # ---------------------------------------------------------------------
    # Parsing helper
    # ---------------------------------------------------------------------

    _TABLE_HEADER = re.compile(r"^\s*-----\+", re.MULTILINE)
    _SCORE_LINE = re.compile(r"^\s*1\s+(-?\d+\.\d+)", re.MULTILINE)

    def _parse_score(self, log_file: Path) -> float | None:
        if not log_file.exists():
            return None
        content = log_file.read_text()
        header = self._TABLE_HEADER.search(content)
        if not header:
            return None
        match = self._SCORE_LINE.search(content[header.end() :])
        if match:
            return float(match.group(1))
        return None

    # ---------------------------------------------------------------------
    # Single docking run (private)
    # ---------------------------------------------------------------------

    def _run_single(self, pdb_file: Path, smiles: str, center: dict, box: dict) -> dict:
        """Run one docking and return result dict with status & score."""
        with tempfile.TemporaryDirectory(prefix="prometheus_dock_") as tmp:
            tmp_dir = Path(tmp)
            receptor = self._prepare_receptor(pdb_file, tmp_dir)
            if receptor is None:
                return {"status": "ERROR", "stage": "prepare_receptor"}

            ligand = self._prepare_ligand(smiles, tmp_dir)
            if ligand is None:
                return {"status": "ERROR", "stage": "prepare_ligand"}

            cfg = self._write_config(receptor, ligand, center, box, tmp_dir)
            run_res = self._run_smina(cfg, tmp_dir)
            if run_res["status"] == "ERROR":
                run_res["stage"] = "run_smina"
                return run_res

            score = self._parse_score(run_res["log"])
            if score is None:
                return {"status": "ERROR", "stage": "parse_score"}

            # Persist docked ligand file so that downstream MD step can access it
            try:
                persistent_path = Path(tempfile.gettempdir()) / f"{smiles[:15].replace('/', '_')}_{pdb_file.stem}_docked.pdbqt"
                shutil.copy(run_res["out_struct"], persistent_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Could not copy docked structure to persistent path: %s", exc)
                persistent_path = run_res["out_struct"]

            return {
                "status": "SUCCESS",
                "score": score,
                "docked_ligand_file": persistent_path,
            }

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def dock(
        self,
        protein_pdb_file: Path,
        ligand_smiles: str,
        center: dict,
        box_size: dict,
        num_runs: int = 1,
    ) -> dict:
        """Run *num_runs* docking simulations and return averaged results."""
        logger.info("Starting docking of molecule (runs=%s)…", num_runs)

        scores: list[float] = []
        first_docked_path: Path | None = None
        for run_idx in range(1, num_runs + 1):
            logger.info("Docking run %d/%d", run_idx, num_runs)
            res = self._run_single(protein_pdb_file, ligand_smiles, center, box_size)
            if res["status"] == "SUCCESS":
                scores.append(res["score"])
                if first_docked_path is None and res.get("docked_ligand_file"):
                    first_docked_path = res["docked_ligand_file"]
                logger.info("Run %d score: %.3f kcal/mol", run_idx, res["score"])
            else:
                logger.error("Docking run %d failed (stage=%s)", run_idx, res.get("stage"))

        if not scores:
            return {"status": "ERROR", "stage": "all_runs_failed"}

        avg = float(np.mean(scores))
        std_dev = float(np.std(scores))
        logger.info("Average binding affinity: %.3f ± %.3f kcal/mol", avg, std_dev)

        return {
            "status": "SUCCESS",
            "average_binding_affinity": avg,
            "std_dev_binding_affinity": std_dev,
            "all_scores": scores,
            "docked_ligand_file": first_docked_path,
        }
