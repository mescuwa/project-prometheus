# prometheus/tools/smina_adapter.py
import subprocess
import re
from pathlib import Path
import logging
import tempfile  # Using a temporary directory for each run

logger = logging.getLogger(__name__)

class SminaAdapter:
    """A wrapper for the Smina command-line tool (a Vina fork)."""

    def __init__(self, smina_executable_path: str):
        """Initializes the adapter with the path to the Smina executable."""
        self.smina_path = smina_executable_path
        self.babel_path = "obabel"  # Assumes obabel is in the conda environment's PATH

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_command(self, command: list, cwd: Path) -> tuple[bool, str, str]:
        """Run an external command and return (success, stdout, stderr)."""
        try:
            logger.info(f"Running command: {' '.join(command)} (cwd={cwd})")
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
                cwd=cwd,
            )
            if process.returncode == 0:
                return True, process.stdout, process.stderr
            logger.error(f"Command failed (rc={process.returncode}): {' '.join(command)}")
            logger.error(f"STDOUT: {process.stdout.strip()}")
            logger.error(f"STDERR: {process.stderr.strip()}")
            return False, process.stdout, process.stderr
        except Exception as e:
            logger.error(f"Exception running command {' '.join(command)}: {e}", exc_info=True)
            return False, "", str(e)

    # ------------------------------------------------------------------
    # Preparation steps
    # ------------------------------------------------------------------

    def _prepare_receptor(self, protein_pdb_file: Path, output_dir: Path) -> Path | None:
        """Convert a protein PDB file to PDBQT, removing non-receptor atoms."""
        logger.info(f"Preparing receptor from: {protein_pdb_file}")
        output_receptor = output_dir / f"{protein_pdb_file.stem}.pdbqt"
        command = [
            self.babel_path,
            "-i",
            "pdb",
            str(protein_pdb_file),
            "-o",
            "pdbqt",
            "-O",
            str(output_receptor),
            "-xr",
        ]
        success, _, stderr = self._run_command(command, output_dir)
        if not success:
            logger.error(f"Failed to prepare receptor: {stderr}")
            return None
        logger.info(f"Receptor prepared: {output_receptor}")
        return output_receptor

    def _prepare_ligand(self, ligand_smiles: str, output_dir: Path) -> Path | None:
        """Convert a ligand SMILES string to PDBQT with 3D coordinates and hydrogens."""
        logger.info(f"Preparing ligand from SMILES: {ligand_smiles[:30]}...")
        output_ligand = output_dir / "ligand.pdbqt"
        command = [
            self.babel_path,
            "-i",
            "smi",
            f"-:{ligand_smiles}",
            "-o",
            "pdbqt",
            "-O",
            str(output_ligand),
            "--gen3d",
            "-p",
            "7.4",
        ]
        success, _, stderr = self._run_command(command, output_dir)
        if not success:
            logger.error(f"Failed to prepare ligand: {stderr}")
            return None
        logger.info(f"Ligand prepared: {output_ligand}")
        return output_ligand

    # ------------------------------------------------------------------
    # Config handling
    # ------------------------------------------------------------------

    def _write_config_file(
        self,
        receptor_pdbqt: Path,
        ligand_pdbqt: Path,
        center: dict,
        box_size: dict,
        output_dir: Path,
    ) -> Path:
        """Create a conf.txt file for Smina/Vina."""
        cfg = output_dir / "conf.txt"
        content = f"""
receptor = {receptor_pdbqt.name}
ligand = {ligand_pdbqt.name}

center_x = {center['x']}
center_y = {center['y']}
center_z = {center['z']}

size_x = {box_size['x']}
size_y = {box_size['y']}
size_z = {box_size['z']}

exhaustiveness = 16
"""
        cfg.write_text(content.strip() + "\n")
        logger.info(f"Config file written: {cfg}")
        return cfg

    # ------------------------------------------------------------------
    # Execution & parsing
    # ------------------------------------------------------------------

    def _run_smina(self, config_file: Path, output_dir: Path) -> dict:
        """Execute Smina docking."""
        log_path = output_dir / "smina_run.log"
        out_struct = output_dir / "docked_ligand.pdbqt"
        cmd = [
            self.smina_path,
            "--config",
            config_file.name,
            "--log",
            log_path.name,
            "--out",
            out_struct.name,
        ]
        success, _, stderr = self._run_command(cmd, output_dir)
        if not success:
            return {"status": "ERROR", "log_file": log_path, "error": stderr}
        return {
            "status": "SUCCESS",
            "log_file": log_path,
            "output_structure_file": out_struct,
            "error": None,
        }

    def _parse_outputs(self, log_file: Path) -> dict:
        """Parses the Smina log file to extract the best binding affinity."""
        try:
            if not log_file.exists():
                return {"status": "ERROR", "error": "Log file not found for parsing."}

            log_content = log_file.read_text()

            # Step 1: Locate the start of the results table (line of dashes)
            table_header_pattern = r"^\s*-----\+"
            header_match = re.search(table_header_pattern, log_content, re.MULTILINE)

            if not header_match:
                logger.warning("Could not find the results table header in the log file.")
                return {"status": "ERROR", "error": "Could not parse score: results table not found."}

            # Step 2: Search only in the section after the header for mode 1
            results_section = log_content[header_match.end():]
            score_pattern = r"^\s*1\s+(-?\d+\.\d+)"
            score_match = re.search(score_pattern, results_section, re.MULTILINE)

            if score_match:
                best_score = float(score_match.group(1))
                logger.info(f"Successfully parsed binding affinity: {best_score} kcal/mol")
                return {"status": "SUCCESS", "binding_affinity_kcal_mol": best_score}

            logger.warning("Found results table, but could not parse score for mode 1.")
            return {"status": "ERROR", "error": "Could not parse score from results table."}

        except Exception as e:
            logger.error(f"Error parsing log file {log_file}: {e}", exc_info=True)
            return {"status": "ERROR", "error": str(e)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dock(
        self,
        protein_pdb_file: Path,
        ligand_smiles: str,
        center: dict,
        box_size: dict,
    ) -> dict:
        """
        Run the full docking workflow and return a structured result dict.
        Uses a temporary directory to keep runs clean.
        """
        with tempfile.TemporaryDirectory(prefix="prometheus_run_") as temp_dir_str:
            out_dir = Path(temp_dir_str)
            logger.info(f"--- Starting Docking Run in Temp Directory: {out_dir} ---")

            # 1. Prepare structures
            receptor = self._prepare_receptor(protein_pdb_file, out_dir)
            if receptor is None:
                return {"status": "ERROR", "stage": "prepare_receptor"}

            ligand = self._prepare_ligand(ligand_smiles, out_dir)
            if ligand is None:
                return {"status": "ERROR", "stage": "prepare_ligand"}

            # 2. Create config file
            cfg = self._write_config_file(receptor, ligand, center, box_size, out_dir)

            # 3. Run Smina
            run_res = self._run_smina(cfg, out_dir)
            if run_res["status"] == "ERROR":
                run_res["stage"] = "run_smina"
                return run_res

            # 4. Parse output log
            parse_res = self._parse_outputs(run_res["log_file"])
            if parse_res["status"] == "ERROR":
                return {
                    "status": "ERROR",
                    "stage": "parse_outputs",
                    "message": parse_res["error"],
                    "log_file_path": str(run_res["log_file"]),
                }

            # 5. Success
            logger.info("--- Docking Run Completed Successfully ---")
            return {
                "status": "SUCCESS",
                "binding_affinity_kcal_mol": parse_res["binding_affinity_kcal_mol"],
                "docked_ligand_file": run_res["output_structure_file"],
            } 