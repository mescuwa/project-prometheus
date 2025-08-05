# prometheus/agents/md_validator_agent.py
import logging
from pathlib import Path
import tempfile
from typing import Dict

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer

logger = logging.getLogger(__name__)

class MDValidatorAgent:
    """Run MD simulation for additional validation of docking results."""
    def __init__(self, config: Dict):
        self._cfg = config["md_simulation"]

        # Determine simulation length based on quick_test flag
        self.is_quick_test = self._cfg.get("quick_test", False)
        if self.is_quick_test:
            self.simulation_steps = self._cfg.get("quick_test_steps", 500)
            logger.warning(
                "MD VALIDATOR IN QUICK TEST MODE. Simulations will run for only %d steps.",
                self.simulation_steps,
            )
        else:
            self.simulation_steps = self._cfg["simulation_steps"]

        logger.info("MDValidatorAgent initialised with %s steps", self.simulation_steps)

    def _combine_protein_ligand(self, protein_pdb: Path, ligand_pdbqt: Path, out_pdb: Path) -> None:
        prot_lines = [ln for ln in protein_pdb.read_text().splitlines() if ln.startswith("ATOM")]
        lig_lines = [ln for ln in ligand_pdbqt.read_text().splitlines() if ln.startswith("HETATM")]
        out_pdb.write_text("\n".join(prot_lines + lig_lines + ["TER", "END"]))

    def run_simulation(self, protein_pdb_file: Path, docked_ligand_file: Path) -> Dict:
        with tempfile.TemporaryDirectory(prefix="md_run_") as tmp:
            tmp_dir = Path(tmp)
            logger.info("Staging MD run in %s", tmp_dir)

            try:
                complex_pdb = tmp_dir / "complex.pdb"
                self._combine_protein_ligand(protein_pdb_file, docked_ligand_file, complex_pdb)

                fixer = PDBFixer(filename=str(complex_pdb))
                fixer.findMissingResidues()
                fixer.findNonstandardResidues()
                fixer.replaceNonstandardResidues()
                fixer.removeHeterogens(True)
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(7.0)

                fixed_pdb = tmp_dir / "fixed.pdb"
                with fixed_pdb.open("w") as fh:
                    app.PDBFile.writeFile(fixer.topology, fixer.positions, fh)

                system_gen = SystemGenerator(
                    forcefields=[self._cfg["protein_force_field"], self._cfg["protein_water_model"]],
                    small_molecule_forcefield=self._cfg["ligand_force_field"],
                    periodic_forcefield_kwargs={"nonbondedMethod": app.PME},
                    forcefield_kwargs={
                        "constraints": app.HBonds,
                        "rigidWater": True,
                        "removeCMMotion": True,
                    },
                )

                pdb = app.PDBFile(str(fixed_pdb))
                modeller = app.Modeller(pdb.topology, pdb.positions)
                modeller.addSolvent(system_gen.forcefield, padding=1.0 * unit.nanometer)
                system = system_gen.create_system(modeller.topology)

                integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
                
                platform = None
                try:
                    plugins_dir = mm.Platform.getDefaultPluginsDirectory()
                    if plugins_dir:
                        mm.Platform.loadPluginsFromDirectory(plugins_dir)
                        logger.info(f"Loaded OpenMM plugins from: {plugins_dir}")
                except Exception as e:
                    logger.warning(f"Could not load OpenMM plugins: {e}")

                for name in ["Metal", "HIP", "CUDA", "OpenCL"]:
                    try:
                        test_platform = mm.Platform.getPlatformByName(name)
                        platform = test_platform
                        logger.info(f"Successfully found and selected '{name}' platform for MD.")
                        break
                    except mm.OpenMMException:
                        continue

                if platform is None:
                    platform = mm.Platform.getPlatformByName("CPU")
                    logger.warning("No accelerated platform found. Falling back to slower CPU platform.")

                sim = app.Simulation(modeller.topology, system, integrator, platform)
                sim.context.setPositions(modeller.positions)

                logger.info("Minimising energy …")
                sim.minimizeEnergy()

                sim.reporters.append(
                    app.StateDataReporter(
                        str(tmp_dir / "md.log"),
                        min(100, self.simulation_steps),
                        step=True,
                        potentialEnergy=True,
                        temperature=True,
                        progress=True,
                        remainingTime=True,
                        totalSteps=self.simulation_steps,
                        separator="\t",
                    )
                )
                sim.reporters.append(app.DCDReporter(str(tmp_dir / "traj.dcd"), min(100, self.simulation_steps)))

                logger.info("Running production MD for %d steps", self.simulation_steps)
                sim.step(self.simulation_steps)

                state = sim.context.getState(getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                logger.info("MD completed. Final potential energy: %.2f kJ/mol", energy)

                return {"status": "SUCCESS", "final_potential_energy_kj_mol": energy, "message": "MD simulation completed."}
            except Exception as exc:
                logger.error("MD simulation failed: %s", exc, exc_info=True)
                return {"status": "ERROR", "message": str(exc)}