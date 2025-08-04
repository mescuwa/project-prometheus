# scripts/run_milestone_2.py
import sys
import logging
from pathlib import Path
import tomli as toml
import json
from dotenv import load_dotenv
load_dotenv()
from typing import Optional

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Import Our Project Components ---
from prometheus.agents import HypothesisAgent, ExperimenterAgent, ValidatorAgent
from prometheus.tools.smina_adapter import SminaAdapter
from prometheus.models import Step, VerificationFailureDetails

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(
    hypothesis_step: Step,
    experimenter: ExperimenterAgent,
    validator: ValidatorAgent,
    tool: SminaAdapter
) -> tuple[dict, str, Optional[VerificationFailureDetails]]:
    """A helper function to run one complete experiment cycle."""

    logger.info(">>> Handing hypothesis to ExperimenterAgent...")
    experiment_result = experimenter.execute(
        step_details=hypothesis_step,
        tool=tool
    )
    logger.info(f"<<< ExperimenterAgent returned result: {experiment_result}")

    logger.info(">>> Handing experiment result to ValidatorAgent...")
    validation_verdict, failure_details = validator.verify(
        hypothesis=hypothesis_step,
        experiment_result=experiment_result
    )
    logger.info(f"<<< ValidatorAgent returned verdict: {validation_verdict}")

    return experiment_result, validation_verdict, failure_details


def main():
    """The main execution script for Prometheus Milestone 2."""
    logger.info("--- Project Prometheus: Milestone 2 - The Autonomous Loop ---")
    logger.info(f"Project Root: {project_root}")

    # 1. Load Configuration
    logger.info("Loading configuration from config.toml...")
    config_path = project_root / 'config.toml'
    with config_path.open("rb") as f:
        config = toml.load(f)
    logger.info("Configuration loaded successfully.")

    # 2. Instantiate Agents and Tools
    logger.info("Instantiating agents and tools...")
    try:
        hypothesis_agent = HypothesisAgent()
        experimenter_agent = ExperimenterAgent()
        validator_agent = ValidatorAgent()
        smina_tool = SminaAdapter(smina_executable_path=config['vina_tool']['executable_path'])
    except Exception as e:
        logger.error(f"Failed to initialize agents or tools: {e}", exc_info=True)
        return

    # --- CYCLE 1: RUN THE BASELINE EXPERIMENT ---
    logger.info("\n" + "="*50)
    logger.info("STEP 1: RUNNING BASELINE EXPERIMENT (ERLOTINIB)")
    logger.info("="*50)

    # A simple agent to generate our baseline hypothesis
    class BaselineHypothesisAgent:
        def generate_hypothesis(self, mission_params: dict, sim_box: dict, project_root: Path) -> Step:
            protein_abs_path = project_root / mission_params['protein_file']
            step = Step(id=1, action="Run baseline docking for Erlotinib", details=None,
                        relevant_file_paths=[str(protein_abs_path)], target_file=str(protein_abs_path))
            step.simulation_parameters = {"ligand_smiles": mission_params['ligand_smiles'],
                                          "center_x": sim_box['center_x'], "center_y": sim_box['center_y'], "center_z": sim_box['center_z'],
                                          "size_x": sim_box['size_x'], "size_y": sim_box['size_y'], "size_z": sim_box['size_z']}
            return step

    baseline_agent = BaselineHypothesisAgent()
    baseline_hypothesis = baseline_agent.generate_hypothesis(config['mission_parameters'], config['simulation_box'], project_root)

    baseline_exp_result, baseline_verdict, baseline_failure = run_experiment(
        baseline_hypothesis, experimenter_agent, validator_agent, smina_tool
    )

    if baseline_verdict != "HYPOTHESIS_VALIDATED":
        logger.error("Baseline experiment failed to validate. Cannot proceed to hypothesis generation.")
        return

    baseline_score = baseline_exp_result.get('binding_affinity_kcal_mol')
    logger.info(f"✅ Baseline experiment successful. Score to beat: {baseline_score:.3f} kcal/mol")

    # --- CYCLE 2: GENERATE AND TEST A NOVEL HYPOTHESIS ---
    logger.info("\n" + "="*50)
    logger.info("STEP 2: GENERATING AND TESTING AI'S NOVEL HYPOTHESIS")
    logger.info("="*50)

    logger.info(">>> Tasking intelligent HypothesisAgent to generate a novel hypothesis...")
    novel_hypothesis = hypothesis_agent.generate_hypothesis(
        baseline_result=baseline_exp_result,
        mission_params=config['mission_parameters'],
        sim_box=config['simulation_box'],
        project_root=project_root
    )

    if not novel_hypothesis:
        logger.error("HypothesisAgent failed to generate a valid novel hypothesis. Halting.")
        return

    logger.info(f"<<< HypothesisAgent proposed a new experiment: '{novel_hypothesis.action}'")

    novel_exp_result, novel_verdict, novel_failure = run_experiment(
        novel_hypothesis, experimenter_agent, validator_agent, smina_tool
    )

    # --- FINAL REPORT ---
    logger.info("\n" + "="*50)
    logger.info("--- FINAL REPORT ---")
    logger.info("="*50)
    logger.info(f"Baseline Erlotinib Score: {baseline_score:.3f} kcal/mol")

    if novel_verdict == "HYPOTHESIS_VALIDATED":
        novel_score = novel_exp_result.get('binding_affinity_kcal_mol')
        logger.info(f"AI's Novel Molecule Score: {novel_score:.3f} kcal/mol")
        if novel_score < baseline_score:
            improvement = baseline_score - novel_score
            logger.info(f"🎉🎉🎉 SUCCESS! The AI discovered a molecule with a predicted improvement of {improvement:.3f} kcal/mol!")
        else:
            logger.info(f"🔬 Result: The AI's proposed molecule did not improve upon the baseline.")
    else:
        logger.error("❌ FAILURE: The AI's novel hypothesis experiment could not be validated.")
        if novel_failure:
            logger.error(f"   Failure Type: {novel_failure.type}")
            logger.error(f"   Message: {novel_failure.message}")


if __name__ == "__main__":
    main() 