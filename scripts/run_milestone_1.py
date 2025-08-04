# scripts/run_milestone_1.py
import sys
import logging
from pathlib import Path
import tomli as toml

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# --- Import Our Project Components ---
from prometheus.agents import HypothesisAgent, ExperimenterAgent, ValidatorAgent
from prometheus.tools.smina_adapter import SminaAdapter
from prometheus.models import VerificationFailureDetails

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry-point for Milestone 1 – executes the full agent pipeline."""

    logger.info("--- Project Prometheus: Milestone 1 ---")
    logger.info(f"Project Root: {project_root}")

    # 1. Load Configuration -------------------------------------------------
    logger.info("Loading configuration from config.toml…")
    config_path = project_root / "config.toml"
    with config_path.open("rb") as f:
        config = toml.load(f)
    logger.info("Configuration loaded.")

    # 2. Instantiate Agents & Tool -----------------------------------------
    logger.info("Instantiating agents and tools…")
    hypothesis_agent = HypothesisAgent()
    experimenter_agent = ExperimenterAgent()
    validator_agent = ValidatorAgent()

    smina_tool = SminaAdapter(
        smina_executable_path=config["vina_tool"]["executable_path"]
    )

    # 3. Generate Hypothesis ------------------------------------------------
    logger.info(">>> Tasking HypothesisAgent to generate baseline hypothesis…")
    hypothesis_step = hypothesis_agent.generate_hypothesis(
        mission_params=config["mission_parameters"],
        sim_box=config["simulation_box"],
        project_root=project_root,
    )
    logger.info(f"<<< HypothesisAgent produced action: '{hypothesis_step.action}'")

    # 4. Run Experiment ------------------------------------------------------
    logger.info(">>> Handing hypothesis to ExperimenterAgent…")
    experiment_result = experimenter_agent.execute(
        step_details=hypothesis_step, tool=smina_tool
    )
    logger.info(f"<<< ExperimenterAgent returned result: {experiment_result}")

    # 5. Validate Result -----------------------------------------------------
    logger.info(">>> Handing experiment result to ValidatorAgent…")
    validation_verdict, failure_details = validator_agent.verify(
        hypothesis=hypothesis_step, experiment_result=experiment_result
    )
    logger.info(f"<<< ValidatorAgent verdict: {validation_verdict}")

    # 6. Final Report --------------------------------------------------------
    logger.info("--- Milestone 1 Complete ---")
    if validation_verdict == "HYPOTHESIS_VALIDATED":
        score = experiment_result.get("binding_affinity_kcal_mol")
        logger.info("✅ SUCCESS: Experiment validated.")
        logger.info(
            f"   Binding Affinity for {config['mission_parameters']['ligand_name']}: {score:.3f} kcal/mol"
        )
    else:
        logger.error("❌ FAILURE: Experiment could not be validated.")
        if failure_details:
            logger.error(f"   Failure Type: {failure_details.type}")
            logger.error(f"   Message: {failure_details.message}")
            if failure_details.test_output:
                logger.error(f"   Tool Log: {failure_details.test_output}")


if __name__ == "__main__":
    main() 