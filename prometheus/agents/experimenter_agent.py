import logging
from pathlib import Path

from prometheus.tools.smina_adapter import SminaAdapter
from prometheus.models import Step

logger = logging.getLogger(__name__)


class ExperimenterAgent:
    """An agent that performs in silico experiments using scientific tools."""

    def __init__(self, config: dict) -> None:
        """Initializes the ExperimenterAgent."""
        self.config = config
        logger.info("ExperimenterAgent initialized.")

    def execute(self, step_details: Step, tool: SminaAdapter) -> dict:
        """Executes a single scientific experiment step.

        Args:
            step_details: The plan step containing the experimental parameters.
            tool: The scientific tool adapter to use for the experiment.

        Returns:
            A dictionary containing the results of the experiment.
        """
        logger.info(f"Executing experiment: '{step_details.action}'")

        # 1. Parse the scientific parameters from the step details.
        try:
            params = step_details.simulation_parameters
            protein_file = Path(step_details.target_file)
            ligand_smiles = params["ligand_smiles"]
            center = {
                "x": params["center_x"],
                "y": params["center_y"],
                "z": params["center_z"],
            }
            box_size = {
                "x": params["size_x"],
                "y": params["size_y"],
                "z": params["size_z"],
            }
        except (KeyError, TypeError) as e:
            error_message = f"Invalid or missing simulation parameters in step details: {e}"
            logger.error(error_message)
            return {
                "status": "ERROR",
                "stage": "parameter_parsing",
                "message": error_message,
            }

        # 2. Call the tool to run the experiment.
        result = tool.dock(
            protein_pdb_file=protein_file,
            ligand_smiles=ligand_smiles,
            center=center,
            box_size=box_size,
        )

        # 3. Return the structured result.
        return result 