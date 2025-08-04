import logging
from typing import Optional

from prometheus.models import Step, VerificationFailureDetails

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """An agent that validates the results of an in-silico experiment."""

    def __init__(self) -> None:
        logger.info("ValidatorAgent initialized.")

    def verify(
        self, *, hypothesis: Step, experiment_result: dict
    ) -> tuple[str, Optional[VerificationFailureDetails]]:
        """Verify the ExperimenterAgent's result.

        Args:
            hypothesis: The original `Step` describing the experiment.
            experiment_result: The dictionary produced by `ExperimenterAgent.execute`.

        Returns
        -------
        tuple
            (verdict, failure_details) where `verdict` is one of
            "HYPOTHESIS_VALIDATED" or "COMPUTATIONAL_ERROR".
        """
        logger.info(f"Validating result for experiment: '{hypothesis.action}'")

        # Check 1: Ensure result object is present and a dict.
        if not experiment_result or not isinstance(experiment_result, dict):
            details = VerificationFailureDetails(
                type="INVALID_RESULT_OBJECT",
                message="ExperimenterAgent returned a null or invalid result object.",
            )
            return "COMPUTATIONAL_ERROR", details

        # Check 2: Confirm the tool reported success.
        if experiment_result.get("status") != "SUCCESS":
            details = VerificationFailureDetails(
                type="TOOL_EXECUTION_FAILED",
                message=(
                    "The scientific tool failed during the "
                    f"'{experiment_result.get('stage', 'unknown')}' stage."
                ),
                test_output=experiment_result.get("message"),
            )
            return "COMPUTATIONAL_ERROR", details

        # Check 3: Verify primary metric is present and numeric.
        binding_affinity = experiment_result.get("binding_affinity_kcal_mol")
        if binding_affinity is None:
            details = VerificationFailureDetails(
                type="MISSING_PRIMARY_METRIC",
                message="Experiment result is missing the 'binding_affinity_kcal_mol' key.",
            )
            return "COMPUTATIONAL_ERROR", details

        if not isinstance(binding_affinity, (float, int)):
            details = VerificationFailureDetails(
                type="INVALID_PRIMARY_METRIC",
                message=(
                    "Binding affinity is not a valid number. Got: "
                    f"{binding_affinity}"
                ),
            )
            return "COMPUTATIONAL_ERROR", details

        # All checks passed.
        logger.info("Experiment validated successfully.")
        return "HYPOTHESIS_VALIDATED", None
