"""run.py
=====================

The *grand orchestration* script. It ties together every agent, tool, and
utility to execute Prometheus’ full autonomous discovery protocol and, at the
end, produce a nicely formatted scientific manuscript.

The implementation purposefully mirrors the structure of earlier milestone
runner scripts so that it is easy to follow. Many detailed steps (docking,
MD simulation, CSV exports, etc.) are still represented as comments – those
pieces already exist in the respective milestone scripts and can be migrated
in incrementally.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import tomli as toml  # `tomli` is the back-port of the 3.11 stdlib parser
from dotenv import load_dotenv  # noqa: F401 – optional but useful
import pandas as pd  # noqa: F401 – for CSV export at the end

# ---------------------------------------------------------------------------
# Prepare import path so that `python scripts/run_final_mission.py` works.
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# pylint: disable=wrong-import-position
from prometheus.agents import (# type: ignore
    ExperimenterAgent,
    HypothesisAgent,
    MDValidatorAgent,
    ReportSynthesizerAgent,
    ResearchAgent,
    ScoringAgent,
    ValidatorAgent,
)
from prometheus.models import ExperimentLog, Step  # noqa: F401 – Step may be used later
from prometheus.tools.smina_adapter import SminaAdapter  # noqa: F401 – Used by ExperimenterAgent
from prometheus.utils import generate_molecule_image, setup_logging  # noqa: F401

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper – load configuration once at start-up.
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    """Load *config.toml* from project root and return as dict."""

    config_path = project_root / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError("config.toml not found at project root.")

    with config_path.open("rb") as fp:
        config: dict = toml.load(fp)
    return config


aSync = asyncio.run  # alias – short hand for bottom of file


async def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Entry-point coroutine for the final Prometheus mission."""

    setup_logging(project_root)
    logger.info("\n%s\nPROJECT PROMETHEUS – FINAL MISSION START\n%s", "=" * 60, "=" * 60)

    # ---------------------------------------------------------------------
    # Mission-specific output directory
    # ---------------------------------------------------------------------
    mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mission_dir = project_root / "reports" / f"mission_{mission_id}"
    mission_images_dir = mission_dir / "images"
    mission_images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Mission ID: %s", mission_id)
    logger.info("Output directory: %s", mission_dir)

    # ---------------------------------------------------------------------
    # 1. Configuration & environment
    # ---------------------------------------------------------------------
    load_dotenv()  # Pull API keys, etc.
    config = _load_config()

    mission_params: dict = config.get("mission_parameters", {})
    loop_config: dict = config.get("autonomous_loop", {})

    # Convenience short-hands from config
    sim_box: dict = config.get("simulation_box", {})
    vina_config: dict = config.get("vina_tool", {})
    scoring_weights: dict = config.get("scoring", {})

    # ---------------------------------------------------------------------
    # 2. Instantiate agents & tools
    # ---------------------------------------------------------------------
    research_agent = ResearchAgent(config=config)
    hypothesis_agent = HypothesisAgent(config=config)
    experimenter_agent = ExperimenterAgent(config=config)
    validator_agent = ValidatorAgent(config=config)
    md_validator = MDValidatorAgent(config=config)
    scoring_agent = ScoringAgent(config=config)
    report_synthesizer = ReportSynthesizerAgent()

    docking_tool = SminaAdapter(config=config)  # Provided to ExperimenterAgent

    # ---------------------------------------------------------------------
    # 3. Initial state
    # ---------------------------------------------------------------------
    best_overall_log: Optional[ExperimentLog] = None
    experiment_history: List[ExperimentLog] = []
    research_briefs: List[str] = []
    tested_smiles: set[str] = set(loop_config.get("tested_smiles", []))

    # ---------------------------------------------------------------------
    # 4. Baseline run (Erlotinib)
    # ---------------------------------------------------------------------
    logger.info("\n%s\nCYCLE 0 – Baseline (Erlotinib)\n%s", "=" * 60, "=" * 60)

    protein_abs = project_root / mission_params["protein_file"]

    # Compile simulation parameters dict for consistency with other steps
    baseline_sim_params = {
        "ligand_smiles": mission_params["ligand_smiles"],
        "center_x": sim_box["center_x"],
        "center_y": sim_box["center_y"],
        "center_z": sim_box["center_z"],
        "size_x": sim_box["size_x"],
        "size_y": sim_box["size_y"],
        "size_z": sim_box["size_z"],
    }

    baseline_step = Step(
        id=0,
        action="Baseline docking for Erlotinib",
        relevant_file_paths=[str(protein_abs)],
        target_file=str(protein_abs),
        simulation_parameters=baseline_sim_params,
    )

    # Run baseline docking through the ExperimenterAgent for architectural consistency
    baseline_docking_result = await asyncio.to_thread(
        experimenter_agent.execute,
        baseline_step,
        docking_tool,
    )

    if baseline_docking_result.get("status") != "SUCCESS":
        logger.error("Baseline docking failed. Aborting mission.")
        return

    baseline_scores = scoring_agent.calculate_scores(mission_params["ligand_smiles"])
    if not baseline_scores:
        logger.error("Baseline scoring failed. Aborting mission.")
        return

    baseline_affinity = baseline_docking_result["average_binding_affinity"]

    composite_score = (
        scoring_weights.get("w_affinity", -1.0) * baseline_affinity
        + scoring_weights.get("w_qed", 5.0) * baseline_scores["qed"]
        + scoring_weights.get("w_sa_score", -1.0) * baseline_scores["sa_score"]
    )

    best_overall_log = ExperimentLog(
        cycle=0,
        smiles=mission_params["ligand_smiles"],
        reasoning="Baseline measurement for Erlotinib.",
        average_binding_affinity=baseline_affinity,
        std_dev_binding_affinity=baseline_docking_result["std_dev_binding_affinity"],
        qed=baseline_scores["qed"],
        sa_score=baseline_scores["sa_score"],
        logp=baseline_scores["logp"],
        composite_score=composite_score,
        verdict="VALIDATED_BY_DEFAULT",
    )
    experiment_history.append(best_overall_log)
    tested_smiles.add(best_overall_log.smiles)

    logger.info(
        "✅ Baseline established. Composite Score: %.3f (Affinity %.3f, QED %.3f, SA %.3f)",
        composite_score,
        baseline_affinity,
        baseline_scores["qed"],
        baseline_scores["sa_score"],
    )

    # Generate baseline molecule representation in mission-specific folder
    baseline_image_filename = mission_images_dir / f"cycle_00_CS_{composite_score:.3f}.png"
    generate_molecule_image(
        smiles=mission_params["ligand_smiles"],
        output_path=baseline_image_filename,
        molecule_name=f"Cycle 0 (Baseline) | CS: {composite_score:.3f}",
    )
    # Store RELATIVE image path in baseline log
    best_overall_log.image_path = str(Path(baseline_image_filename).relative_to(mission_dir))

    # ---------------------------------------------------------------------
    # 5. Autonomous discovery loop
    # ---------------------------------------------------------------------
    max_cycles: int = int(loop_config.get("max_cycles", 0))
    for cycle in range(1, max_cycles + 1):
        logger.info("\n%s\nCYCLE %d – START\n%s", "=" * 50, cycle, "=" * 50)

        # A) Live research --------------------------------------------------
        if best_overall_log is None:
            reference_smiles = mission_params.get("ligand_smiles", "")
        else:
            reference_smiles = best_overall_log.smiles

        research_prompt = (
            "Research brief for designing a new EGFR inhibitor. Current best "
            f"molecule: {reference_smiles}. Focus on strategies to improve the "
            "composite score (affinity, QED, SA) specifically addressing the "
            "T790M resistance."
        )

        knowledge_context: str | None = None
        if research_agent.config.get("enabled", False):
            research_report = await asyncio.to_thread(research_agent.conduct_research, research_prompt)
            if research_report:
                knowledge_context = research_report
                research_briefs.append(research_report)

        if knowledge_context is None:
            logger.warning("Live research failed or was disabled – falling back to static knowledge base.")
            knowledge_context = await asyncio.to_thread(  # noqa: SLF001
                hypothesis_agent._search_knowledge_base, research_prompt
            )

        # B) Hypothesis generation ----------------------------------------
                # Build concise history summary for the agent
        history_data = [
            {
                "cycle": log.cycle,
                "smiles_tested": log.smiles,
                "composite_score": log.composite_score or 0.0,
                "affinity": log.average_binding_affinity,
                "qed": log.qed or 0.0,
                "sa_score": log.sa_score or 0.0,
            }
            for log in experiment_history
        ]

        candidate_steps = await hypothesis_agent.generate_hypothesis(
            current_best_log=best_overall_log,
            knowledge_context=knowledge_context,
            mission_params=mission_params,
            sim_box=sim_box,
            project_root=project_root,
            history=history_data,
            scoring_weights=scoring_weights,
            batch_size=int(loop_config.get("hypothesis_batch_size", 1)),
        )

        # Handle empty or failed generation
        if not candidate_steps:
            logger.warning("HypothesisAgent failed to generate a batch for cycle %d – skipping cycle.", cycle)
            continue

        # C) Screening (docking + scoring) ---------------------------------
        screened_candidates: List[dict] = []

        for idx, step in enumerate(candidate_steps, start=1):
            smiles = step.simulation_parameters.get("ligand_smiles") if step.simulation_parameters else None
            if not smiles:
                logger.warning("Candidate %d missing SMILES – skipping.", idx)
                continue

            # Run the docking screen in a background thread since it is CPU-bound
            docking_result = await asyncio.to_thread(experimenter_agent.execute, step, docking_tool)
            if docking_result.get("status") != "SUCCESS":
                logger.warning("Docking failed for candidate %s – skipping.", smiles)
                continue

            # Property scores (QED, SA, LogP)
            prop_scores = scoring_agent.calculate_scores(smiles)
            if prop_scores is None:
                logger.warning("Scoring failed for %s – skipping candidate.", smiles)
                continue

            # Composite score – weights come from the config
            weights = config.get("scoring", {})
            composite = (
                docking_result.get("average_binding_affinity", 0.0) * weights.get("w_affinity", -1.0)
                + prop_scores["qed"] * weights.get("w_qed", 5.0)
                + prop_scores["sa_score"] * weights.get("w_sa_score", -1.0)
            )

            screened_candidates.append(
                {
                    "step": step,
                    "smiles": smiles,
                    "docking": docking_result,
                    "scores": prop_scores,
                    "composite": composite,
                }
            )

        if not screened_candidates:
            logger.warning("All candidates failed in screening – skipping cycle %d.", cycle)
            continue

        best_candidate = max(screened_candidates, key=lambda c: c["composite"])

        # D) MD validation --------------------------------------------------
        promoted_docking_result = best_candidate["docking"]
        docked_ligand_file_path = promoted_docking_result.get("docked_ligand_file")

        if not docked_ligand_file_path or not Path(docked_ligand_file_path).exists():
            logger.error(
                "Docked ligand file not found for MD simulation. Skipping validation for candidate %s",
                best_candidate["smiles"],
            )
            md_status = "ERROR"
            md_result = {"status": "ERROR", "message": "Docked ligand PDBQT file was not generated or found."}
        else:
            logger.info("Submitting promoted candidate for MD Validation...")
            md_result = await asyncio.to_thread(
                md_validator.run_simulation,
                protein_abs,
                Path(docked_ligand_file_path),
            )
            md_status = md_result.get("status")

        md_status = md_result.get("status")
        log_entry_data = {
            "cycle": cycle,
            "smiles": best_candidate["smiles"],
            "average_binding_affinity": promoted_docking_result.get("average_binding_affinity"),
            "std_dev_binding_affinity": promoted_docking_result.get("std_dev_binding_affinity"),
            "qed": best_candidate["scores"]["qed"],
            "sa_score": best_candidate["scores"]["sa_score"],
            "logp": best_candidate["scores"]["logp"],
            "composite_score": best_candidate["composite"],
            "reasoning": best_candidate["step"].action,
            "verdict": "VALIDATED" if md_status == "SUCCESS" else "FAILED_MD",
        }
        experiment_history.append(ExperimentLog(**log_entry_data))

        if md_status == "SUCCESS":
            logger.info("✅ MD Validation Successful.")
            
            # --- Generate 2-D image for the newly validated molecule ---
            current_log_entry = experiment_history[-1]  # Most recent log is the validated candidate
            image_filename = mission_images_dir / f"cycle_{current_log_entry.cycle:02d}_CS_{current_log_entry.composite_score:.3f}.png"
            generate_molecule_image(
                smiles=current_log_entry.smiles,
                output_path=image_filename,
                molecule_name=f"Cycle {current_log_entry.cycle} | CS: {current_log_entry.composite_score:.3f}"
            )
            # Attach RELATIVE image path to the log entry for downstream reporting
            current_log_entry.image_path = str(Path(image_filename).relative_to(mission_dir))
            
            if best_overall_log is None or best_candidate["composite"] > (best_overall_log.composite_score or -float('inf')):
                logger.info(
                    "🎉 NEW CHAMPION! CS: %.3f (Affinity: %.3f, QED: %.3f, SA: %.3f)",
                    best_candidate["composite"],
                    promoted_docking_result["average_binding_affinity"],
                    best_candidate["scores"]["qed"],
                    best_candidate["scores"]["sa_score"],
                )
                best_overall_log = experiment_history[-1]
        else:
            logger.error("❌ MD Validation Failed: %s", md_result.get("message"))

    # ---------------------------------------------------------------------
    # 6. Final AI-written report
    # ---------------------------------------------------------------------
    logger.info("\n%s\nPHASE 4 – GENERATING FINAL SCIENTIFIC PAPER\n%s", "=" * 60, "=" * 60)

    final_markdown = await report_synthesizer.generate_report(
        config=config,
        mission_params=mission_params,
        experiment_history=experiment_history,
        research_briefs=research_briefs,
    )

    # Persist report in the mission directory -----------------------------
    report_path = mission_dir / f"prometheus_final_report_{mission_id}.md"
    report_path.write_text(final_markdown, encoding="utf-8")
    logger.info("✅ Final scientific paper saved to: %s", report_path)

    # Optionally export structured CSV as in earlier milestones -------------
    # if experiment_history:
    #     df = pd.DataFrame([log.dict() for log in experiment_history])
    #     csv_path = reports_dir / f"prometheus_run_data_{timestamp}.csv"
    #     df.to_csv(csv_path, index=False)
    #     logger.info("Structured results saved to: %s", csv_path)


if __name__ == "__main__":
    aSync(main())
