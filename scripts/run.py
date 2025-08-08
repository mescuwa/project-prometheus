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
    ScoringAgent,
    ValidatorAgent,
)
from prometheus.models import ExperimentLog, Step  # noqa: F401 – Step may be used later
from prometheus.tools.smina_adapter import SminaAdapter  # noqa: F401 – Used by ExperimenterAgent
from prometheus.utils import generate_molecule_image, setup_logging  # noqa: F401
from prometheus.llm_utils import call_llm

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
    hypothesis_agent = HypothesisAgent(config=config)
    experimenter_agent = ExperimenterAgent(config=config)
    validator_agent = ValidatorAgent(config=config)
    md_validator = MDValidatorAgent(config=config)
    scoring_agent = ScoringAgent(config=config)
    report_synthesizer = ReportSynthesizerAgent(config=config)

    docking_tool = SminaAdapter(config=config)  # Provided to ExperimenterAgent

    # ---------------------------------------------------------------------
    # 3. Initial state
    # ---------------------------------------------------------------------
    best_overall_log: Optional[ExperimentLog] = None
    experiment_history: List[ExperimentLog] = []
    research_briefs: List[str] = []
    tested_smiles: set[str] = set(loop_config.get("tested_smiles", []))
    failed_smiles: set[str] = set()

    # Dynamic affinity gate – starts lenient, tightens with progress
    best_affinity_so_far: float = float(config.get("scoring", {}).get("affinity_threshold", 0.0))
    logger.info("Initial affinity gate set to: %.3f", best_affinity_so_far)

    # ---------------------------------------------------------------------
    # 4. Baseline run (optional; skipped for de novo missions)
    # ---------------------------------------------------------------------
    protein_abs = project_root / mission_params["protein_file"]

    if mission_params.get("ligand_smiles"):
        logger.info("\n%s\nCYCLE 0 – Baseline\n%s", "=" * 60, "=" * 60)

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
            action="Baseline docking",
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
            reasoning="Baseline measurement.",
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
    else:
        logger.info("No baseline SMILES provided – starting mission de novo (no cycle 0).")

    # ---------------------------------------------------------------------
    # 5. Sanity Check (Negative Control: Aspirin)
    # ---------------------------------------------------------------------
    logger.info("--- Running Sanity Check with Negative Control (Aspirin) ---")
    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    aspirin_scores = scoring_agent.calculate_scores(aspirin_smiles)

    aspirin_sim_params = {
        "ligand_smiles": aspirin_smiles,
        "center_x": sim_box["center_x"],
        "center_y": sim_box["center_y"],
        "center_z": sim_box["center_z"],
        "size_x": sim_box["size_x"],
        "size_y": sim_box["size_y"],
        "size_z": sim_box["size_z"],
    }

    aspirin_step = Step(
        id=-1,
        action="Negative Control Test (Aspirin)",
        relevant_file_paths=[str(protein_abs)],
        target_file=str(protein_abs),
        simulation_parameters=aspirin_sim_params,
    )
    aspirin_docking_result = await asyncio.to_thread(
        experimenter_agent.execute, aspirin_step, docking_tool
    )
    logger.info("Aspirin Docking Result: %s", aspirin_docking_result)
    logger.info("Aspirin Property Scores: %s", aspirin_scores)
    logger.info("--- Sanity Check Complete ---")

    # ---------------------------------------------------------------------
    # 6. Autonomous discovery loop
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
            f"Research brief for designing a new inhibitor for the {mission_params.get('pdb_id', 'protein target')}. Current best "
            f"molecule: {reference_smiles}. Focus on strategies to improve the "
            "composite score (affinity, QED, SA), paying attention to any known "
            "resistance mechanisms or binding challenges."
        )

        logger.info("Searching static knowledge base for context...")
        knowledge_context = await asyncio.to_thread(  # noqa: SLF001
            hypothesis_agent._search_knowledge_base, research_prompt
        )
        research_briefs.append(knowledge_context)

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

        # Determine the top 2 best hits across all validated cycles
        validated_logs = [log for log in experiment_history if "VALIDATED" in log.verdict]
        best_hits = sorted(
            validated_logs,
            key=lambda l: l.composite_score or -float('inf'),
            reverse=True,
        )[:2]

        # --- NEW "STRATEGIC REVIEW" STEP ---
        strategic_injection = ""
        if cycle > 1 and experiment_history:
            logger.info("--- Conducting Strategic Review with GPT-5 Consultant ---")
                        # Build full history briefing for consultant
            logs_by_cycle = {}
            for log in experiment_history:
                logs_by_cycle.setdefault(log.cycle, []).append(log)

            full_history_summary = []
            for c in sorted(logs_by_cycle.keys()):
                full_history_summary.append(f"--- Cycle {c} Results ---")
                cycle_logs = logs_by_cycle[c]
                summary_lines = [
                    f"- Molecule {idx+1}: Verdict: {log.verdict}, Affinity: {log.average_binding_affinity:.3f}, CS: {log.composite_score if log.composite_score is not None else 'N/A'}"
                    for idx, log in enumerate(cycle_logs)
                ]
                full_history_summary.extend(summary_lines)

            history_briefing = "\n".join(full_history_summary)
            # (Research Analyst step removed – direct briefing used)

            consultant_prompt = f"""
You are a world-class medicinal chemist. The project is stuck in a potency plateau.
Here is the executive summary of the project's history:
---
{history_briefing}
---
The current champion is: {best_overall_log.smiles if best_overall_log else 'None'}.

Your task: Provide 2-3 novel strategic directions.
**CRITICAL INSTRUCTION:** For each strategy, you MUST provide TWO simple, valid SMILES strings:
1. A "Core Scaffold" for the main part of the molecule.
2. A "Key Fragment" (a linker, warhead, or side chain) that should be attached to the core.

Format your response as follows:
**Strategy 1:** [Your strategic advice...]
**Core 1:** [Valid SMILES for the core]
**Fragment 1:** [Valid SMILES for the fragment to be attached]

**Strategy 2:** [Your second piece of advice...]
**Core 2:** [Valid SMILES for the core]
**Fragment 2:** [Valid SMILES for the fragment]
"""
            consultant_model = config.get("llm_models", {}).get("consultant_model", "gpt-5-2025-08-07")

            consultant_response = await call_llm(
                prompt=consultant_prompt,
                model_name=consultant_model,
                temperature=0.7
            )

            if consultant_response and not consultant_response.get("error"):
                strategic_injection = consultant_response.get("raw_text_output", "")
                logger.info("Consultant's Strategic Advice:\n%s", strategic_injection)
            else:
                logger.warning("Could not get strategic advice from the consultant model.")
        # --- END OF NEW STEP ---

        candidate_steps = await hypothesis_agent.generate_hypothesis(
            current_best_log=best_overall_log,
            knowledge_context=knowledge_context,
            mission_params=mission_params,
            sim_box=sim_box,
            project_root=project_root,
            history=history_data,
            scoring_weights=scoring_weights,
            best_hits=best_hits,
            failed_smiles=list(failed_smiles),
            batch_size=int(loop_config.get("hypothesis_batch_size", 1)),
            strategic_injection=strategic_injection
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

            # Skip molecules that have already been tested or are known to be invalid
            if smiles in failed_smiles or smiles in tested_smiles:
                logger.debug("Skipping previously failed or tested SMILES: %s", smiles)
                continue

            # Run the docking screen in a background thread since it is CPU-bound
            docking_result = await asyncio.to_thread(experimenter_agent.execute, step, docking_tool)
            if docking_result.get("status") != "SUCCESS":
                logger.warning("Docking failed for candidate %s – skipping.", smiles)
                failed_smiles.add(smiles)
                continue

            docked_file = docking_result.get("docked_ligand_file")
            if not docked_file or not Path(docked_file).exists():
                logger.warning("Docked ligand file missing for %s – skipping scoring.", smiles)
                failed_smiles.add(smiles)
                continue

            # Property scores (QED, SA, LogP) AND Interaction Fingerprint (IFP)
            prop_scores = scoring_agent.calculate_scores(
                smiles=smiles,
                protein_file=protein_abs,
                docked_ligand_file=Path(docked_file),
                key_interactions=mission_params.get("key_interactions", []),
            )
            if prop_scores is None:
                logger.warning("Scoring failed for %s – skipping candidate.", smiles)
                failed_smiles.add(smiles)
                continue

            # --- AFFINITY GATE (dynamic) ---
            affinity_gate = best_affinity_so_far
            current_affinity = docking_result.get("average_binding_affinity", 0.0)

            # Skip the gate entirely for the first cycle to encourage exploration
            if cycle > 1 and current_affinity > affinity_gate:
                logger.warning(
                    "Candidate %s FAILED AFFINITY GATE. Affinity %.3f is worse than threshold %.3f. Discarding.",
                    smiles,
                    current_affinity,
                    affinity_gate,
                )
                failed_smiles.add(smiles)
                log_entry_data = {
                    "cycle": cycle,
                    "smiles": smiles,
                    "average_binding_affinity": current_affinity,
                    "qed": prop_scores["qed"],
                    "sa_score": prop_scores["sa_score"],
                    "logp": prop_scores["logp"],
                    "composite_score": None,  # Failed the gate, no composite score
                    "reasoning": step.action,
                    "verdict": "FAILED_AFFINITY_GATE",
                }
                experiment_history.append(ExperimentLog(**log_entry_data))
                continue

            # Composite score – weights come from the config
            weights = config.get("scoring", {})
            composite = (
                current_affinity * weights.get("w_affinity", -1.0)
                + prop_scores["qed"] * weights.get("w_qed", 5.0)
                + prop_scores["sa_score"] * weights.get("w_sa_score", -1.0)
                + prop_scores["ifp_score"] * weights.get("w_ifp", 10.0)
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

        # --- Update dynamic affinity gate based on this cycle's results ---
        cycle_affinities = [c["docking"].get("average_binding_affinity", 0.0) for c in screened_candidates]
        if cycle_affinities:
            best_cycle_affinity = min(cycle_affinities)
            if best_cycle_affinity < best_affinity_so_far:
                best_affinity_so_far = best_cycle_affinity
                logger.info(
                    "🏆 New best affinity achieved this cycle: %.3f. Affinity gate for next cycle updated.",
                    best_affinity_so_far,
                )

        best_candidate = max(screened_candidates, key=lambda c: c["composite"])

        # D) Tiered Validation --------------------------------------------------
        is_new_champion = (
            best_overall_log is None
            or best_candidate["composite"] > (best_overall_log.composite_score or -float("inf"))
        )

        promoted_docking_result = best_candidate["docking"]

        if is_new_champion:
            docked_ligand_file_path = promoted_docking_result.get("docked_ligand_file")
            if not docked_ligand_file_path or not Path(docked_ligand_file_path).exists():
                logger.error(
                    "Docked ligand file not found for MD simulation. Skipping validation for candidate %s",
                    best_candidate["smiles"],
                )
                md_status = "ERROR"
                md_result = {"status": "ERROR", "message": "Docked ligand PDBQT file was not generated or found."}
            else:
                logger.info("🎉 Potential new champion found! Submitting for expensive MD Validation...")
                md_result = await asyncio.to_thread(
                    md_validator.run_simulation,
                    protein_abs,
                    Path(docked_ligand_file_path),
                )
                md_status = md_result.get("status")
        else:
            logger.info(
                "Best candidate (CS: %.3f) did not beat the current champion (CS: %.3f). Skipping MD.",
                best_candidate["composite"],
                best_overall_log.composite_score if best_overall_log else float("-inf"),
            )
            md_status = "SKIPPED"
            md_result = {"status": "SKIPPED"}

        # Determine verdict for the best candidate
        if md_status == "SUCCESS":
            verdict_best = "VALIDATED"
        elif md_status == "SKIPPED":
            verdict_best = "VALIDATED_BY_SCORE"
        else:
            verdict_best = "FAILED_MD"

        # Log every screened candidate with appropriate verdict
        for cand in screened_candidates:
            v = verdict_best if cand is best_candidate else "VALIDATED_BY_SCORE"
            log_entry_data = {
                "cycle": cycle,
                "smiles": cand["smiles"],
                "average_binding_affinity": cand["docking"].get("average_binding_affinity"),
                "std_dev_binding_affinity": cand["docking"].get("std_dev_binding_affinity"),
                "qed": cand["scores"]["qed"],
                "sa_score": cand["scores"]["sa_score"],
                "logp": cand["scores"]["logp"],
                "composite_score": cand["composite"],
                "reasoning": cand["step"].action,
                "verdict": v,
            }
            experiment_history.append(ExperimentLog(**log_entry_data))

            # If this is the validated new champion, update image and overall log
            if v == "VALIDATED" and cand is best_candidate:
                logger.info("✅ MD Validation Successful. A new champion is crowned!")

                # Generate 2-D image for the newly validated molecule
                image_filename = mission_images_dir / f"cycle_{cycle:02d}_CS_{cand['composite']:.3f}.png"
                generate_molecule_image(
                    smiles=cand["smiles"],
                    output_path=image_filename,
                    molecule_name=f"Cycle {cycle} | CS: {cand['composite']:.3f}",
                )
                # Attach RELATIVE image path to the log entry for downstream reporting
                experiment_history[-1].image_path = str(Path(image_filename).relative_to(mission_dir))

                best_overall_log = experiment_history[-1]

        # If MD failed for the potential champion, log the error message
        if md_status not in {"SUCCESS", "SKIPPED"}:
            logger.error("❌ MD Validation Failed for potential champion: %s", md_result.get("message"))

    # ---------------------------------------------------------------------
    # 7. Final AI-written report
    # ---------------------------------------------------------------------
    logger.info("\n%s\nPHASE 4 – GENERATING FINAL SCIENTIFIC PAPER\n%s", "=" * 60, "=" * 60)

    final_markdown = await report_synthesizer.generate_report(
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
