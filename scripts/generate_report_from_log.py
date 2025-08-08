#!/usr/bin/env python
"""Temporary helper script to generate a Prometheus scientific report **after** a
successful autonomous mission run.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
from prometheus.models import ExperimentLog  # noqa: E402
from prometheus.agents.report_synthesizer_agent import ReportSynthesizerAgent  # noqa: E402
from prometheus.llm_utils import configure_llm_utils  # noqa: E402

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# --- Regex patterns for log parsing (FINAL, ROBUST VERSION) ---
RE_REASONING = re.compile(r"Executing experiment: 'Test hypothesis: (.*?)'")
RE_AFFINITY = re.compile(r"Average binding affinity:\s+([\-\d.]+)")
RE_SCORES = re.compile(
    r"Computed scores for\s+(?P<smiles>.*?)\s*…\s*QED=([0-9.]+)\s+SA=([0-9.]+)\s+LogP=([0-9.]+)",
)
RE_MD_SUCCESS = re.compile(r"MD Validation Successful")
RE_MD_FAIL = re.compile(r"MD Validation Failed")
RE_CYCLE_START = re.compile(r"CYCLE\s+(?P<cycle>\d+)\s+–")


def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config.toml"
    with cfg_path.open("rb") as fh:
        return tomllib.load(fh)


def _compute_composite_score(
    affinity: float,
    qed: float,
    sa: float,
    weights: dict[str, float],
) -> Optional[float]:
    try:
        return (
            weights.get("w_affinity", -1.0) * affinity
            + weights.get("w_qed", 1.0) * qed
            + weights.get("w_sa_score", -1.0) * sa
        )
    except Exception:
        return None


# --- FINAL, ROBUST PARSING LOGIC ---

def parse_experiment_history(log_path: Path, scoring_weights: dict) -> List[ExperimentLog]:
    """Parse the log file and build a list of ExperimentLog entries."""
    history: List[ExperimentLog] = []
    candidates_in_cycle: List[Dict] = []
    current_cycle = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            # Cycle header ---------------------------------------------------
            if (m_cycle := RE_CYCLE_START.search(line)):
                # Flush any completed candidates from the previous cycle
                for cand in candidates_in_cycle:
                    if "smiles" in cand:
                        history.append(ExperimentLog(**cand))
                candidates_in_cycle = []
                current_cycle = int(m_cycle.group("cycle"))
                continue

            # Hypothesis reasoning ------------------------------------------
            if (m_reason := RE_REASONING.search(line)):
                candidates_in_cycle.append(
                    {
                        "cycle": current_cycle,
                        "reasoning": m_reason.group(1).strip(),
                    }
                )
                continue

            # Docking affinity ----------------------------------------------
            if (m_aff := RE_AFFINITY.search(line)) and candidates_in_cycle:
                candidates_in_cycle[-1]["average_binding_affinity"] = float(m_aff.group(1))
                continue

            # Property scores + SMILES --------------------------------------
            if (m_scores := RE_SCORES.search(line)) and candidates_in_cycle:
                smiles = m_scores.group("smiles")
                qed = float(m_scores.group(2))
                sa_score = float(m_scores.group(3))
                logp = float(m_scores.group(4))

                # Attach to the last candidate lacking SMILES
                for cand in reversed(candidates_in_cycle):
                    if "smiles" not in cand:
                        affinity = cand.get("average_binding_affinity", 0.0)
                        cand.update(
                            {
                                "smiles": smiles,
                                "qed": qed,
                                "sa_score": sa_score,
                                "logp": logp,
                                "composite_score": _compute_composite_score(
                                    affinity, qed, sa_score, scoring_weights
                                ),
                                "verdict": "VALIDATED_BY_SCORE",
                            }
                        )
                        break
                continue

            # MD outcomes ----------------------------------------------------
            if candidates_in_cycle:
                if RE_MD_SUCCESS.search(line):
                    candidates_in_cycle[-1]["verdict"] = "VALIDATED"
                    continue
                if RE_MD_FAIL.search(line):
                    candidates_in_cycle[-1]["verdict"] = "FAILED_MD"
                    continue
            if history:
                if RE_MD_SUCCESS.search(line):
                    history[-1].verdict = "VALIDATED"
                    continue
                if RE_MD_FAIL.search(line):
                    history[-1].verdict = "FAILED_MD"
                    continue

    # Flush any remaining candidates from the final cycle
    for cand in candidates_in_cycle:
        if "smiles" in cand:
            history.append(ExperimentLog(**cand))

    return history


async def _async_main(args: argparse.Namespace) -> None:
    load_dotenv()

    cfg = _load_config()
    scoring_weights = cfg.get("scoring", {})

    log_file = Path(args.log_file).expanduser().resolve()
    if not log_file.exists():
        print(f"[ERROR] Log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)

    experiment_history = parse_experiment_history(log_file, scoring_weights)
    if not experiment_history:
        print("[ERROR] No experiment history could be reconstructed from the log.", file=sys.stderr)
        sys.exit(1)

    configure_llm_utils(default_temperature=cfg.get("llm_models", {}).get("default_temperature"))

    agent = ReportSynthesizerAgent(cfg)
    mission_params = cfg.get("mission_parameters", {})

    print(f"[INFO] Generating report with {len(experiment_history)} experiment logs …")
    report_markdown = await agent.generate_report(
        mission_params=mission_params,
        experiment_history=experiment_history,
        research_briefs=[],
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.write_text(report_markdown, encoding="utf-8")
    print(f"[OK] Report saved to: {output_path}")


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate Prometheus report from a run log.")
    parser.add_argument("log_file", help="Path to the Prometheus run .log file to parse")
    parser.add_argument(
        "-o",
        "--output",
        default="prometheus_report_from_log.md",
        help="Destination path for the generated Markdown report",
    )
    asyncio.run(_async_main(parser.parse_args()))


if __name__ == "__main__":  # pragma: no cover
    main()
