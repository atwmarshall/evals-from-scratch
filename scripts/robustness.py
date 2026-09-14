"""robustness.py — measure model robustness across adversarially perturbed inputs.

Flow:
  load dataset → generate perturbations (or reuse saved) →
  run each perturbation through model → RobustnessReporter → save results

Part B of Challenge 6: model robustness. Hold scorer fixed, vary the model input.
Question: "how much does the model degrade under realistic input stress?"

Only run this after Part A (sensitivity.py) has established an acceptable scorer
noise floor. If scorer variance is ±0.1 and a robustness delta is 0.08, the signal
is within noise and uninterpretable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.core import Dataset, EvalConfig, RunResult
from evals.perturbation_generator import PerturbationGenerator
from evals.robustness_reporter import RobustnessReporter
from evals.runner import Runner
from evals.scorer_factory import SCORER_CHOICES, build_scorer

logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    import argparse

    parser = argparse.ArgumentParser(
        description="Robustness analysis: measure model score degradation under adversarial input perturbations"
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--scorer", required=True, help=f"Scorer to use ({SCORER_CHOICES})")
    parser.add_argument("--model", default=None, help="Model to evaluate (overrides DEFAULT_MODEL)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=None,
        help="Perturbation types to run (default: all 5 — typos colloquial verbose indirect multilingual)",
    )
    parser.add_argument(
        "--no-save-perturbations",
        action="store_true",
        help="Skip saving generated perturbations to datasets/generated/robustness/",
    )
    parser.add_argument(
        "--reuse-perturbations",
        default=None,
        metavar="DIR",
        help="Reuse previously saved perturbations from this directory",
    )
    parser.add_argument("--output", default=None, help="Results directory (overrides RESULTS_DIR)")
    # Scorer-specific flags
    parser.add_argument(
        "--pattern", default=None, help="Regex pattern for regex/multi-regex scorers"
    )
    parser.add_argument("--schema", default=None, help="Path to JSON schema file for schema scorer")
    parser.add_argument("--scale", type=int, default=5, help="Judge score scale (default: 5)")
    parser.add_argument("--judge-model", default=None, help="Model for LLM judge scorer")
    parser.add_argument(
        "--fast-tier",
        default="normalised",
        choices=["exact", "normalised"],
        help="Fast tier for cascade scorer (default: normalised)",
    )
    parser.add_argument(
        "--cascade-threshold",
        type=float,
        default=1.0,
        help="Fast-tier threshold for cascade scorer (default: 1.0)",
    )
    args = parser.parse_args()

    # --- resolve models ---
    config = EvalConfig(model=args.model) if args.model else EvalConfig()
    perturbation_model = os.environ.get("PERTURBATION_MODEL") or os.environ.get(
        "DEFAULT_MODEL", "llama3.2:3b"
    )

    # No model separation check — for robustness testing the perturbation model
    # can equal the evaluated model. There is no circularity concern here (unlike
    # sensitivity analysis where the same model generating and judging variations
    # measures self-consistency rather than scorer reliability).

    # --- load dataset ---
    ds = Dataset.from_jsonl(args.dataset, limit=args.limit)
    dataset_name = Path(args.dataset).stem
    print(f"Loaded {len(ds)} samples from {args.dataset}")

    # --- build scorer ---
    scorer = build_scorer(args, evaluated_model=config.model)

    # --- generate or load perturbations ---
    if args.reuse_perturbations:
        reuse_dir = Path(args.reuse_perturbations)
        _warn_reuse_metadata(reuse_dir)
        perturbations = PerturbationGenerator.load_perturbations(reuse_dir)
        # load_perturbations() never returns "baseline" — inject from original dataset
        perturbations = {"baseline": ds, **perturbations}
        if args.perturbations:
            keep = {"baseline"} | set(args.perturbations)
            perturbations = {k: v for k, v in perturbations.items() if k in keep}
        print(f"Reusing {len(perturbations) - 1} perturbation type(s) from {reuse_dir}")
    else:
        gen = PerturbationGenerator()
        print(f"Generating perturbations using {gen.model}...")
        perturbations = gen.generate(ds, perturbations=args.perturbations)

        if not args.no_save_perturbations:
            saved_dir = gen.save_perturbations(
                perturbations=perturbations,
                source_path=args.dataset,
            )
            print(f"Perturbations saved → {saved_dir}")

    # --- run each perturbation through the model ---
    n_perturbations = len([k for k in perturbations if k != "baseline"])
    print(
        f"Running {len(perturbations)} perturbation(s) ({n_perturbations} + baseline) through {config.model}..."
    )

    results_by_perturbation: dict[str, list[RunResult]] = {}
    for name, dataset in perturbations.items():
        if len(dataset) == 0:
            logger.warning("skipping %r — empty dataset after perturbation failures", name)
            continue
        print(f"  {name} ({len(dataset)} samples)...")
        results_by_perturbation[name] = Runner().run(dataset, scorer, config)

    if not results_by_perturbation:
        sys.exit("No perturbation results — all datasets were empty.")

    # --- report ---
    reporter_kwargs: dict[str, Any] = {}
    if args.output:
        reporter_kwargs["results_dir"] = Path(args.output)
    reporter = RobustnessReporter(**reporter_kwargs)

    output_str, rob_dir = reporter.report(
        results_by_perturbation,
        dataset_name=dataset_name,
        scorer_name=args.scorer,
        model=config.model,
        run_config={
            "perturbation_model": perturbation_model,
            "cascade_threshold": args.cascade_threshold,
            "limit": args.limit,
            "reused_from": str(args.reuse_perturbations) if args.reuse_perturbations else None,
        },
    )
    print()
    print(output_str)
    print(f"\nSaved → {rob_dir}")


def _warn_reuse_metadata(reuse_dir: Path) -> None:
    """Log the model used to generate saved perturbations."""
    meta_path = reuse_dir / "generation_metadata.json"
    if not meta_path.exists():
        print(
            f"Warning: no generation_metadata.json found in {reuse_dir} — "
            "cannot verify perturbation model used at generation time."
        )
        return

    meta = json.loads(meta_path.read_text())
    stored_model = meta.get("perturbation_model", "unknown")
    print(f"Reusing perturbations (generated with model={stored_model!r})")


if __name__ == "__main__":
    main()
