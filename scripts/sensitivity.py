"""sensitivity.py — measure scorer stability across semantically equivalent input variations.

Flow:
  load dataset → generate variations (or reuse saved) → validate (judge) →
  run each variation through model → SensitivityReporter → save results

Part A of Challenge 6: scorer reliability. Hold input + model fixed, vary the
scorer prompt. Question: "can I trust my ruler?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from evals.core import Dataset, EvalConfig
from evals.scorer_factory import SCORER_CHOICES, build_scorer
from evals.scorers.llm_judge import LLMJudgeScorer
from evals.sensitivity_reporter import SensitivityReporter, run_variations
from evals.variation_generator import VariationGenerator


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: measure scorer stability across input variations"
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--scorer", required=True, help=f"Scorer to test ({SCORER_CHOICES})")
    parser.add_argument("--model", default=None, help="Model to evaluate (overrides DEFAULT_MODEL)")
    parser.add_argument(
        "--variations", nargs="+", default=None,
        help="Variation types to run (default: all 5 — synonym_swap rephrase add_noise formal concise)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument(
        "--validation-threshold", type=float, default=0.8,
        help="Min judge score for a variation to be considered meaning-preserving (default: 0.8)",
    )
    parser.add_argument(
        "--no-save-variations", action="store_true",
        help="Skip saving generated variations to datasets/generated/sensitivity/ "
             "(ignored when --reuse-variations is set)",
    )
    parser.add_argument(
        "--reuse-variations", default=None, metavar="DIR",
        help="Reuse previously saved variations from this directory instead of generating new ones",
    )
    parser.add_argument("--output", default=None, help="Results directory (overrides RESULTS_DIR)")
    # Scorer-specific flags (passed through to build_scorer)
    parser.add_argument("--pattern", default=None, help="Regex pattern for regex/multi-regex scorers")
    parser.add_argument("--schema", default=None, help="Path to JSON schema file for schema scorer")
    parser.add_argument("--scale", type=int, default=5, help="Judge score scale (default: 5)")
    parser.add_argument(
        "--judge-model", default=None,
        help="Model for LLM judge — used for both eval scorer and variation validation (overrides JUDGE_MODEL)",
    )
    parser.add_argument(
        "--fast-tier", default="normalised", choices=["exact", "normalised"],
        help="Fast tier for cascade scorer (default: normalised)",
    )
    parser.add_argument(
        "--cascade-threshold", type=float, default=1.0,
        help="Fast-tier threshold for cascade scorer (default: 1.0). "
             "Separate from --validation-threshold which controls variation filtering.",
    )
    args = parser.parse_args()

    # --- resolve models ---
    config = EvalConfig(model=args.model) if args.model else EvalConfig()
    variation_model = (
        os.environ.get("VARIATION_MODEL") or os.environ.get("DEFAULT_MODEL", "llama3.2:3b")
    )
    judge_model = args.judge_model or os.environ.get("JUDGE_MODEL", "llama3.2:3b")

    # --- model separation check (skip when reusing saved variations) ---
    if not args.reuse_variations and variation_model == judge_model:
        sys.exit(
            f"Error: VARIATION_MODEL ({variation_model!r}) must differ from JUDGE_MODEL ({judge_model!r}).\n"
            "Set VARIATION_MODEL in .env to use a different model for variation generation.\n"
            "Using the same model to generate and score variations measures self-consistency, "
            "not scorer reliability — the wrong thing to measure."
        )

    # --- load dataset ---
    ds = Dataset.from_jsonl(args.dataset, limit=args.limit)
    dataset_name = Path(args.dataset).stem
    print(f"Loaded {len(ds)} samples from {args.dataset}")

    # --- build eval scorer ---
    scorer = build_scorer(args, evaluated_model=config.model)

    # --- build validation judge (always a judge, regardless of main scorer) ---
    validation_judge = LLMJudgeScorer(
        scale=args.scale,
        evaluated_model=None,
        **({"model": args.judge_model} if args.judge_model else {}),
    )

    # --- generate or load variations ---
    if args.reuse_variations:
        reuse_dir = Path(args.reuse_variations)
        _warn_reuse_metadata(reuse_dir, args.validation_threshold, judge_model)
        variations = VariationGenerator.load_variations(reuse_dir)
        # load_variations() never returns "baseline" — inject from original dataset
        variations = {"baseline": ds, **variations}
        if args.variations:
            keep = {"baseline"} | set(args.variations)
            variations = {k: v for k, v in variations.items() if k in keep}
        print(f"Reusing {len(variations) - 1} variation type(s) from {reuse_dir}")
    else:
        gen = VariationGenerator()
        print(f"Generating variations using {gen.model}...")
        raw_variations = gen.generate(ds, variations=args.variations)

        print(f"Validating variations (threshold={args.validation_threshold}, judge={judge_model})...")
        variations, discards = gen.validate_variations(
            raw_variations,
            validation_scorer=validation_judge,
            threshold=args.validation_threshold,
        )
        if discards:
            print(f"  {len(discards)} sample(s) discarded by validity filter (see discarded.jsonl)")

        if not args.no_save_variations:
            saved_dir = gen.save_variations(
                validated=variations,
                original=raw_variations,
                source_path=args.dataset,
                threshold=args.validation_threshold,
                discards=discards,
            )
            print(f"Variations saved → {saved_dir}")

    # --- run variations ---
    n_variations = len([k for k in variations if k != "baseline"])
    print(f"Running {len(variations)} variation(s) ({n_variations} + baseline) through {config.model}...")
    results_by_variation = run_variations(variations, scorer, config)

    if not results_by_variation:
        sys.exit("No variation results — all datasets were empty after validation.")

    # --- report ---
    reporter_kwargs: dict = {}
    if args.output:
        reporter_kwargs["results_dir"] = Path(args.output)
    reporter = SensitivityReporter(**reporter_kwargs)

    output_str, sens_dir = reporter.report(
        results_by_variation,
        dataset_name=dataset_name,
        scorer_name=args.scorer,
        model=config.model,
        run_config={
            "variation_model": variation_model,
            "judge_model": judge_model,
            "validation_threshold": args.validation_threshold,
            "cascade_threshold": args.cascade_threshold,
            "limit": args.limit,
            "reused_from": str(args.reuse_variations) if args.reuse_variations else None,
        },
    )
    print()
    print(output_str)
    print(f"\nSaved → {sens_dir}")


def _warn_reuse_metadata(reuse_dir: Path, current_threshold: float, current_judge: str) -> None:
    """Warn if reused variations were validated with different settings."""
    meta_path = reuse_dir / "generation_metadata.json"
    if not meta_path.exists():
        print(
            f"Warning: no generation_metadata.json found in {reuse_dir} — "
            "cannot verify validation settings used at generation time."
        )
        return

    meta = json.loads(meta_path.read_text())
    stored_threshold = meta.get("threshold")
    stored_model = meta.get("variation_model", "unknown")

    if stored_threshold is not None and stored_threshold != current_threshold:
        print(
            f"Warning: reusing variations validated with threshold={stored_threshold}, "
            f"current --validation-threshold={current_threshold}. "
            "Results may include samples that would fail the current threshold."
        )

    print(f"Reusing variations (generated with model={stored_model!r})")


if __name__ == "__main__":
    main()
