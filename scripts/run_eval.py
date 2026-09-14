from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from evals.core import Dataset, EvalConfig
from evals.reporters import Reporter
from evals.runner import Runner
from evals.scorer_factory import SCORER_CHOICES, build_scorer


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run an LLM eval")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--scorer", required=True, help=f"Scorer name ({SCORER_CHOICES})")
    parser.add_argument("--model", default=None, help="Model ID (overrides DEFAULT_MODEL env var)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run")
    parser.add_argument("--output", default=None, help="Results directory (overrides RESULTS_DIR env var)")
    # Scorer-specific args
    parser.add_argument("--pattern", default=None, help="Regex pattern(s) for regex/multi-regex scorers")
    parser.add_argument("--schema", default=None, help="Path to JSON schema file for schema scorer")
    parser.add_argument("--scale", type=int, default=5, help="Score scale for judge scorer (default: 5)")
    parser.add_argument("--judge-model", default=None, help="Model ID for judge (overrides JUDGE_MODEL env var)")
    parser.add_argument("--fast-tier", default="normalised", choices=["exact", "normalised"], help="Fast tier for cascade scorer (default: normalised)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Fast-tier threshold for cascade scorer (default: 1.0)")
    args = parser.parse_args()

    config = EvalConfig(model=args.model) if args.model else EvalConfig()
    scorer = build_scorer(args, evaluated_model=config.model)

    ds = Dataset.from_jsonl(args.dataset, limit=args.limit)

    wrapped = tqdm(ds, total=len(ds), desc="Running eval")
    results = Runner().run(wrapped, scorer, config)

    reporter_kwargs = {}
    if args.output:
        reporter_kwargs["results_dir"] = Path(args.output)
    reporter = Reporter(**reporter_kwargs)

    dataset_name = Path(args.dataset).stem
    scorer_name = args.scorer

    output, path = reporter.report(results, dataset_name, scorer_name, model=config.model)
    print(output)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
