from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path

import questionary
from dotenv import load_dotenv
from tqdm import tqdm

from evals.core import Dataset, EvalConfig
from evals.reporters import Reporter
from evals.runner import Runner
from evals.scorer_factory import SCORER_CHOICES, build_scorer


def _list_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running 'ollama list': {e}", file=sys.stderr)
        sys.exit(1)

    lines = result.stdout.strip().splitlines()
    models = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a benchmark across multiple models")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--scorer", required=True, help=f"Scorer name ({SCORER_CHOICES})")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run")
    parser.add_argument("--output", default=None, help="Results directory (overrides RESULTS_DIR env var)")
    # Scorer-specific args
    parser.add_argument("--pattern", default=None, help="Regex pattern(s) for regex/multi-regex scorers")
    parser.add_argument("--schema", default=None, help="Path to JSON schema file for schema scorer")
    parser.add_argument("--scale", type=int, default=5, help="Score scale for judge scorer (default: 5)")
    parser.add_argument("--judge-model", default=None, help="Model ID for judge (overrides JUDGE_MODEL env var)")
    parser.add_argument("--fast-tier", default="normalised", choices=["exact", "normalised"], help="Fast tier for cascade scorer")
    parser.add_argument("--threshold", type=float, default=1.0, help="Fast-tier threshold for cascade scorer (default: 1.0)")
    parser.add_argument("--models", default=None, help="Comma-separated list of model IDs (skips interactive selection)")
    parser.add_argument("--timeout", type=int, default=120, help="Per-sample timeout in seconds (default: 120)")
    args = parser.parse_args()

    if args.models:
        selected = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = _list_ollama_models()
        if not models:
            print("No models found in 'ollama list'.", file=sys.stderr)
            sys.exit(1)

        selected = questionary.checkbox(
            "Select models to benchmark (space to toggle, enter to confirm):",
            choices=models,
        ).ask()

        if not selected:
            print("No models selected — exiting.")
            sys.exit(0)

    scorer = build_scorer(args, evaluated_model=selected[0])
    ds = Dataset.from_jsonl(args.dataset, limit=args.limit)
    base_config = EvalConfig(timeout_seconds=args.timeout)

    reporter_kwargs = {}
    if args.output:
        reporter_kwargs["results_dir"] = Path(args.output)
    reporter = Reporter(**reporter_kwargs)

    dataset_name = Path(args.dataset).stem
    scorer_name = args.scorer

    print(f"\nRunning {scorer_name} on {dataset_name} ({len(ds)} samples) across {len(selected)} model(s)...\n")

    all_model_results = []
    for model_id in selected:
        print(f"  → {model_id}")
        wrapped = tqdm(ds, total=len(ds), desc=f"  {model_id}", leave=False)
        if hasattr(scorer, "set_evaluated_model"):
            scorer.set_evaluated_model(model_id)
        config = dataclasses.replace(base_config, model=model_id)
        results = Runner().run(wrapped, scorer, config)
        all_model_results.append((model_id, results))

    table_str, bench_dir = reporter.benchmark_report(all_model_results, dataset_name, scorer_name)
    print("\n" + table_str)
    print(f"\nSaved → {bench_dir}")


if __name__ == "__main__":
    main()
