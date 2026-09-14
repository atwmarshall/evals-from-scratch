from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

from evals.core import AnyScorer, Dataset, EvalConfig, RunResult
from evals.runner import Runner

logger = logging.getLogger(__name__)


def run_variations(
    variations: dict[str, Dataset],
    scorer: AnyScorer,
    config: EvalConfig,
) -> dict[str, list[RunResult]]:
    """Run each variation dataset through Runner and return results keyed by variation name.

    Iterates the variations dict (output of VariationGenerator.validate_variations),
    calling Runner().run() for each. Empty datasets are skipped with a warning.

    Args:
        variations: Dict mapping variation name to Dataset. Typically the validated
                    output of VariationGenerator.validate_variations() — includes
                    "baseline" plus each variation type.
        scorer: The scorer to evaluate completions with.
        config: EvalConfig controlling model, temperature, retries, etc.

    Returns:
        Dict mapping variation name to list[RunResult]. Same keys as input,
        minus any empty datasets that were skipped.
    """
    results: dict[str, list[RunResult]] = {}
    for name, dataset in variations.items():
        if len(dataset) == 0:
            logger.warning("run_variations: skipping %r — empty dataset", name)
            continue
        logger.info("run_variations: running %r (%d samples)", name, len(dataset))
        results[name] = Runner().run(dataset, scorer, config)
    return results


@dataclass
class SensitivityReporter:
    """Computes and reports scorer sensitivity across input variations.

    Takes dict[str, list[RunResult]] (output of run_variations) and produces:
    - Per-sample variance table: how much each sample's score varies across variations
    - Per-variation summary: which variation types destabilise scores most

    Variance formula: statistics.variance(scores) across all variation columns for
    a given sample (requires ≥ 2 non-None scores). Verdict: "unstable" if
    variance > 0.05 else "ok". Samples with fewer than 2 valid scores get "n/a".

    The "baseline" variation is the anchor: its delta_from_baseline and
    mean_variance are null by design. Baseline is the reference point, not a
    variation under test. Including it in its own variance calculation would
    artificially deflate variance for variations that agree with baseline.
    """

    results_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("RESULTS_DIR", "results"))
    )

    def report(
        self,
        results_by_variation: dict[str, list[RunResult]],
        dataset_name: str,
        scorer_name: str,
        model: str = "unknown",
        run_config: dict[str, Any] | None = None,
    ) -> tuple[str, Path]:
        """Compute sensitivity metrics, format tables, save artefacts.

        Args:
            results_by_variation: Output of run_variations().
            dataset_name: Source dataset stem — used in output path and saved JSON.
            scorer_name: Scorer name — used in output path and saved JSON.
            model: Model ID that was evaluated.
            run_config: Optional dict of run-time settings to record in sensitivity.json
                        for auditability (variation_model, judge_model, thresholds, etc.).
                        All values are stored as-is — caller decides what to include.

        Returns:
            (output_string, sens_dir) — the combined table string and saved directory.
        """
        variation_names = list(results_by_variation.keys())

        per_sample = self._compute_per_sample(results_by_variation, variation_names)
        per_variation = self._compute_per_variation(
            results_by_variation, per_sample, variation_names
        )

        n_unstable = sum(1 for r in per_sample if r["verdict"] == "unstable")
        n_total = len(per_sample)

        non_baseline = [
            r
            for r in per_variation
            if r["variation"] != "baseline" and r["mean_variance"] is not None
        ]
        most_destabilising = (
            max(non_baseline, key=lambda r: r["mean_variance"])["variation"]
            if non_baseline
            else None
        )

        baseline_n = len(results_by_variation.get("baseline", []))
        most_dest_n = (
            len(results_by_variation.get(most_destabilising, [])) if most_destabilising else None
        )

        # Per-sample table
        ps_headers = ["id"] + variation_names + ["variance", "verdict"]
        ps_rows = []
        for row in per_sample:
            ps_row = [row["id"]]
            for name in variation_names:
                score = row.get(name)
                ps_row.append(f"{score:.2f}" if score is not None else "—")
            ps_row.append(f"{row['variance']:.4f}" if row["variance"] is not None else "—")
            ps_row.append(row["verdict"])
            ps_rows.append(ps_row)
        ps_table = tabulate(ps_rows, headers=ps_headers, tablefmt="simple")

        # Per-variation summary table
        pv_headers = ["variation", "mean_score", "delta_from_baseline", "mean_variance"]
        pv_rows = []
        for row in per_variation:
            pv_rows.append(
                [
                    row["variation"],
                    f"{row['mean_score']:.3f}" if row["mean_score"] is not None else "—",
                    f"{row['delta_from_baseline']:+.3f}"
                    if row["delta_from_baseline"] is not None
                    else "—",
                    f"{row['mean_variance']:.4f}" if row["mean_variance"] is not None else "—",
                ]
            )
        pv_table = tabulate(pv_rows, headers=pv_headers, tablefmt="simple")

        summary_line = f"{n_unstable} unstable / {n_total} samples"
        if most_destabilising:
            if most_dest_n is not None and most_dest_n != baseline_n:
                summary_line += f"  (most destabilising: {most_destabilising}, n={most_dest_n} vs baseline n={baseline_n})"
            else:
                summary_line += f"  (most destabilising: {most_destabilising})"

        output_str = (
            f"── PER-SAMPLE VARIANCE ──\n{ps_table}\n\n"
            f"── PER-VARIATION SUMMARY ──\n{pv_table}\n\n"
            f"{summary_line}"
        )

        # Save artefacts
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        sens_dir = (
            self.results_dir / "sensitivity" / date / f"{time_str}_{dataset_name}_{scorer_name}"
        )
        sens_dir.mkdir(parents=True, exist_ok=True)

        for name, results in results_by_variation.items():
            with (sens_dir / f"{name}.jsonl").open("w") as f:
                for r in results:
                    f.write(
                        json.dumps(
                            {
                                "id": r.sample.id,
                                "expected": r.sample.expected,
                                "score": r.score,
                                "latency_ms": r.latency_ms,
                                "completion": r.completion,
                                "error": r.error,
                                "scorer_metadata": r.metadata,
                            }
                        )
                        + "\n"
                    )

        payload = {
            "dataset": dataset_name,
            "scorer": scorer_name,
            "model": model,
            "timestamp": now.isoformat(timespec="seconds"),
            "materialised_at": now.isoformat(timespec="milliseconds"),
            "run_config": run_config or {},
            "variation_names": variation_names,
            "schema_notes": {
                "baseline_delta_from_baseline": (
                    "null by design — baseline is the reference point, not a variation under test"
                ),
                "baseline_mean_variance": (
                    "null by design — baseline is the anchor, excluded from variance calculations; "
                    "including it would artificially deflate variance for variations that agree with baseline"
                ),
            },
            "per_sample": per_sample,
            "per_variation": per_variation,
            "summary": {
                "n_unstable": n_unstable,
                "n_total": n_total,
                "most_destabilising": most_destabilising,
                "most_destabilising_n": most_dest_n,
                "baseline_n": baseline_n,
            },
        }
        (sens_dir / "sensitivity.json").write_text(json.dumps(payload, indent=2))

        return output_str, sens_dir

    def _compute_per_sample(
        self,
        results_by_variation: dict[str, list[RunResult]],
        variation_names: list[str],
    ) -> list[dict[str, Any]]:
        # Build lookup: variation_name -> sample_id -> score
        scores_lookup: dict[str, dict[str, float | None]] = {}
        all_ids_ordered: list[str] = []
        seen_ids: set[str] = set()

        for name in variation_names:
            scores_lookup[name] = {}
            for r in results_by_variation.get(name, []):
                scores_lookup[name][r.sample.id] = r.score
                if r.sample.id not in seen_ids:
                    all_ids_ordered.append(r.sample.id)
                    seen_ids.add(r.sample.id)

        rows = []
        for sample_id in sorted(all_ids_ordered):
            row: dict[str, Any] = {"id": sample_id}

            for name in variation_names:
                row[name] = scores_lookup[name].get(sample_id)

            valid_scores = [row[name] for name in variation_names if row[name] is not None]
            if len(valid_scores) >= 2:
                var = statistics.variance(valid_scores)
                verdict = "unstable" if var > 0.05 else "ok"
            else:
                var = None
                verdict = "n/a"

            row["variance"] = var
            row["verdict"] = verdict
            rows.append(row)

        return rows

    def _compute_per_variation(
        self,
        results_by_variation: dict[str, list[RunResult]],
        per_sample_rows: list[dict[str, Any]],
        variation_names: list[str],
    ) -> list[dict[str, Any]]:
        # Baseline mean for delta calculation
        baseline_mean: float | None = None
        if "baseline" in variation_names:
            baseline_scores = [
                row["baseline"] for row in per_sample_rows if row.get("baseline") is not None
            ]
            baseline_mean = statistics.mean(baseline_scores) if baseline_scores else None

        summary_rows = []
        for name in variation_names:
            variation_scores = [row[name] for row in per_sample_rows if row.get(name) is not None]
            mean_score = statistics.mean(variation_scores) if variation_scores else None

            if name == "baseline":
                delta = None
                mean_variance = None  # anchor — null by design
            else:
                delta = (
                    (mean_score - baseline_mean)
                    if mean_score is not None and baseline_mean is not None
                    else None
                )
                variances = [
                    row["variance"]
                    for row in per_sample_rows
                    if row.get(name) is not None and row["variance"] is not None
                ]
                mean_variance = statistics.mean(variances) if variances else None

            summary_rows.append(
                {
                    "variation": name,
                    "mean_score": mean_score,
                    "delta_from_baseline": delta,
                    "mean_variance": mean_variance,
                }
            )

        return summary_rows
