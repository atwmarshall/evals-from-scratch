from __future__ import annotations

import json
import logging
import os
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

from evals.core import RunResult

logger = logging.getLogger(__name__)


@dataclass
class RobustnessReporter:
    """Computes and reports model robustness across adversarially perturbed inputs.

    Takes dict[str, list[RunResult]] keyed by perturbation name and produces:
    - Per-sample degradation table: how much each sample's score drops across perturbations
    - Per-perturbation summary: which perturbation types cause the most degradation

    Degradation formula: baseline_score - mean(non-None perturbation scores) for a given
    sample. The metric is directional — positive means the model scored lower under
    perturbation. Verdicts:
      "robust"  — degradation < 0.1   (strictly less than)
      "fragile" — 0.1 <= degradation < 0.3
      "brittle" — degradation >= 0.3
      "n/a"     — baseline score is None, or no valid perturbation scores to average

    The "baseline" perturbation is the anchor: its delta_from_baseline is null by design.
    Per-perturbation delta_from_baseline = perturbation_mean - baseline_mean (negative
    when the perturbation degrades performance). This is a different quantity from the
    per-sample degradation metric.

    Results are saved under results/robustness/ keyed by model (not scorer), because
    robustness measures how the model responds to adversarial inputs — the scorer is
    held fixed, the model is what's under test.
    """

    results_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("RESULTS_DIR", "results"))
    )

    def report(
        self,
        results_by_perturbation: dict[str, list[RunResult]],
        dataset_name: str,
        scorer_name: str,
        model: str = "unknown",
        run_config: dict | None = None,
    ) -> tuple[str, Path]:
        """Compute robustness metrics, format tables, save artefacts.

        Args:
            results_by_perturbation: Dict mapping perturbation name to list[RunResult].
                                     Produced by calling Runner().run() per perturbation.
            dataset_name: Source dataset stem — used in output path and saved JSON.
            scorer_name: Scorer name — recorded in robustness.json for auditability.
            model: Model ID that was evaluated — used in output path and saved JSON.
            run_config: Optional dict of run-time settings to record in robustness.json.

        Returns:
            (output_string, robustness_dir) — the combined table string and saved directory.
        """
        perturbation_names = list(results_by_perturbation.keys())

        per_sample = self._compute_per_sample(results_by_perturbation, perturbation_names)
        per_perturbation = self._compute_per_perturbation(results_by_perturbation, per_sample, perturbation_names)

        n_robust = sum(1 for r in per_sample if r["verdict"] == "robust")
        n_fragile = sum(1 for r in per_sample if r["verdict"] == "fragile")
        n_brittle = sum(1 for r in per_sample if r["verdict"] == "brittle")
        n_na = sum(1 for r in per_sample if r["verdict"] == "n/a")
        n_total = len(per_sample)
        n_degraded = n_fragile + n_brittle

        non_baseline = [
            r for r in per_perturbation
            if r["perturbation"] != "baseline" and r["delta_from_baseline"] is not None
        ]
        most_degrading = (
            min(non_baseline, key=lambda r: r["delta_from_baseline"])["perturbation"]
            if non_baseline else None
        )

        baseline_n = len(results_by_perturbation.get("baseline", []))
        most_degrading_n = (
            len(results_by_perturbation.get(most_degrading, []))
            if most_degrading else None
        )

        # Per-sample table
        ps_headers = ["id"] + perturbation_names + ["degradation", "verdict"]
        ps_rows = []
        for row in per_sample:
            ps_row = [row["id"]]
            for name in perturbation_names:
                score = row.get(name)
                ps_row.append(f"{score:.2f}" if score is not None else "—")
            ps_row.append(f"{row['degradation']:.4f}" if row["degradation"] is not None else "—")
            ps_row.append(row["verdict"])
            ps_rows.append(ps_row)
        ps_table = tabulate(ps_rows, headers=ps_headers, tablefmt="simple")

        # Per-perturbation summary table
        pv_headers = ["perturbation", "mean_score", "delta_from_baseline"]
        pv_rows = []
        for row in per_perturbation:
            pv_rows.append([
                row["perturbation"],
                f"{row['mean_score']:.3f}" if row["mean_score"] is not None else "—",
                f"{row['delta_from_baseline']:+.3f}" if row["delta_from_baseline"] is not None else "—",
            ])
        pv_table = tabulate(pv_rows, headers=pv_headers, tablefmt="simple")

        summary_line = f"{n_degraded} degraded / {n_total} samples"
        if most_degrading:
            if most_degrading_n is not None and most_degrading_n != baseline_n:
                summary_line += f"  (most degrading: {most_degrading}, n={most_degrading_n} vs baseline n={baseline_n})"
            else:
                summary_line += f"  (most degrading: {most_degrading})"

        output_str = (
            f"── PER-SAMPLE DEGRADATION ──\n{ps_table}\n\n"
            f"── PER-PERTURBATION SUMMARY ──\n{pv_table}\n\n"
            f"{summary_line}"
        )

        # Save artefacts
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        model_slug = re.sub(r"[:/]", "_", model)
        rob_dir = (
            self.results_dir / "robustness" / date
            / f"{time_str}_{dataset_name}_{model_slug}"
        )
        rob_dir.mkdir(parents=True, exist_ok=True)

        for name, results in results_by_perturbation.items():
            with (rob_dir / f"{name}.jsonl").open("w") as f:
                for r in results:
                    f.write(json.dumps({
                        "id": r.sample.id,
                        "expected": r.sample.expected,
                        "score": r.score,
                        "latency_ms": r.latency_ms,
                        "completion": r.completion,
                        "error": r.error,
                        "scorer_metadata": r.metadata,
                    }) + "\n")

        payload = {
            "dataset": dataset_name,
            "scorer": scorer_name,
            "model": model,
            "timestamp": now.isoformat(timespec="seconds"),
            "materialised_at": now.isoformat(timespec="milliseconds"),
            "run_config": run_config or {},
            "perturbation_names": perturbation_names,
            "schema_notes": {
                "baseline_delta_from_baseline": (
                    "null by design — baseline is the reference point, not a perturbation under test"
                ),
                "degradation_vs_delta": (
                    "per_sample degradation = baseline_score - mean(perturbation_scores), positive = model degraded; "
                    "per_perturbation delta_from_baseline = perturbation_mean - baseline_mean, negative = degraded"
                ),
            },
            "per_sample": per_sample,
            "per_perturbation": per_perturbation,
            "summary": {
                "n_robust": n_robust,
                "n_fragile": n_fragile,
                "n_brittle": n_brittle,
                "n_na": n_na,
                "n_total": n_total,
                "most_degrading": most_degrading,
                "most_degrading_n": most_degrading_n,
                "baseline_n": baseline_n,
            },
        }
        (rob_dir / "robustness.json").write_text(json.dumps(payload, indent=2))

        return output_str, rob_dir

    def _compute_per_sample(
        self,
        results_by_perturbation: dict[str, list[RunResult]],
        perturbation_names: list[str],
    ) -> list[dict]:
        # Build lookup: perturbation_name -> sample_id -> score
        scores_lookup: dict[str, dict[str, float | None]] = {}
        all_ids_ordered: list[str] = []
        seen_ids: set[str] = set()

        for name in perturbation_names:
            scores_lookup[name] = {}
            for r in results_by_perturbation.get(name, []):
                scores_lookup[name][r.sample.id] = r.score
                if r.sample.id not in seen_ids:
                    all_ids_ordered.append(r.sample.id)
                    seen_ids.add(r.sample.id)

        rows = []
        for sample_id in sorted(all_ids_ordered):
            row: dict = {"id": sample_id}

            for name in perturbation_names:
                row[name] = scores_lookup[name].get(sample_id)

            baseline_score = row.get("baseline")
            perturbation_scores = [
                row[name] for name in perturbation_names
                if name != "baseline" and row.get(name) is not None
            ]

            if baseline_score is None or not perturbation_scores:
                degradation = None
                verdict = "n/a"
            else:
                degradation = baseline_score - statistics.mean(perturbation_scores)
                if degradation < 0.1:    # strictly less than 0.1
                    verdict = "robust"
                elif degradation < 0.3:  # 0.1 <= degradation < 0.3
                    verdict = "fragile"
                else:                    # 0.3 and above
                    verdict = "brittle"

            row["degradation"] = degradation
            row["verdict"] = verdict
            rows.append(row)

        return rows

    def _compute_per_perturbation(
        self,
        results_by_perturbation: dict[str, list[RunResult]],
        per_sample_rows: list[dict],
        perturbation_names: list[str],
    ) -> list[dict]:
        # Baseline mean for delta calculation
        baseline_mean: float | None = None
        if "baseline" in perturbation_names:
            baseline_scores = [
                row["baseline"] for row in per_sample_rows
                if row.get("baseline") is not None
            ]
            baseline_mean = statistics.mean(baseline_scores) if baseline_scores else None

        summary_rows = []
        for name in perturbation_names:
            perturbation_scores = [
                row[name] for row in per_sample_rows if row.get(name) is not None
            ]
            mean_score = statistics.mean(perturbation_scores) if perturbation_scores else None

            if name == "baseline":
                delta = None  # anchor — null by design
            else:
                delta = (
                    (mean_score - baseline_mean)
                    if mean_score is not None and baseline_mean is not None
                    else None
                )

            summary_rows.append({
                "perturbation": name,
                "mean_score": mean_score,
                "delta_from_baseline": delta,
            })

        return summary_rows
