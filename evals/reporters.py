from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

from evals.core import RunResult


def _sanitise_model(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


@dataclass
class Reporter:
    results_dir: Path = field(default_factory=lambda: Path(os.environ.get("RESULTS_DIR", "results")))

    def _summarise(self, results: list[RunResult]) -> dict[str, Any]:
        latencies = [r.latency_ms for r in results]
        scores = [r.score for r in results if r.score is not None]
        api_errors = sum(1 for r in results if r.error and r.score is None and r.completion is None)
        parse_failures = sum(1 for r in results if r.error and r.score is None and r.completion is not None)
        n = len(results)
        mean_score = statistics.mean(scores) if scores else None
        p50_latency = int(statistics.median(latencies)) if latencies else 0
        sorted_latencies = sorted(latencies)
        p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))] if sorted_latencies else 0
        total_errors = api_errors + parse_failures
        error_rate = total_errors / n if n else 0.0

        clean_rate: float | None = None
        format_pass_rate: float | None = None
        repair_failure_rate: float | None = None
        statuses = [r.metadata["format_status"] for r in results if "format_status" in r.metadata]
        if statuses:
            n_status = len(statuses)
            clean_rate = sum(1 for s in statuses if s == "clean") / n_status
            format_pass_rate = sum(1 for s in statuses if s in ("clean", "repaired")) / n_status
            repair_failure_rate = sum(1 for s in statuses if s == "repair_failed") / n_status

        tiers = [r.metadata["tier_used"] for r in results if "tier_used" in r.metadata]
        judge_rate = sum(1 for t in tiers if t == "judge") / len(tiers) if tiers else None

        return {
            "mean_score": mean_score,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p95_low_confidence": n < 20,
            "n": n,
            "api_errors": api_errors,
            "parse_failures": parse_failures,
            "error_rate": error_rate,
            "clean_rate": clean_rate,
            "format_pass_rate": format_pass_rate,
            "repair_failure_rate": repair_failure_rate,
            "judge_rate": judge_rate,
        }

    @staticmethod
    def _outcome_str(r: RunResult) -> str:
        if r.score is None:
            return "error"
        if r.score >= 1.0:
            return "pass"
        if r.score >= 0.5:
            return "partial"
        return "fail"

    def report(
        self,
        results: list[RunResult],
        dataset_name: str,
        scorer_name: str,
        model: str = "unknown",
    ) -> tuple[str, Path]:
        has_tier   = any("tier_used"     in r.metadata for r in results)
        has_format = any("format_status" in r.metadata for r in results)

        rows = []
        for r in results:
            row = [
                r.sample.id,
                f"{r.score:.2f}" if r.score is not None else "—",
                self._outcome_str(r),
                r.latency_ms,
            ]
            if has_tier:
                sm = r.metadata
                if "tier_used" in sm:
                    tier = sm["tier_used"]
                    if tier == "fast":
                        fast = sm.get("fast_score")
                        tier_str = f"fast/{fast:.2f}" if fast is not None else "fast"
                    else:
                        tier_str = "judge"
                else:
                    tier_str = ""
                row.append(tier_str)
            if has_format:
                row.append(r.metadata.get("format_status", ""))
            row.append(r.error or "")
            rows.append(row)

        headers = ["id", "score", "outcome", "latency_ms"]
        if has_tier:
            headers.append("tier_used")
        if has_format:
            headers.append("format_status")
        headers.append("error")

        table_str = tabulate(rows, headers=headers, tablefmt="simple")

        summary = self._summarise(results)
        mean_score = summary["mean_score"]
        p50_latency = summary["p50_latency_ms"]
        p95_latency = summary["p95_latency_ms"]
        n = summary["n"]

        p95_str = f"{p95_latency}ms"
        if summary["p95_low_confidence"]:
            p95_str += f" (n={n} ⚠)"

        summary_parts = [
            f"mean_score={mean_score:.3f}" if mean_score is not None else "mean_score=—",
            f"p50_latency={p50_latency}ms",
            f"p95_latency={p95_str}",
            f"api_errors={summary['api_errors']}",
            f"parse_failures={summary['parse_failures']}",
            f"error_rate={summary['error_rate']:.1%}",
        ]
        if summary["clean_rate"] is not None:
            summary_parts += [
                f"clean_rate={summary['clean_rate']:.1%}",
                f"fmt_pass_rate={summary['format_pass_rate']:.1%}",
                f"repair_fail_rate={summary['repair_failure_rate']:.1%}",
            ]
        if summary["judge_rate"] is not None:
            summary_parts.append(f"judge_rate={summary['judge_rate']:.1%}")
        summary_str = "  ".join(summary_parts)

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        safe_model = _sanitise_model(model)
        run_dir = (
            self.results_dir / "runs" / date / f"{time_str}_{safe_model}_{dataset_name}_{scorer_name}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        run_payload = {
            "dataset": dataset_name,
            "scorer": scorer_name,
            "model": model,
            "timestamp": now.isoformat(timespec="seconds"),
            "summary": {k: v for k, v in summary.items() if k != "p95_low_confidence"},
        }
        (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2))

        with (run_dir / "samples.jsonl").open("w") as f:
            for r in results:
                record: dict[str, Any] = {
                    "id": r.sample.id,
                    "expected": r.sample.expected,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "completion": r.completion,
                    "error": r.error,
                    "scorer_metadata": r.metadata,
                }
                # Preserve sample metadata for show.py (e.g. context chunks for RAG)
                if r.sample.metadata:
                    record["sample_metadata"] = {
                        k: v for k, v in r.sample.metadata.items() if k != "id"
                    }
                f.write(json.dumps(record) + "\n")

        return f"{table_str}\n\n{summary_str}", run_dir

    def benchmark_report(
        self,
        model_results: list[tuple[str, list[RunResult]]],
        dataset_name: str,
        scorer_name: str,
    ) -> tuple[str, Path]:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        bench_dir = (
            self.results_dir / "benchmarks" / date / f"{time_str}_{dataset_name}_{scorer_name}"
        )
        bench_dir.mkdir(parents=True, exist_ok=True)

        summaries: dict[str, dict[str, Any]] = {}
        table_rows = []

        for model_id, results in model_results:
            s = self._summarise(results)
            summaries[model_id] = s

            mean_score_str = f"{s['mean_score']:.3f}" if s["mean_score"] is not None else "—"
            p95_str = f"{s['p95_latency_ms']}ms"
            if s["p95_low_confidence"]:
                p95_str += f" (n={s['n']} ⚠)"

            table_rows.append([
                model_id,
                mean_score_str,
                f"{s['p50_latency_ms']}ms",
                p95_str,
                f"{s['error_rate']:.1%}",
            ])

            safe_model = _sanitise_model(model_id)
            with (bench_dir / f"{safe_model}.jsonl").open("w") as f:
                for r in results:
                    record: dict[str, Any] = {
                        "id": r.sample.id,
                        "expected": r.sample.expected,
                        "score": r.score,
                        "latency_ms": r.latency_ms,
                        "completion": r.completion,
                        "error": r.error,
                        "scorer_metadata": r.metadata,
                    }
                    if r.sample.metadata:
                        record["sample_metadata"] = {
                            k: v for k, v in r.sample.metadata.items() if k != "id"
                        }
                    f.write(json.dumps(record) + "\n")

        has_format = any(s.get("clean_rate") is not None for s in summaries.values())
        has_judge = any(s.get("judge_rate") is not None for s in summaries.values())

        headers = ["model", "mean_score", "p50_latency", "p95_latency", "error_rate"]
        if has_format:
            headers += ["clean_rate", "fmt_pass_rate", "repair_fail_rate"]
        if has_judge:
            headers.append("judge_rate")

        if has_format or has_judge:
            extended_rows = []
            for (model_id, _), base_row in zip(model_results, table_rows, strict=True):
                s = summaries[model_id]
                row = list(base_row)
                if has_format:
                    row.append(f"{s['clean_rate']:.1%}" if s["clean_rate"] is not None else "—")
                    row.append(f"{s['format_pass_rate']:.1%}" if s["format_pass_rate"] is not None else "—")
                    row.append(f"{s['repair_failure_rate']:.1%}" if s["repair_failure_rate"] is not None else "—")
                if has_judge:
                    row.append(f"{s['judge_rate']:.1%}" if s["judge_rate"] is not None else "—")
                extended_rows.append(row)
            table_rows = extended_rows

        table_str = tabulate(
            table_rows,
            headers=headers,
            tablefmt="simple",
        )

        benchmark_payload = {
            "dataset": dataset_name,
            "scorer": scorer_name,
            "timestamp": now.isoformat(timespec="seconds"),
            "models": {
                model_id: {k: v for k, v in s.items() if k != "p95_low_confidence"}
                for model_id, s in summaries.items()
            },
        }
        (bench_dir / "benchmark.json").write_text(json.dumps(benchmark_payload, indent=2))

        return table_str, bench_dir
