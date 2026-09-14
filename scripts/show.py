"""show.py — read benchmark/run/sample results without opening files manually.

Usage:
  show.py <path>              auto-detect type (benchmark dir / run dir / .jsonl)
  show.py <path> --id c3-001  show one sample in full (jsonl or benchmark model jsonl)
  show.py <path> --verbose    include full completions in error listings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tabulate import tabulate

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _short(text: str | None, n: int = 200) -> str:
    if text is None:
        return "(none)"
    text = text.replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


def _is_failure(r: dict[str, Any]) -> bool:
    s = r.get("score")
    return s is None or s < 1.0


def _is_strict_failure(r: dict[str, Any]) -> bool:
    s = r.get("score")
    return s is None or s < 0.5


def _outcome(row: dict[str, Any]) -> str:
    score = row.get("score")
    if score is not None:
        if score >= 1.0:
            return "pass"
        if score >= 0.5:
            return "partial"
        return "fail"
    return "error"


def _find_trace(
    results_dir: Path, date: str, model_id: str, sample_id: str
) -> dict[str, Any] | None:
    """Locate a judge trace JSON for a given model+sample under results/judge_traces/{date}/."""
    traces_root = results_dir / "judge_traces" / date
    if not traces_root.exists():
        return None
    # session dirs are named {time}_{model_id} — match by suffix
    suffix = f"_{model_id}"
    for session_dir in sorted(traces_root.iterdir()):
        if session_dir.name.endswith(suffix):
            trace_file = session_dir / f"{sample_id}.json"
            if trace_file.exists():
                trace: dict[str, Any] = json.loads(trace_file.read_text())
                return trace
    return None


# ---------------------------------------------------------------------------
# sensitivity mode
# ---------------------------------------------------------------------------


def inspect_sensitivity(
    sens_dir: Path,
    verbose: bool,
    sample_id: str | None = None,
    failures_only: bool = False,
) -> None:
    """Display sensitivity analysis results from a sensitivity.json directory.

    --verbose is accepted but ignored — completions are not stored in sensitivity.json.
    --failures-only filters the per-sample table to verdict=unstable rows only.
    """
    meta = json.loads((sens_dir / "sensitivity.json").read_text())
    run_cfg = meta.get("run_config") or {}

    print(
        f"SENSITIVITY  dataset={meta['dataset']}  scorer={meta['scorer']}  "
        f"model={meta['model']}  {meta['timestamp']}"
    )
    if run_cfg.get("variation_model"):
        print(
            f"             variation_model={run_cfg['variation_model']}  "
            f"judge={run_cfg.get('judge_model', '?')}  "
            f"validation_threshold={run_cfg.get('validation_threshold', '?')}"
        )
    print(f"dir: {sens_dir}\n")

    variation_names = meta.get("variation_names", [])
    per_sample = meta.get("per_sample", [])

    # --id: show one sample's scores across all variations
    if sample_id:
        matches = [r for r in per_sample if r.get("id") == sample_id]
        if not matches:
            print(f"Sample {sample_id!r} not found in sensitivity results.")
            sys.exit(1)
        row = matches[0]
        print(f"── sample {sample_id} ──\n")
        table_rows = [
            [
                name,
                f"{row[name]:.3f}" if row.get(name) is not None else "—",
                "anchor"
                if name == "baseline"
                else "discarded"
                if row.get(name) is None
                else row.get("verdict", "—"),
            ]
            for name in variation_names
        ]
        print(tabulate(table_rows, headers=["variation", "score", "note"], tablefmt="simple"))
        v = row.get("variance")
        print(f"\nvariance: {v:.4f}" if v is not None else "\nvariance: —")
        print(f"verdict:  {row.get('verdict', '—')}")
        return

    # Per-sample variance table
    if failures_only:
        display_rows = [r for r in per_sample if r.get("verdict") == "unstable"]
        label = f"  (unstable only: {len(display_rows)}/{len(per_sample)})"
    else:
        display_rows = per_sample
        label = f"  ({len(per_sample)} samples)"

    ps_headers = ["id"] + variation_names + ["variance", "verdict"]
    ps_rows = []
    for row in display_rows:
        ps_row = [row["id"]]
        for name in variation_names:
            score = row.get(name)
            ps_row.append(f"{score:.2f}" if score is not None else "—")
        v = row.get("variance")
        ps_row.append(f"{v:.4f}" if v is not None else "—")
        ps_row.append(row.get("verdict", "—"))
        ps_rows.append(ps_row)

    print(f"── PER-SAMPLE VARIANCE{label} ──")
    print(tabulate(ps_rows, headers=ps_headers, tablefmt="simple"))

    # Per-variation summary (always shown regardless of --failures-only)
    per_variation = meta.get("per_variation", [])
    pv_headers = ["variation", "mean_score", "delta_from_baseline", "mean_variance"]
    pv_rows = []
    for row in per_variation:
        ms = row.get("mean_score")
        delta = row.get("delta_from_baseline")
        mv = row.get("mean_variance")
        pv_rows.append(
            [
                row["variation"],
                f"{ms:.3f}" if ms is not None else "—",
                f"{delta:+.3f}" if delta is not None else "—",
                f"{mv:.4f}" if mv is not None else "—",
            ]
        )

    print("\n── PER-VARIATION SUMMARY ──")
    print(tabulate(pv_rows, headers=pv_headers, tablefmt="simple"))

    summary = meta.get("summary", {})
    n_unstable = summary.get("n_unstable", 0)
    n_total = summary.get("n_total", 0)
    most_dest = summary.get("most_destabilising")
    summary_line = f"\n{n_unstable} unstable / {n_total} samples"
    if most_dest:
        summary_line += f"  (most destabilising: {most_dest})"
    print(summary_line)


# ---------------------------------------------------------------------------
# benchmark mode
# ---------------------------------------------------------------------------


def inspect_benchmark(
    bench_dir: Path,
    verbose: bool,
    sample_id: str | None = None,
    failures_only: bool = False,
    strict: bool = False,
) -> None:
    meta = json.loads((bench_dir / "benchmark.json").read_text())
    date = bench_dir.parent.name  # results/benchmarks/{date}/...
    results_dir = bench_dir.parent.parent.parent  # results/

    no_traces_note = ""
    if meta.get("scorer") not in ("judge", "cascade"):
        no_traces_note = "  (no judge traces — scorer is not judge/cascade)"
    print(
        f"BENCHMARK  dataset={meta['dataset']}  scorer={meta['scorer']}  {meta['timestamp']}{no_traces_note}"
    )
    print(f"dir: {bench_dir}\n")

    # --- --id: show one sample across all models ---
    if sample_id:
        _benchmark_sample(
            bench_dir,
            meta,
            results_dir,
            date,
            sample_id,
            verbose,
            failures_only=failures_only,
            strict=strict,
        )
        return

    # --- comparison table from benchmark.json (suppressed with --failures-only) ---
    if not failures_only:
        models = meta["models"]
        has_format = any(s.get("clean_rate") is not None for s in models.values())
        has_judge = any(s.get("judge_rate") is not None for s in models.values())

        headers = ["model", "mean_score", "p50", "p95", "error_rate", "n_errors"]
        if has_format:
            headers += ["clean_rate", "fmt_pass_rate", "repair_fail_rate"]
        if has_judge:
            headers.append("judge_rate")

        table_rows = []
        for model_id, s in models.items():
            mean = f"{s['mean_score']:.3f}" if s["mean_score"] is not None else "—"
            p95_str = f"{s['p95_latency_ms']}ms"
            if s.get("n", 99) < 20:
                p95_str += f" (n={s['n']}⚠)"
            row = [
                model_id,
                mean,
                f"{s['p50_latency_ms']}ms",
                p95_str,
                f"{s['error_rate']:.1%}",
                s["api_errors"] + s["parse_failures"],
            ]
            if has_format:
                row.append(f"{s['clean_rate']:.1%}" if s.get("clean_rate") is not None else "—")
                row.append(
                    f"{s['format_pass_rate']:.1%}" if s.get("format_pass_rate") is not None else "—"
                )
                row.append(
                    f"{s['repair_failure_rate']:.1%}"
                    if s.get("repair_failure_rate") is not None
                    else "—"
                )
            if has_judge:
                row.append(f"{s['judge_rate']:.1%}" if s.get("judge_rate") is not None else "—")
            table_rows.append(row)
        print(tabulate(table_rows, headers=headers, tablefmt="simple"))

    # --- error breakdown ---
    any_errors = False
    for jsonl_path in sorted(bench_dir.glob("*.jsonl")):
        # reverse-engineer model_id from filename: replace _ with : for the colon, but
        # we can't perfectly invert sanitisation — use benchmark.json model keys instead
        rows = _load_jsonl(jsonl_path)
        failures = [r for r in rows if _outcome(r) not in ("pass", "partial")]
        if not failures:
            continue

        if not any_errors:
            print("\n── ERROR BREAKDOWN " + "─" * 60)
            any_errors = True

        # match filename stem back to a model id from benchmark.json
        stem = jsonl_path.stem
        model_id = next(
            (m for m in meta["models"] if stem == m.replace(":", "_").replace("/", "_")),
            stem,
        )
        print(f"\n[{model_id}]  {len(failures)}/{len(rows)} failed")

        for r in failures:
            err = r.get("error") or ""
            print(f"  {r['id']}  outcome=error  latency={r['latency_ms']}ms")

            if "timed out" in err:
                print(f"    error: {err}")
            elif "scorer returned None" in err:
                # try to find the trace
                trace = _find_trace(results_dir, date, model_id, r["id"])
                if trace:
                    raw = trace.get("raw_response", "")
                    print(f"    judge raw: {_short(raw, 300 if verbose else 150)}")
                    if trace.get("error"):
                        print(f"    parse error: {trace['error'][:120]}")
                else:
                    print(f"    (no trace found — check results/judge_traces/{date}/)")
                if verbose and r.get("completion"):
                    print(f"    completion: {_short(r['completion'], 400)}")
            elif err:
                print(f"    error: {err}")

    if not any_errors:
        scorer = meta.get("scorer", "")
        if scorer not in ("judge", "cascade"):
            print(f"\nNo errors.  (scorer={scorer!r} — no judge traces generated)")
        else:
            print("\nNo errors.")


# ---------------------------------------------------------------------------
# benchmark --id helper: one sample across all models
# ---------------------------------------------------------------------------


def _benchmark_sample(
    bench_dir: Path,
    meta: dict[str, Any],
    results_dir: Path,
    date: str,
    sample_id: str,
    verbose: bool,
    failures_only: bool = False,
    strict: bool = False,
) -> None:
    print(f"── sample {sample_id} across all models ──\n")
    found_any = False
    expected_shown = False
    for jsonl_path in sorted(bench_dir.glob("*.jsonl")):
        rows = _load_jsonl(jsonl_path)
        matches = [r for r in rows if r.get("id") == sample_id]
        if not matches:
            continue
        r = matches[0]

        if strict and not _is_strict_failure(r):
            continue
        elif failures_only and not _is_failure(r):
            continue

        found_any = True
        # show expected once at the top (same for all models)
        if not expected_shown and r.get("expected") is not None:
            print(f"expected:  {r['expected']}\n")
            expected_shown = True

        stem = jsonl_path.stem
        model_id = next(
            (m for m in meta["models"] if stem == m.replace(":", "_").replace("/", "_")),
            stem,
        )
        oc = _outcome(r)
        score_str = f"{r['score']:.3f}" if r.get("score") is not None else "—"
        print(f"[{model_id}]  score={score_str}  latency={r['latency_ms']}ms  outcome={oc}")

        err = r.get("error") or ""
        if "scorer returned None" in err:
            trace = _find_trace(results_dir, date, model_id, sample_id)
            if trace:
                print(f"  judge raw: {_short(trace.get('raw_response', ''), 200)}")

        if "timed out" in err:
            print(f"  error: {r.get('error')}")

        if verbose and r.get("completion"):
            print(f"  completion: {_short(r['completion'], 600)}")
        print()

    if not found_any:
        if strict:
            print(f"No strict failures found for sample {sample_id!r} across all models.")
        elif failures_only:
            print(f"No failures found for sample {sample_id!r} across all models.")
        else:
            print(f"Sample {sample_id!r} not found in any model jsonl under {bench_dir}")


# ---------------------------------------------------------------------------
# run mode
# ---------------------------------------------------------------------------


def inspect_run(
    run_dir: Path,
    verbose: bool,
    sample_id: str | None = None,
    failures_only: bool = False,
    strict: bool = False,
) -> None:
    meta = json.loads((run_dir / "run.json").read_text())
    rows = _load_jsonl(run_dir / "samples.jsonl")

    print(
        f"RUN  model={meta['model']}  dataset={meta['dataset']}  scorer={meta['scorer']}  {meta['timestamp']}"
    )
    s = meta["summary"]
    mean = f"{s['mean_score']:.3f}" if s["mean_score"] is not None else "—"
    print(
        f"     mean_score={mean}  p50={s['p50_latency_ms']}ms  errors={s['api_errors'] + s['parse_failures']}/{s['n']}\n"
    )

    if sample_id:
        matches = [r for r in rows if r.get("id") == sample_id]
        if not matches:
            print(f"Sample {sample_id!r} not found.")
            sys.exit(1)
        r = matches[0]
        print(
            f"id={r['id']}  score={r.get('score')}  latency={r.get('latency_ms')}ms  outcome={_outcome(r)}"
        )
        if r.get("expected") is not None:
            print(f"expected:  {r['expected']}")
        sm = r.get("scorer_metadata") or {}
        if sm.get("format_status"):
            print(f"format_status: {sm['format_status']}")
        if sm.get("judge_format_status"):
            print(f"judge_format_status: {sm['judge_format_status']}")
        if sm.get("tier_used"):
            fast = sm.get("fast_score")
            fast_str = f"{fast:.3f}" if fast is not None else "None"
            print(f"tier_used: {sm['tier_used']}  fast_score={fast_str}")
        print(f"error: {r.get('error')}")
        # Show context chunks if present (RAG samples)
        sample_meta = r.get("sample_metadata") or {}
        context = sample_meta.get("context")
        if context:
            chunks = [context] if isinstance(context, str) else context
            print("\ncontext:")
            for i, chunk in enumerate(chunks):
                print(f"  [{i}] {chunk}")
        # Sufficiency details (context-sufficiency scorer)
        if sm.get("sufficiency_prompt"):
            print(f"\nprompt:\n{sm['sufficiency_prompt']}")
        if sm.get("sufficiency_raw_response"):
            print(f"\nraw response:\n{sm['sufficiency_raw_response']}")
        if sm.get("sufficiency_reasoning"):
            print(f"\nreasoning:  {sm['sufficiency_reasoning']}")
        # Suppress completion for dataset scorers (latency=0, completion="")
        completion = r.get("completion")
        if completion == "":
            print("\ncompletion: (not applicable — dataset scorer)")
        else:
            print(f"\ncompletion:\n{completion or '(none)'}")
        return

    if strict:
        display_rows = [r for r in rows if _is_strict_failure(r)]
    elif failures_only:
        display_rows = [r for r in rows if _is_failure(r)]
    else:
        display_rows = rows

    has_tier = any("tier_used" in (r.get("scorer_metadata") or {}) for r in display_rows)
    has_format = any("format_status" in (r.get("scorer_metadata") or {}) for r in display_rows)

    table = []
    for r in display_rows:
        sm = r.get("scorer_metadata") or {}
        row = [
            r["id"],
            f"{r['score']:.2f}" if r["score"] is not None else "—",
            _outcome(r),
            r["latency_ms"],
        ]
        if has_tier:
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
            row.append(sm.get("format_status", ""))
        row.append(_short(r.get("error") or "", 60))
        table.append(row)

    headers = ["id", "score", "outcome", "latency_ms"]
    if has_tier:
        headers.append("tier_used")
    if has_format:
        headers.append("format_status")
    headers.append("error")

    if strict:
        print(f"(strict failures only: {len(display_rows)}/{len(rows)})\n")
    elif failures_only:
        print(f"(failures only: {len(display_rows)}/{len(rows)})\n")
    print(tabulate(table, headers=headers, tablefmt="simple"))

    failures = [r for r in rows if _outcome(r) != "pass"]
    if failures and verbose:
        print("\n── COMPLETIONS FOR FAILURES " + "─" * 50)
        for r in failures:
            print(f"\n[{r['id']}]")
            print(f"  completion: {_short(r.get('completion'), 400)}")
            print(f"  error:      {r.get('error')}")


# ---------------------------------------------------------------------------
# jsonl/sample mode
# ---------------------------------------------------------------------------


def inspect_jsonl(
    path: Path,
    sample_id: str | None,
    verbose: bool,
    failures_only: bool = False,
    strict: bool = False,
) -> None:
    rows = _load_jsonl(path)

    if sample_id:
        matches = [r for r in rows if r.get("id") == sample_id]
        if not matches:
            print(f"Sample {sample_id!r} not found in {path}")
            sys.exit(1)
        r = matches[0]
        print(f"id={r['id']}  score={r.get('score')}  latency={r.get('latency_ms')}ms")
        if r.get("expected") is not None:
            print(f"expected:  {r['expected']}")
        print(f"error: {r.get('error')}")
        print(f"\ncompletion:\n{r.get('completion') or '(none)'}")
        return

    if strict:
        display_rows = [r for r in rows if _is_strict_failure(r)]
    elif failures_only:
        display_rows = [r for r in rows if _is_failure(r)]
    else:
        display_rows = rows

    has_tier = any("tier_used" in (r.get("scorer_metadata") or {}) for r in display_rows)
    has_format = any("format_status" in (r.get("scorer_metadata") or {}) for r in display_rows)

    table = []
    for r in display_rows:
        sm = r.get("scorer_metadata") or {}
        row = [
            r.get("id"),
            f"{r['score']:.2f}" if r.get("score") is not None else "—",
            _outcome(r),
            r.get("latency_ms"),
        ]
        if has_tier:
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
            row.append(sm.get("format_status", ""))
        row.append(_short(r.get("error") or "", 60))
        table.append(row)

    headers = ["id", "score", "outcome", "latency_ms"]
    if has_tier:
        headers.append("tier_used")
    if has_format:
        headers.append("format_status")
    headers.append("error")

    if strict:
        label = f"  (strict failures only: {len(display_rows)}/{len(rows)})"
    elif failures_only:
        label = f"  (failures only: {len(display_rows)}/{len(rows)})"
    else:
        label = f"  ({len(rows)} samples)"
    print(f"{path}{label}\n")
    print(tabulate(table, headers=headers, tablefmt="simple"))

    if verbose:
        print()
        for r in rows:
            print(f"── {r['id']}  score={r.get('score')} ──")
            print(r.get("completion") or "(none)")
            print()


# ---------------------------------------------------------------------------
# traces mode
# ---------------------------------------------------------------------------


def inspect_traces(traces_dir: Path, verbose: bool, failures_only: bool = False) -> None:
    """Summarise judge traces under a session directory."""
    trace_files = sorted(traces_dir.glob("*.json"))
    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        sys.exit(1)

    all_rows = []
    for tf in trace_files:
        t = json.loads(tf.read_text())
        status = "ok" if t.get("final_score") is not None else "fail"
        all_rows.append((tf, t, status))

    display = [(tf, t, s) for tf, t, s in all_rows if s != "ok"] if failures_only else all_rows
    label = (
        f"  (failures only: {len(display)}/{len(all_rows)})"
        if failures_only
        else f"  ({len(all_rows)} samples)"
    )
    print(f"TRACES  dir={traces_dir}{label}\n")

    rows = [
        [
            t.get("sample_id", tf.stem),
            t.get("evaluated_model", "?"),
            f"{t['final_score']:.2f}" if t.get("final_score") is not None else "—",
            t.get("parsed_score"),
            s,
            _short(t.get("error") or "", 80),
        ]
        for tf, t, s in display
    ]
    print(
        tabulate(
            rows,
            headers=["sample_id", "evaluated_model", "score", "raw_score", "status", "error"],
            tablefmt="simple",
        )
    )

    if verbose:
        for tf, t, _s in display:
            if t.get("error"):
                print(f"\n── {tf.stem} ──")
                print(f"raw_response: {_short(t.get('raw_response', ''), 400)}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect eval results")
    parser.add_argument("path", help="Benchmark dir / run dir / .jsonl file / traces dir")
    parser.add_argument("--id", default=None, help="Show a specific sample by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full completions")
    parser.add_argument(
        "--failures-only",
        "-f",
        action="store_true",
        help="Show only samples with score < 1.0 or score=None",
    )
    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        help="Show only genuine failures (score < 0.5 or score=None). Implies --failures-only.",
    )
    args = parser.parse_args()

    if args.strict:
        args.failures_only = True

    path = Path(args.path)
    if not path.exists():
        print(f"Path not found: {path}")
        sys.exit(1)

    if path.is_file() and path.suffix == ".jsonl":
        inspect_jsonl(
            path, args.id, args.verbose, failures_only=args.failures_only, strict=args.strict
        )
    elif path.is_dir() and (path / "sensitivity.json").exists():
        inspect_sensitivity(path, args.verbose, sample_id=args.id, failures_only=args.failures_only)
    elif path.is_dir() and (path / "benchmark.json").exists():
        inspect_benchmark(
            path,
            args.verbose,
            sample_id=args.id,
            failures_only=args.failures_only,
            strict=args.strict,
        )
    elif path.is_dir() and (path / "run.json").exists():
        inspect_run(
            path,
            args.verbose,
            sample_id=args.id,
            failures_only=args.failures_only,
            strict=args.strict,
        )
    elif path.is_dir() and any(path.glob("*.json")):
        inspect_traces(path, args.verbose, failures_only=args.failures_only)
    else:
        print(f"Cannot determine type of {path}")
        if path.is_dir():
            candidates = sorted(d for d in path.iterdir() if d.is_dir() and any(d.glob("*.json")))
            if candidates:
                print("\nDid you mean one of these?")
                for c in candidates:
                    print(f"  {c.name}/")
                sys.exit(1)
        print(
            "Expected: benchmark dir (has benchmark.json), run dir (has run.json), .jsonl file, or traces dir"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
