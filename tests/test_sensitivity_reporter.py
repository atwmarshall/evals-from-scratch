from __future__ import annotations

import json
import statistics

import pytest

from evals.core import RunResult, Sample
from evals.sensitivity_reporter import SensitivityReporter

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sample(sid: str) -> Sample:
    return Sample(id=sid, input=f"q {sid}", expected=f"a {sid}", metadata={})


def _result(sid: str, score: float | None) -> RunResult:
    return RunResult(
        sample=_sample(sid),
        completion="c",
        score=score,
        latency_ms=100,
        error=None if score is not None else "scorer returned None",
    )


def _reporter(tmp_path) -> SensitivityReporter:
    return SensitivityReporter(results_dir=tmp_path)


# ---------------------------------------------------------------------------
# _compute_per_sample
# ---------------------------------------------------------------------------

class TestComputePerSample:
    def test_all_pass_gives_zero_variance(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "rephrase":  [_result("a", 1.0), _result("b", 1.0)],
        }
        rows = r._compute_per_sample(rbv, ["baseline", "rephrase"])
        for row in rows:
            assert row["variance"] == pytest.approx(0.0)
            assert row["verdict"] == "ok"

    def test_mixed_scores_mark_unstable(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0)],
            "rephrase":  [_result("a", 0.0)],
        }
        rows = r._compute_per_sample(rbv, ["baseline", "rephrase"])
        assert rows[0]["variance"] == pytest.approx(statistics.variance([1.0, 0.0]))
        assert rows[0]["verdict"] == "unstable"

    def test_none_score_excluded_from_variance(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0)],
            "rephrase":  [_result("a", None)],   # scorer failure
            "formal":    [_result("a", 1.0)],
        }
        rows = r._compute_per_sample(rbv, ["baseline", "rephrase", "formal"])
        row = rows[0]
        assert row["rephrase"] is None
        # variance computed from only [1.0, 1.0] (None excluded)
        assert row["variance"] == pytest.approx(0.0)
        assert row["verdict"] == "ok"

    def test_single_valid_score_gives_n_a_verdict(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0)],
            "rephrase":  [_result("a", None)],
        }
        rows = r._compute_per_sample(rbv, ["baseline", "rephrase"])
        assert rows[0]["variance"] is None
        assert rows[0]["verdict"] == "n/a"

    def test_missing_sample_in_variation_is_none(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "rephrase":  [_result("a", 0.5)],           # "b" not present
        }
        rows = r._compute_per_sample(rbv, ["baseline", "rephrase"])
        b_row = next(row for row in rows if row["id"] == "b")
        assert b_row["rephrase"] is None
        assert b_row["verdict"] == "n/a"   # only one valid score

    def test_baseline_always_first_column(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0)],
            "rephrase":  [_result("a", 1.0)],
        }
        rows = r._compute_per_sample(rbv, ["baseline", "rephrase"])
        row = rows[0]
        keys = list(row.keys())
        assert keys[1] == "baseline"    # id is first, baseline second

    def test_threshold_is_strict_greater_than(self, tmp_path):
        r = _reporter(tmp_path)
        # variance([0.9, 0.95]) ≈ 0.00125 — clearly "ok"
        # variance([1.0, 0.0])  = 0.5     — clearly "unstable"
        # The verdict rule is variance > 0.05 (strict), so values just below
        # the threshold should be "ok".
        rbv_ok = {
            "baseline": [_result("a", 0.9)],
            "rephrase":  [_result("a", 0.95)],
        }
        rows_ok = r._compute_per_sample(rbv_ok, ["baseline", "rephrase"])
        assert rows_ok[0]["verdict"] == "ok"

        rbv_unstable = {
            "baseline": [_result("a", 1.0)],
            "rephrase":  [_result("a", 0.0)],
        }
        rows_unstable = r._compute_per_sample(rbv_unstable, ["baseline", "rephrase"])
        assert rows_unstable[0]["verdict"] == "unstable"

    def test_rows_sorted_by_sample_id(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {"baseline": [_result("c", 1.0), _result("a", 1.0), _result("b", 1.0)]}
        rows = r._compute_per_sample(rbv, ["baseline"])
        assert [row["id"] for row in rows] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _compute_per_variation
# ---------------------------------------------------------------------------

class TestComputePerVariation:
    def test_baseline_delta_and_variance_are_null(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "rephrase":  [_result("a", 0.8), _result("b", 0.8)],
        }
        variation_names = ["baseline", "rephrase"]
        per_sample = r._compute_per_sample(rbv, variation_names)
        summary = r._compute_per_variation(rbv, per_sample, variation_names)
        baseline_row = next(s for s in summary if s["variation"] == "baseline")
        assert baseline_row["delta_from_baseline"] is None
        assert baseline_row["mean_variance"] is None

    def test_delta_computed_correctly(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "rephrase":  [_result("a", 0.8), _result("b", 0.8)],
        }
        variation_names = ["baseline", "rephrase"]
        per_sample = r._compute_per_sample(rbv, variation_names)
        summary = r._compute_per_variation(rbv, per_sample, variation_names)
        rephrase_row = next(s for s in summary if s["variation"] == "rephrase")
        assert rephrase_row["delta_from_baseline"] == pytest.approx(-0.2)

    def test_no_baseline_all_deltas_null(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {"rephrase": [_result("a", 0.8)], "formal": [_result("a", 0.9)]}
        variation_names = ["rephrase", "formal"]
        per_sample = r._compute_per_sample(rbv, variation_names)
        summary = r._compute_per_variation(rbv, per_sample, variation_names)
        for s in summary:
            assert s["delta_from_baseline"] is None

    def test_mean_variance_uses_cross_variation_per_sample_values(self, tmp_path):
        r = _reporter(tmp_path)
        # Give rephrase a very different score to produce high variance
        rbv = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "rephrase":  [_result("a", 0.0), _result("b", 0.0)],
        }
        variation_names = ["baseline", "rephrase"]
        per_sample = r._compute_per_sample(rbv, variation_names)
        # Each sample's variance = var([1.0, 0.0]) = 0.5
        assert all(row["variance"] == pytest.approx(0.5) for row in per_sample)

        summary = r._compute_per_variation(rbv, per_sample, variation_names)
        rephrase_row = next(s for s in summary if s["variation"] == "rephrase")
        # mean_variance for rephrase = mean of per-sample variances = 0.5
        assert rephrase_row["mean_variance"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SensitivityReporter.report() — integration with tmp_path
# ---------------------------------------------------------------------------

class TestReport:
    def _make_rbv(self) -> dict[str, list[RunResult]]:
        return {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "rephrase":  [_result("a", 0.5), _result("b", 1.0)],
            "formal":    [_result("a", 1.0), _result("b", 0.5)],
        }

    def test_creates_sensitivity_json(self, tmp_path):
        r = _reporter(tmp_path)
        r.report(self._make_rbv(), "extraction", "schema")
        files = list(tmp_path.rglob("sensitivity.json"))
        assert len(files) == 1

    def test_creates_per_variation_jsonl(self, tmp_path):
        r = _reporter(tmp_path)
        _, sens_dir = r.report(self._make_rbv(), "extraction", "schema")
        jsonls = {p.stem for p in sens_dir.glob("*.jsonl")}
        assert jsonls == {"baseline", "rephrase", "formal"}

    def test_returns_path_inside_results_dir(self, tmp_path):
        r = _reporter(tmp_path)
        _, sens_dir = r.report(self._make_rbv(), "extraction", "schema")
        assert str(sens_dir).startswith(str(tmp_path))

    def test_sensitivity_json_structure(self, tmp_path):
        r = _reporter(tmp_path)
        _, sens_dir = r.report(self._make_rbv(), "extraction", "schema")
        data = json.loads((sens_dir / "sensitivity.json").read_text())
        assert "dataset" in data
        assert "scorer" in data
        assert "model" in data
        assert "timestamp" in data
        assert "materialised_at" in data
        assert "run_config" in data
        assert "variation_names" in data
        assert "per_sample" in data
        assert "per_variation" in data
        assert "summary" in data
        assert "schema_notes" in data

    def test_summary_n_unstable_count(self, tmp_path):
        r = _reporter(tmp_path)
        # rephrase drops "a" to 0.5, formal drops "b" to 0.5 → both samples unstable
        _, sens_dir = r.report(self._make_rbv(), "extraction", "schema")
        data = json.loads((sens_dir / "sensitivity.json").read_text())
        assert data["summary"]["n_unstable"] == 2
        assert data["summary"]["n_total"] == 2

    def test_most_destabilising_identified(self, tmp_path):
        r = _reporter(tmp_path)
        # Arrange coverage so each non-baseline variation touches a different sample:
        #   rephrase covers "b" (variance=0.0 — stable sample)
        #   add_noise covers "a" (variance=0.5 — unstable sample)
        # This produces different mean_variance per variation, identifying add_noise.
        rbv = {
            "baseline":  [_result("a", 1.0), _result("b", 1.0)],
            "rephrase":  [_result("b", 1.0)],   # stable sample → low mean_variance
            "add_noise": [_result("a", 0.0)],   # unstable sample → high mean_variance
        }
        _, sens_dir = r.report(rbv, "extraction", "schema")
        data = json.loads((sens_dir / "sensitivity.json").read_text())
        assert data["summary"]["most_destabilising"] == "add_noise"

    def test_table_str_contains_unstable_label(self, tmp_path):
        r = _reporter(tmp_path)
        output_str, _ = r.report(self._make_rbv(), "extraction", "schema")
        assert "unstable" in output_str

    def test_none_score_renders_as_dash_in_output(self, tmp_path):
        r = _reporter(tmp_path)
        rbv = {
            "baseline": [_result("a", 1.0)],
            "rephrase":  [_result("a", None)],
            "formal":    [_result("a", 1.0)],
        }
        output_str, _ = r.report(rbv, "extraction", "schema")
        assert "—" in output_str

    def test_run_config_stored_in_sensitivity_json(self, tmp_path):
        r = _reporter(tmp_path)
        cfg = {"variation_model": "mistral:7b", "validation_threshold": 0.8}
        _, sens_dir = r.report(self._make_rbv(), "extraction", "schema", run_config=cfg)
        data = json.loads((sens_dir / "sensitivity.json").read_text())
        assert data["run_config"]["variation_model"] == "mistral:7b"
        assert data["run_config"]["validation_threshold"] == 0.8

    def test_baseline_mean_variance_is_null_in_saved_json(self, tmp_path):
        r = _reporter(tmp_path)
        _, sens_dir = r.report(self._make_rbv(), "extraction", "schema")
        data = json.loads((sens_dir / "sensitivity.json").read_text())
        baseline_pv = next(row for row in data["per_variation"] if row["variation"] == "baseline")
        assert baseline_pv["mean_variance"] is None
        assert baseline_pv["delta_from_baseline"] is None

    def test_path_contains_dataset_and_scorer_name(self, tmp_path):
        r = _reporter(tmp_path)
        _, sens_dir = r.report(self._make_rbv(), "extraction", "cascade")
        assert "extraction" in sens_dir.name
        assert "cascade" in sens_dir.name
