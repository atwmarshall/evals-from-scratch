from __future__ import annotations

import json

import pytest

from evals.core import RunResult, Sample
from evals.robustness_reporter import RobustnessReporter

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


def _reporter(tmp_path) -> RobustnessReporter:
    return RobustnessReporter(results_dir=tmp_path)


# ---------------------------------------------------------------------------
# _compute_per_sample
# ---------------------------------------------------------------------------


class TestComputePerSample:
    def test_degradation_computed_correctly(self, tmp_path):
        r = _reporter(tmp_path)
        # baseline=1.0, typos=0.75, colloquial=1.0 → mean_perturbation=0.875
        # degradation = 1.0 - 0.875 = 0.125
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 0.75)],
            "colloquial": [_result("a", 1.0)],
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos", "colloquial"])
        assert rows[0]["degradation"] == pytest.approx(0.125)

    def test_robust_verdict_when_degradation_below_0_1(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 1.0)],  # degradation = 0.0
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos"])
        assert rows[0]["verdict"] == "robust"

    def test_fragile_verdict_when_degradation_between_0_1_and_0_3(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 0.8)],  # degradation = 0.2
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos"])
        assert rows[0]["verdict"] == "fragile"

    def test_brittle_verdict_when_degradation_gte_0_3(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 0.5)],  # degradation = 0.5
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos"])
        assert rows[0]["verdict"] == "brittle"

    def test_verdict_threshold_just_above_0_1_is_fragile(self, tmp_path):
        """degradation < 0.1 is robust; >= 0.1 is fragile. 0.9 - 1.0 = -0.1 in float
        but we use 1.0 - 0.875 = 0.125 (exact in binary) to avoid float boundary noise."""
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 0.875)],  # degradation = 0.125, exact in binary
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos"])
        assert rows[0]["degradation"] == pytest.approx(0.125)
        assert rows[0]["verdict"] == "fragile"

    def test_verdict_threshold_at_exactly_0_3_is_brittle(self, tmp_path):
        """degradation < 0.3 is fragile; 0.3 itself is brittle."""
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 0.7)],  # degradation = 0.3 exactly
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos"])
        assert rows[0]["degradation"] == pytest.approx(0.3)
        assert rows[0]["verdict"] == "brittle"

    def test_na_verdict_when_no_perturbation_scores(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", None)],
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos"])
        assert rows[0]["degradation"] is None
        assert rows[0]["verdict"] == "n/a"

    def test_na_verdict_when_baseline_score_is_none(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", None)],
            "typos": [_result("a", 0.5)],
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos"])
        assert rows[0]["degradation"] is None
        assert rows[0]["verdict"] == "n/a"

    def test_none_perturbation_scores_excluded_from_mean(self, tmp_path):
        r = _reporter(tmp_path)
        # typos=None excluded, only colloquial=0.8 counts → degradation = 1.0 - 0.8 = 0.2
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", None)],
            "colloquial": [_result("a", 0.8)],
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos", "colloquial"])
        assert rows[0]["degradation"] == pytest.approx(0.2)
        assert rows[0]["verdict"] == "fragile"

    def test_missing_sample_in_perturbation_treated_as_none(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "typos": [_result("a", 0.5)],  # "b" missing
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos"])
        b_row = next(row for row in rows if row["id"] == "b")
        assert b_row["typos"] is None
        assert b_row["verdict"] == "n/a"

    def test_rows_sorted_by_sample_id(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {"baseline": [_result("c", 1.0), _result("a", 1.0), _result("b", 1.0)]}
        rows = r._compute_per_sample(rbp, ["baseline"])
        assert [row["id"] for row in rows] == ["a", "b", "c"]

    def test_baseline_not_included_in_perturbation_mean(self, tmp_path):
        """Baseline score must not appear in the mean used to compute degradation."""
        r = _reporter(tmp_path)
        # If baseline were included: mean([1.0, 0.0, 1.0]) = 0.667, degradation = 0.333
        # Correct: mean([0.0, 1.0]) = 0.5, degradation = 0.5
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 0.0)],
            "colloquial": [_result("a", 1.0)],
        }
        rows = r._compute_per_sample(rbp, ["baseline", "typos", "colloquial"])
        assert rows[0]["degradation"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _compute_per_perturbation
# ---------------------------------------------------------------------------


class TestComputePerPerturbation:
    def test_baseline_delta_is_null(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 0.8)],
        }
        perturbation_names = ["baseline", "typos"]
        per_sample = r._compute_per_sample(rbp, perturbation_names)
        summary = r._compute_per_perturbation(rbp, per_sample, perturbation_names)
        baseline_row = next(s for s in summary if s["perturbation"] == "baseline")
        assert baseline_row["delta_from_baseline"] is None

    def test_delta_is_negative_when_perturbation_degrades(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "typos": [_result("a", 0.8), _result("b", 0.8)],
        }
        perturbation_names = ["baseline", "typos"]
        per_sample = r._compute_per_sample(rbp, perturbation_names)
        summary = r._compute_per_perturbation(rbp, per_sample, perturbation_names)
        typos_row = next(s for s in summary if s["perturbation"] == "typos")
        assert typos_row["delta_from_baseline"] == pytest.approx(-0.2)

    def test_delta_can_be_positive_when_perturbation_improves(self, tmp_path):
        """Verbose perturbation can occasionally improve LLM-judge scores (verbosity bias)."""
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 0.8)],
            "verbose": [_result("a", 1.0)],
        }
        perturbation_names = ["baseline", "verbose"]
        per_sample = r._compute_per_sample(rbp, perturbation_names)
        summary = r._compute_per_perturbation(rbp, per_sample, perturbation_names)
        verbose_row = next(s for s in summary if s["perturbation"] == "verbose")
        assert verbose_row["delta_from_baseline"] == pytest.approx(0.2)

    def test_no_baseline_all_deltas_null(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {"typos": [_result("a", 0.8)], "colloquial": [_result("a", 0.9)]}
        perturbation_names = ["typos", "colloquial"]
        per_sample = r._compute_per_sample(rbp, perturbation_names)
        summary = r._compute_per_perturbation(rbp, per_sample, perturbation_names)
        for s in summary:
            assert s["delta_from_baseline"] is None


# ---------------------------------------------------------------------------
# RobustnessReporter.report() — integration with tmp_path
# ---------------------------------------------------------------------------


class TestReport:
    def _make_rbp(self) -> dict[str, list[RunResult]]:
        return {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "typos": [_result("a", 0.75), _result("b", 0.5)],
            "colloquial": [_result("a", 1.0), _result("b", 1.0)],
        }

    def test_creates_robustness_json(self, tmp_path):
        r = _reporter(tmp_path)
        r.report(self._make_rbp(), "extraction", "schema", model="llama3.2")
        files = list(tmp_path.rglob("robustness.json"))
        assert len(files) == 1

    def test_creates_per_perturbation_jsonl(self, tmp_path):
        r = _reporter(tmp_path)
        _, rob_dir = r.report(self._make_rbp(), "extraction", "schema", model="llama3.2")
        jsonls = {p.stem for p in rob_dir.glob("*.jsonl")}
        assert jsonls == {"baseline", "typos", "colloquial"}

    def test_output_dir_keyed_by_model_not_scorer(self, tmp_path):
        r = _reporter(tmp_path)
        _, rob_dir = r.report(self._make_rbp(), "extraction", "exact", model="mistral:7b")
        assert "mistral_7b" in rob_dir.name
        assert "exact" not in rob_dir.name

    def test_output_dir_under_robustness_subdir(self, tmp_path):
        r = _reporter(tmp_path)
        _, rob_dir = r.report(self._make_rbp(), "extraction", "schema", model="llama3.2")
        assert "robustness" in str(rob_dir)

    def test_returns_path_inside_results_dir(self, tmp_path):
        r = _reporter(tmp_path)
        _, rob_dir = r.report(self._make_rbp(), "extraction", "schema", model="llama3.2")
        assert str(rob_dir).startswith(str(tmp_path))

    def test_robustness_json_structure(self, tmp_path):
        r = _reporter(tmp_path)
        _, rob_dir = r.report(self._make_rbp(), "extraction", "schema", model="llama3.2")
        data = json.loads((rob_dir / "robustness.json").read_text())
        for key in [
            "dataset",
            "scorer",
            "model",
            "timestamp",
            "materialised_at",
            "run_config",
            "perturbation_names",
            "per_sample",
            "per_perturbation",
            "summary",
            "schema_notes",
        ]:
            assert key in data, f"missing key: {key}"

    def test_summary_counts_fragile_and_brittle(self, tmp_path):
        r = _reporter(tmp_path)
        # "a": mean_perturbation = mean([0.75, 1.0]) = 0.875, degradation=0.125 → fragile
        # "b": mean_perturbation = mean([0.5, 1.0]) = 0.75, degradation=0.25 → fragile
        _, rob_dir = r.report(self._make_rbp(), "extraction", "schema", model="llama3.2")
        data = json.loads((rob_dir / "robustness.json").read_text())
        assert data["summary"]["n_fragile"] == 2
        assert data["summary"]["n_brittle"] == 0
        assert data["summary"]["n_total"] == 2

    def test_most_degrading_identified(self, tmp_path):
        r = _reporter(tmp_path)
        # typos drops "a" to 0.0 (baseline 1.0, mean_delta=-1.0)
        # colloquial drops "a" to 0.8 (mean_delta=-0.2)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", 0.0)],
            "colloquial": [_result("a", 0.8)],
        }
        _, rob_dir = r.report(rbp, "extraction", "schema", model="llama3.2")
        data = json.loads((rob_dir / "robustness.json").read_text())
        assert data["summary"]["most_degrading"] == "typos"

    def test_summary_line_n_note_when_counts_differ(self, tmp_path):
        r = _reporter(tmp_path)
        # typos only has 1 sample ("a"), baseline has 2
        rbp = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "typos": [_result("a", 0.0)],
        }
        output_str, _ = r.report(rbp, "extraction", "schema", model="llama3.2")
        assert "n=1" in output_str
        assert "baseline n=2" in output_str

    def test_summary_line_no_n_note_when_counts_match(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0), _result("b", 1.0)],
            "typos": [_result("a", 0.5), _result("b", 0.5)],
        }
        output_str, _ = r.report(rbp, "extraction", "schema", model="llama3.2")
        assert "vs baseline" not in output_str

    def test_none_score_renders_as_dash(self, tmp_path):
        r = _reporter(tmp_path)
        rbp = {
            "baseline": [_result("a", 1.0)],
            "typos": [_result("a", None)],
            "colloquial": [_result("a", 1.0)],
        }
        output_str, _ = r.report(rbp, "extraction", "schema", model="llama3.2")
        assert "—" in output_str

    def test_run_config_stored_in_robustness_json(self, tmp_path):
        r = _reporter(tmp_path)
        cfg = {"perturbation_model": "mistral:7b", "cascade_threshold": 1.0}
        _, rob_dir = r.report(
            self._make_rbp(), "extraction", "schema", model="llama3.2", run_config=cfg
        )
        data = json.loads((rob_dir / "robustness.json").read_text())
        assert data["run_config"]["perturbation_model"] == "mistral:7b"

    def test_baseline_delta_null_in_saved_json(self, tmp_path):
        r = _reporter(tmp_path)
        _, rob_dir = r.report(self._make_rbp(), "extraction", "schema", model="llama3.2")
        data = json.loads((rob_dir / "robustness.json").read_text())
        baseline_pp = next(
            row for row in data["per_perturbation"] if row["perturbation"] == "baseline"
        )
        assert baseline_pp["delta_from_baseline"] is None

    def test_table_str_contains_verdict_labels(self, tmp_path):
        r = _reporter(tmp_path)
        output_str, _ = r.report(self._make_rbp(), "extraction", "schema", model="llama3.2")
        assert "fragile" in output_str or "brittle" in output_str or "robust" in output_str
