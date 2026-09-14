from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from evals.core import Dataset, EvalConfig, RunResult, Sample, ScorerContext
from evals.reporters import Reporter
from evals.scorers.llm_judge import LLMJudgeScorer
from runners.benchmark import BenchmarkRunner


def _make_sample(id: str = "s1") -> Sample:
    return Sample(id=id, input="q", expected="a", metadata={})


def _make_result(
    id: str = "s1",
    score: float | None = 1.0,
    latency_ms: int = 100,
    error: str | None = None,
    completion: str | None = "answer",
) -> RunResult:
    return RunResult(
        sample=_make_sample(id),
        completion=completion,
        score=score,
        latency_ms=latency_ms,
        error=error,
    )


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_called_once_per_model(self):
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = [_make_result()]
        scorer = MagicMock()
        ds = MagicMock(spec=Dataset)

        with patch("runners.benchmark.Runner", return_value=mock_runner_instance):
            result = BenchmarkRunner().run(ds, scorer, ["model-a", "model-b"])

        assert mock_runner_instance.run.call_count == 2
        assert [m for m, _ in result] == ["model-a", "model-b"]

    def test_config_model_set_per_model(self):
        captured_configs = []

        def fake_run(dataset, scorer, config):
            captured_configs.append(config.model)
            return [_make_result()]

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = fake_run
        scorer = MagicMock()
        ds = MagicMock(spec=Dataset)

        with patch("runners.benchmark.Runner", return_value=mock_runner_instance):
            BenchmarkRunner().run(ds, scorer, ["alpha", "beta"], base_config=EvalConfig(model="ignored"))

        assert captured_configs == ["alpha", "beta"]

    def test_set_evaluated_model_called_on_judge(self):
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = [_make_result()]
        scorer = MagicMock(spec=LLMJudgeScorer)
        ds = MagicMock(spec=Dataset)

        with patch("runners.benchmark.Runner", return_value=mock_runner_instance):
            BenchmarkRunner().run(ds, scorer, ["x", "y"])

        calls = [c.args[0] for c in scorer.set_evaluated_model.call_args_list]
        assert calls == ["x", "y"]

    def test_scorer_without_set_evaluated_model_is_fine(self):
        """Plain callables (no set_evaluated_model) should not raise."""
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = [_make_result()]
        scorer = MagicMock(spec=[])  # no attributes
        ds = MagicMock(spec=Dataset)

        with patch("runners.benchmark.Runner", return_value=mock_runner_instance):
            result = BenchmarkRunner().run(ds, scorer, ["m1"])

        assert len(result) == 1


# ---------------------------------------------------------------------------
# Reporter._summarise
# ---------------------------------------------------------------------------

class TestSummarise:
    def test_empty_scores_returns_none_mean(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result(score=None, error="api error", completion=None)]
        s = r._summarise(results)
        assert s["mean_score"] is None

    def test_normal_scores(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result(score=1.0), _make_result(score=0.5, id="s2")]
        s = r._summarise(results)
        assert s["mean_score"] == pytest.approx(0.75)

    def test_p95_low_confidence_when_n_lt_20(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result(id=str(i)) for i in range(5)]
        s = r._summarise(results)
        assert s["p95_low_confidence"] is True

    def test_p95_not_low_confidence_when_n_gte_20(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result(id=str(i)) for i in range(20)]
        s = r._summarise(results)
        assert s["p95_low_confidence"] is False

    def test_no_divide_by_zero_on_empty(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        s = r._summarise([])
        assert s["mean_score"] is None
        assert s["error_rate"] == 0.0


# ---------------------------------------------------------------------------
# Reporter.report — new directory structure
# ---------------------------------------------------------------------------

class TestReport:
    def test_creates_runs_directory_structure(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result()]
        _, run_dir = r.report(results, "mydata", "exact", model="llama3.2:3b")
        assert (run_dir / "run.json").exists()
        assert (run_dir / "samples.jsonl").exists()
        # path: results/runs/{date}/{time}_{model}_{dataset}_{scorer}
        assert run_dir.parent.parent.name == "runs"

    def test_run_json_contains_model(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result()]
        _, run_dir = r.report(results, "ds", "exact", model="llama3.2:3b")
        payload = json.loads((run_dir / "run.json").read_text())
        assert payload["model"] == "llama3.2:3b"

    def test_samples_jsonl_one_line_per_result(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result(id="a"), _make_result(id="b")]
        _, run_dir = r.report(results, "ds", "exact", model="m")
        lines = (run_dir / "samples.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "a"

    def test_samples_jsonl_contains_expected(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result()]
        _, run_dir = r.report(results, "ds", "exact", model="m")
        row = json.loads((run_dir / "samples.jsonl").read_text().strip())
        assert row["expected"] == "a"

    def test_model_colon_sanitised_in_dirname(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result()]
        _, run_dir = r.report(results, "ds", "exact", model="llama3.2:3b")
        assert ":" not in run_dir.name

    def test_none_mean_score_in_summary_string(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        results = [_make_result(score=None, error="err", completion=None)]
        summary_str, _ = r.report(results, "ds", "exact", model="m")
        assert "mean_score=—" in summary_str


# ---------------------------------------------------------------------------
# Reporter.benchmark_report
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_creates_benchmarks_directory_structure(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        model_results = [
            ("model-a", [_make_result()]),
            ("model-b", [_make_result()]),
        ]
        _, bench_dir = r.benchmark_report(model_results, "ds", "judge")
        assert (bench_dir / "benchmark.json").exists()
        assert bench_dir.parent.parent.name == "benchmarks"

    def test_per_model_jsonl_files_created(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        model_results = [
            ("llama3.2:3b", [_make_result()]),
            ("mistral:7b", [_make_result()]),
        ]
        _, bench_dir = r.benchmark_report(model_results, "ds", "judge")
        assert (bench_dir / "llama3.2_3b.jsonl").exists()
        assert (bench_dir / "mistral_7b.jsonl").exists()

    def test_none_mean_score_renders_as_dash(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        # all errors → mean_score=None
        model_results = [
            ("model-x", [_make_result(score=None, error="err", completion=None)]),
        ]
        table_str, _ = r.benchmark_report(model_results, "ds", "judge")
        assert "—" in table_str

    def test_benchmark_json_contains_all_models(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        model_results = [
            ("m1", [_make_result(score=1.0)]),
            ("m2", [_make_result(score=0.5)]),
        ]
        _, bench_dir = r.benchmark_report(model_results, "ds", "exact")
        payload = json.loads((bench_dir / "benchmark.json").read_text())
        assert set(payload["models"].keys()) == {"m1", "m2"}

    def test_per_model_jsonl_contains_expected(self, tmp_path):
        r = Reporter(results_dir=tmp_path)
        model_results = [("m1", [_make_result()])]
        _, bench_dir = r.benchmark_report(model_results, "ds", "exact")
        row = json.loads((bench_dir / "m1.jsonl").read_text().strip())
        assert row["expected"] == "a"


# ---------------------------------------------------------------------------
# LLMJudgeScorer — evaluated_model + set_evaluated_model
# ---------------------------------------------------------------------------

class TestLLMJudgeScorerEvaluatedModel:
    def test_trace_dir_uses_evaluated_model(self, tmp_path):
        judge = LLMJudgeScorer(results_dir=tmp_path, evaluated_model="my-model")
        assert "my-model" in str(judge._trace_dir)

    def test_trace_dir_unknown_when_not_set(self, tmp_path):
        judge = LLMJudgeScorer(results_dir=tmp_path)
        assert "unknown" in str(judge._trace_dir)

    def test_set_evaluated_model_updates_trace_dir(self, tmp_path):
        judge = LLMJudgeScorer(results_dir=tmp_path)
        judge.set_evaluated_model("new-model")
        assert "new-model" in str(judge._trace_dir)
        assert "unknown" not in str(judge._trace_dir)

    def test_trace_filename_is_sample_id(self, tmp_path):
        judge = LLMJudgeScorer(results_dir=tmp_path, evaluated_model="m")
        judge._write_trace(
            prompt="p",
            raw_response='{"score": 3, "reasoning": "ok"}',
            parsed_score=3,
            final_score=0.5,
            error=None,
            ctx=ScorerContext(input="q", metadata={"id": "sample-007"}),
        )
        trace_dir = judge._trace_dir
        assert (trace_dir / "sample-007.json").exists()
