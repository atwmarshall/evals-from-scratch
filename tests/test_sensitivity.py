from __future__ import annotations

from unittest.mock import MagicMock, patch

from evals.core import Dataset, EvalConfig, RunResult, Sample
from evals.sensitivity_reporter import run_variations


def _make_dataset(*ids: str) -> Dataset:
    return Dataset(samples=[
        Sample(id=i, input=f"input {i}", expected=f"expected {i}", metadata={})
        for i in ids
    ])


def _make_result(sample_id: str) -> RunResult:
    return RunResult(
        sample=Sample(id=sample_id, input="", expected="", metadata={}),
        completion="answer",
        score=1.0,
        latency_ms=100,
        error=None,
    )


_config = EvalConfig(model="llama3.2:3b")
_scorer = MagicMock(return_value=1.0)


class TestRunVariations:
    def test_calls_runner_for_each_variation(self):
        variations = {
            "baseline": _make_dataset("a", "b"),
            "rephrase": _make_dataset("a", "b"),
        }
        fake_results = [_make_result("a"), _make_result("b")]

        with patch("evals.sensitivity_reporter.Runner") as MockRunner:
            MockRunner.return_value.run.return_value = fake_results
            results = run_variations(variations, _scorer, _config)

        assert MockRunner.return_value.run.call_count == 2
        assert set(results.keys()) == {"baseline", "rephrase"}

    def test_results_keyed_by_variation_name(self):
        variations = {"baseline": _make_dataset("a"), "formal": _make_dataset("a")}
        fake = [_make_result("a")]

        with patch("evals.sensitivity_reporter.Runner") as MockRunner:
            MockRunner.return_value.run.return_value = fake
            results = run_variations(variations, _scorer, _config)

        assert list(results.keys()) == ["baseline", "formal"]

    def test_empty_dataset_skipped(self):
        variations = {
            "baseline": _make_dataset("a"),
            "rephrase": Dataset(samples=[]),  # empty — should be skipped
        }
        fake = [_make_result("a")]

        with patch("evals.sensitivity_reporter.Runner") as MockRunner:
            MockRunner.return_value.run.return_value = fake
            results = run_variations(variations, _scorer, _config)

        assert "rephrase" not in results
        assert "baseline" in results
        assert MockRunner.return_value.run.call_count == 1

    def test_passes_scorer_and_config_to_runner(self):
        variations = {"baseline": _make_dataset("a")}
        fake = [_make_result("a")]

        with patch("evals.sensitivity_reporter.Runner") as MockRunner:
            MockRunner.return_value.run.return_value = fake
            run_variations(variations, _scorer, _config)

        call_args = MockRunner.return_value.run.call_args
        _, passed_scorer, passed_config = call_args.args
        assert passed_scorer is _scorer
        assert passed_config is _config

    def test_returns_empty_dict_when_all_datasets_empty(self):
        variations = {"rephrase": Dataset(samples=[])}

        with patch("evals.sensitivity_reporter.Runner") as MockRunner:
            results = run_variations(variations, _scorer, _config)

        MockRunner.return_value.run.assert_not_called()
        assert results == {}
