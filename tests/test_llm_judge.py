from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evals.core import ScorerContext

# _parse_response is tested by instantiating LLMJudgeScorer with a tmp trace dir
# to avoid touching the filesystem in an unexpected place.


@pytest.fixture
def judge(tmp_path):
    from evals.scorers.llm_judge import LLMJudgeScorer
    return LLMJudgeScorer(scale=5, results_dir=tmp_path)


class TestParseResponse:
    def test_valid_response(self, judge):
        score, err, _ = judge._parse_response('{"score": 3, "reasoning": "ok"}')
        assert score == 3
        assert err is None

    def test_minimum_score(self, judge):
        score, err, _ = judge._parse_response('{"score": 1, "reasoning": "bad"}')
        assert score == 1
        assert err is None

    def test_maximum_score(self, judge):
        score, err, _ = judge._parse_response('{"score": 5, "reasoning": "great"}')
        assert score == 5
        assert err is None

    def test_markdown_code_fence(self, judge):
        raw = '```json\n{"score": 4, "reasoning": "good"}\n```'
        score, err, _ = judge._parse_response(raw)
        assert score == 4
        assert err is None

    def test_markdown_fence_no_language(self, judge):
        raw = '```\n{"score": 2, "reasoning": "ok"}\n```'
        score, err, _ = judge._parse_response(raw)
        assert score == 2
        assert err is None

    def test_not_json(self, judge):
        score, err, _ = judge._parse_response("I think the score is 3 out of 5.")
        assert score is None
        assert err is not None

    def test_missing_score_field(self, judge):
        score, err, _ = judge._parse_response('{"reasoning": "looks good"}')
        assert score is None
        assert err is not None
        assert "score" in err

    def test_score_is_numeric_string(self, judge):
        # int("3") succeeds in Python, so "3" is accepted
        score, err, _ = judge._parse_response('{"score": "3", "reasoning": "ok"}')
        assert score == 3
        assert err is None

    def test_score_is_float(self, judge):
        # floats are not ints — int("3.5") raises, int(3.5) truncates
        # json parses 3.5 as float; int(3.5) = 3 which IS valid — document the behaviour
        score, err, _ = judge._parse_response('{"score": 3.5, "reasoning": "ok"}')
        # int(3.5) == 3, which is in range — acceptable truncation
        assert score == 3
        assert err is None

    def test_score_below_range(self, judge):
        score, err, _ = judge._parse_response('{"score": 0, "reasoning": "terrible"}')
        assert score is None
        assert err is not None

    def test_score_above_range(self, judge):
        score, err, _ = judge._parse_response('{"score": 6, "reasoning": "amazing"}')
        assert score is None
        assert err is not None

    def test_empty_string(self, judge):
        score, err, _ = judge._parse_response("")
        assert score is None
        assert err is not None

    def test_extra_fields_ignored(self, judge):
        score, err, _ = judge._parse_response('{"score": 3, "reasoning": "ok", "extra": "ignored"}')
        assert score == 3
        assert err is None

    def test_backticks_inside_json_not_stripped(self, judge):
        # Backticks inside the JSON content (e.g. in reasoning) must not be stripped
        raw = '{"score": 3, "reasoning": "use `x` for this"}'
        score, err, _ = judge._parse_response(raw)
        assert score == 3
        assert err is None

    def test_fence_with_backticks_in_content(self, judge):
        # Leading/trailing fences stripped, inner backticks preserved
        raw = '```json\n{"score": 2, "reasoning": "try `foo` instead"}\n```'
        score, err, _ = judge._parse_response(raw)
        assert score == 2
        assert err is None


class TestCallValidation:
    def test_empty_expected_raises(self, judge):
        ctx = ScorerContext(input="What is 2+2?")
        with pytest.raises(ValueError, match="non-empty expected"):
            judge("four", "", ctx)

    def test_whitespace_only_expected_raises(self, judge):
        ctx = ScorerContext(input="What is 2+2?")
        with pytest.raises(ValueError, match="non-empty expected"):
            judge("four", "   ", ctx)


class TestCallJudgeFormatStatus:
    """Verify that __call__ propagates judge_format_status to ctx.metadata_out."""

    def _make_response(self, content: str):
        msg = MagicMock()
        msg.content = content
        resp = MagicMock()
        resp.message = msg
        return resp

    def test_clean_response_sets_judge_format_status(self, judge):
        judge._client.chat = MagicMock(
            return_value=self._make_response('{"score": 3, "reasoning": "ok"}')
        )
        ctx = ScorerContext(input="q")
        judge("answer", "rubric", ctx)
        assert ctx.metadata_out["judge_format_status"] == "clean"

    def test_repaired_response_sets_judge_format_status(self, judge):
        # Truncated JSON — missing closing brace — repair should succeed
        judge._client.chat = MagicMock(
            return_value=self._make_response('{"score": 3, "reasoning": "ok"')
        )
        ctx = ScorerContext(input="q")
        judge("answer", "rubric", ctx)
        assert ctx.metadata_out["judge_format_status"] == "repaired"

    def test_api_error_leaves_judge_format_status_absent(self, judge):
        judge._client.chat = MagicMock(side_effect=RuntimeError("connection refused"))
        ctx = ScorerContext(input="q")
        judge("answer", "rubric", ctx)
        assert "judge_format_status" not in ctx.metadata_out


class TestFixtureIsolation:
    def test_no_directory_created_on_construction(self, tmp_path):
        from evals.scorers.llm_judge import LLMJudgeScorer
        judge = LLMJudgeScorer(results_dir=tmp_path)
        trace_dir = judge._trace_dir
        assert not trace_dir.exists(), "trace dir should not be created until _write_trace is called"
