from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evals.core import DatasetScorer, ScorerContext
from evals.scorers.context_sufficiency import ContextSufficiencyScorer
from evals.scorers.faithfulness import FaithfulnessScorer


def _ctx(context=None, input_text="What is the capital of Australia?", metadata=None):
    meta = dict(metadata or {})
    if context is not None:
        meta["context"] = context
    return ScorerContext(input=input_text, metadata=meta)


def _mock_response(content: str):
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


class TestFaithfulnessScorer:
    @pytest.fixture
    def scorer(self):
        return FaithfulnessScorer(scale=5)

    def test_returns_none_when_no_context_in_metadata(self, scorer):
        ctx = _ctx()  # no context key
        result = scorer("Sydney is the capital.", "Canberra", ctx)
        assert result is None

    def test_returns_none_when_context_is_empty_list(self, scorer):
        ctx = _ctx(context=[])
        result = scorer("Sydney is the capital.", "Canberra", ctx)
        assert result is None

    def test_api_error_returns_none(self, scorer):
        ctx = _ctx(context=["Australia has many cities."])
        scorer._client = MagicMock()
        scorer._client.chat.side_effect = RuntimeError("connection refused")
        result = scorer("Sydney.", "Canberra", ctx)
        assert result is None

    def test_score_normalised_correctly(self, scorer):
        ctx = _ctx(context=["The capital of Australia is Canberra."])
        scorer._client = MagicMock()

        scorer._client.chat.return_value = _mock_response('{"score": 5, "reasoning": "fully supported"}')
        assert scorer("Canberra.", "Canberra", ctx) == pytest.approx(1.0)

        scorer._client.chat.return_value = _mock_response('{"score": 1, "reasoning": "no support"}')
        assert scorer("Sydney.", "Canberra", ctx) == pytest.approx(0.0)

        scorer._client.chat.return_value = _mock_response('{"score": 3, "reasoning": "partial"}')
        assert scorer("Maybe Canberra.", "Canberra", ctx) == pytest.approx(0.5)

    def test_context_joined_as_newline_in_prompt(self, scorer):
        chunks = ["Chunk one.", "Chunk two."]
        ctx = _ctx(context=chunks)
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"score": 4, "reasoning": "ok"}')
        scorer("Some answer.", "expected", ctx)

        call_args = scorer._client.chat.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Chunk one.\nChunk two." in prompt

    def test_expected_is_ignored(self, scorer):
        ctx1 = _ctx(context=["The sky is blue."])
        ctx2 = _ctx(context=["The sky is blue."])
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"score": 5, "reasoning": "ok"}')

        scorer("The sky is blue.", "expected A", ctx1)
        call1 = scorer._client.chat.call_args[1]["messages"][0]["content"]

        scorer("The sky is blue.", "expected B", ctx2)
        call2 = scorer._client.chat.call_args[1]["messages"][0]["content"]

        assert call1 == call2

    def test_format_status_written_to_metadata_out(self, scorer):
        ctx = _ctx(context=["Some context."])
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"score": 4, "reasoning": "ok"}')
        scorer("answer", "expected", ctx)
        assert ctx.metadata_out.get("faithfulness_format_status") == "clean"

    def test_parse_failure_returns_none(self, scorer):
        ctx = _ctx(context=["Some context."])
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response("I think it's a 4 out of 5.")
        result = scorer("answer", "expected", ctx)
        assert result is None


class TestContextSufficiencyScorer:
    @pytest.fixture
    def scorer(self):
        return ContextSufficiencyScorer()

    def test_is_dataset_scorer(self, scorer):
        assert isinstance(scorer, DatasetScorer)

    def test_no_completion_arg(self, scorer):
        import inspect
        sig = inspect.signature(scorer.__call__)
        assert list(sig.parameters) == ["expected", "ctx"]

    def test_returns_0_when_no_context(self, scorer):
        ctx = _ctx()  # no context key
        assert scorer("expected", ctx) == 0.0

    def test_returns_0_when_context_is_empty_list(self, scorer):
        ctx = _ctx(context=[])
        assert scorer("expected", ctx) == 0.0

    def test_yes_response_returns_1(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"answer": "YES"}')
        ctx = _ctx(context=["The capital of Australia is Canberra."])
        assert scorer("Canberra", ctx) == 1.0

    def test_no_response_returns_0(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"answer": "NO", "reasoning": "Canberra is not mentioned."}')
        ctx = _ctx(context=["Sydney is a city in New South Wales."])
        assert scorer("Canberra", ctx) == 0.0

    def test_api_error_returns_none(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.side_effect = RuntimeError("connection refused")
        ctx = _ctx(context=["some context"])
        assert scorer("expected", ctx) is None

    def test_parse_failure_returns_none(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response("I'm not sure about this.")
        ctx = _ctx(context=["some context"])
        assert scorer("expected", ctx) is None

    def test_context_joined_with_newlines_in_prompt(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"answer": "YES"}')
        ctx = _ctx(context=["Chunk one.", "Chunk two."])
        scorer("expected", ctx)
        prompt = scorer._client.chat.call_args[1]["messages"][0]["content"]
        assert "Chunk one.\nChunk two." in prompt

    def test_context_as_string_not_list(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"answer": "YES"}')
        ctx = _ctx(context="The capital of France is Paris.")
        assert scorer("Paris", ctx) == 1.0

    def test_format_status_clean_on_yes(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"answer": "YES"}')
        ctx = _ctx(context=["Some context."])
        scorer("expected", ctx)
        assert ctx.metadata_out.get("context_sufficiency_format_status") == "clean"
        assert ctx.metadata_out.get("sufficiency_reasoning") is None

    def test_format_status_clean_on_no(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response(
            '{"answer": "NO", "reasoning": "Context only mentions Sydney, not Canberra."}'
        )
        ctx = _ctx(context=["Some context."])
        scorer("expected", ctx)
        assert ctx.metadata_out.get("context_sufficiency_format_status") == "clean"
        assert ctx.metadata_out.get("sufficiency_reasoning") == "Context only mentions Sydney, not Canberra."

    def test_format_status_repair_failed_on_parse_failure(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response("I'm not sure.")
        ctx = _ctx(context=["Some context."])
        scorer("expected", ctx)
        assert ctx.metadata_out.get("context_sufficiency_format_status") == "repair_failed"

    def test_yes_case_insensitive(self, scorer):
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"answer": "yes"}')
        ctx = _ctx(context=["The capital of Australia is Canberra."])
        assert scorer("Canberra", ctx) == 1.0

    def test_not_enough_information_returns_none(self, scorer):
        # Plausible non-JSON LLM response — no valid answer field
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response("Not enough information to determine this.")
        ctx = _ctx(context=["Some context."])
        assert scorer("expected", ctx) is None

    def test_yes_reasoning_is_none(self, scorer):
        # YES responses don't need reasoning — store None, not an empty string
        scorer._client = MagicMock()
        scorer._client.chat.return_value = _mock_response('{"answer": "YES", "reasoning": ""}')
        ctx = _ctx(context=["Some context."])
        scorer("expected", ctx)
        assert ctx.metadata_out.get("sufficiency_reasoning") is None
