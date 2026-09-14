from __future__ import annotations

import re

import pytest

from evals.core import ScorerContext
from evals.scorers.regex import MultiRegexScorer, RegexScorer

_ctx = ScorerContext()


class TestRegexScorer:
    def test_match(self):
        assert RegexScorer(r"\d+")("answer is 42", "", _ctx) == 1.0

    def test_no_match(self):
        assert RegexScorer(r"\d+")("no numbers here", "", _ctx) == 0.0

    def test_partial_match_counts(self):
        # search() not fullmatch — partial match in string returns 1.0
        assert RegexScorer(r"hello")("say hello world", "", _ctx) == 1.0

    def test_empty_completion(self):
        assert RegexScorer(r"\d+")("", "", _ctx) == 0.0

    def test_empty_pattern_matches_everything(self):
        assert RegexScorer(r"")("anything", "", _ctx) == 1.0

    def test_empty_pattern_matches_empty(self):
        assert RegexScorer(r"")("", "", _ctx) == 1.0

    def test_case_insensitive_by_default(self):
        assert RegexScorer(r"Hello")("hello", "", _ctx) == 1.0

    def test_case_sensitive_with_flags_zero(self):
        assert RegexScorer(r"Hello", flags=0)("hello", "", _ctx) == 0.0

    def test_case_insensitive_matches_upper(self):
        assert RegexScorer(r"acme corp")("ACME CORP paid the invoice", "", _ctx) == 1.0

    def test_invalid_pattern_raises(self):
        with pytest.raises(re.error):
            RegexScorer(r"[invalid")

    def test_expected_arg_ignored(self):
        # expected is not used by RegexScorer
        assert RegexScorer(r"yes")("yes", "no", _ctx) == 1.0
        assert RegexScorer(r"yes")("no", "yes", _ctx) == 0.0

    def test_multiline_anchor(self):
        assert RegexScorer(r"(?m)^bar$")("foo\nbar\nbaz", "", _ctx) == 1.0


class TestMultiRegexScorer:
    def test_empty_patterns_returns_zero(self):
        assert MultiRegexScorer([])("anything", "", _ctx) == 0.0

    def test_all_match(self):
        assert MultiRegexScorer([r"foo", r"bar"])("foobar", "", _ctx) == 1.0

    def test_none_match(self):
        assert MultiRegexScorer([r"foo", r"bar"])("baz", "", _ctx) == 0.0

    def test_partial_match(self):
        assert MultiRegexScorer([r"foo", r"bar"])("foo only", "", _ctx) == 0.5

    def test_single_pattern_match(self):
        assert MultiRegexScorer([r"\d+"])("42", "", _ctx) == 1.0

    def test_single_pattern_no_match(self):
        assert MultiRegexScorer([r"\d+"])("no digits", "", _ctx) == 0.0

    def test_duplicate_patterns(self):
        # each pattern counted separately, duplicates both match → 2/2 = 1.0
        assert MultiRegexScorer([r"a", r"a"])("a", "", _ctx) == 1.0

    def test_three_patterns_one_third(self):
        result = MultiRegexScorer([r"foo", r"bar", r"baz"])("foo only", "", _ctx)
        assert abs(result - 1 / 3) < 1e-9

    def test_case_insensitive_by_default(self):
        assert MultiRegexScorer([r"foo", r"bar"])("FOO BAR", "", _ctx) == 1.0
