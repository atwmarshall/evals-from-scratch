from __future__ import annotations

from evals.core import ScorerContext
from evals.scorers.exact import exact_match, normalised_match

_ctx = ScorerContext()


class TestExactMatch:
    def test_identical(self):
        assert exact_match("hello", "hello", _ctx) == 1.0

    def test_different(self):
        assert exact_match("hello", "world", _ctx) == 0.0

    def test_strips_whitespace(self):
        assert exact_match("  hello  ", "hello", _ctx) == 1.0

    def test_strips_newline(self):
        assert exact_match("hello\n", "hello", _ctx) == 1.0

    def test_case_sensitive(self):
        assert exact_match("Hello", "hello", _ctx) == 0.0

    def test_empty_both(self):
        assert exact_match("", "", _ctx) == 1.0

    def test_empty_vs_nonempty(self):
        assert exact_match("", "hello", _ctx) == 0.0

    def test_whitespace_only_vs_empty(self):
        assert exact_match("   ", "", _ctx) == 1.0

    def test_unicode(self):
        assert exact_match("café", "café", _ctx) == 1.0

    def test_unicode_mismatch(self):
        assert exact_match("cafe", "café", _ctx) == 0.0


class TestNormalisedMatch:
    def test_identical(self):
        assert normalised_match("hello", "hello", _ctx) == 1.0

    def test_different(self):
        assert normalised_match("cat", "dog", _ctx) == 0.0

    def test_case_insensitive(self):
        assert normalised_match("Hello World", "hello world", _ctx) == 1.0

    def test_strips_punctuation(self):
        assert normalised_match("hello, world!", "hello world", _ctx) == 1.0

    def test_collapses_whitespace(self):
        assert normalised_match("hello  world", "hello world", _ctx) == 1.0

    def test_empty_both(self):
        assert normalised_match("", "", _ctx) == 1.0

    def test_punctuation_only_vs_empty(self):
        assert normalised_match("!!!", "", _ctx) == 1.0

    def test_hyphens_stripped(self):
        # hyphens are punctuation and get stripped: "2024-03-14" → "20240314"
        assert normalised_match("2024-03-14", "20240314", _ctx) == 1.0

    def test_underscore_preserved(self):
        # underscore is \w so it is NOT stripped
        assert normalised_match("hello_world", "hello world", _ctx) == 0.0

    def test_mixed_case_and_punctuation(self):
        assert normalised_match("The Answer: YES.", "the answer yes", _ctx) == 1.0
