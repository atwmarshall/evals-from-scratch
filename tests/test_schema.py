from __future__ import annotations

from evals.core import ScorerContext
from evals.scorers.schema import JSONSchemaScorer

_ctx = ScorerContext()

_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name"],
}


class TestJSONSchemaScorer:
    def test_valid_json_passes_schema(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"name": "Alice", "age": 30}', "", _ctx) == 1.0

    def test_valid_json_missing_required_field(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"age": 30}', "", _ctx) == 0.5

    def test_valid_json_wrong_type(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"name": 123}', "", _ctx) == 0.5

    def test_not_json(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer("not json at all", "", _ctx) == 0.0

    def test_empty_string(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer("", "", _ctx) == 0.0

    def test_partial_json_repaired_passes_schema(self):
        # truncated JSON that repairs to a valid, schema-conforming object
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"name": "Al', "", _ctx) == 1.0

    def test_partial_json_irreparable(self):
        # completely malformed — not even truncation, repair returns None
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer("not json at all", "", _ctx) == 0.0

    def test_json_with_leading_whitespace(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('  {"name": "Bob"}  ', "", _ctx) == 1.0

    def test_empty_object_against_schema_with_required(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer("{}", "", _ctx) == 0.5

    def test_array_schema(self):
        scorer = JSONSchemaScorer({"type": "array", "items": {"type": "integer"}})
        assert scorer("[1, 2, 3]", "", _ctx) == 1.0
        assert scorer('["a", "b"]', "", _ctx) == 0.5
        assert scorer("not an array", "", _ctx) == 0.0

    def test_expected_arg_ignored(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"name": "X"}', "completely ignored", _ctx) == 1.0

    def test_markdown_json_fence(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('```json\n{"name": "Alice"}\n```', "", _ctx) == 1.0

    def test_markdown_fence_no_language(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('```\n{"name": "Alice"}\n```', "", _ctx) == 1.0

    def test_markdown_fence_invalid_schema(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('```json\n{"age": 30}\n```', "", _ctx) == 0.5
