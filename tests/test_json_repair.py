from __future__ import annotations

import json

from evals.scorers._json_utils import _repair_truncated_json


def test_missing_closing_brace():
    assert _repair_truncated_json('{"a": 1') == '{"a": 1}'


def test_truncated_string_value():
    assert (
        _repair_truncated_json('{"score": 3, "reasoning": "tru')
        == '{"score": 3, "reasoning": "tru"}'
    )


def test_missing_closing_bracket():
    assert _repair_truncated_json("[1, 2, 3") == "[1, 2, 3]"


def test_nested_truncated():
    assert _repair_truncated_json('{"a": [1, 2') == '{"a": [1, 2]}'


def test_trailing_comma():
    result = _repair_truncated_json('{"a": 1,')
    assert result is not None
    assert json.loads(result) == {"a": 1}


def test_nothing_to_repair_object():
    assert _repair_truncated_json('{"a": 1}') is None


def test_nothing_to_repair_array():
    assert _repair_truncated_json("[1, 2]") is None


def test_double_nested():
    repaired = _repair_truncated_json('{"a": {"b": 1')
    assert repaired is not None
    assert json.loads(repaired) == {"a": {"b": 1}}
