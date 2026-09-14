from __future__ import annotations

import json
import logging
import re
from typing import Any

import jsonschema

from evals.core import ScorerContext
from evals.scorers._json_utils import _repair_truncated_json

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


def _extract_json(text: str) -> str:
    """Strip markdown code fences and surrounding whitespace."""
    return _FENCE_RE.sub("", text).strip()


class JSONSchemaScorer:
    """Scorer that validates a completion as JSON conforming to a schema.

    Partial credit logic:
      - 0.0  — completion is not valid JSON (or empty)
      - 0.5  — valid JSON but fails schema validation
      - 1.0  — valid JSON and passes schema validation

    `expected` and `ctx` are not used — the schema passed at construction time
    defines what a correct response looks like.

    Handles completions wrapped in markdown code fences (```json ... ```).
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def __call__(self, completion: str, expected: str, ctx: ScorerContext) -> float:
        cleaned = _extract_json(completion)
        format_status = "clean"
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(cleaned)
            if repaired is None:
                ctx.metadata_out["format_status"] = "repair_failed"
                return 0.0
            try:
                parsed = json.loads(repaired)
                format_status = "repaired"
                logger.warning("json repaired (truncated): %s…", cleaned[:80])
            except json.JSONDecodeError:
                ctx.metadata_out["format_status"] = "repair_failed"
                return 0.0

        ctx.metadata_out["format_status"] = format_status
        try:
            jsonschema.validate(parsed, self._schema)
            return 1.0
        except jsonschema.ValidationError as e:
            logger.debug("schema validation failed: %s", e.message)
            return 0.5
