from __future__ import annotations

import json
import logging
import os
import re

import ollama

from evals.core import DatasetScorer, ScorerContext
from evals.scorers._json_utils import _repair_truncated_json

logger = logging.getLogger(__name__)

_SUFFICIENCY_PROMPT = """\
Does the following context contain enough information to answer this question,
even if the answer would use different words than the context?

Context: {context}
Expected answer: {expected}

Respond with JSON only:
{{"answer": "YES" or "NO", "reasoning": "one sentence — only required if NO"}}"""


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


class ContextSufficiencyScorer(DatasetScorer):
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("JUDGE_MODEL", "llama3.2:3b")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def __call__(self, expected: str, ctx: ScorerContext) -> float | None:
        context = ctx.metadata.get("context")
        if not context:
            return 0.0

        chunks = [context] if isinstance(context, str) else list(context)
        if not chunks:
            return 0.0

        prompt = _SUFFICIENCY_PROMPT.format(context="\n".join(chunks), expected=expected)

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw = response.message.content or ""
        except Exception as e:
            logger.error("context_sufficiency api error: %s", e)
            return None

        ctx.metadata_out["sufficiency_prompt"] = prompt
        ctx.metadata_out["sufficiency_raw_response"] = raw

        answer, reasoning, format_status = self._parse_response(raw)
        if format_status is not None:
            ctx.metadata_out["context_sufficiency_format_status"] = format_status

        if answer == "YES":
            ctx.metadata_out["sufficiency_reasoning"] = None
            return 1.0
        elif answer == "NO":
            ctx.metadata_out["sufficiency_reasoning"] = reasoning
            return 0.0
        else:
            logger.warning("context_sufficiency unexpected response: %r", raw)
            return None

    def _parse_response(self, raw: str) -> tuple[str | None, str | None, str | None]:
        text = _strip_fences(raw)
        format_status = "clean"
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(text)
            if repaired is not None:
                try:
                    obj = json.loads(repaired)
                    format_status = "repaired"
                except json.JSONDecodeError:
                    return None, None, "repair_failed"
            else:
                return None, None, "repair_failed"

        answer = str(obj.get("answer", "")).strip().upper()
        reasoning = obj.get("reasoning") or None
        if answer not in ("YES", "NO"):
            return None, None, format_status
        return answer, reasoning, format_status
