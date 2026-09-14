from __future__ import annotations

import json
import logging
import os
import re

import ollama

from evals.core import ScorerContext
from evals.scorers._json_utils import _repair_truncated_json

logger = logging.getLogger(__name__)

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an answer is faithful to the provided context.
An answer is faithful if every claim in it is supported by the context.
An answer is unfaithful if it introduces facts not in the context, even if those facts are true.

Context:
{context}

Question:
{question}

Answer:
{answer}

Score: 1 (completely unfaithful) to {scale} (completely faithful)
Respond with JSON only: {{"score": <integer 1-{scale}>, "reasoning": "<one sentence>"}}\
"""


class FaithfulnessScorer:
    """LLM judge that evaluates whether a completion is grounded in the provided context.

    Reads ctx.metadata["context"] (list of strings) and ctx.input (the question).
    Ignores `expected` — faithfulness is about grounding, not correctness.
    Returns None if context is absent/empty, or on parse/API failure.
    Writes ctx.metadata_out["faithfulness_format_status"].
    """

    def __init__(self, scale: int = 5, model: str | None = None) -> None:
        self.scale = scale
        self.model = model or os.environ.get("JUDGE_MODEL", "llama3.2:3b")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def __call__(self, completion: str, expected: str, ctx: ScorerContext) -> float | None:
        context_chunks = ctx.metadata.get("context", [])
        if not context_chunks:
            return None

        context_str = "\n".join(context_chunks)
        prompt = _FAITHFULNESS_PROMPT.format(
            context=context_str,
            question=ctx.input,
            answer=completion,
            scale=self.scale,
        )

        raw_response = ""
        format_status: str | None = None
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw_response = response.message.content or ""
            score, error, format_status = self._parse_response(raw_response)
        except Exception as e:
            logger.error("faithfulness api error: %s", e)
            return None
        finally:
            if format_status is not None:
                ctx.metadata_out["faithfulness_format_status"] = format_status

        if score is None:
            return None
        return (score - 1) / (self.scale - 1)

    def _parse_response(self, raw: str) -> tuple[int | None, str | None, str]:
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
                    return (
                        None,
                        f"faithfulness response not valid JSON even after repair: {raw!r}",
                        "repair_failed",
                    )
            else:
                return None, f"faithfulness response not valid JSON: {raw!r}", "repair_failed"

        if "score" not in obj:
            return None, f"faithfulness response missing 'score' field: {obj!r}", format_status

        try:
            score = int(obj["score"])
        except (TypeError, ValueError):
            return None, f"faithfulness 'score' not an integer: {obj['score']!r}", format_status

        if not (1 <= score <= self.scale):
            return None, f"faithfulness score {score} out of range 1–{self.scale}", format_status

        return score, None, format_status


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()
