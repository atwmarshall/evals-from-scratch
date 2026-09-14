from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import ollama

from evals.core import ScorerContext
from evals.scorers._json_utils import _repair_truncated_json

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an impartial evaluator. Score the following answer using the criteria below.

Criteria:
{criteria}

Question:
{question}

Answer:
{answer}

Respond with a JSON object with two fields:
- "score": an integer from 1 to {scale} (1 = worst, {scale} = best)
- "reasoning": one sentence explaining your score

Respond with JSON only. No other text."""


class LLMJudgeScorer:
    """LLM-based scorer that evaluates completions against a rubric using a judge model.

    Internally uses a 1–scale integer rating, normalised to 0.0–1.0 via
    (score - 1) / (scale - 1). So score=1 → 0.0, score=scale → 1.0.

    Unlike other scorers, __call__ returns float | None. None signals a judge
    failure (API error or unparseable response). The runner records this as an
    error in RunResult so reporters can distinguish parse failures from genuine
    low scores.

    `expected` provides the per-sample scoring rubric and must be non-empty.
    `ctx.input` provides the original question shown to the model being evaluated.

    Traces (prompt, raw response, parsed score, error) are written as JSON to
    results/judge_traces/{date}/{time}_{evaluated_model}/{sample_id}.json.
    """

    def __init__(
        self,
        scale: int = 5,
        model: str | None = None,
        results_dir: Path | None = None,
        evaluated_model: str | None = None,
    ) -> None:
        self.scale = scale
        self.model = model or os.environ.get("JUDGE_MODEL", "llama3.2:3b")
        self.results_dir = results_dir or Path(os.environ.get("RESULTS_DIR", "results"))
        self._session_dt = datetime.now()
        self._session_ts = self._session_dt.strftime("%H%M%S")
        self._evaluated_model = evaluated_model
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    @property
    def _trace_dir(self) -> Path:
        date = self._session_dt.strftime("%Y-%m-%d")
        return (
            self.results_dir
            / "judge_traces"
            / date
            / f"{self._session_ts}_{self._evaluated_model or 'unknown'}"
        )

    def set_evaluated_model(self, model_id: str) -> None:
        self._evaluated_model = model_id

    def __call__(self, completion: str, expected: str, ctx: ScorerContext) -> float | None:
        if not expected.strip():
            raise ValueError(
                "LLMJudgeScorer requires a non-empty expected value — "
                "it provides the per-sample scoring criteria"
            )

        prompt = _PROMPT_TEMPLATE.format(
            criteria=expected,
            question=ctx.input,
            answer=completion,
            scale=self.scale,
        )

        raw_response: str = ""
        parsed_score: int | None = None
        final_score: float | None = None
        error: str | None = None

        judge_format_status: str | None = None
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw_response = response.message.content or ""
            parsed_score, error, judge_format_status = self._parse_response(raw_response)
            if parsed_score is not None:
                final_score = (parsed_score - 1) / (self.scale - 1)
        except Exception as e:
            error = str(e)
            logger.error("llm_judge api error: %s", e)

        if judge_format_status is not None:
            ctx.metadata_out["judge_format_status"] = judge_format_status

        self._write_trace(prompt, raw_response, parsed_score, final_score, error, ctx)

        if error and final_score is None:
            return None
        return final_score

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
                    logger.warning("judge response repaired (truncated): %s…", raw[:80])
                except json.JSONDecodeError:
                    return (
                        None,
                        f"judge response not valid JSON even after repair: {raw!r}",
                        "repair_failed",
                    )
            else:
                return None, f"judge response not valid JSON: {raw!r}", "repair_failed"

        if "score" not in obj:
            return None, f"judge response missing 'score' field: {obj!r}", format_status

        try:
            score = int(obj["score"])
        except (TypeError, ValueError):
            return None, f"judge 'score' not an integer: {obj['score']!r}", format_status

        if not (1 <= score <= self.scale):
            return None, f"judge score {score} out of range 1–{self.scale}", format_status

        return score, None, format_status

    def _write_trace(
        self,
        prompt: str,
        raw_response: str,
        parsed_score: int | None,
        final_score: float | None,
        error: str | None,
        ctx: ScorerContext,
    ) -> None:
        trace_dir = self._trace_dir
        trace_dir.mkdir(parents=True, exist_ok=True)
        sample_id = ctx.metadata.get("id", "unknown")
        trace_path = trace_dir / f"{sample_id}.json"
        trace = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "model": self.model,
            "evaluated_model": self._evaluated_model,
            "sample_id": sample_id,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_score": parsed_score,
            "final_score": final_score,
            "error": error,
        }
        trace_path.write_text(json.dumps(trace, indent=2))
        logger.debug("judge trace written to %s", trace_path)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()
