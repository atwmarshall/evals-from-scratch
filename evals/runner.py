from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from typing import cast

import ollama

from evals.core import (
    AnyScorer,
    DatasetScorer,
    EvalConfig,
    RunResult,
    Sample,
    ScorerCallable,
    ScorerContext,
)

logger = logging.getLogger(__name__)


class Runner:
    def run(
        self,
        dataset: Iterable[Sample],
        scorer: AnyScorer,
        config: EvalConfig,
    ) -> list[RunResult]:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=host, timeout=config.timeout_seconds)

        results: list[RunResult] = []

        for sample in dataset:
            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            messages.append({"role": "user", "content": sample.input})

            completion: str | None = None
            score: float | None = None
            error: str | None = None
            latency_ms: int = 0
            last_error: Exception | None = None
            ctx = ScorerContext(
                input=sample.input,
                metadata={**sample.metadata, "id": sample.id},
                metadata_out={},
            )

            if isinstance(scorer, DatasetScorer):
                # Dataset scorers evaluate data quality — no model call needed.
                completion = ""
                latency_ms = 0
                try:
                    ctx.metadata_out = {}
                    score = scorer(sample.expected, ctx)
                    logger.info("sample=%s score=%s (dataset scorer)", sample.id, score)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    error = str(e)
                    logger.error("sample=%s dataset scorer error: %s", sample.id, e)
            else:
                for attempt in range(config.max_retries):
                    try:
                        t0 = time.monotonic()
                        response = client.chat(
                            model=config.model,
                            messages=messages,
                            options={
                                "temperature": config.temperature,
                                "num_predict": config.max_tokens,
                            },
                        )
                        latency_ms = int((time.monotonic() - t0) * 1000)
                        completion = response.message.content or ""
                        ctx.metadata_out = {}
                        score = cast(ScorerCallable, scorer)(completion, sample.expected, ctx)
                        if score is None and last_error is None:
                            error = "scorer returned None — see judge traces for details"
                        logger.info("sample=%s score=%s", sample.id, score)
                        last_error = None
                        break
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        latency_ms = int((time.monotonic() - t0) * 1000)
                        last_error = e
                        if attempt < config.max_retries - 1:
                            logger.warning(
                                "sample=%s attempt=%d error=%s — retrying in %ds",
                                sample.id, attempt, e, 2 ** attempt,
                            )
                            time.sleep(2 ** attempt)
                        else:
                            logger.error("sample=%s final failure: %s", sample.id, e)

                if last_error is not None:
                    error = str(last_error)

            results.append(RunResult(
                sample=sample,
                completion=completion,
                score=score,
                latency_ms=latency_ms,
                error=error,
                metadata=ctx.metadata_out,
            ))

        return results
