from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


@dataclass
class Sample:
    id: str
    input: str
    expected: str
    metadata: dict[str, Any]


@dataclass
class Dataset:
    samples: list[Sample]

    @classmethod
    def from_jsonl(cls, path: str | Path, limit: int | None = None) -> Dataset:
        path = Path(path)
        samples: list[Sample] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                metadata = dict(obj.get("metadata") or {})
                for key in obj:
                    if key not in {"id", "input", "expected", "metadata"}:
                        metadata[key] = obj[key]
                samples.append(Sample(
                    id=obj["id"],
                    input=obj["input"],
                    expected=obj["expected"],
                    metadata=metadata,
                ))
        if limit is not None:
            samples = samples[:limit]
        return cls(samples=samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)


@dataclass
class RunResult:
    sample: Sample
    completion: str | None
    score: float | None
    latency_ms: int
    error: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScorerContext:
    """Extra context passed to every scorer alongside completion and expected.

    Pure scorers (exact_match, RegexScorer, etc.) accept this parameter but
    ignore it. Scorers that need the original question or metadata (LLMJudge,
    faithfulness) read from it.

    Populated by Runner from the current Sample before each scorer call.
    Scorers write diagnostic metadata to `metadata_out`; Runner copies it to
    RunResult.metadata after the call.
    """

    input: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_out: dict[str, Any] = field(default_factory=dict)


ScorerCallable = Callable[[str, str, ScorerContext], float | None]


class DatasetScorer:
    """Marker interface. Runner detects this and skips the model call entirely.

    Subclasses evaluate dataset quality, not model output. Their __call__
    signature drops `completion` — it is not ignored, it is structurally absent.

    Run before model evals to validate dataset construction. If context
    sufficiency is low for a sample, remove it — a bad sample produces
    misleading model scores regardless of scorer quality.
    """
    pass


DatasetScorerCallable = Callable[[str, ScorerContext], float | None]
AnyScorer = Union[ScorerCallable, DatasetScorerCallable]


@dataclass
class EvalConfig:
    model: str = field(default_factory=lambda: os.environ.get("DEFAULT_MODEL", "llama3.2:3b"))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "1024")))
    temperature: float = 0.0
    system_prompt: str = ""
    max_retries: int = 3
    timeout_seconds: int = 30
