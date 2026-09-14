from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.core import AnyScorer
from evals.scorers.cascade import CascadeScorer
from evals.scorers.exact import exact_match, normalised_match
from evals.scorers.llm_judge import LLMJudgeScorer
from evals.scorers.regex import MultiRegexScorer, RegexScorer
from evals.scorers.schema import JSONSchemaScorer

SCORER_CHOICES = "exact, normalised, regex, multi-regex, schema, judge, cascade, faithfulness, context-sufficiency"

_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "amount": {"type": "number"},
    },
    "required": ["company", "date", "amount"],
    "additionalProperties": False,
}


def build_scorer(
    args: argparse.Namespace,
    evaluated_model: str | None = None,
) -> AnyScorer:
    match args.scorer:
        case "exact":
            return exact_match
        case "normalised":
            return normalised_match
        case "regex":
            if not args.pattern:
                sys.exit(
                    "--pattern is required for scorer 'regex' "
                    r"(e.g. --pattern '\d{4}-\d{2}-\d{2}')"
                )
            return RegexScorer(args.pattern)
        case "multi-regex":
            if not args.pattern:
                sys.exit(
                    "--pattern is required for scorer 'multi-regex' "
                    r"(e.g. --pattern 'company,date,amount' — comma-separated patterns)"
                )
            patterns = [p.strip() for p in args.pattern.split(",")]
            return MultiRegexScorer(patterns)
        case "schema":
            if args.schema:
                schema = json.loads(Path(args.schema).read_text())
            else:
                schema = _EXTRACTION_SCHEMA
            return JSONSchemaScorer(schema)
        case "judge":
            return LLMJudgeScorer(
                scale=args.scale,
                evaluated_model=evaluated_model,
                **({"model": args.judge_model} if args.judge_model else {}),
            )
        case "cascade":
            fast = normalised_match if args.fast_tier == "normalised" else exact_match
            judge = LLMJudgeScorer(
                scale=args.scale,
                evaluated_model=evaluated_model,
                **({"model": args.judge_model} if args.judge_model else {}),
            )
            # sensitivity.py uses --cascade-threshold to avoid confusion with
            # --validation-threshold; other CLIs use --threshold. Support both.
            threshold: float | None = getattr(args, "cascade_threshold", None)
            if threshold is None:
                threshold = float(getattr(args, "threshold", 1.0))
            return CascadeScorer(fast=fast, judge=judge, threshold=threshold)
        case "faithfulness":
            from evals.scorers.faithfulness import FaithfulnessScorer

            return FaithfulnessScorer(
                scale=args.scale,
                **({"model": args.judge_model} if args.judge_model else {}),
            )
        case "context-sufficiency":
            from evals.scorers.context_sufficiency import ContextSufficiencyScorer

            return ContextSufficiencyScorer(
                **({"model": args.judge_model} if args.judge_model else {})
            )
        case _:
            sys.exit(f"Unknown scorer: {args.scorer!r}. Choose from: {SCORER_CHOICES}")
