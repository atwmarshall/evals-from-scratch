from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import ollama

from evals.core import Dataset, Sample, ScorerCallable, ScorerContext

logger = logging.getLogger(__name__)

VARIATION_PROMPTS: dict[str, str] = {
    "synonym_swap": "Rewrite this question replacing key verbs and nouns with synonyms. Keep the exact meaning.",
    "rephrase":     "Rephrase this question completely differently. Same meaning, different sentence structure.",
    "add_noise":    "Add one irrelevant sentence to this question. The added sentence should be unrelated to the task.",
    "formal":       "Rewrite this question in a more formal register.",
    "concise":      "Rewrite this question more concisely. Remove all unnecessary words.",
}

_VARIATION_PROMPT = """\
{instruction}

Text:
{input}

Return only the rewritten text. Do not add any explanation or preamble.\
"""


class VariationGenerator:
    """Generates semantically equivalent dataset variations by rewriting inputs via LLM.

    Each variation type applies a different rewriting instruction (synonym swap,
    rephrase, noise injection, etc.) while preserving sample.id, sample.expected,
    and sample.metadata unchanged.

    A dedicated VARIATION_MODEL env var is provided so the variation model can
    differ from both the evaluated model (DEFAULT_MODEL) and the judge (JUDGE_MODEL).
    Using the same model to generate and score variations would measure
    self-consistency rather than scorer reliability.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("VARIATION_MODEL") or os.environ.get("DEFAULT_MODEL", "llama3.2:3b")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def generate(
        self,
        dataset: Dataset,
        variations: list[str] | None = None,
    ) -> dict[str, Dataset]:
        """Generate varied datasets for each variation type.

        Args:
            dataset: Original dataset to vary.
            variations: Subset of VARIATION_PROMPTS keys to apply. Defaults to all.

        Returns:
            Dict mapping variation name to Dataset. Always includes "baseline"
            with the original dataset unchanged.
        """
        if variations is None:
            variations = list(VARIATION_PROMPTS.keys())

        unknown = [v for v in variations if v not in VARIATION_PROMPTS]
        if unknown:
            raise ValueError(f"unknown variation types: {unknown!r}. Valid: {list(VARIATION_PROMPTS)}")

        result: dict[str, Dataset] = {"baseline": dataset}

        for variation_name in variations:
            instruction = VARIATION_PROMPTS[variation_name]
            varied_samples: list[Sample] = []
            for sample in dataset:
                try:
                    varied_input = self._vary_sample(sample.input, instruction)
                except Exception as e:
                    logger.warning(
                        "variation failed for sample %s (%s): %s",
                        sample.id, variation_name, e,
                    )
                    varied_input = sample.input
                varied_samples.append(Sample(
                    id=sample.id,
                    input=varied_input,
                    expected=sample.expected,
                    metadata=sample.metadata,
                ))
            result[variation_name] = Dataset(samples=varied_samples)

        return result

    def validate_variations(
        self,
        variations: dict[str, Dataset],
        validation_scorer: ScorerCallable,
        threshold: float = 0.8,
    ) -> tuple[dict[str, Dataset], list[dict]]:
        """Return a filtered copy of `variations` where per-sample validity is confirmed,
        plus a list of discarded entries for audit.

        For each variation type (excluding "baseline"), scores the original expected
        answer against the varied input using `validation_scorer`. Any (sample, variation)
        pair where the scorer returns None or a score below `threshold` is discarded.

        `validation_scorer` should always be a context-aware scorer (LLMJudgeScorer or
        CascadeScorer). Pure scorers (exact_match, regex, schema) compare completion
        against expected directly and ignore ctx.input — since both are the ground-truth
        expected value here, they always return 1.0 and provide no signal. Use the judge
        for validation regardless of which scorer you're running sensitivity analysis with.

        The "baseline" key is passed through unchanged without scoring.
        Empty variation Datasets (all samples filtered) are kept as keys with samples=[]
        so downstream reporters see them rather than silently missing them.

        Returns:
            (validated, discards) where discards is a list of dicts with keys:
            id, variation, varied_input, expected, validity_score (float | None).
        """
        validated: dict[str, Dataset] = {}
        discards: list[dict] = []

        for variation_name, dataset in variations.items():
            if variation_name == "baseline":
                validated["baseline"] = dataset
                continue

            valid_samples: list[Sample] = []

            for varied_sample in dataset:
                ctx = ScorerContext(
                    input=varied_sample.input,
                    metadata={**varied_sample.metadata, "id": varied_sample.id},
                    metadata_out={},
                )
                try:
                    score = validation_scorer(varied_sample.expected, varied_sample.expected, ctx)
                except Exception as exc:
                    logger.warning(
                        "validate_variations: scorer raised for sample %s variation %r: %s — discarding",
                        varied_sample.id, variation_name, exc,
                    )
                    discards.append({
                        "id": varied_sample.id,
                        "variation": variation_name,
                        "varied_input": varied_sample.input,
                        "expected": varied_sample.expected,
                        "validity_score": None,
                        "reason": "scorer_exception",
                    })
                    continue

                if score is None:
                    logger.warning(
                        "validate_variations: discarded sample %s from variation %r — scorer_returned_none (threshold=%.3f)",
                        varied_sample.id, variation_name, threshold,
                    )
                    discards.append({
                        "id": varied_sample.id,
                        "variation": variation_name,
                        "varied_input": varied_sample.input,
                        "expected": varied_sample.expected,
                        "validity_score": None,
                        "reason": "scorer_returned_none",
                    })
                    continue

                if score < threshold:
                    logger.warning(
                        "validate_variations: discarded sample %s from variation %r — score_below_threshold (score=%.3f, threshold=%.3f)",
                        varied_sample.id, variation_name, score, threshold,
                    )
                    discards.append({
                        "id": varied_sample.id,
                        "variation": variation_name,
                        "varied_input": varied_sample.input,
                        "expected": varied_sample.expected,
                        "validity_score": score,
                        "reason": "score_below_threshold",
                    })
                    continue

                valid_samples.append(varied_sample)

            if not valid_samples:
                logger.warning(
                    "validate_variations: variation %r has zero valid samples after filtering",
                    variation_name,
                )

            validated[variation_name] = Dataset(samples=valid_samples)

        return validated, discards

    def save_variations(
        self,
        validated: dict[str, Dataset],
        original: dict[str, Dataset],
        source_path: str | Path,
        threshold: float,
        discards: list[dict] | None = None,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Save validated variation datasets to datasets/generated/sensitivity/.

        Writes one JSONL per variation type (excluding "baseline", which is
        identical to the source) plus a generation_metadata.json file recording
        provenance: source dataset, variation model, timestamp, threshold, and
        per-variation discard counts.

        Args:
            validated: Output of validate_variations() — the filtered datasets.
            original: Output of generate() — the pre-validation datasets (used to
                      compute discard counts by comparing lengths).
            source_path: Path to the source JSONL that was passed to generate().
            threshold: The validation threshold that was used.
            discards: Optional list of discard records from validate_variations().
                      Written to discarded.jsonl if provided and non-empty.
            output_dir: Override the output directory. Defaults to
                        datasets/generated/sensitivity/{date}_{source_stem}_{model_slug}/.

        Returns:
            Path to the directory where files were written.
        """
        if output_dir is None:
            date = datetime.now().strftime("%Y-%m-%d")
            source_stem = Path(source_path).stem
            model_slug = re.sub(r"[:/]", "_", self.model)
            output_dir = Path("datasets") / "generated" / "sensitivity" / f"{date}_{source_stem}_{model_slug}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        discard_counts: dict[str, int] = {}
        for name, ds in validated.items():
            if name == "baseline":
                continue
            orig_count = len(original.get(name, Dataset(samples=[])))
            discard_counts[name] = orig_count - len(ds)
            jsonl_path = output_dir / f"{name}.jsonl"
            with jsonl_path.open("w") as f:
                for sample in ds:
                    f.write(json.dumps({
                        "id": sample.id,
                        "input": sample.input,
                        "expected": sample.expected,
                        "metadata": sample.metadata,
                    }) + "\n")
            logger.info("saved %d samples to %s", len(ds), jsonl_path)

        if discards:
            discarded_path = output_dir / "discarded.jsonl"
            with discarded_path.open("w") as f:
                for entry in discards:
                    f.write(json.dumps(entry) + "\n")
            logger.info("saved %d discarded samples to %s", len(discards), discarded_path)

        metadata = {
            "source_path": str(Path(source_path).resolve()),
            "variation_model": self.model,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "threshold": threshold,
            "variation_types": [k for k in validated if k != "baseline"],
            "discard_counts": discard_counts,
            "sample_counts": {k: len(v) for k, v in validated.items() if k != "baseline"},
        }
        (output_dir / "generation_metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("saved generation metadata to %s", output_dir / "generation_metadata.json")

        return output_dir

    @classmethod
    def load_variations(cls, directory: str | Path) -> dict[str, Dataset]:
        """Load previously saved variation datasets from a generated directory.

        Reads all *.jsonl files in the directory and returns them as a dict keyed
        by variation name (filename stem). The "baseline" key is never present —
        load the original source dataset separately if needed.

        Args:
            directory: Path to a directory written by save_variations().

        Returns:
            Dict mapping variation name to Dataset.

        Raises:
            FileNotFoundError: if the directory doesn't exist.
            ValueError: if no JSONL files are found.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"variation directory not found: {directory}")

        jsonl_files = sorted(
            p for p in directory.glob("*.jsonl") if p.stem != "discarded"
        )
        if not jsonl_files:
            raise ValueError(f"no JSONL files found in {directory}")

        result: dict[str, Dataset] = {}
        for path in jsonl_files:
            result[path.stem] = Dataset.from_jsonl(path)
            logger.info("loaded %d samples from %s", len(result[path.stem]), path)

        return result

    def _vary_sample(self, input_text: str, instruction: str) -> str:
        """Call the LLM to rewrite input_text according to instruction.

        Raises on Ollama error — caller catches and falls back to original input.
        """
        prompt = _VARIATION_PROMPT.format(instruction=instruction, input=input_text)
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return (response.message.content or "").strip()
