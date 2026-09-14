from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import ollama

from evals.core import Dataset, Sample

logger = logging.getLogger(__name__)

PERTURBATION_PROMPTS: dict[str, str] = {
    "typos": "Introduce 2-3 realistic typos into this text as a human might make.",
    "colloquial": "Rewrite this in casual conversational language.",
    "verbose": "Add redundant words and padding while keeping the same question.",
    "indirect": "Rephrase this as an indirect or implicit question.",
    "multilingual": "Translate key terms into another language but keep the structure.",
}

_PERTURBATION_PROMPT = """\
{instruction}

Text:
{input}

Return only the rewritten text. Do not add any explanation or preamble.\
"""


class PerturbationGenerator:
    """Generates adversarially perturbed dataset variants to stress-test model robustness.

    Each perturbation type applies a different rewriting instruction that intentionally
    degrades or transforms the input. Unlike VariationGenerator, there is no validation
    step — the expected degradation in model scores IS the signal. A missing score is
    more honest than a fake one: if the perturbation LLM fails to generate a perturbation
    for a sample, that sample is excluded from the perturbation column rather than falling
    back to the original input (which would measure robustness to a perturbation that
    never happened).

    Env var resolution: PERTURBATION_MODEL → DEFAULT_MODEL. VARIATION_MODEL is
    intentionally excluded — it was set specifically to avoid circular validation in
    sensitivity analysis, which is a different concern entirely.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = (
            model
            or os.environ.get("PERTURBATION_MODEL")
            or os.environ.get("DEFAULT_MODEL", "llama3.2:3b")
        )
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def generate(
        self,
        dataset: Dataset,
        perturbations: list[str] | None = None,
    ) -> dict[str, Dataset]:
        """Generate perturbed datasets for each perturbation type.

        Args:
            dataset: Original dataset to perturb.
            perturbations: Subset of PERTURBATION_PROMPTS keys to apply. Defaults to all.

        Returns:
            Dict mapping perturbation name to Dataset. Always includes "baseline"
            with the original dataset unchanged. Samples where the perturbation LLM
            fails are excluded from that perturbation's Dataset — they will appear as
            missing scores (rendered as —) rather than silently reusing the original input.
        """
        if perturbations is None:
            perturbations = list(PERTURBATION_PROMPTS.keys())

        unknown = [p for p in perturbations if p not in PERTURBATION_PROMPTS]
        if unknown:
            raise ValueError(
                f"unknown perturbation types: {unknown!r}. Valid: {list(PERTURBATION_PROMPTS)}"
            )

        result: dict[str, Dataset] = {"baseline": dataset}

        for perturbation_name in perturbations:
            instruction = PERTURBATION_PROMPTS[perturbation_name]
            perturbed_samples: list[Sample] = []
            for sample in dataset:
                try:
                    perturbed_input = self._perturb_sample(sample.input, instruction)
                except Exception as e:
                    logger.warning(
                        "perturbation failed for sample %s (%s): %s — excluding from this perturbation",
                        sample.id,
                        perturbation_name,
                        e,
                    )
                    continue  # exclude, don't fall back — a missing score is honest
                perturbed_samples.append(
                    Sample(
                        id=sample.id,
                        input=perturbed_input,
                        expected=sample.expected,
                        metadata=sample.metadata,
                    )
                )
            result[perturbation_name] = Dataset(samples=perturbed_samples)

        return result

    def save_perturbations(
        self,
        perturbations: dict[str, Dataset],
        source_path: str | Path,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Save perturbation datasets to datasets/generated/robustness/.

        Writes one JSONL per perturbation type (excluding "baseline") plus
        generation_metadata.json recording provenance.

        Args:
            perturbations: Output of generate() — includes baseline and all perturbation types.
            source_path: Path to the source JSONL passed to generate().
            output_dir: Override the output directory. Defaults to
                        datasets/generated/robustness/{date}_{source_stem}_{model_slug}/.

        Returns:
            Path to the directory where files were written.
        """
        if output_dir is None:
            date = datetime.now().strftime("%Y-%m-%d")
            source_stem = Path(source_path).stem
            model_slug = re.sub(r"[:/]", "_", self.model)
            output_dir = (
                Path("datasets") / "generated" / "robustness" / f"{date}_{source_stem}_{model_slug}"
            )
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for name, ds in perturbations.items():
            if name == "baseline":
                continue
            jsonl_path = output_dir / f"{name}.jsonl"
            with jsonl_path.open("w") as f:
                for sample in ds:
                    f.write(
                        json.dumps(
                            {
                                "id": sample.id,
                                "input": sample.input,
                                "expected": sample.expected,
                                "metadata": sample.metadata,
                            }
                        )
                        + "\n"
                    )
            logger.info("saved %d samples to %s", len(ds), jsonl_path)

        metadata = {
            "source_path": str(Path(source_path).resolve()),
            "perturbation_model": self.model,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "perturbation_types": [k for k in perturbations if k != "baseline"],
            "sample_counts": {k: len(v) for k, v in perturbations.items() if k != "baseline"},
        }
        (output_dir / "generation_metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("saved generation metadata to %s", output_dir / "generation_metadata.json")

        return output_dir

    @classmethod
    def load_perturbations(cls, directory: str | Path) -> dict[str, Dataset]:
        """Load previously saved perturbation datasets from a generated directory.

        Reads all *.jsonl files in the directory. generation_metadata.json is not
        a JSONL file and is excluded automatically by the glob pattern.

        Args:
            directory: Path to a directory written by save_perturbations().

        Returns:
            Dict mapping perturbation name to Dataset. "baseline" is never present —
            load the original source dataset separately if needed.

        Raises:
            FileNotFoundError: if the directory doesn't exist.
            ValueError: if no JSONL files are found.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"perturbation directory not found: {directory}")

        jsonl_files = sorted(directory.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"no JSONL files found in {directory}")

        result: dict[str, Dataset] = {}
        for path in jsonl_files:
            result[path.stem] = Dataset.from_jsonl(path)
            logger.info("loaded %d samples from %s", len(result[path.stem]), path)

        return result

    def _perturb_sample(self, input_text: str, instruction: str) -> str:
        """Call the LLM to perturb input_text according to instruction.

        Raises on Ollama error — caller catches and excludes the sample.
        """
        prompt = _PERTURBATION_PROMPT.format(instruction=instruction, input=input_text)
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return (response.message.content or "").strip()
