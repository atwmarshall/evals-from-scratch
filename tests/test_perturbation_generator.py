from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from evals.core import Dataset, Sample
from evals.perturbation_generator import PerturbationGenerator


def _make_dataset(*ids: str) -> Dataset:
    return Dataset(samples=[
        Sample(id=i, input=f"input {i}", expected=f"expected {i}", metadata={})
        for i in ids
    ])


def _gen() -> PerturbationGenerator:
    """PerturbationGenerator with no Ollama client."""
    return PerturbationGenerator.__new__(PerturbationGenerator)


class TestGenerate:
    def test_baseline_always_present(self):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen._perturb_sample = MagicMock(return_value="perturbed")
        result = gen.generate(ds, perturbations=["typos"])
        assert "baseline" in result
        assert result["baseline"] is ds

    def test_returns_dataset_for_each_requested_type(self):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a", "b")
        gen._perturb_sample = MagicMock(return_value="perturbed")
        result = gen.generate(ds, perturbations=["typos", "colloquial"])
        assert set(result.keys()) == {"baseline", "typos", "colloquial"}

    def test_successful_perturbation_replaces_input(self):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen._perturb_sample = MagicMock(return_value="teh perturbed text")
        result = gen.generate(ds, perturbations=["typos"])
        assert result["typos"].samples[0].input == "teh perturbed text"

    def test_api_error_excludes_sample(self):
        """A failed perturbation excludes the sample — does not fall back to original input."""
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a", "b")
        gen._perturb_sample = MagicMock(side_effect=[RuntimeError("api down"), "perturbed b"])
        result = gen.generate(ds, perturbations=["typos"])
        # "a" excluded, "b" present
        assert len(result["typos"].samples) == 1
        assert result["typos"].samples[0].id == "b"

    def test_api_error_does_not_fall_back_to_original(self):
        """Regression: falling back to original input would silently measure robustness
        to a perturbation that never happened."""
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen._perturb_sample = MagicMock(side_effect=RuntimeError("timeout"))
        result = gen.generate(ds, perturbations=["typos"])
        assert len(result["typos"].samples) == 0

    def test_expected_and_metadata_preserved(self):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = Dataset(samples=[Sample(id="x", input="q", expected="ans", metadata={"cat": "test"})])
        gen._perturb_sample = MagicMock(return_value="different q")
        result = gen.generate(ds, perturbations=["typos"])
        s = result["typos"].samples[0]
        assert s.expected == "ans"
        assert s.metadata["cat"] == "test"
        assert s.id == "x"

    def test_unknown_perturbation_raises_value_error(self):
        gen = _gen()
        gen.model = "mistral:7b"
        with pytest.raises(ValueError, match="unknown perturbation types"):
            gen.generate(_make_dataset("a"), perturbations=["nonexistent"])

    def test_all_perturbation_types_generated_by_default(self):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen._perturb_sample = MagicMock(return_value="perturbed")
        result = gen.generate(ds)
        assert {"typos", "colloquial", "verbose", "indirect", "multilingual"}.issubset(result.keys())


class TestSavePerturbations:
    def test_writes_jsonl_per_perturbation(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a", "b")
        perturbations = {"baseline": ds, "typos": ds}
        gen.save_perturbations(perturbations, "datasets/test.jsonl", output_dir=tmp_path)
        assert (tmp_path / "typos.jsonl").exists()

    def test_baseline_not_written_as_jsonl(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen.save_perturbations({"baseline": ds, "typos": ds}, "datasets/test.jsonl", output_dir=tmp_path)
        assert not (tmp_path / "baseline.jsonl").exists()

    def test_jsonl_content_is_correct(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen.save_perturbations({"typos": ds}, "datasets/test.jsonl", output_dir=tmp_path)
        lines = (tmp_path / "typos.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["id"] == "a"
        assert obj["input"] == "input a"
        assert obj["expected"] == "expected a"

    def test_generation_metadata_written(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a", "b")
        perturbations = {"baseline": ds, "typos": Dataset(samples=[ds.samples[0]])}
        gen.save_perturbations(perturbations, "datasets/test.jsonl", output_dir=tmp_path)
        meta = json.loads((tmp_path / "generation_metadata.json").read_text())
        assert meta["perturbation_model"] == "mistral:7b"
        assert "generated_at" in meta
        assert "typos" in meta["perturbation_types"]
        assert meta["sample_counts"]["typos"] == 1

    def test_metadata_has_no_threshold_or_discard_keys(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen.save_perturbations({"typos": ds}, "datasets/test.jsonl", output_dir=tmp_path)
        meta = json.loads((tmp_path / "generation_metadata.json").read_text())
        assert "threshold" not in meta
        assert "discard_counts" not in meta

    def test_no_discarded_jsonl_written(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen.save_perturbations({"typos": ds}, "datasets/test.jsonl", output_dir=tmp_path)
        assert not (tmp_path / "discarded.jsonl").exists()

    def test_default_output_dir_uses_robustness_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        out = gen.save_perturbations({"typos": ds}, "datasets/challenge2/extraction.jsonl")
        assert "robustness" in str(out)
        assert "extraction" in str(out)
        assert "mistral_7b" in str(out)
        assert out.exists()


class TestLoadPerturbations:
    def test_loads_saved_perturbations(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a", "b")
        gen.save_perturbations({"typos": ds, "colloquial": ds}, "datasets/test.jsonl", output_dir=tmp_path)
        loaded = PerturbationGenerator.load_perturbations(tmp_path)
        assert set(loaded.keys()) == {"typos", "colloquial"}
        assert len(loaded["typos"].samples) == 2

    def test_round_trip_preserves_sample_fields(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        sample = Sample(id="x", input="hello", expected="world", metadata={"cat": "test"})
        ds = Dataset(samples=[sample])
        gen.save_perturbations({"typos": ds}, "datasets/test.jsonl", output_dir=tmp_path)
        loaded = PerturbationGenerator.load_perturbations(tmp_path)
        s = loaded["typos"].samples[0]
        assert s.id == "x"
        assert s.input == "hello"
        assert s.expected == "world"
        assert s.metadata["cat"] == "test"

    def test_generation_metadata_not_loaded_as_perturbation(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen.save_perturbations({"typos": ds}, "datasets/test.jsonl", output_dir=tmp_path)
        loaded = PerturbationGenerator.load_perturbations(tmp_path)
        assert "generation_metadata" not in loaded

    def test_skips_non_jsonl_files(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        gen.save_perturbations({"typos": ds}, "datasets/test.jsonl", output_dir=tmp_path)
        (tmp_path / "notes.txt").write_text("some notes")
        loaded = PerturbationGenerator.load_perturbations(tmp_path)
        assert "notes" not in loaded

    def test_raises_if_directory_missing(self):
        with pytest.raises(FileNotFoundError):
            PerturbationGenerator.load_perturbations("/nonexistent/path/xyz")

    def test_raises_if_no_jsonl_files(self, tmp_path):
        with pytest.raises(ValueError, match="no JSONL files"):
            PerturbationGenerator.load_perturbations(tmp_path)
