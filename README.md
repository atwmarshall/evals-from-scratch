# evals-from-scratch

[![CI](https://github.com/atwmarshall/evals-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/atwmarshall/evals-from-scratch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)

An LLM evaluation framework built from first principles — no eval libraries, no wrappers.

Nine scorers, from exact match through JSON schema with partial credit to LLM judge and
RAG faithfulness, plus two analyses most frameworks leave out:

- **Sensitivity** — hold the model fixed, vary the input phrasing, measure how far the
  *scorer's* output moves. Can I trust my ruler?
- **Robustness** — hold the scorer fixed, perturb the input adversarially, measure how far
  the *model* degrades.

Run sensitivity first. If scorer variance is ±0.1, a robustness delta of 0.08 is noise and
means nothing. Sensitivity analysis separates three model roles — evaluated model, judge,
and variation generator — and exits if the variation generator and judge are the same
model, because using one model to both generate and score variations measures
self-consistency, not reliability.

**Stack**: Python 3.11+, Ollama.

Built by working through every layer, to understand how evals behave and why they are hard.

---

## Project structure

```
evals/
├── evals/
│   ├── core.py                    # Sample, Dataset, RunResult, ScorerContext, EvalConfig, ScorerCallable, DatasetScorer, AnyScorer
│   ├── runner.py                  # Calls the model, populates RunResult
│   ├── reporters.py               # Aggregates scores, prints tables, saves JSON
│   ├── scorer_factory.py          # build_scorer() — maps CLI args to scorer instances
│   ├── sensitivity_reporter.py    # SensitivityReporter — scorer reliability (variance metric)
│   ├── robustness_reporter.py     # RobustnessReporter — model robustness (degradation metric)
│   ├── variation_generator.py     # VariationGenerator — meaning-preserving rewrites for sensitivity
│   ├── perturbation_generator.py  # PerturbationGenerator — adversarial rewrites for robustness
│   └── scorers/
│       ├── exact.py               # exact_match, normalised_match
│       ├── regex.py               # RegexScorer, MultiRegexScorer
│       ├── schema.py              # JSONSchemaScorer
│       ├── llm_judge.py           # LLMJudgeScorer
│       ├── cascade.py             # CascadeScorer (fast + judge two-tier)
│       ├── faithfulness.py        # FaithfulnessScorer (RAG — LLM judge)
│       ├── context_sufficiency.py # ContextSufficiencyScorer (RAG — LLM YES/NO entailment check, DatasetScorer)
│       └── _json_utils.py         # _repair_truncated_json (internal)
├── scripts/
│   ├── run_eval.py                # CLI: run a single eval
│   ├── benchmark.py               # CLI: multi-model comparison
│   ├── sensitivity.py             # CLI: scorer sensitivity analysis
│   ├── robustness.py              # CLI: model robustness analysis
│   └── show.py                    # CLI: inspect result artifacts
├── datasets/
│   ├── extraction/                # Structured extraction samples (JSONL)
│   ├── open_ended/                # Open-ended QA samples (JSONL)
│   ├── rag/                       # RAG eval samples (JSONL)
│   └── generated/
│       ├── sensitivity/           # Auto-created by sensitivity.py
│       └── robustness/            # Auto-created by robustness.py
├── tests/
├── results/                       # Auto-created
│   ├── runs/
│   ├── benchmarks/
│   ├── sensitivity/
│   ├── robustness/
│   └── judge_traces/
├── pyproject.toml
└── .env.example
```

---

## Setup

```bash
uv sync
cp .env.example .env
# edit .env — set OLLAMA_HOST and model vars (see Environment variables below)
```

---

## Scripts

### `run_eval.py` — single eval run

```bash
uv run python scripts/run_eval.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer exact --model llama3.2:3b

# Schema scorer with a JSON schema file
uv run python scripts/run_eval.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer schema --schema datasets/extraction/schema.json

# LLM judge, limit to 20 samples
uv run python scripts/run_eval.py \
  --dataset datasets/open_ended/qa.jsonl \
  --scorer judge --limit 20

# Cascade: normalised match first, judge fallback
uv run python scripts/run_eval.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer cascade --fast-tier normalised
```

Flags: `--dataset`, `--scorer`, `--model`, `--limit`, `--output`, plus scorer-specific flags (see Scorers table).

Results saved to `results/runs/{date}/{time}_{model}_{dataset}_{scorer}/`.

---

### `benchmark.py` — multi-model comparison

Runs the same scorer across multiple models and prints a comparison table (mean score, p50/p95 latency, error rate).

```bash
# Pass models directly
uv run python scripts/benchmark.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer schema --schema datasets/extraction/schema.json \
  --models llama3.2:3b,mistral:7b

# Interactive model selection (questionary checkbox)
uv run python scripts/benchmark.py \
  --dataset datasets/open_ended/qa.jsonl \
  --scorer judge
```

Flags: `--dataset`, `--scorer`, `--models` (comma-separated, skips interactive), plus scorer-specific flags.

Results saved to `results/benchmarks/{date}/{time}_{dataset}_{scorer}/`.

---

### `sensitivity.py` — scorer sensitivity analysis (Part A)

Holds the model fixed and varies input phrasing across 5 variation types. Measures score variance to test scorer reliability: **"can I trust my ruler?"**

Run this before `robustness.py`. If scorer variance is ±0.1 and a robustness delta is 0.08, the robustness signal is within noise and uninterpretable.

#### The pipeline

```
load dataset
  → VariationGenerator.generate()      # 5 variation types via LLM
  → validate_variations()              # LLM judge filters meaning-changed samples
  → Runner().run() per variation       # eval scorer on each variation
  → SensitivityReporter.report()       # variance tables + sensitivity.json
```

#### Three-model requirement

Sensitivity analysis uses three distinct models — this is intentional and enforced:

| Role | Env var | Description |
|------|---------|-------------|
| Evaluated model | `DEFAULT_MODEL` | The model whose completions are scored |
| Judge | `JUDGE_MODEL` | Scores completions via `LLMJudgeScorer`; also validates variations |
| Variation generator | `VARIATION_MODEL` | Rewrites inputs via `VariationGenerator` |

**`VARIATION_MODEL` must differ from `JUDGE_MODEL`.** Using the same model to generate and score variations measures self-consistency, not scorer reliability — the script exits on misconfiguration.

#### Variation types

| Type | What it does |
|------|-------------|
| `synonym_swap` | Replaces key words with synonyms |
| `rephrase` | Rewrites the sentence structure |
| `add_noise` | Adds minor phrasing noise |
| `formal` | Makes the phrasing more formal |
| `concise` | Shortens the input |

#### Validation

Before running the scorer, each variation is checked by the judge for meaning preservation. Variations scoring below `--validation-threshold` (default `0.8`) are dropped per-sample — if a variation fails for one sample, only that sample is excluded for that variation type. Discarded samples are written to `discarded.jsonl` alongside the saved variations for audit.

#### Reusing saved variations

Variations are saved to `datasets/generated/sensitivity/` after generation and validation. To re-run with a different scorer on the same inputs:

```bash
--reuse-variations datasets/generated/sensitivity/2026-03-15_extraction_llama3.2:3b/
```

#### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | required | Path to JSONL dataset |
| `--scorer` | required | Scorer to test for reliability |
| `--model` | `DEFAULT_MODEL` | Model being evaluated |
| `--limit` | none | Max samples |
| `--variations` | all 5 | Space-separated list of variation types to run |
| `--validation-threshold` | `0.8` | Min judge score for meaning preservation |
| `--no-save-variations` | off | Skip saving generated variations |
| `--reuse-variations DIR` | none | Load variations from a saved directory |
| `--output` | `RESULTS_DIR` | Override output directory |
| `--judge-model` | `JUDGE_MODEL` | Override judge model for scoring and validation |
| `--scale` | `5` | Judge score scale |
| `--fast-tier` | `normalised` | Fast tier for cascade scorer |
| `--cascade-threshold` | `1.0` | Fast-tier threshold (separate from `--validation-threshold`) |

#### Output

Per-sample variance table and per-variation summary table printed to stdout. Verdict is `unstable` if variance > 0.05, `ok` otherwise. Summary line notes when the most-destabilising variation ran on fewer samples than baseline (e.g. due to validation discards).

Results saved to `results/sensitivity/{date}/{time}_{dataset}_{scorer}/`.

#### Examples

```bash
# Full run — generates and saves variations
uv run python scripts/sensitivity.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer judge --limit 10

# Reuse previously saved variations (different scorer)
uv run python scripts/sensitivity.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer cascade \
  --reuse-variations datasets/generated/sensitivity/2026-03-15_extraction_llama3.2:3b/

# Run only specific variation types
uv run python scripts/sensitivity.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer judge --variations rephrase formal
```

---

### `robustness.py` — model robustness analysis (Part B)

Holds the scorer fixed and applies adversarial perturbations to the input. Measures score degradation to test model robustness: **"how much does the model degrade under realistic input stress?"**

Only run after `sensitivity.py` has established an acceptable scorer noise floor — if scorer variance is ±0.1, a degradation delta of 0.08 is within noise.

#### The pipeline

```
load dataset
  → PerturbationGenerator.generate()   # 5 adversarial perturbation types via LLM
  → Runner().run() per perturbation    # eval scorer on each perturbation
  → RobustnessReporter.report()        # degradation tables + robustness.json
```

#### Key difference from sensitivity

Perturbations are designed to stress the model — some will intentionally degrade performance. There is no validation step: the degradation itself is the signal. A missing score is more honest than a fake one: if the perturbation LLM fails to generate a perturbation for a sample, that sample is excluded from that column rather than silently falling back to the original input.

#### Perturbation types

| Type | What it does |
|------|-------------|
| `typos` | Introduces 2–3 realistic typos |
| `colloquial` | Rewrites in casual conversational language |
| `verbose` | Adds redundant words and padding |
| `indirect` | Rephrases as an indirect or implicit question |
| `multilingual` | Translates key terms into another language |

#### Verdicts

| Verdict | Condition |
|---------|-----------|
| `robust` | degradation < 0.1 |
| `fragile` | 0.1 ≤ degradation < 0.3 |
| `brittle` | degradation ≥ 0.3 |
| `n/a` | baseline score is `None`, or no valid perturbation scores |

Degradation = `baseline_score − mean(perturbation_scores)` per sample. Positive = model scored lower under perturbation.

#### Env var

No three-model separation requirement. The perturbation model can equal the evaluated model — there is no circularity concern here. Env var resolution: `PERTURBATION_MODEL` → `DEFAULT_MODEL`.

#### Reusing saved perturbations

Perturbations are saved to `datasets/generated/robustness/` after generation. To re-run with a different scorer:

```bash
--reuse-perturbations datasets/generated/robustness/2026-03-15_extraction_llama3.2:3b/
```

#### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | required | Path to JSONL dataset |
| `--scorer` | required | Scorer held fixed during robustness test |
| `--model` | `DEFAULT_MODEL` | Model being evaluated |
| `--limit` | none | Max samples |
| `--perturbations` | all 5 | Space-separated list of perturbation types to run |
| `--no-save-perturbations` | off | Skip saving generated perturbations |
| `--reuse-perturbations DIR` | none | Load perturbations from a saved directory |
| `--output` | `RESULTS_DIR` | Override output directory |
| `--judge-model` | `JUDGE_MODEL` | Override judge model (for judge/cascade scorer) |
| `--scale` | `5` | Judge score scale |
| `--fast-tier` | `normalised` | Fast tier for cascade scorer |
| `--cascade-threshold` | `1.0` | Fast-tier threshold |

#### Output

Per-sample degradation table and per-perturbation summary printed to stdout. Summary line notes when the most-degrading perturbation ran on fewer samples than baseline.

Results saved to `results/robustness/{date}/{time}_{dataset}_{model}/` — keyed by model (not scorer), because robustness measures how the model responds to adversarial inputs.

#### Examples

```bash
# Full run
uv run python scripts/robustness.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer judge --limit 10

# Specific perturbation types only
uv run python scripts/robustness.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer exact --perturbations typos indirect

# Reuse saved perturbations
uv run python scripts/robustness.py \
  --dataset datasets/extraction/extraction.jsonl \
  --scorer cascade \
  --reuse-perturbations datasets/generated/robustness/2026-03-15_extraction_llama3.2:3b/
```

#### What to watch for

- `indirect` and `typos` typically cause the most degradation
- `verbose` can sometimes *improve* scores on judge-evaluated tasks due to verbosity bias — this is worth noting if it happens, as it reveals a scorer blind spot rather than genuine robustness
- Degradation that exceeds scorer variance (established by `sensitivity.py`) is real signal; degradation within that noise floor is uninterpretable

---

### `show.py` — inspect result artifacts

Auto-detects the path type in this order: sensitivity dir → benchmark dir → run dir → JSONL → traces dir.

> **Known gap**: robustness result directories (`robustness.json` present) are not yet auto-detected. Per-perturbation JSONL files inside the directory are readable by the `.jsonl` handler, but the summary and degradation table are not shown. See `docs/backlog/show-robustness-support.md`.

```bash
# Inspect a run
uv run python scripts/show.py results/runs/<date>/<run_dir>/

# Inspect a benchmark
uv run python scripts/show.py results/benchmarks/<date>/<bench_dir>/

# Inspect a sensitivity result
uv run python scripts/show.py results/sensitivity/<date>/<sens_dir>/

# Show only unstable samples (sensitivity mode)
uv run python scripts/show.py results/sensitivity/<date>/<sens_dir>/ --failures-only

# Show one sample by ID
uv run python scripts/show.py results/runs/<date>/<run_dir>/ --id sample-001

# Show full completions
uv run python scripts/show.py results/runs/<date>/<run_dir>/ --verbose

# Inspect a raw JSONL file (works for any JSONL, including per-perturbation files)
uv run python scripts/show.py datasets/extraction/extraction.jsonl

# Inspect judge traces
uv run python scripts/show.py results/judge_traces/<date>/<session_dir>/
```

In sensitivity mode, `--failures-only` filters the per-sample table to rows where `verdict == "unstable"`.

---

## Scorers

There are two scorer contracts:

**Model scorers** — evaluate model output. Runner calls the model first, then passes completion to the scorer:

| Name | CLI flag | Returns | Notes |
|------|----------|---------|-------|
| `exact_match` | `--scorer exact` | `0.0` or `1.0` | Strict equality after strip |
| `normalised_match` | `--scorer normalised` | `0.0` or `1.0` | Lowercase, strip punctuation, collapse whitespace |
| `RegexScorer` | `--scorer regex --pattern "..."` | `0.0` or `1.0` | Pattern found in completion |
| `MultiRegexScorer` | `--scorer multi-regex --pattern "a" --pattern "b"` | `0.0`–`1.0` | Fraction of patterns matched |
| `JSONSchemaScorer` | `--scorer schema --schema path/to/schema.json` | `0.0`, `0.5`, or `1.0` | See partial credit below |
| `LLMJudgeScorer` | `--scorer judge` | `0.0`–`1.0` or `None` | Rubric-based, uses judge model |
| `CascadeScorer` | `--scorer cascade` | `0.0`–`1.0` or `None` | Fast tier first, judge as fallback |
| `FaithfulnessScorer` | `--scorer faithfulness` | `0.0`–`1.0` or `None` | RAG: LLM judge grounding check against `context` chunks |

**Dataset scorers** — evaluate dataset quality. Runner skips the model call entirely (`completion=""`, `latency_ms=0`). The `latency_ms=0` records evaluated-model latency only — dataset scorers may make their own LLM calls internally, but that time is not attributed to the model under test. Signature drops `completion` structurally — it is not ignored, it is absent:

| Name | CLI flag | Returns | Notes |
|------|----------|---------|-------|
| `ContextSufficiencyScorer` | `--scorer context-sufficiency` | `0.0`, `1.0`, or `None` | RAG: LLM YES/NO check — does the context contain enough information to answer? Uses `JUDGE_MODEL`. Returns `None` on parse failure (unexpected response), `0.0` when context absent/empty. |

Run dataset scorers **before** model evals. If context sufficiency is low for a sample, remove it — a bad sample produces misleading model scores regardless of scorer quality.

**JSONSchemaScorer partial credit:**
- `1.0` — valid JSON that passes the schema
- `0.5` — valid JSON that fails the schema
- `0.0` — not valid JSON (after repair attempt)

**Score values:**
- `0.0`–`1.0` — normalised score (1.0 = correct / fully sufficient)
- `None` — scorer failure (API error or unparseable judge response) — recorded as an error in results, distinct from a genuine low score

In `sensitivity.py`, the scorer is tested for reliability — the same scorer runs across all variation types to measure whether it produces consistent scores.

### Scorer-specific flags

| Flag | Scorer | Default |
|------|--------|---------|
| `--pattern` | `regex`, `multi-regex` | required |
| `--schema` | `schema` | required |
| `--scale` | `judge`, `cascade`, `faithfulness` | `5` |
| `--judge-model` | `judge`, `cascade`, `faithfulness`, `context-sufficiency` | `JUDGE_MODEL` env |
| `--fast-tier` | `cascade` | `normalised` |
| `--cascade-threshold` | `cascade` (in `sensitivity.py`) | `1.0` |

---

## Results directory structure

```
results/
├── runs/{date}/{time}_{model}_{dataset}_{scorer}/
│   ├── run.json
│   └── samples.jsonl
├── benchmarks/{date}/{time}_{dataset}_{scorer}/
│   ├── benchmark.json
│   └── {model}.jsonl
├── sensitivity/{date}/{time}_{dataset}_{scorer}/      ← keyed by scorer
│   ├── sensitivity.json
│   ├── baseline.jsonl
│   ├── rephrase.jsonl
│   ├── discarded.jsonl                                ← samples filtered by validation
│   └── ...
├── robustness/{date}/{time}_{dataset}_{model}/        ← keyed by model
│   ├── robustness.json
│   ├── baseline.jsonl
│   ├── typos.jsonl
│   └── ...
└── judge_traces/{date}/{time}_{model}/
    └── {sample_id}.json
```

Sensitivity results are keyed by scorer (what's under test is the scorer's reliability). Robustness results are keyed by model (what's under test is the model's resilience to adversarial inputs).

### `sensitivity.json` fields

| Field | Contents |
|-------|---------|
| `dataset` | Source dataset name |
| `scorer` | Scorer used for reliability testing |
| `model` | Evaluated model |
| `timestamp` | ISO timestamp |
| `materialised_at` | When the file was written |
| `run_config` | `variation_model`, `judge_model`, `validation_threshold`, `cascade_threshold`, `limit`, `reused_from` |
| `variation_names` | List of variation types that were run (including `"baseline"`) |
| `schema_notes` | Internal schema versioning notes |
| `per_sample` | List of per-sample rows: `id`, one score per variation, `variance`, `verdict` |
| `per_variation` | List of per-variation rows: `variation`, `mean_score`, `delta_from_baseline`, `mean_variance` |
| `summary` | `n_unstable`, `n_total`, `most_destabilising`, `most_destabilising_n`, `baseline_n` |

### `robustness.json` fields

| Field | Contents |
|-------|---------|
| `dataset` | Source dataset name |
| `scorer` | Scorer held fixed during the test |
| `model` | Evaluated model |
| `timestamp` | ISO timestamp |
| `materialised_at` | When the file was written |
| `run_config` | `perturbation_model`, `cascade_threshold`, `limit`, `reused_from` |
| `perturbation_names` | List of perturbation types that were run (including `"baseline"`) |
| `schema_notes` | Notes on metric definitions (degradation vs delta) |
| `per_sample` | List of per-sample rows: `id`, one score per perturbation, `degradation`, `verdict` |
| `per_perturbation` | List of per-perturbation rows: `perturbation`, `mean_score`, `delta_from_baseline` |
| `summary` | `n_robust`, `n_fragile`, `n_brittle`, `n_na`, `n_total`, `most_degrading`, `most_degrading_n`, `baseline_n` |

---

## Output fields

### Summary line (printed after each run)

| Field | Meaning |
|-------|---------|
| `mean_score` | Mean of all non-None scores. `—` if every sample errored. |
| `p50_latency` | Median latency across all samples (ms) |
| `p95_latency` | 95th-percentile latency (ms). Flagged with `⚠` when n < 20 (low confidence) |
| `api_errors` | Count of samples where the model API call failed entirely (no completion) |
| `parse_failures` | Count of samples where the scorer returned `None` (completion exists but scorer couldn't score it) |
| `error_rate` | `(api_errors + parse_failures) / n` as a percentage |
| `clean_rate` | % of samples where JSON parsed without any repair needed. Only shown for `schema` scorer. |
| `fmt_pass_rate` | % of samples where JSON was valid (clean or repaired). Only shown for `schema` scorer. |
| `repair_fail_rate` | % of samples where JSON could not be parsed or repaired. Only shown for `schema` scorer. |
| `judge_rate` | % of samples that fell through to the judge tier. Only shown for `cascade` scorer. |

### `scorer_metadata` per sample (in `samples.jsonl`)

Set by scorers to record per-sample diagnostic state. Keys present depend on which scorer was used.

| Key | Set by | Values |
|-----|--------|--------|
| `format_status` | `JSONSchemaScorer` | `"clean"` — parsed without repair<br>`"repaired"` — truncated JSON was fixed automatically<br>`"repair_failed"` — could not parse or repair |
| `judge_format_status` | `LLMJudgeScorer` | Same values as above, for the judge's own response. Absent on API error. |
| `faithfulness_format_status` | `FaithfulnessScorer` | Same values as `judge_format_status`, for the faithfulness judge response. |
| `context_sufficiency_format_status` | `ContextSufficiencyScorer` | `"clean"` — LLM responded YES or NO<br>`"repair_failed"` — response started with neither (e.g. "Not enough information") |
| `tier_used` | `CascadeScorer` | `"fast"` — fast scorer met threshold, judge not called<br>`"judge"` — fast scorer missed, judge was called |
| `fast_score` | `CascadeScorer` | The raw score from the fast tier (`float` or `None`). Always present. |

---

## Dataset format

Each line in a JSONL dataset is:

```json
{"id": "001", "input": "...", "expected": "...", "metadata": {"category": "..."}}
```

- `input` — the prompt sent to the model
- `expected` — the target output (for deterministic scorers) or scoring rubric (for LLM judge)
- `metadata` — arbitrary key/value pairs passed through to `ScorerContext` and results

For LLM-judge datasets, `expected` is a rubric string describing what a good answer looks like, not a literal target answer.

For RAG datasets, samples include an additional `context` field:

```json
{"id": "rag-01", "input": "What is the capital of France?", "expected": "Paris", "context": ["France is a country in Western Europe. Its capital is Paris."], "metadata": {"category": "factual"}}
```

---

## Environment variables

| Variable | Used by | Default | Notes |
|----------|---------|---------|-------|
| `OLLAMA_HOST` | Runner, all scorers | `http://localhost:11434` | |
| `DEFAULT_MODEL` | `EvalConfig` | `llama3.2:3b` | Model being evaluated |
| `JUDGE_MODEL` | `LLMJudgeScorer`, `CascadeScorer`, `FaithfulnessScorer`, `ContextSufficiencyScorer` | `llama3.2:3b` | Scores completions |
| `VARIATION_MODEL` | `VariationGenerator` | falls back to `DEFAULT_MODEL` | Must differ from `JUDGE_MODEL` in sensitivity analysis |
| `PERTURBATION_MODEL` | `PerturbationGenerator` | falls back to `DEFAULT_MODEL` | Can equal evaluated model — no circularity concern |
| `MAX_TOKENS` | `EvalConfig` | `1024` | |
| `RESULTS_DIR` | `Reporter`, `SensitivityReporter`, `RobustnessReporter` | `results/` | |

---

## RAG eval

The dataset at `datasets/rag/rag_qa.jsonl` is a fixed-context eval — `context` fields are pre-populated with relevant chunks. This isolates faithfulness and answer quality from retrieval quality.

### Three dimensions, three scorers

| Dimension | Scorer | What it measures |
|-----------|--------|-----------------|
| Context quality | `context-sufficiency` | Can the expected answer be derived from the context? (dataset property) |
| Faithfulness | `faithfulness` | Is the model's answer grounded in the context? (model behaviour) |
| Answer quality | `normalised` | Does the model's answer match the expected answer? (correctness) |

```bash
# Step 1: validate dataset — which samples have insufficient context?
uv run python scripts/run_eval.py \
  --dataset datasets/rag/rag_qa.jsonl \
  --scorer context-sufficiency

# Step 2: faithfulness — is the model grounded in what the context says?
uv run python scripts/run_eval.py \
  --dataset datasets/rag/rag_qa.jsonl \
  --scorer faithfulness --model llama3.2:3b

# Step 3: accuracy — does the model produce the correct answer?
uv run python scripts/run_eval.py \
  --dataset datasets/rag/rag_qa.jsonl \
  --scorer normalised --model llama3.2:3b
```

### rag-006 — the adversarial case

Sample rag-006 ("What is the capital of Australia?") has "Canberra" deliberately absent from its context chunks (which mention Sydney and Melbourne). This creates a forced tension:

- A model that answers "Canberra" is **correct but hallucinating** — high accuracy, low faithfulness
- A model that hedges ("The context doesn't say") is **grounded but wrong** — high faithfulness, low accuracy

`context-sufficiency` will score rag-006 as `0.0` — the LLM correctly identifies that Canberra is not derivable from context about Sydney and Melbourne, even though all three are Australian cities. Use `show.py --id rag-006` to see the context chunks alongside the score — without the context visible, the `0.0` looks like a bug.

## Key things to watch for

1. **Your scorer will be wrong.** Models produce correct answers that scorers mark as wrong (whitespace, capitalisation, phrasing). Understand *why* before making the scorer smarter.

2. **LLM judges have biases.** Position bias, verbosity bias, self-preference. You'll see these clearly with the judge scorer.

3. **Benchmark rankings are task-dependent.** The model that wins on extraction loses on open-ended QA.

4. **Eval variance is real.** Small prompt changes can swing scores ±10%. Sensitivity analysis is dedicated to measuring this.

5. **Dataset quality dominates everything.** A great scorer on a bad dataset gives false confidence.

6. **`None` ≠ `0.0`.** A scorer returning `None` means it couldn't score the sample (API down, unparseable response). It's recorded separately from a genuine low score and excluded from mean calculation.

7. **Three-model separation matters in sensitivity analysis.** If `VARIATION_MODEL` equals `JUDGE_MODEL`, you're measuring self-consistency, not scorer reliability. The script enforces separation and exits on misconfiguration.

8. **Robustness and sensitivity measure different things.** Sensitivity holds the input fixed and varies the scorer's perspective; robustness holds the scorer fixed and varies the input adversarially. Sensitivity answers "can I trust my ruler?"; robustness answers "how fragile is the model to realistic input noise?". Run sensitivity first — if your scorer is noisy, robustness deltas are uninterpretable.

9. **`verbose` perturbation can improve scores.** If verbose inputs score *higher* than baseline under an LLM judge, that is verbosity bias in the judge, not genuine robustness. It is a scorer blind spot, not a good result.
