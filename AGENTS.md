# AGENTS.md

Guidance for AI coding agents working in this repository. Human contributors:
the same rules apply, they are just written for the reader who has no memory of
last week.

This is a single-maintainer project built to understand how evals behave, not a
framework chasing feature coverage. That shapes what belongs here: depth on a
small surface beats breadth. Most "add another scorer / provider / metric" ideas
should be argued for in an issue before any code is written.

## Before opening a PR

1. **Start from a demonstrated problem.** A reproduction or a failing test, not
   a hunch. If the need is speculative, open an issue with the evidence and
   stop there.
2. **Check `docs/backlog/` first.** Several ideas are already written up with a
   decision attached. If your change is in there, follow what it says or make
   the case for changing it — do not silently re-litigate it.
3. **One concern per PR.** No bundled drive-by refactors, no reformatting a file
   you happened to open.
4. **Run the checks and report the results honestly:**
   ```
   uv run pytest -q
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy
   ```
   All four must pass. If you cannot make one pass, say so in the PR
   description rather than weakening the check.
5. **Disclose agent involvement** in the PR description: which model and tool
   wrote the change, and whether a separate review pass ran over it.

## Invariants — do not "fix" these

These look like bugs and are not. Changing them silently corrupts results.

- **A scorer returns `None` when it could not score, never `0.0`.** `None` means
  "no judgement" (unparseable judge response, model error). `0.0` means "judged,
  and wrong". Collapsing the two turns infrastructure failures into fake
  evidence that a model is bad — the single most dangerous failure mode in this
  codebase. Reporters must keep counting them separately.
- **Run artifacts are named by local date and time** (`results/YYYY-MM-DD/HHMMSS`,
  and `results/judge_traces/{date}/{time}_{model}/`). They are read by a human on
  this machine; UTC would put an evening run in tomorrow's folder. The `DTZ`
  ruff rules are off for this reason, not by oversight.
- **Sensitivity analysis exits when the variation generator and the judge are the
  same model.** That configuration measures a model's self-consistency, not
  scorer reliability. The guard is the point of the analysis.
- **Sensitivity is run before robustness.** A robustness delta smaller than the
  scorer's own variance is noise. Do not present one without the other.

## Conventions

- **Python 3.11+, `uv` for everything.** `uv sync --all-groups`, `uv run <cmd>`.
- **Scorers** live in `evals/scorers/`, take `(completion, expected, ctx)` and
  return `float | None`. Dataset-level scorers subclass `DatasetScorer` and take
  `(expected, ctx)`. Wire new ones into `evals/scorer_factory.py`.
- **`response.message.content` is `str | None`.** Always `or ""` it. Three
  scorers already do; match them.
- **Tests never touch the network.** The suite runs in under a second because
  every Ollama call is mocked. Keep it that way — a test that needs a live model
  is not a test, it is an eval.
- **Line length is not enforced** (`E501` is off). Scorer prompts read better
  unwrapped.
- **Typing is checked but not `--strict`.** New code should be annotated. Do not
  add `# type: ignore` without a comment saying why.

## Where things are

| Path | What |
|---|---|
| `evals/core.py` | `Sample`, `Dataset`, `RunResult`, `ScorerContext`, `EvalConfig`, scorer types |
| `evals/runner.py` | Calls the model, populates `RunResult` |
| `evals/scorers/` | The nine scorers |
| `evals/*_reporter.py` | Aggregation and tables for sensitivity / robustness |
| `scripts/` | CLI entry points |
| `docs/ARCHITECTURE.md` | How the pieces fit together — read before structural changes |
| `docs/FAILURE_MODES.md` | Known ways evals lie to you |
| `docs/OPEN_PROBLEMS.md` | Unsolved, and why |
| `docs/backlog/` | Written-up ideas with decisions attached |
