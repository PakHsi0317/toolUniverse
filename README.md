# SpecForge

SpecForge is a research prototype for one focused question:

> If an LLM is given a tool specification that is slightly wrong, can we
> automatically diagnose which field is causing bad tool calls and rewrite only
> that field until invocation accuracy improves?

The repo has two cooperating parts:

- `Discoverer`: turns a plain-English tool description into a structured `ToolSpec` plus a Python stub.
- `Optimizer`: stress-tests that spec in multi-tool competition, classifies failures mechanically, diagnoses the likely bad field, and applies a single-field rewrite.

The implementation is inspired by the ToolUniverse framing, but the code here is a self-contained experimental pipeline rather than a production agent runtime.

## At a glance

```text
Natural-language description
    -> Discoverer
       -> ToolSpec JSON
       -> Python stub
    -> Baseline evaluation
       -> multi-tool invocation accuracy
    -> Optimizer loop
       -> test -> classify -> diagnose -> rewrite
       -> improved ToolSpec (or needs_redesign)
```

What is already in the repo:

- Source code for the Discoverer and Optimizer
- Checked-in generated artifacts under `data/`
- Multiple experiment scripts for degradation, confusion, low-quality inputs, and redesign detection
- Unit tests that run locally with a mock LLM

What this repo is not:

- A real tool execution framework
- A package with polished install tooling
- A benchmark that depends on hidden data; most outputs are already committed for inspection

## Core idea

The central object is a `ToolSpec` with:

- `name`
- `description`
- `parameters`
- `return_schema`

The Optimizer does not ask an LLM to judge correctness. Instead, it uses a deterministic checker to score the returned tool call against the target spec:

- correct tool name
- required arguments present
- no hallucinated arguments
- argument types match
- in the dimension-eval experiment, salient values are preserved

That means the headline numbers in this repo come from explicit comparison logic, not LLM-as-a-judge scoring.

## Quick start

### 1. Install dependencies

There is no `requirements.txt` yet, so install the small dependency set directly:

```bash
python3 -m pip install openai pydantic diskcache python-dotenv truststore pytest
```

### 2. Add environment variables

Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Notes:

- All LLM-backed scripts use OpenAI through `src/llm_client.py`.
- All LLM calls are cached to `.llm_cache/`, so rerunning the same experiment is much cheaper.

### 3. Verify the local test suite

For a fully local run with no network call:

```bash
python3 -m pytest tests/ -q -k 'not live_smoke'
```

Current expected result in this repo:

```text
99 passed, 1 deselected
```

Why the filter matters:

- `tests/test_llm_client.py` contains one real OpenAI smoke test.
- If `OPENAI_API_KEY` is present, plain `pytest` may try that live request.
- The `-k 'not live_smoke'` command is the reliable "local-only" check.

## Main workflow

If you want the core pipeline, run these three scripts in order:

```bash
python3 scripts/run_discoverer.py
python3 scripts/run_baseline.py
python3 scripts/run_optimization.py
```

What each step does:

| Step | Script | Purpose | Main outputs |
|---|---|---|---|
| 0 | `scripts/run_discoverer.py` | Generate `ToolSpec` JSON and Python stubs from natural-language descriptions | `data/discovered_specs/`, `data/discovered_stubs/`, `data/discovery_report.json` |
| 1 | `scripts/run_baseline.py` | Measure invocation accuracy of the discovered specs under multi-tool competition | `data/test_prompts/`, `data/logs/baseline.jsonl` |
| 2 | `scripts/run_optimization.py` | Run the optimizer loop and compare original vs optimized specs on the same baseline prompts | `data/optimized_specs/`, `data/logs/optimization.jsonl`, `data/logs/final_eval.jsonl`, `data/optimization_report.json` |

The repo already contains generated artifacts under `data/`, so you can inspect results immediately even before rerunning anything.

## Results snapshot

The checked-in artifacts currently show:

| Experiment | Current checked-in result |
|---|---|
| Discovery validity | `11/11` generated specs validated |
| Baseline invocation accuracy | `89.1%` (`49/55`) |
| After optimization | `90.9%` (`+1.8pp`) |
| Five-dimension eval | `51/55` calls correct on all five dimensions |
| `values_ok` dimension | `42/42` applicable cases passed |
| Confusion-pair experiment | `90.0% -> 95.0%` on 8 pair-member tools |
| Low-quality description experiment | `64.0% -> 64.0%` |
| No-seed Discoverer experiment | `70.9% -> 76.4%`, with `5/11` tools flagged `needs_redesign` |
| Redesign detection | `3/5` structural defects flagged, `0/1` healthy controls false-flagged |

Those numbers come from these files:

- `data/discovery_report.json`
- `data/optimization_report.json`
- `data/dimension_eval.json`
- `data/confusion_report.json`
- `data/lowqual/lowqual_report.json`
- `data/noseed/noseed_report.json`
- `data/redesign_detection.json`

## Additional experiments

The script names reflect the history of the project, so the numbering is not perfectly linear. The most useful follow-on scripts are:

| Script | Question | Main outputs |
|---|---|---|
| `scripts/run_phase3a.py` | What failure patterns already exist in the logs? | `data/phase3a_analysis.json` |
| `scripts/run_degradation.py` | If we inject controlled spec bugs, can the Optimizer recover? | `data/degradation_results.json`, `data/logs/degradation.jsonl` |
| `scripts/run_dimension_eval.py` | How do tool calls break down across five independent correctness dimensions? | `data/dimension_eval.json`, `data/logs/dimension_eval.jsonl`, `data/test_prompts_gt/` |
| `scripts/run_confusion_experiment.py` | Can optimization recover accuracy when semantically overlapping tools compete? | `data/confusion_report.json`, `data/confusion_optimized/`, `data/logs/confusion.jsonl` |
| `scripts/run_phase3c.py` | Which adversarial tool pairs are auto-discovered from baseline confusion? | `data/phase3c_results.json`, `data/logs/phase3c.jsonl` |
| `scripts/run_lowqual_experiment.py` | What happens when the Discoverer is given vague descriptions? | `data/lowqual/` |
| `scripts/run_noseed_experiment.py` | What happens when the Discoverer loses its few-shot seed templates? | `data/noseed/` |
| `scripts/run_redesign_detection.py` | Can the system tell when a defect is structural and should be sent back for redesign? | `data/redesign_detection.json` |

Helpful inspection/export utilities:

- `scripts/inspect_failures.py data/logs/baseline.jsonl`
- `scripts/export_degraded_specs.py`
- `scripts/export_one_example.py`

## Project map

```text
src/
  schema.py                 core data model: ToolSpec, FailureRecord, helpers
  llm_client.py             cached OpenAI client + mock client
  discoverer/
    discoverer.py           top-level discovery pipeline
    pattern_retriever.py    picks few-shot seeds
    spec_generator.py       LLM: description -> ToolSpec
    stub_generator.py       ToolSpec -> Python stub
    static_validator.py     AST/rule validation for generated outputs
  optimizer/
    test_prompt_generator.py
    invocation_tester.py
    failure_diagnoser.py
    spec_rewriter.py
    loop.py                 iterative optimization controller

scripts/
  run_discoverer.py
  run_baseline.py
  run_optimization.py
  ... experiment-specific scripts

data/
  tool_descriptions.json    starting descriptions
  seed_templates/           few-shot examples for discovery
  discovered_specs/         generated specs
  discovered_stubs/         generated Python stubs
  test_prompts/             baseline prompts
  optimized_specs/          optimized specs
  logs/                     jsonl records for experiments
  *_report.json             summary outputs

data_v1_backup/
  archived earlier version of the project outputs
```

## Files worth reading first

If you want to understand the repo quickly, start here:

1. `src/schema.py`
2. `src/discoverer/discoverer.py`
3. `src/optimizer/loop.py`
4. `scripts/run_discoverer.py`
5. `scripts/run_optimization.py`

That path gives you the data model, the generation path, and the optimization path in the smallest number of files.

## Design choices

- Deterministic evaluation: correctness is computed by explicit comparison logic, not by an LLM judge.
- Single-field rewrites: each optimizer step edits one blamed field rather than regenerating the whole spec.
- Multi-tool competition: each test presents competing tools so tool selection is genuinely hard.
- Do-no-harm guard: a candidate rewrite is rejected if it lowers current-round accuracy.
- Redesign detection: when field-level edits cannot lift held-out performance, the system can return `needs_redesign` instead of pretending another rewrite will help.

## Important caveats

- Generated Python stubs are placeholders, not fully implemented external tools.
- Most scripts require a working OpenAI API key and network access.
- The repo is experimental and some utility scripts still reflect earlier tool names or earlier phases.
- There is no packaging or CLI layer yet; the `scripts/` directory is the entrypoint.

## One-line summary

SpecForge is a compact experimental repo for generating tool specs, stress-testing them under multi-tool competition, and iteratively repairing the exact spec fields that cause invocation failures.
