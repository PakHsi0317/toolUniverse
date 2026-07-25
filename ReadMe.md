# toolUniverse Assignment

This repository is my assignment project based on the ToolUniverse-style idea of
tool specification generation and optimization.

The main goal of this assignment is:

- start from a natural-language tool description
- generate a structured tool spec automatically
- test whether an LLM can call that tool correctly
- improve the spec through an optimization loop when the tool call fails

In short, this project asks:

> Can we automatically discover a tool spec, test it in multi-tool competition,
> find what field is causing failure, and rewrite only that part to improve
> invocation accuracy?

## What I built

I split the project into two main parts:

### 1. Discoverer

The Discoverer takes a plain-English tool description and produces:

- a `ToolSpec` JSON
- a Python stub for that tool

The pipeline is:

```text
tool description
-> retrieve a few similar seed templates
-> generate a ToolSpec with the LLM
-> generate a deterministic Python stub
-> validate the result
```

### 2. Optimizer

The Optimizer checks whether the LLM can use the generated tool spec correctly.
If the call is wrong, it:

- generates test prompts
- runs the tool-calling evaluation
- classifies the failure type
- diagnoses which field in the spec is most likely responsible
- rewrites only that field
- repeats for up to 3 iterations

So the full idea is:

```text
description
-> discovered spec
-> baseline test
-> diagnose failures
-> rewrite one field
-> test again
-> final optimized spec
```

## My idea / design thinking

My main thought process for this assignment was:

1. Keep generation and optimization separate.
   The Discoverer is responsible for creating a reasonable first draft, and the
   Optimizer is responsible for improving it.

2. Use deterministic evaluation instead of LLM-as-judge.
   I do not use the LLM to decide whether a tool call is correct. I compare the
   tool name, parameters, required fields, and types mechanically so the result
   is reproducible.

3. Test in multi-tool competition, not single-tool mode.
   The model sees competing tools, so tool selection is actually challenging.
   This better reflects how tool use works in real agent systems.

4. Rewrite only one field at a time.
   Instead of regenerating the entire spec, the optimizer changes only the
   blamed field. This makes the optimization process easier to analyze.

5. Add guardrails.
   I added a do-no-harm guard so a rewrite is rejected if it makes performance
   worse on the current prompts.

6. Explore failure cases beyond the main pipeline.
   I also added experiments for degraded specs, confusion pairs, low-quality
   descriptions, no-seed discovery, dimension-based evaluation, and structural
   `needs_redesign` detection.

## Project purpose

The purpose of this project is not to build real production tools. The purpose
is to study whether tool specs can be:

- generated automatically
- evaluated systematically
- improved automatically through feedback

This makes the project more of a research / assignment prototype than an app.

## How to run the code

### 1. Install dependencies

```bash
python3 -m pip install openai pydantic diskcache python-dotenv truststore pytest
```

### 2. Add API key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Main run order

If you want to run the core assignment pipeline, use these scripts in order:

```bash
python3 scripts/run_discoverer.py
python3 scripts/run_baseline.py
python3 scripts/run_optimization.py
```

What they do:

- `run_discoverer.py`
  Reads the natural-language tool descriptions and generates discovered specs
  and mock Python stubs.

- `run_baseline.py`
  Tests the discovered specs in multi-tool competition and records baseline
  invocation accuracy.

- `run_optimization.py`
  Runs the optimization loop and compares before/after accuracy on the same
  baseline prompts.

## Important output files

After running the pipeline, the main outputs are:

- `data/discovered_specs/`
  Generated tool specs

- `data/discovered_stubs/`
  Generated Python mock stubs

- `data/test_prompts/`
  Baseline prompts used for evaluation

- `data/optimized_specs/`
  Final optimized specs

- `data/logs/baseline.jsonl`
  Baseline failure records

- `data/logs/optimization.jsonl`
  Optimization-loop records

- `data/logs/final_eval.jsonl`
  Final after-optimization evaluation records

- `data/optimization_report.json`
  Before/after summary

## Extra experiments

Besides the main pipeline, I added several extra scripts to study the assignment
from different angles:

- `scripts/run_degradation.py`
  Inject controlled bugs into specs and test whether the optimizer can recover.

- `scripts/run_dimension_eval.py`
  Break correctness into five dimensions instead of using only one final label.

- `scripts/run_confusion_experiment.py`
  Add overlapping tools and test whether optimization helps with wrong-tool
  selection.

- `scripts/run_lowqual_experiment.py`
  Test what happens when the input descriptions are vague.

- `scripts/run_noseed_experiment.py`
  Test what happens when the Discoverer loses few-shot seed templates.

- `scripts/run_redesign_detection.py`
  Detect when the problem is structural and should be sent back for redesign
  instead of continuing field-level rewrites.

## Testing

For local tests without making a live API request:

```bash
python3 -m pytest tests/ -q -k 'not live_smoke'
```

Expected result in the current repo:

```text
99 passed, 1 deselected
```

## Current result summary

From the checked-in outputs in this repo:

- 11 out of 11 tool descriptions were successfully turned into valid specs
- baseline invocation accuracy is about `89.1%`
- after optimization it improves to about `90.9%`
- in the confusion-pair experiment, pair-member accuracy improves from `90.0%`
  to `95.0%`

These numbers are not the only point of the project, but they show that the
pipeline can improve some specs and also reveal when some failures are
structural rather than just local field mistakes.

## Code structure

```text
src/
  schema.py
  llm_client.py
  discoverer/
  optimizer/

scripts/
  run_discoverer.py
  run_baseline.py
  run_optimization.py
  ...extra experiment scripts

data/
  discovered_specs/
  discovered_stubs/
  optimized_specs/
  logs/
```

## Short summary

This assignment builds a full pipeline for:

- discovering tool specs from natural-language descriptions
- evaluating tool-use quality
- diagnosing why tool calls fail
- improving the spec through targeted rewrites

The project is meant to show both the implementation and the reasoning behind
automatic tool specification optimization.
