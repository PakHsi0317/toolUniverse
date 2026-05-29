"""TestPromptGenerator: produces natural-language prompts that should trigger the tool.

In later rounds (iteration > 0), it can be given the previous round's
failures as feedback so it generates harder, more targeted prompts
(this is the paper's 'adaptive test generation' principle).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.llm_client import LLMClient
from src.schema import FailureRecord, ToolSpec


class _PromptList(BaseModel):
    """Wrapper so LLM returns a structured list."""
    prompts: list[str] = Field(..., min_length=1, max_length=20)


_SYSTEM = """You generate realistic user tasks that should cause an AI agent to call \
a specific tool. Each task is a single sentence in natural language, written as if \
a real user (scientist, clinician, student) is asking the question."""


_USER_TEMPLATE = """Tool specification:
{spec_json}

Generate exactly {n} REALISTIC user tasks that should trigger this tool. These will \
be used to stress-test whether an LLM can correctly identify and invoke this tool — \
so they must be REALISTIC, not toy examples that spoon-feed the parameter values.

Hard constraints for prompt difficulty (follow ALL of them):
1. DO NOT echo parameter names from the spec. If a parameter is 'weight_kg', the user \
   should say something like '70 kilograms' or '70kg' or 'weighs 70', NOT 'weight_kg=70'.
2. Use INDIRECT references where realistic — e.g. cite a gene by its disease ("the gene \
   linked to Li-Fraumeni syndrome" → TP53), a unit needing conversion (cm → m, lbs → kg), \
   or a time range needing computation ("last 5 years" → year filter).
3. Vary the prompt style: some questions, some imperatives, some scenarios ("Dr. Smith \
   wants to know..."), some terse, some verbose.
4. AT MOST 2 of the {n} prompts should explicitly state ALL required parameter values. \
   The rest should leave at least one parameter implicit (defaults, derivable from context, \
   or requiring the LLM to make a sensible choice).
5. Use REAL domain entities: real gene symbols (BRCA1, TP53), real drugs (aspirin, statin \
   names), real PDB IDs (1HVR, 7A4N), real conditions (hypertension, Alzheimer's).

{feedback_block}

Return JSON: {{"prompts": ["task1", "task2", ...]}}. Each prompt is one sentence."""


_FEEDBACK_TEMPLATE = """Previous-round failure patterns (the prior tests REVEALED these problems; \
generate NEW tasks that probe these and related edge cases):
{failure_summary}"""


def _summarize_failures(failures: list[FailureRecord]) -> str:
    by_type: dict[str, int] = {}
    for f in failures:
        if f.failure_type == "correct":
            continue
        by_type[f.failure_type] = by_type.get(f.failure_type, 0) + 1
    if not by_type:
        return "(none — all previous prompts succeeded; explore harder edge cases)"
    return "; ".join(f"{t}={c}" for t, c in sorted(by_type.items()))


class TestPromptGenerator:
    __test__ = False  # Tell pytest this is not a test class (name collision)

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(
        self,
        spec: ToolSpec,
        n: int = 5,
        prior_failures: Optional[list[FailureRecord]] = None,
    ) -> list[str]:
        feedback_block = ""
        if prior_failures:
            feedback_block = _FEEDBACK_TEMPLATE.format(
                failure_summary=_summarize_failures(prior_failures)
            )
        prompt = _USER_TEMPLATE.format(
            spec_json=spec.model_dump_json(indent=2),
            n=n,
            feedback_block=feedback_block,
        )
        result = self.llm.complete_structured(
            prompt=prompt,
            schema=_PromptList,
            system=_SYSTEM,
            temperature=0.3,  # Some diversity in prompts
            max_tokens=1500,
        )
        return result.prompts[:n]
