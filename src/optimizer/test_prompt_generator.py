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


class PromptWithGroundTruth(BaseModel):
    """A test prompt paired with the literal values that MUST land in some
    argument of a correct call (used to evaluate dimension ⑤ values_ok)."""
    prompt: str = Field(..., description="One realistic user task, single sentence.")
    salient_values: list[str] = Field(
        default_factory=list,
        description=(
            "Literal values that appear VERBATIM in the prompt and must be passed "
            "as an argument value for the call to be semantically correct — e.g. a "
            "PMID '12345678', a PDB id '1HVR', a SMILES string. Leave EMPTY if the "
            "prompt only refers to entities indirectly (e.g. 'caffeine') with no "
            "literal value the model could copy verbatim."
        ),
    )


class _PromptWithGTList(BaseModel):
    """Wrapper so LLM returns a structured list of prompt+ground-truth pairs."""
    items: list[PromptWithGroundTruth] = Field(..., min_length=1, max_length=20)


_SALIENT_INSTRUCTIONS = """
For EACH task you generate, also extract its `salient_values`: the list of literal \
values that appear VERBATIM in your task sentence AND must be passed as an argument \
value for the call to count as semantically correct.

Rules for salient_values (critical — follow exactly):
- Include a value ONLY if it appears character-for-character in the prompt and a \
  correct call must copy it into an argument (e.g. a PMID '12345678', a PDB id '1HVR', \
  a raw SMILES string 'CC(=O)O', an explicit threshold '0.85').
- DO NOT include entities the model must translate/look up (e.g. a gene name that maps \
  to an ID, 'caffeine' that maps to a SMILES) — those are NOT verbatim-copyable. \
  Leave salient_values EMPTY for such prompts.
- It is correct and expected for many prompts to have an EMPTY salient_values list."""


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

    def generate_with_salient(
        self,
        spec: ToolSpec,
        n: int = 5,
        prior_failures: Optional[list[FailureRecord]] = None,
    ) -> list[PromptWithGroundTruth]:
        """Like generate(), but each prompt is paired with salient_values —
        the verbatim literal values a correct call must copy into an argument.
        Enables dimension ⑤ (values_ok) in diagnose_dimensions().

        A post-filter drops any salient value that does NOT actually appear in
        its prompt string, so the ground truth can never be hallucinated.
        """
        feedback_block = ""
        if prior_failures:
            feedback_block = _FEEDBACK_TEMPLATE.format(
                failure_summary=_summarize_failures(prior_failures)
            )
        prompt = _USER_TEMPLATE.format(
            spec_json=spec.model_dump_json(indent=2),
            n=n,
            feedback_block=feedback_block,
        ) + _SALIENT_INSTRUCTIONS
        result = self.llm.complete_structured(
            prompt=prompt,
            schema=_PromptWithGTList,
            system=_SYSTEM,
            temperature=0.3,
            max_tokens=2000,
        )
        items = result.items[:n]
        # Safety net: keep only salient values that literally occur in the prompt
        # (case-insensitive). Guards against the model inventing ground truth.
        for it in items:
            low = it.prompt.lower()
            it.salient_values = [
                v for v in it.salient_values if str(v).strip().lower() in low
            ]
        return items
