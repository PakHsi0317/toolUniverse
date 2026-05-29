"""FailureDiagnoser: LLM reasons over (spec, prompt, actual_call) to identify
WHICH FIELD of the spec caused the failure, WHY, and HOW to fix it.

This is the lynchpin component. Every Phase 3 analysis (confusion matrix,
field ablation, adversarial pair) is downstream of these structured
diagnoses. If diagnosis output is vague, all downstream analyses are vague.

Output schema (forced by Pydantic):
    blamed_field:       path like "parameters[2].description" or "description"
                        or "(unfixable)" if the failure is not a spec issue
    root_cause:         human-readable explanation in terms of the named field
    suggested_rewrite:  {field, new_value} ready to apply with SpecRewriter
                        (None if blamed_field == '(unfixable)')
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.llm_client import LLMClient
from src.schema import FailureRecord, SuggestedRewrite, ToolSpec


class DiagnosisResult(BaseModel):
    """Structured diagnosis. Used to enrich a FailureRecord."""

    blamed_field: str = Field(
        ...,
        description=(
            "Path to the spec field at fault, e.g. 'description', "
            "'parameters[2].description', 'parameters[0].type', "
            "or the literal string '(unfixable)' if the failure is not a spec issue."
        ),
    )
    root_cause: str = Field(
        ...,
        description="Concise explanation citing the named field. 1-3 sentences.",
    )
    suggested_rewrite: Optional[SuggestedRewrite] = Field(
        None,
        description=(
            "How to rewrite the blamed field. Omit (null) if blamed_field is '(unfixable)'."
        ),
    )


_SYSTEM = """You diagnose tool-specification bugs that caused an LLM to invoke a tool incorrectly. \
You think like a careful PR reviewer: cite the EXACT field at fault, explain WHY it caused the \
observed behavior, and propose a MINIMAL rewrite that addresses THIS SPECIFIC failure.

Critical rules:
- Blame the SMALLEST possible field. Prefer 'parameters[2].description' over 'description'.
- Path syntax: 'name', 'description', 'parameters[N].name', 'parameters[N].type', \
'parameters[N].description', 'parameters[N].required', 'return_schema.description'.

WHEN TO RETURN '(unfixable)' (very important — be honest):
1. The user prompt fundamentally lacks data the tool requires (e.g. asks for a t-test but provides \
   no numeric arrays). No spec wording can make the LLM compute without inputs.
2. The LLM did the RIGHT thing by chaining tools (e.g. for "convert DNA from gene BRCA1", the LLM \
   correctly calls get_gene_info first to retrieve sequence, then would convert). This is correct \
   multi-step planning, not a spec bug.
3. The failure is intermittent / depends on prompt phrasing that isn't tied to any spec field.
4. The LLM correctly identified ambiguity and declined to act.

In any of these cases, set blamed_field='(unfixable)' and suggested_rewrite=null.

WHEN PROPOSING A REWRITE:
- The new_value MUST address the SPECIFIC behavior observed. If the LLM chained tools, the rewrite \
  should explicitly instruct: 'If the user refers to a gene by name rather than providing an \
  explicit sequence, ask for the sequence rather than calling this tool.' Generic 'be more clear' \
  rewrites are useless.
- A rewrite is only valid if a reasonable LLM, reading the NEW spec, would behave differently on \
  THIS SAME PROMPT. If you can't articulate that, return (unfixable).
- The new_value type must match the original field (string for descriptions, etc.)."""


_USER_TEMPLATE = """A tool specification was given to an LLM, which produced a wrong tool call.

SPEC (the tool that SHOULD have been invoked correctly):
{spec_json}

TEST PROMPT (what the user asked):
{prompt}

EXPECTED CALL (what a correct invocation looks like):
{expected_json}

ACTUAL CALL (what the LLM did):
{actual_json}

MECHANICAL CLASSIFICATION: {failure_type}

Diagnose the failure. Which field of the SPEC is most responsible? Why does that wording \
produce this LLM behavior on this prompt? What concrete rewrite fixes it?

If this failure is NOT a spec issue (e.g. the user prompt is missing data the tool needs, \
or the LLM is correctly chaining tools), say so with blamed_field='(unfixable)'."""


def _format_call(call) -> str:
    if call is None:
        return "(no tool call produced — LLM returned text only or declined to call any tool)"
    return json.dumps(call, indent=2, ensure_ascii=False)


class FailureDiagnoser:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def diagnose(
        self,
        spec: ToolSpec,
        failure_record: FailureRecord,
    ) -> DiagnosisResult:
        """Diagnose a single failure. Returns structured diagnosis."""
        prompt = _USER_TEMPLATE.format(
            spec_json=spec.model_dump_json(indent=2),
            prompt=failure_record.test_prompt,
            expected_json=_format_call(failure_record.expected_call),
            actual_json=_format_call(failure_record.actual_call),
            failure_type=failure_record.failure_type,
        )
        diagnosis = self.llm.complete_structured(
            prompt=prompt,
            schema=DiagnosisResult,
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=1000,
        )
        # Enforce: if unfixable, suggested_rewrite must be None
        if diagnosis.blamed_field == "(unfixable)":
            diagnosis.suggested_rewrite = None
        return diagnosis

    def enrich(self, spec: ToolSpec, record: FailureRecord) -> FailureRecord:
        """Diagnose and write fields back into the FailureRecord."""
        if record.failure_type == "correct":
            return record  # nothing to diagnose
        d = self.diagnose(spec, record)
        record.blamed_field = d.blamed_field
        record.root_cause = d.root_cause
        record.suggested_rewrite = d.suggested_rewrite
        return record
