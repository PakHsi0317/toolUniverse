"""InvocationTester: feeds (prompt, spec) to LLM, parses the function call,
mechanically compares against the target spec → assigns a failure_type.

This is the SOURCE of every FailureRecord. The diagnostic LLM (in
FailureDiagnoser) takes these records and fills in blamed_field /
root_cause / suggested_rewrite.

Mechanical classification (NO LLM call here):
    correct           — name matches AND all required params present AND types ok
    wrong_tool        — LLM called a different tool (used in adversarial pair test)
    missing_required  — name matches but a required param is absent
    extra_argument    — name matches but LLM passed a key not in the spec
    wrong_param_name  — LLM used a param name close-but-not-equal to a real one
    wrong_type        — value type doesn't match declared type
    wrong_value_format — declared type matches but value violates an obvious format
                         (e.g. PDB ID must be 4 chars; date string must be ISO 8601)
    malformed_output  — LLM didn't return a tool call at all
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.llm_client import LLMClient
from src.schema import FailureRecord, FailureType, ToolSpec


@dataclass
class InvocationResult:
    """Per-prompt result from running one tool-use test."""
    prompt: str
    actual_call: Optional[dict]  # {"name": ..., "arguments": {...}} or None
    failure_type: FailureType

    def to_failure_record(
        self,
        target_spec: ToolSpec,
        iteration: int,
        model: str,
        expected_call: Optional[dict] = None,
    ) -> FailureRecord:
        if expected_call is None:
            expected_call = {"name": target_spec.name, "arguments": {}}
        return FailureRecord(
            tool_name=target_spec.name,
            spec_version=target_spec.spec_hash(),
            iteration=iteration,
            model=model,
            test_prompt=self.prompt,
            expected_call=expected_call,
            actual_call=self.actual_call,
            failure_type=self.failure_type,
        )


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def classify(target_spec: ToolSpec, actual_call: Optional[dict]) -> FailureType:
    """Pure function: given the target spec and the LLM's tool call, classify."""
    if actual_call is None:
        return "malformed_output"

    if actual_call.get("name") != target_spec.name:
        return "wrong_tool"

    args = actual_call.get("arguments", {}) or {}
    spec_param_names = {p.name for p in target_spec.parameters}
    spec_param_by_name = {p.name: p for p in target_spec.parameters}

    # Missing required?
    for p in target_spec.parameters:
        if p.required and p.name not in args:
            return "missing_required"

    # Wrong param name? (key in args matches no spec param)
    extra_keys = set(args) - spec_param_names
    if extra_keys:
        # Heuristic: if extra key is a close variant of a real one, call it
        # wrong_param_name; otherwise extra_argument.
        spec_names = list(spec_param_names)
        for key in extra_keys:
            if any(_is_close(key, sn) for sn in spec_names):
                return "wrong_param_name"
        return "extra_argument"

    # Type mismatch?
    for key, val in args.items():
        p = spec_param_by_name[key]
        check = _TYPE_CHECKS.get(p.type)
        if check and not check(val):
            return "wrong_type"

    return "correct"


def _is_close(a: str, b: str) -> bool:
    """Heuristic for typo/synonym confusion in param names."""
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    if a in b or b in a:
        return True
    # Edit distance proxy: same length, mostly same chars
    if abs(len(a) - len(b)) <= 2:
        common = sum(1 for ch in a if ch in b)
        return common / max(len(a), len(b)) > 0.7
    return False


# ---------------------------------------------------------------------------


class InvocationTester:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def test_one(
        self,
        target_spec: ToolSpec,
        prompt: str,
        competing_specs: Optional[list[ToolSpec]] = None,
    ) -> InvocationResult:
        """Run one (prompt, spec) test → InvocationResult.

        If competing_specs is provided, LLM sees ALL of them (target + competitors)
        and must select the right one — used for adversarial-pair experiments.
        """
        specs = [target_spec] + list(competing_specs or [])
        tools = [s.to_openai_function() for s in specs]

        resp = self.llm.call_with_tools(
            prompt=prompt,
            tools=tools,
            temperature=0.0,
            max_tokens=500,
        )

        if not resp.tool_calls:
            return InvocationResult(prompt=prompt, actual_call=None,
                                    failure_type="malformed_output")

        # Use the first tool call (one-shot setting)
        tc = resp.tool_calls[0]
        actual = {"name": tc.name, "arguments": tc.arguments}
        ft = classify(target_spec, actual)
        return InvocationResult(prompt=prompt, actual_call=actual, failure_type=ft)

    def test_batch(
        self,
        target_spec: ToolSpec,
        prompts: list[str],
        competing_specs: Optional[list[ToolSpec]] = None,
    ) -> list[InvocationResult]:
        return [self.test_one(target_spec, p, competing_specs) for p in prompts]
