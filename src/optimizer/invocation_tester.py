"""InvocationTester: feeds (prompt, spec) to LLM, parses the function call,
mechanically compares against the target spec.

This is the SOURCE of every FailureRecord. The diagnostic LLM (in
FailureDiagnoser) takes these records and fills in blamed_field /
root_cause / suggested_rewrite.

Two complementary checkers (NEITHER calls an LLM):

1. classify() → a SINGLE label (FailureType). Short-circuits at the first
   problem found, top-down. This is the historical output consumed by
   FailureRecord.failure_type and all Phase-3 scripts. UNCHANGED.

       correct           — name matches AND all required params present AND types ok
       wrong_tool        — LLM called a different tool
       missing_required  — name matches but a required param is absent
       extra_argument    — name matches but LLM passed a key not in the spec
       wrong_param_name  — LLM used a param name close-but-not-equal to a real one
       wrong_type        — value type doesn't match declared type
       malformed_output  — LLM didn't return a tool call at all

2. diagnose_dimensions() → a 5-DIMENSION vector (CallDimensions). Evaluates
   every dimension INDEPENDENTLY (no short-circuit), so a call that is both
   wrong-tool AND wrong-type records BOTH — giving unbiased per-dimension
   statistics for "which error class did the Optimizer fix?". The single
   label and the vector agree by construction:
       classify(...) == "correct"  ⟺  diagnose_dimensions(...).all_correct
   (when no salient ground-truth values are supplied).
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
    # Optional 5-dimension breakdown (parallel to failure_type). None for
    # callers/paths that don't request it — keeps the historical shape intact.
    dimensions: Optional["CallDimensions"] = None

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
# Five-dimension diagnostic checker (parallel to classify(); no short-circuit)
# ---------------------------------------------------------------------------


@dataclass
class CallDimensions:
    """Independent pass/fail on each of 5 correctness dimensions.

    Unlike classify() (one label, short-circuits), every field here is
    evaluated on its own so co-occurring defects are both visible. Aggregate
    per-dimension pass rates across many calls to see WHICH error class the
    Optimizer fixed.

    values_ok is None when no salient ground-truth values were supplied for
    the test case (it is then excluded from all_correct, so the vector stays
    consistent with classify()).
    """
    tool_name: bool          # ① right tool selected
    required_present: bool    # ② all required params present
    no_hallucination: bool    # ③ no args outside the spec's parameter set
    types_ok: bool            # ④ every supplied value matches its declared type
    values_ok: Optional[bool] = None  # ⑤ salient prompt values landed in some arg

    @property
    def all_correct(self) -> bool:
        """True iff every APPLICABLE dimension passed (values_ok=None is skipped)."""
        core = (
            self.tool_name
            and self.required_present
            and self.no_hallucination
            and self.types_ok
        )
        if self.values_ok is None:
            return core
        return core and self.values_ok

    def failed_dimensions(self) -> list[str]:
        """Names of the dimensions that failed (for logging / aggregation)."""
        out: list[str] = []
        if not self.tool_name:
            out.append("tool_name")
        if not self.required_present:
            out.append("required_present")
        if not self.no_hallucination:
            out.append("no_hallucination")
        if not self.types_ok:
            out.append("types_ok")
        if self.values_ok is False:
            out.append("values_ok")
        return out

    def as_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "required_present": self.required_present,
            "no_hallucination": self.no_hallucination,
            "types_ok": self.types_ok,
            "values_ok": self.values_ok,
            "all_correct": self.all_correct,
        }


def _canon(v: object) -> str:
    """Normalize a scalar for tolerant equality: numbers compare numerically,
    strings compare case/space-insensitively. Avoids 0.80 != '0.8' false negatives.

    A numeric STRING like '-7.5' is normalized through the float branch too, so
    the prompt token '-7.5' matches the argument value -7.5."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    s = str(v).strip()
    try:  # numeric string → compare as a number
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return " ".join(s.lower().split())


def _flatten_canon(v: object) -> set[str]:
    """Canonical tokens contained in an argument value. Scalars yield one token;
    lists/tuples yield one per element (so a prompt value can match an element of
    an array argument); dicts yield one per value."""
    if isinstance(v, (list, tuple)):
        out: set[str] = set()
        for el in v:
            out |= _flatten_canon(el)
        return out
    if isinstance(v, dict):
        out = set()
        for el in v.values():
            out |= _flatten_canon(el)
        return out
    return {_canon(v)}


def diagnose_dimensions(
    target_spec: ToolSpec,
    actual_call: Optional[dict],
    salient_values: Optional[list] = None,
) -> CallDimensions:
    """Evaluate all 5 dimensions independently (no short-circuit).

    salient_values: literal values from the prompt that MUST land in some
    argument (e.g. a PMID '12345678'). Omit/empty → values_ok stays None and
    is excluded from all_correct, keeping parity with classify().
    """
    # A missing tool call fails every structural dimension at once.
    if actual_call is None:
        vok = None if not salient_values else False
        return CallDimensions(
            tool_name=False, required_present=False,
            no_hallucination=False, types_ok=False, values_ok=vok,
        )

    args = actual_call.get("arguments", {}) or {}
    spec_param_names = {p.name for p in target_spec.parameters}
    spec_param_by_name = {p.name: p for p in target_spec.parameters}

    # ① right tool
    tool_name_ok = actual_call.get("name") == target_spec.name

    # ② all required present
    required_present = all(
        p.name in args for p in target_spec.parameters if p.required
    )

    # ③ no hallucinated args (every key is a real spec param)
    no_hallucination = set(args).issubset(spec_param_names)

    # ④ types: only check keys that ARE real params (hallucinated keys covered by ③)
    types_ok = True
    for key, val in args.items():
        p = spec_param_by_name.get(key)
        if p is None:
            continue
        check = _TYPE_CHECKS.get(p.type)
        if check and not check(val):
            types_ok = False
            break

    # ⑤ salient values landed in SOME argument (semantic correctness).
    # Flatten array/dict argument values so a prompt token can match an element
    # of an array argument (e.g. '-7.5' matches binding_scores=[-7.5, -8.2]).
    values_ok: Optional[bool] = None
    if salient_values:
        arg_canons: set[str] = set()
        for v in args.values():
            arg_canons |= _flatten_canon(v)
        values_ok = all(_canon(sv) in arg_canons for sv in salient_values)

    return CallDimensions(
        tool_name=tool_name_ok,
        required_present=required_present,
        no_hallucination=no_hallucination,
        types_ok=types_ok,
        values_ok=values_ok,
    )


# ---------------------------------------------------------------------------


class InvocationTester:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def test_one(
        self,
        target_spec: ToolSpec,
        prompt: str,
        competing_specs: Optional[list[ToolSpec]] = None,
        salient_values: Optional[list] = None,
    ) -> InvocationResult:
        """Run one (prompt, spec) test → InvocationResult.

        If competing_specs is provided, LLM sees ALL of them (target + competitors)
        and must select the right one — used for adversarial-pair experiments.

        salient_values: optional literal ground-truth values from the prompt;
        when given, the 5-dimension breakdown also checks dimension ⑤ values_ok.
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
            return InvocationResult(
                prompt=prompt, actual_call=None,
                failure_type="malformed_output",
                dimensions=diagnose_dimensions(target_spec, None, salient_values),
            )

        # Use the first tool call (one-shot setting)
        tc = resp.tool_calls[0]
        actual = {"name": tc.name, "arguments": tc.arguments}
        ft = classify(target_spec, actual)
        dims = diagnose_dimensions(target_spec, actual, salient_values)
        return InvocationResult(prompt=prompt, actual_call=actual,
                                failure_type=ft, dimensions=dims)

    def test_batch(
        self,
        target_spec: ToolSpec,
        prompts: list[str],
        competing_specs: Optional[list[ToolSpec]] = None,
    ) -> list[InvocationResult]:
        return [self.test_one(target_spec, p, competing_specs) for p in prompts]
