"""Tests for FailureDiagnoser, SpecRewriter, and OptimizerLoop (mock LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm_client import LLMResponse, MockLLMClient, ToolCall
from src.optimizer import (
    DiagnosisResult,
    FailureDiagnoser,
    OptimizerLoop,
    SpecRewriter,
)
from src.schema import (
    FailureRecord,
    Parameter,
    ReturnSchema,
    SuggestedRewrite,
    ToolSpec,
)


@pytest.fixture
def bmi_spec() -> ToolSpec:
    return ToolSpec(
        name="compute_bmi",
        description="Compute body mass index from weight and height.",
        parameters=[
            Parameter(name="weight_kg", type="number",
                      description="weight in kg", required=True),
            Parameter(name="height_m", type="number",
                      description="height in m", required=True),
        ],
        return_schema=ReturnSchema(type="object", properties={}, description="BMI."),
    )


# ---------------------------------------------------------------------------
# SpecRewriter
# ---------------------------------------------------------------------------

class TestSpecRewriter:
    def test_rewrite_top_level(self, bmi_spec):
        rw = SpecRewriter()
        new_spec = rw.apply(
            bmi_spec,
            SuggestedRewrite(field="description", new_value="Better description."),
        )
        assert new_spec.description == "Better description."
        assert bmi_spec.description != "Better description."  # immutable

    def test_rewrite_param_field(self, bmi_spec):
        rw = SpecRewriter()
        new_spec = rw.apply(
            bmi_spec,
            SuggestedRewrite(
                field="parameters[0].description",
                new_value="Body weight in kilograms (positive float).",
            ),
        )
        assert "positive float" in new_spec.parameters[0].description


# ---------------------------------------------------------------------------
# FailureDiagnoser
# ---------------------------------------------------------------------------

class TestFailureDiagnoser:
    def test_diagnoses_failure(self, bmi_spec, tmp_path):
        mock = MockLLMClient(cache_dir=tmp_path)
        mock.register(
            "MECHANICAL CLASSIFICATION",
            lambda p, ctx: LLMResponse(text=json.dumps({
                "blamed_field": "parameters[0].description",
                "root_cause": "Description is too short; LLM couldn't infer the expected unit.",
                "suggested_rewrite": {
                    "field": "parameters[0].description",
                    "new_value": "Body weight in kilograms (positive float)."
                }
            })),
        )
        d = FailureDiagnoser(mock)
        rec = FailureRecord(
            tool_name="compute_bmi", spec_version="abc", iteration=0, model="mock",
            test_prompt="bmi for 70 and 1.75", expected_call={}, actual_call={},
            failure_type="wrong_type",
        )
        result = d.diagnose(bmi_spec, rec)
        assert result.blamed_field == "parameters[0].description"
        assert result.suggested_rewrite is not None

    def test_unfixable_clears_rewrite(self, bmi_spec, tmp_path):
        mock = MockLLMClient(cache_dir=tmp_path)
        mock.register(
            "MECHANICAL CLASSIFICATION",
            lambda p, ctx: LLMResponse(text=json.dumps({
                "blamed_field": "(unfixable)",
                "root_cause": "User didn't supply data.",
                "suggested_rewrite": {
                    "field": "description", "new_value": "x",  # diagnoser may forget; we enforce
                },
            })),
        )
        d = FailureDiagnoser(mock)
        rec = FailureRecord(
            tool_name="compute_bmi", spec_version="abc", iteration=0, model="mock",
            test_prompt="x", expected_call={}, failure_type="malformed_output",
        )
        result = d.diagnose(bmi_spec, rec)
        assert result.blamed_field == "(unfixable)"
        assert result.suggested_rewrite is None  # enforced regardless of LLM output

    def test_enrich_skips_correct(self, bmi_spec, tmp_path):
        mock = MockLLMClient(cache_dir=tmp_path)
        d = FailureDiagnoser(mock)
        rec = FailureRecord(
            tool_name="compute_bmi", spec_version="abc", iteration=0, model="mock",
            test_prompt="x", expected_call={}, failure_type="correct",
        )
        enriched = d.enrich(bmi_spec, rec)
        assert enriched.blamed_field is None  # not touched
        assert mock.call_log == []  # no LLM call made


# ---------------------------------------------------------------------------
# OptimizerLoop — end-to-end with mock
# ---------------------------------------------------------------------------

class TestOptimizerLoop:
    def test_target_reached_in_one_iteration(self, bmi_spec, tmp_path):
        mock = MockLLMClient(cache_dir=tmp_path)
        # All prompts result in correct calls
        mock.register("Tool specification:", lambda p, ctx: LLMResponse(
            text='{"prompts": ["a", "b", "c", "d", "e"]}'
        ))
        mock.register("", lambda p, ctx: LLMResponse(tool_calls=[
            ToolCall(name="compute_bmi", arguments={"weight_kg": 70.0, "height_m": 1.75})
        ]))

        loop = OptimizerLoop(llm=mock, log_path=tmp_path / "log.jsonl",
                             n_prompts=5, max_iterations=3)
        result = loop.optimize(bmi_spec)
        assert result.final_accuracy == 1.0
        assert len(result.iterations) == 1
        assert "target_reached" in result.terminated_reason

    def test_loop_continues_on_failures(self, bmi_spec, tmp_path):
        """Failures should trigger diagnose + rewrite + next iteration."""
        mock = MockLLMClient(cache_dir=tmp_path)
        # Always return prompts when generating
        mock.register("Tool specification:", lambda p, ctx: LLMResponse(
            text='{"prompts": ["test1", "test2", "test3"]}'
        ))
        # Always return diagnoses when diagnosing
        mock.register("MECHANICAL CLASSIFICATION", lambda p, ctx: LLMResponse(text=json.dumps({
            "blamed_field": "parameters[0].description",
            "root_cause": "x",
            "suggested_rewrite": {"field": "parameters[0].description", "new_value": "Better."}
        })))
        # Always return wrong tool call (missing weight_kg)
        mock.register("", lambda p, ctx: LLMResponse(tool_calls=[
            ToolCall(name="compute_bmi", arguments={"height_m": 1.75})  # missing required
        ]))

        # detect_redesign=False to test the raw max_iterations exit in isolation
        # (with it on, a persistently-stuck spec is reclassified needs_redesign —
        # covered by TestNeedsRedesign below).
        loop = OptimizerLoop(llm=mock, log_path=tmp_path / "log.jsonl",
                             n_prompts=3, max_iterations=3, detect_redesign=False)
        result = loop.optimize(bmi_spec)
        # Did not reach target, ran all 3 iterations
        assert len(result.iterations) == 3
        assert "max_iterations" in result.terminated_reason
        # Spec was rewritten each round (description changed)
        assert result.final_spec.parameters[0].description != result.initial_spec.parameters[0].description


# ---------------------------------------------------------------------------
# Do-no-harm guard
# ---------------------------------------------------------------------------

class TestNoHarmGuard:
    """A rewrite that LOWERS accuracy must be rejected; the loop keeps the
    last known-good spec and stops with a no_harm_guard reason."""

    @staticmethod
    def _target_desc(ctx: dict) -> str:
        """The target spec is always the FIRST tool presented; return its description."""
        tools = ctx.get("tools") or []
        if not tools:
            return ""
        return tools[0].get("function", {}).get("description", "")

    def _make_mock(self, tmp_path):
        """Deterministic mock keyed on the TARGET SPEC's description (carried in
        ctx['tools'][0], since InvocationTester puts the target spec first):
        - initial spec (description starts "Compute body"): fail only on "t5" → 80%.
        - rewritten spec (description == "Harmful rewrite."): ALWAYS fail → 0%.
        Branching on spec content (not a call counter) keeps it order-independent.
        """
        mock = MockLLMClient(cache_dir=tmp_path)
        mock.register("Tool specification:", lambda p, ctx: LLMResponse(
            text='{"prompts": ["t1", "t2", "t3", "t4", "t5"]}'
        ))
        mock.register("MECHANICAL CLASSIFICATION", lambda p, ctx: LLMResponse(text=json.dumps({
            "blamed_field": "description",
            "root_cause": "ambiguous",
            "suggested_rewrite": {"field": "description", "new_value": "Harmful rewrite."}
        })))

        fail_call = LLMResponse(tool_calls=[ToolCall(name="compute_bmi",
                                arguments={"height_m": 1.75})])           # missing required
        ok_call = LLMResponse(tool_calls=[ToolCall(name="compute_bmi",
                              arguments={"weight_kg": 70.0, "height_m": 1.75})])

        def tool_responder(p, ctx):
            desc = self._target_desc(ctx)
            if "Harmful rewrite." in desc:
                return fail_call                       # regressed spec → 0%
            return fail_call if "t5" in p else ok_call  # initial spec → 80%

        mock.register("", tool_responder)
        return mock

    def test_guard_rejects_regression(self, bmi_spec, tmp_path):
        mock = self._make_mock(tmp_path)
        # detect_redesign=False to isolate the guard's rejection behaviour.
        loop = OptimizerLoop(llm=mock, log_path=tmp_path / "log.jsonl",
                             n_prompts=5, max_iterations=3, do_no_harm=True,
                             detect_redesign=False)
        result = loop.optimize(bmi_spec)
        # Initial spec = 80%; harmful rewrite → 0% must be rejected.
        assert "no_harm_guard" in result.terminated_reason
        assert result.final_spec.description == bmi_spec.description  # unchanged

    def test_guard_disabled_allows_regression(self, bmi_spec, tmp_path):
        mock = self._make_mock(tmp_path)
        loop = OptimizerLoop(llm=mock, log_path=tmp_path / "log.jsonl",
                             n_prompts=5, max_iterations=3, do_no_harm=False)
        result = loop.optimize(bmi_spec)
        # Without the guard, the harmful rewrite IS committed (description changes).
        assert result.final_spec.description == "Harmful rewrite."
        assert "no_harm_guard" not in result.terminated_reason


# ---------------------------------------------------------------------------
# needs_redesign — structural-defect detection
# ---------------------------------------------------------------------------

class TestNeedsRedesign:
    """When the diagnoser's best field-level fix cannot lift a failing spec,
    the loop concludes the defect is STRUCTURAL and flags needs_redesign."""

    @staticmethod
    def _target_desc(ctx: dict) -> str:
        tools = ctx.get("tools") or []
        return tools[0].get("function", {}).get("description", "") if tools else ""

    def test_guard_path_flags_redesign(self, bmi_spec, tmp_path):
        """Initial spec fails (60%), the one field fix the diagnoser proposes
        makes it worse → guard rejects → structural → needs_redesign."""
        mock = MockLLMClient(cache_dir=tmp_path)
        mock.register("Tool specification:", lambda p, ctx: LLMResponse(
            text='{"prompts": ["t1", "t2", "t3", "t4", "t5"]}'
        ))
        mock.register("MECHANICAL CLASSIFICATION", lambda p, ctx: LLMResponse(text=json.dumps({
            "blamed_field": "description",
            "root_cause": "scope too broad",
            "suggested_rewrite": {"field": "description", "new_value": "Worse scope."}
        })))
        fail = LLMResponse(tool_calls=[ToolCall(name="compute_bmi",
                           arguments={"height_m": 1.75})])
        ok = LLMResponse(tool_calls=[ToolCall(name="compute_bmi",
                         arguments={"weight_kg": 70.0, "height_m": 1.75})])

        def responder(p, ctx):
            if "Worse scope." in self._target_desc(ctx):
                return fail                              # rewrite → 0% (worse)
            return ok if p.endswith("t1") or p.endswith("t2") or p.endswith("t3") else fail  # 60%

        mock.register("", responder)
        loop = OptimizerLoop(llm=mock, log_path=tmp_path / "log.jsonl",
                             n_prompts=5, max_iterations=3,
                             do_no_harm=True, detect_redesign=True)
        result = loop.optimize(bmi_spec)
        assert result.needs_redesign
        assert "needs_redesign" in result.terminated_reason
        assert result.final_spec.description == bmi_spec.description  # untouched

    def test_clean_spec_not_flagged(self, bmi_spec, tmp_path):
        """A spec that reaches target must NOT be flagged for redesign."""
        mock = MockLLMClient(cache_dir=tmp_path)
        mock.register("Tool specification:", lambda p, ctx: LLMResponse(
            text='{"prompts": ["t1", "t2", "t3", "t4", "t5"]}'
        ))
        mock.register("", lambda p, ctx: LLMResponse(tool_calls=[
            ToolCall(name="compute_bmi", arguments={"weight_kg": 70.0, "height_m": 1.75})
        ]))
        loop = OptimizerLoop(llm=mock, log_path=tmp_path / "log.jsonl",
                             n_prompts=5, max_iterations=3, detect_redesign=True)
        result = loop.optimize(bmi_spec)
        assert not result.needs_redesign
        assert "target_reached" in result.terminated_reason
