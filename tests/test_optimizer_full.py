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

        loop = OptimizerLoop(llm=mock, log_path=tmp_path / "log.jsonl",
                             n_prompts=3, max_iterations=3)
        result = loop.optimize(bmi_spec)
        # Did not reach target, ran all 3 iterations
        assert len(result.iterations) == 3
        assert "max_iterations" in result.terminated_reason
        # Spec was rewritten each round (description changed)
        assert result.final_spec.parameters[0].description != result.initial_spec.parameters[0].description
