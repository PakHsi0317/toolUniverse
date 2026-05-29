"""Tests for TestPromptGenerator and InvocationTester (mock-only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.llm_client import LLMResponse, MockLLMClient, ToolCall
from src.optimizer import InvocationTester, TestPromptGenerator
from src.optimizer.invocation_tester import classify, _is_close
from src.schema import Parameter, ReturnSchema, ToolSpec


@pytest.fixture
def bmi_spec() -> ToolSpec:
    return ToolSpec(
        name="compute_bmi",
        description="Compute body mass index given weight in kg and height in m.",
        parameters=[
            Parameter(name="weight_kg", type="number",
                      description="Weight in kg, positive.", required=True),
            Parameter(name="height_m", type="number",
                      description="Height in meters, positive.", required=True),
            Parameter(name="round_digits", type="integer",
                      description="Decimal places for rounding.", required=False),
        ],
        return_schema=ReturnSchema(type="object", properties={"bmi": {"type": "number"}},
                                   description="Object with bmi field."),
    )


# ---------------------------------------------------------------------------
# classify() — pure function, no LLM
# ---------------------------------------------------------------------------

class TestClassify:
    def test_correct(self, bmi_spec):
        call = {"name": "compute_bmi", "arguments": {"weight_kg": 70.0, "height_m": 1.75}}
        assert classify(bmi_spec, call) == "correct"

    def test_correct_with_optional(self, bmi_spec):
        call = {"name": "compute_bmi",
                "arguments": {"weight_kg": 70.0, "height_m": 1.75, "round_digits": 2}}
        assert classify(bmi_spec, call) == "correct"

    def test_wrong_tool(self, bmi_spec):
        call = {"name": "compute_bmr", "arguments": {}}
        assert classify(bmi_spec, call) == "wrong_tool"

    def test_missing_required(self, bmi_spec):
        call = {"name": "compute_bmi", "arguments": {"weight_kg": 70.0}}
        assert classify(bmi_spec, call) == "missing_required"

    def test_extra_argument(self, bmi_spec):
        call = {"name": "compute_bmi",
                "arguments": {"weight_kg": 70.0, "height_m": 1.75, "age": 30}}
        assert classify(bmi_spec, call) == "extra_argument"

    def test_wrong_param_name(self, bmi_spec):
        # 'weight' is close to 'weight_kg' → wrong_param_name (typo-ish)
        call = {"name": "compute_bmi",
                "arguments": {"weight": 70.0, "height_m": 1.75}}
        ft = classify(bmi_spec, call)
        # Missing required (weight_kg) takes precedence over wrong_param_name
        # because we check missing-required first; verify our chosen behavior:
        assert ft == "missing_required"

    def test_wrong_type(self, bmi_spec):
        call = {"name": "compute_bmi",
                "arguments": {"weight_kg": "seventy", "height_m": 1.75}}
        assert classify(bmi_spec, call) == "wrong_type"

    def test_malformed(self, bmi_spec):
        assert classify(bmi_spec, None) == "malformed_output"


class TestIsClose:
    def test_identical(self):
        assert _is_close("weight_kg", "weight_kg")

    def test_substring(self):
        assert _is_close("weight", "weight_kg")

    def test_unrelated(self):
        assert not _is_close("weight_kg", "patient_id")


# ---------------------------------------------------------------------------
# InvocationTester with mock LLM
# ---------------------------------------------------------------------------

class TestInvocationTester:
    def test_correct_invocation(self, bmi_spec, tmp_path):
        mock = MockLLMClient(cache_dir=tmp_path)
        mock.register(
            "compute body mass",
            lambda p, ctx: LLMResponse(tool_calls=[
                ToolCall(name="compute_bmi",
                         arguments={"weight_kg": 70.0, "height_m": 1.75})
            ]),
        )
        tester = InvocationTester(mock)
        result = tester.test_one(bmi_spec, "Please compute body mass index for me.")
        assert result.failure_type == "correct"

    def test_failure_record_conversion(self, bmi_spec, tmp_path):
        mock = MockLLMClient(cache_dir=tmp_path)
        mock.register(
            "test",
            lambda p, ctx: LLMResponse(tool_calls=[
                ToolCall(name="compute_bmi", arguments={"weight_kg": 70.0})
            ]),
        )
        tester = InvocationTester(mock)
        r = tester.test_one(bmi_spec, "test prompt")
        rec = r.to_failure_record(bmi_spec, iteration=0, model="mock")
        assert rec.failure_type == "missing_required"
        assert rec.tool_name == "compute_bmi"
        assert rec.iteration == 0


# ---------------------------------------------------------------------------
# TestPromptGenerator
# ---------------------------------------------------------------------------

class TestTestPromptGenerator:
    def test_generates_n_prompts(self, bmi_spec, tmp_path):
        mock = MockLLMClient(cache_dir=tmp_path)
        mock.register(
            "Tool specification:",
            lambda p, ctx: LLMResponse(text='{"prompts": ["a", "b", "c", "d", "e"]}'),
        )
        gen = TestPromptGenerator(mock)
        prompts = gen.generate(bmi_spec, n=5)
        assert len(prompts) == 5
