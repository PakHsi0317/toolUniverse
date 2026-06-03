"""Tests for TestPromptGenerator and InvocationTester (mock-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm_client import LLMResponse, MockLLMClient, ToolCall
from src.optimizer import InvocationTester, TestPromptGenerator
from src.optimizer.invocation_tester import (
    classify,
    diagnose_dimensions,
    _canon,
    _is_close,
)
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
# diagnose_dimensions() — 5-dimension parallel checker
# ---------------------------------------------------------------------------

class TestDiagnoseDimensions:
    def test_all_correct(self, bmi_spec):
        call = {"name": "compute_bmi", "arguments": {"weight_kg": 70.0, "height_m": 1.75}}
        d = diagnose_dimensions(bmi_spec, call)
        assert d.all_correct
        assert d.failed_dimensions() == []
        # values_ok not requested → stays None, excluded from all_correct
        assert d.values_ok is None

    def test_wrong_tool_isolated(self, bmi_spec):
        # Wrong tool, but otherwise well-formed args for THIS spec's params.
        call = {"name": "compute_bmr", "arguments": {"weight_kg": 70.0, "height_m": 1.75}}
        d = diagnose_dimensions(bmi_spec, call)
        assert not d.tool_name
        assert d.required_present and d.no_hallucination and d.types_ok
        assert d.failed_dimensions() == ["tool_name"]

    def test_co_occurring_defects_both_recorded(self, bmi_spec):
        # BOTH wrong tool AND a wrong type — the key advantage over classify(),
        # which would short-circuit to "wrong_tool" only.
        call = {"name": "compute_bmr",
                "arguments": {"weight_kg": "seventy", "height_m": 1.75}}
        d = diagnose_dimensions(bmi_spec, call)
        assert not d.tool_name
        assert not d.types_ok
        assert set(d.failed_dimensions()) == {"tool_name", "types_ok"}

    def test_hallucinated_param(self, bmi_spec):
        call = {"name": "compute_bmi",
                "arguments": {"weight_kg": 70.0, "height_m": 1.75, "age": 30}}
        d = diagnose_dimensions(bmi_spec, call)
        assert not d.no_hallucination
        assert d.tool_name and d.required_present and d.types_ok

    def test_malformed_fails_all_structural(self, bmi_spec):
        d = diagnose_dimensions(bmi_spec, None)
        assert not d.all_correct
        assert set(d.failed_dimensions()) == {
            "tool_name", "required_present", "no_hallucination", "types_ok"
        }

    def test_values_ok_pass(self):
        spec = ToolSpec(
            name="fetch_pubmed_abstract",
            description="Fetch a PubMed abstract by PMID.",
            parameters=[Parameter(name="pmid", type="string",
                                  description="PubMed id.", required=True)],
            return_schema=ReturnSchema(type="object", properties={},
                                       description="abstract"),
        )
        call = {"name": "fetch_pubmed_abstract", "arguments": {"pmid": "12345678"}}
        d = diagnose_dimensions(spec, call, salient_values=["12345678"])
        assert d.values_ok is True
        assert d.all_correct

    def test_values_ok_fail(self):
        spec = ToolSpec(
            name="fetch_pubmed_abstract",
            description="Fetch a PubMed abstract by PMID.",
            parameters=[Parameter(name="pmid", type="string",
                                  description="PubMed id.", required=True)],
            return_schema=ReturnSchema(type="object", properties={},
                                       description="abstract"),
        )
        # LLM hallucinated a different PMID than the one in the prompt.
        call = {"name": "fetch_pubmed_abstract", "arguments": {"pmid": "99999999"}}
        d = diagnose_dimensions(spec, call, salient_values=["12345678"])
        assert d.values_ok is False
        assert not d.all_correct
        assert d.failed_dimensions() == ["values_ok"]


class TestClassifyDimensionsParity:
    """The single label and the 5-vector must agree on correctness
    (when no salient values are supplied)."""

    @pytest.mark.parametrize("call", [
        {"name": "compute_bmi", "arguments": {"weight_kg": 70.0, "height_m": 1.75}},
        {"name": "compute_bmi",
         "arguments": {"weight_kg": 70.0, "height_m": 1.75, "round_digits": 2}},
        {"name": "compute_bmr", "arguments": {}},
        {"name": "compute_bmi", "arguments": {"weight_kg": 70.0}},
        {"name": "compute_bmi",
         "arguments": {"weight_kg": 70.0, "height_m": 1.75, "age": 30}},
        {"name": "compute_bmi",
         "arguments": {"weight_kg": "seventy", "height_m": 1.75}},
        None,
    ])
    def test_parity(self, bmi_spec, call):
        is_correct_label = classify(bmi_spec, call) == "correct"
        is_correct_vector = diagnose_dimensions(bmi_spec, call).all_correct
        assert is_correct_label == is_correct_vector


class TestCanon:
    def test_numeric_equivalence(self):
        assert _canon(0.80) == _canon("0.8")
        assert _canon(5) == _canon("5")

    def test_numeric_string_matches_number(self):
        # prompt token is a string '-7.5'; argument value is the float -7.5
        assert _canon("-7.5") == _canon(-7.5)

    def test_string_normalization(self):
        assert _canon("  Hello World ") == _canon("hello world")

    def test_distinct(self):
        assert _canon("12345678") != _canon("99999999")


class TestValuesOkArrayFlattening:
    """Regression: salient values must match ELEMENTS of an array argument,
    not the array as a whole (the false-negative found on rank_drug_compounds)."""

    def _array_spec(self) -> ToolSpec:
        return ToolSpec(
            name="rank_drug_compounds",
            description="Rank candidate compounds by score arrays.",
            parameters=[
                Parameter(name="binding_affinity_scores", type="array",
                          description="List of binding affinities.", required=True),
                Parameter(name="toxicity_risks", type="array",
                          description="List of toxicity risks.", required=True),
            ],
            return_schema=ReturnSchema(type="object", properties={},
                                       description="ranking"),
        )

    def test_values_in_array_pass(self):
        spec = self._array_spec()
        call = {"name": "rank_drug_compounds", "arguments": {
            "binding_affinity_scores": [-7.5, -8.2, -7.0],
            "toxicity_risks": [0.2, 0.4, 0.1],
        }}
        # prompt literals are strings; they must match numeric array elements
        sal = ["-7.5", "-8.2", "-7.0", "0.2", "0.4", "0.1"]
        d = diagnose_dimensions(spec, call, salient_values=sal)
        assert d.values_ok is True
        assert d.all_correct

    def test_missing_array_value_fails(self):
        spec = self._array_spec()
        call = {"name": "rank_drug_compounds", "arguments": {
            "binding_affinity_scores": [-7.5, -8.2],   # -7.0 dropped
            "toxicity_risks": [0.2, 0.4, 0.1],
        }}
        sal = ["-7.5", "-8.2", "-7.0"]
        d = diagnose_dimensions(spec, call, salient_values=sal)
        assert d.values_ok is False


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

    def test_generate_with_salient_filters_non_verbatim(self, bmi_spec, tmp_path):
        """salient_values not literally present in the prompt must be dropped,
        so ground truth can never be hallucinated by the generator."""
        mock = MockLLMClient(cache_dir=tmp_path)
        payload = {
            "items": [
                # '70' IS in the prompt → kept; '999' is NOT → dropped
                {"prompt": "A patient weighs 70 kg at 1.75 m tall.",
                 "salient_values": ["70", "999"]},
                # neither value is in the prompt → both dropped → empty list
                {"prompt": "Compute BMI for this patient please.",
                 "salient_values": ["80", "1.9"]},
            ]
        }
        mock.register("Tool specification:", lambda p, ctx: LLMResponse(text=json.dumps(payload)))
        gen = TestPromptGenerator(mock)
        items = gen.generate_with_salient(bmi_spec, n=2)
        assert len(items) == 2
        assert items[0].salient_values == ["70"]       # 999 filtered out
        assert items[1].salient_values == []            # all filtered out
