"""Tests for the Discoverer pipeline.

Uses MockLLMClient so no API calls (and no $) needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.discoverer import (
    Discoverer,
    PatternRetriever,
    SpecGenerator,
    StaticValidator,
    generate_stub,
    load_seed_templates,
)
from src.llm_client import LLMResponse, MockLLMClient
from src.schema import Parameter, ReturnSchema, ToolSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeds() -> list[ToolSpec]:
    return load_seed_templates()


@pytest.fixture
def sample_spec() -> ToolSpec:
    return ToolSpec(
        name="compute_bmi",
        description="Compute body mass index given weight and height. Use this for adult BMI; not for pediatric.",
        parameters=[
            Parameter(name="weight_kg", type="number",
                      description="Body weight in kilograms, must be positive.", required=True),
            Parameter(name="height_m", type="number",
                      description="Body height in meters, must be positive.", required=True),
        ],
        return_schema=ReturnSchema(
            type="object",
            properties={"bmi": {"type": "number"}},
            description="Object with 'bmi' field as a float.",
        ),
    )


# ---------------------------------------------------------------------------
# PatternRetriever
# ---------------------------------------------------------------------------

class TestPatternRetriever:
    def test_returns_at_most_k(self, seeds):
        r = PatternRetriever(seeds)
        assert len(r.retrieve("anything", k=2)) == 2
        assert len(r.retrieve("anything", k=5)) == len(seeds)

    def test_relevant_seed_ranks_first(self, seeds):
        """A query about literature search should rank search_pubmed first."""
        r = PatternRetriever(seeds)
        results = r.retrieve(
            "Search PubMed for articles about CRISPR gene editing.",
            k=1,
        )
        assert results[0].name == "search_pubmed"

    def test_protein_query_ranks_uniprot(self, seeds):
        r = PatternRetriever(seeds)
        results = r.retrieve(
            "Fetch protein information from UniProt by accession.",
            k=1,
        )
        assert results[0].name == "get_uniprot_protein_info"


# ---------------------------------------------------------------------------
# StubGenerator (pure function)
# ---------------------------------------------------------------------------

class TestStubGenerator:
    def test_basic_structure(self, sample_spec):
        stub = generate_stub(sample_spec)
        assert "def compute_bmi(" in stub
        assert "weight_kg: float" in stub
        assert "height_m: float" in stub
        assert '"""' in stub  # has docstring

    def test_optional_params_have_default_and_union_none(self):
        spec = ToolSpec(
            name="x", description="A tool that does x and not y.",
            parameters=[
                Parameter(name="req", type="string", description="A required string parameter.", required=True),
                Parameter(name="opt", type="integer", description="An optional integer parameter.", required=False),
            ],
            return_schema=ReturnSchema(type="object", properties={}, description="A result object."),
        )
        stub = generate_stub(spec)
        assert "req: str" in stub
        assert "opt: int | None = None" in stub
        # Required must come before optional (Python rule)
        assert stub.index("req:") < stub.index("opt:")

    def test_stub_parses_as_python(self, sample_spec):
        import ast
        stub = generate_stub(sample_spec)
        ast.parse(stub)  # raises if invalid

    def test_no_params(self):
        spec = ToolSpec(
            name="ping", description="A no-arg ping tool that always returns pong.",
            parameters=[],
            return_schema=ReturnSchema(type="object", properties={}, description="A pong response."),
        )
        stub = generate_stub(spec)
        import ast
        ast.parse(stub)
        assert "def ping(" in stub


# ---------------------------------------------------------------------------
# StaticValidator
# ---------------------------------------------------------------------------

class TestStaticValidator:
    def test_clean_spec_passes(self, sample_spec):
        stub = generate_stub(sample_spec)
        errors = StaticValidator().validate(sample_spec, stub)
        assert errors == []

    def test_camel_case_name_flagged(self):
        spec = ToolSpec(
            name="ComputeBMI",  # bad
            description="A tool description that is long enough.",
            parameters=[],
            return_schema=ReturnSchema(type="object", properties={}, description="A result."),
        )
        errors = StaticValidator().validate(spec, generate_stub(spec))
        # Generate stub will fail to parse because of capital letters? actually Python allows it
        # but our validator should flag the spec name
        assert any(e.field == "name" for e in errors)

    def test_stub_function_name_mismatch(self, sample_spec):
        bad_stub = "def wrong_name(weight_kg: float, height_m: float) -> dict:\n    '''doc'''\n    pass"
        errors = StaticValidator().validate(sample_spec, bad_stub)
        assert any("No function named" in e.message for e in errors)

    def test_stub_missing_arg(self, sample_spec):
        bad_stub = "def compute_bmi(weight_kg: float) -> dict:\n    '''doc'''\n    pass"
        errors = StaticValidator().validate(sample_spec, bad_stub)
        assert any("missing args" in e.message for e in errors)

    def test_stub_syntax_error(self, sample_spec):
        bad_stub = "def compute_bmi( :::: not valid"
        errors = StaticValidator().validate(sample_spec, bad_stub)
        assert any("syntax error" in e.message for e in errors)


# ---------------------------------------------------------------------------
# End-to-end with mock LLM
# ---------------------------------------------------------------------------

class TestDiscovererE2E:
    def test_end_to_end_with_mock(self, seeds, tmp_path: Path, sample_spec):
        mock = MockLLMClient(cache_dir=tmp_path)

        # Mock: when SpecGenerator asks for the spec, return our sample
        spec_json = sample_spec.model_dump_json()
        mock.register(
            "REQUEST:",
            lambda prompt, ctx: LLMResponse(text=spec_json),
        )

        d = Discoverer(llm=mock, seed_templates=seeds)
        result = d.discover(
            "Compute body mass index given weight in kg and height in meters."
        )

        assert result.spec.name == "compute_bmi"
        assert result.is_valid, f"Validation errors: {result.validation_errors}"
        assert "def compute_bmi(" in result.stub_source
        assert len(result.seeds_used) >= 1
