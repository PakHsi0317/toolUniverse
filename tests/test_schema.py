"""Smoke tests for schema.py. Run with: python -m pytest tests/"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schema import (
    FailureRecord,
    Parameter,
    ReturnSchema,
    SuggestedRewrite,
    ToolSpec,
    append_record,
    load_records,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_spec() -> ToolSpec:
    return ToolSpec(
        name="search_pubmed",
        description="Search PubMed for biomedical articles matching a keyword query.",
        parameters=[
            Parameter(
                name="query",
                type="string",
                description="Free-text keyword query, e.g. 'BRCA1 breast cancer'.",
                required=True,
            ),
            Parameter(
                name="max_results",
                type="integer",
                description="Maximum number of articles to return (1-100).",
                required=False,
            ),
        ],
        return_schema=ReturnSchema(
            type="object",
            properties={"articles": {"type": "array"}},
            description="A list of article metadata objects.",
        ),
    )


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------

def test_spec_round_trip(sample_spec: ToolSpec) -> None:
    raw = sample_spec.model_dump_json()
    restored = ToolSpec.model_validate_json(raw)
    assert restored == sample_spec


def test_spec_hash_stable(sample_spec: ToolSpec) -> None:
    assert sample_spec.spec_hash() == sample_spec.spec_hash()
    assert len(sample_spec.spec_hash()) == 12


def test_spec_hash_changes_with_content(sample_spec: ToolSpec) -> None:
    h1 = sample_spec.spec_hash()
    edited = sample_spec.set_field("description", "Something different.")
    assert edited.spec_hash() != h1


def test_to_openai_function(sample_spec: ToolSpec) -> None:
    fn = sample_spec.to_openai_function()
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "search_pubmed"
    assert fn["function"]["parameters"]["required"] == ["query"]
    assert "query" in fn["function"]["parameters"]["properties"]


def test_to_anthropic_tool(sample_spec: ToolSpec) -> None:
    tool = sample_spec.to_anthropic_tool()
    assert tool["name"] == "search_pubmed"
    assert tool["input_schema"]["required"] == ["query"]


def test_field_paths_complete(sample_spec: ToolSpec) -> None:
    paths = sample_spec.field_paths()
    assert "name" in paths
    assert "description" in paths
    assert "return_schema.description" in paths
    assert "parameters[0].name" in paths
    assert "parameters[1].description" in paths


def test_get_field(sample_spec: ToolSpec) -> None:
    assert sample_spec.get_field("name") == "search_pubmed"
    assert sample_spec.get_field("parameters[0].name") == "query"
    assert sample_spec.get_field("parameters[1].required") is False


def test_set_field_immutable(sample_spec: ToolSpec) -> None:
    edited = sample_spec.set_field("parameters[0].description", "New desc.")
    assert sample_spec.get_field("parameters[0].description") != "New desc."
    assert edited.get_field("parameters[0].description") == "New desc."


def test_set_field_nested(sample_spec: ToolSpec) -> None:
    edited = sample_spec.set_field("return_schema.description", "New return.")
    assert edited.return_schema.description == "New return."


def test_extra_field_rejected() -> None:
    with pytest.raises(Exception):
        ToolSpec.model_validate({
            "name": "x", "description": "y", "parameters": [],
            "return_schema": {"type": "object", "properties": {}, "description": "z"},
            "unexpected": "field",
        })


# ---------------------------------------------------------------------------
# FailureRecord
# ---------------------------------------------------------------------------

def test_failure_record_correct() -> None:
    r = FailureRecord(
        tool_name="search_pubmed",
        spec_version="abc123",
        iteration=0,
        model="claude-opus-4-7",
        test_prompt="Find papers on CRISPR.",
        expected_call={"name": "search_pubmed", "arguments": {"query": "CRISPR"}},
        actual_call={"name": "search_pubmed", "arguments": {"query": "CRISPR"}},
        failure_type="correct",
    )
    assert r.is_correct()


def test_failure_record_with_diagnosis() -> None:
    r = FailureRecord(
        tool_name="search_pubmed",
        spec_version="abc123",
        iteration=1,
        model="claude-opus-4-7",
        test_prompt="Find papers on CRISPR.",
        expected_call={"name": "search_pubmed", "arguments": {"query": "CRISPR"}},
        actual_call={"name": "search_articles", "arguments": {"q": "CRISPR"}},
        failure_type="wrong_tool",
        blamed_field="description",
        root_cause="Tool description is too generic; overlaps with search_articles.",
        suggested_rewrite=SuggestedRewrite(
            field="description",
            new_value="Search PubMed (biomedical literature only) for articles...",
        ),
    )
    assert not r.is_correct()
    assert r.blamed_field == "description"


# ---------------------------------------------------------------------------
# JSONL log
# ---------------------------------------------------------------------------

def test_jsonl_round_trip(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    records = [
        FailureRecord(
            tool_name="t", spec_version="v", iteration=i, model="m",
            test_prompt="p", expected_call={}, failure_type="correct",
        )
        for i in range(3)
    ]
    for r in records:
        append_record(log, r)
    loaded = load_records(log)
    assert len(loaded) == 3
    assert [r.iteration for r in loaded] == [0, 1, 2]
