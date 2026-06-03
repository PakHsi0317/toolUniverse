"""Sanity checks on the NL tool descriptions."""

from __future__ import annotations

import json
from pathlib import Path

DESC_PATH = Path(__file__).parent.parent / "data" / "tool_descriptions.json"

EXPECTED_COUNT = 11

LEAKAGE_TERMS = [
    "kilogram", "kg", "meter", "uniprot", "ncbi", "hgnc", "drugbank",
    "pubchem", "rcsb", "iso",
]


def _load() -> list[str]:
    return json.loads(DESC_PATH.read_text(encoding="utf-8"))["tools"]


def test_tool_count() -> None:
    """Exactly EXPECTED_COUNT tool descriptions present."""
    assert len(_load()) == EXPECTED_COUNT


def test_all_strings() -> None:
    """Every entry is a non-empty string."""
    for item in _load():
        assert isinstance(item, str) and len(item.strip()) > 0


def test_descriptions_are_substantive() -> None:
    """Each description is long enough to be meaningful (>= 8 words)."""
    for desc in _load():
        n_words = len(desc.split())
        assert n_words >= 8, f"Too short ({n_words} words): {desc!r}"


def test_no_duplicates() -> None:
    descriptions = _load()
    assert len(set(descriptions)) == len(descriptions), "Duplicate descriptions found"


def test_no_leakage_terms() -> None:
    """Descriptions should not leak implementation-specific identifiers."""
    for desc in _load():
        desc_lower = desc.lower()
        for term in LEAKAGE_TERMS:
            assert term not in desc_lower, f"Description leaks {term!r}: {desc!r}"
