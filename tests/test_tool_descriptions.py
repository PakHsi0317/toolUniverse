"""Sanity checks on the 10 NL tool descriptions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DESC_PATH = Path(__file__).parent.parent / "data" / "tool_descriptions.json"


def _load() -> list[dict]:
    return json.loads(DESC_PATH.read_text(encoding="utf-8"))["tools"]


def test_twenty_tools_total() -> None:
    """10 detailed + 10 vague = 20 total."""
    assert len(_load()) == 20


def test_ten_per_style() -> None:
    counts = Counter(t["style"] for t in _load())
    assert counts == {"detailed": 10, "vague": 10}


def test_three_categories_covered_per_style() -> None:
    for style in ("detailed", "vague"):
        cats = {t["category"] for t in _load() if t["style"] == style}
        assert cats == {"computation", "api_wrapper", "data_retrieval"}, style


def test_difficulty_distribution_per_style() -> None:
    """Each style: 5 easy, 3 medium, 2 hard."""
    for style in ("detailed", "vague"):
        counts = Counter(
            t["difficulty"] for t in _load() if t["style"] == style
        )
        assert counts == {"easy": 5, "medium": 3, "hard": 2}, style


def test_adversarial_pair_present_per_style() -> None:
    """Each style needs an adversarial pair for Phase 3-C."""
    for style in ("detailed", "vague"):
        tools = [t for t in _load() if t["style"] == style]
        paired = [t for t in tools if "adversarial_pair" in t]
        assert len(paired) >= 2, style
        ids = {t["id"] for t in paired}
        for t in paired:
            assert t["adversarial_pair"] in ids, t["id"]


def test_ids_unique_and_snake_case() -> None:
    ids = [t["id"] for t in _load()]
    assert len(set(ids)) == len(ids), "Duplicate IDs"
    for tid in ids:
        assert tid.islower() and " " not in tid and "-" not in tid


def test_detailed_descriptions_substantive() -> None:
    for t in _load():
        if t["style"] == "detailed":
            assert len(t["description"]) >= 60, f"{t['id']}: detailed should be substantive"


def test_vague_descriptions_are_short() -> None:
    """Vague descriptions should be SHORT — they're meant to force inference."""
    for t in _load():
        if t["style"] == "vague":
            n_words = len(t["description"].split())
            assert n_words <= 8, f"{t['id']}: vague description has {n_words} words (>8)"


def test_vague_descriptions_dont_leak_param_names() -> None:
    """Vague descriptions should NOT specify parameter names, units, or IDs."""
    leakage_terms = [
        "kilogram", "kg", "meter", "uniprot", "ncbi", "hgnc", "drugbank",
        "pubchem", "rcsb", "pdb id", "iso", "tanimoto",
    ]
    for t in _load():
        if t["style"] == "vague":
            desc_lower = t["description"].lower()
            for term in leakage_terms:
                assert term not in desc_lower, f"{t['id']}: vague description leaks '{term}'"
