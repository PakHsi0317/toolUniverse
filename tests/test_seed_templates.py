"""Verify seed templates are valid ToolSpec JSON and follow conventions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schema import ToolSpec

SEED_DIR = Path(__file__).parent.parent / "data" / "seed_templates"


def _seed_files() -> list[Path]:
    return sorted(SEED_DIR.glob("*.json"))


def test_seed_dir_exists() -> None:
    assert SEED_DIR.exists(), f"Seed dir missing: {SEED_DIR}"
    assert len(_seed_files()) >= 3, "Need at least 3 seed templates (1 per category)"


@pytest.mark.parametrize("seed_path", _seed_files(), ids=lambda p: p.stem)
def test_seed_validates(seed_path: Path) -> None:
    """Each seed must parse as a valid ToolSpec."""
    data = json.loads(seed_path.read_text(encoding="utf-8"))
    spec = ToolSpec.model_validate(data)
    # File name should match tool name (convention)
    assert spec.name == seed_path.stem


@pytest.mark.parametrize("seed_path", _seed_files(), ids=lambda p: p.stem)
def test_seed_quality(seed_path: Path) -> None:
    """Sanity checks on description quality."""
    spec = ToolSpec.model_validate_json(seed_path.read_text(encoding="utf-8"))
    # Description should be substantive (not a one-liner)
    assert len(spec.description) >= 80, f"{spec.name}: description too short"
    # Should have at least one required parameter
    assert any(p.required for p in spec.parameters), f"{spec.name}: no required params"
    # Each parameter description must be substantive
    for p in spec.parameters:
        assert len(p.description) >= 30, f"{spec.name}.{p.name}: param description too short"
    # Return schema description must exist
    assert len(spec.return_schema.description) >= 20, f"{spec.name}: return desc too short"


def test_three_categories_covered() -> None:
    """The three seeds should span data-retrieval, computation, API-wrapper."""
    names = {p.stem for p in _seed_files()}
    # Loose check: we have these three by name
    expected = {"search_pubmed", "compute_blast_alignment", "get_uniprot_protein_info"}
    assert expected.issubset(names), f"Missing seeds: {expected - names}"
