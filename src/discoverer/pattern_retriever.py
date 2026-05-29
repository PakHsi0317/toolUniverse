"""PatternRetriever: select few-shot examples from the seed template library.

For 3 seeds covering 3 categories, the simplest robust strategy is to
score by keyword overlap with the NL request and return the top-k.

In a full system (600+ tools, like ToolUniverse), this would be embedding
search. For the assignment scale, keyword overlap is sufficient and
inspectable. The interface stays the same — drop in an embedding backend
later without changing callers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.schema import ToolSpec


_STOP = {
    "a", "an", "the", "of", "for", "by", "on", "in", "to", "from", "with",
    "and", "or", "is", "are", "this", "that", "these", "those",
    "get", "fetch", "retrieve", "find", "search", "compute", "calculate",
    "given", "based", "using", "into", "data",
}


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", text.lower())
        if len(t) > 2 and t not in _STOP
    }


def _score(nl: str, spec: ToolSpec) -> float:
    """Jaccard-like overlap between NL tokens and the spec's name+description."""
    nl_tokens = _tokens(nl)
    spec_tokens = _tokens(spec.name + " " + spec.description)
    if not nl_tokens or not spec_tokens:
        return 0.0
    overlap = nl_tokens & spec_tokens
    return len(overlap) / (len(nl_tokens) + len(spec_tokens) - len(overlap))


class PatternRetriever:
    def __init__(self, seed_templates: list[ToolSpec]):
        if not seed_templates:
            raise ValueError("Need at least one seed template")
        self.seeds = seed_templates

    def retrieve(self, nl_description: str, k: int = 3) -> list[ToolSpec]:
        """Return up to k most relevant seeds. Ties broken by original order."""
        scored = [(s, _score(nl_description, s)) for s in self.seeds]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:k]]


def load_seed_templates(
    seed_dir: Path | str = Path(__file__).parent.parent.parent / "data" / "seed_templates",
) -> list[ToolSpec]:
    seed_dir = Path(seed_dir)
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed dir not found: {seed_dir}")
    out: list[ToolSpec] = []
    for path in sorted(seed_dir.glob("*.json")):
        out.append(ToolSpec.model_validate_json(path.read_text(encoding="utf-8")))
    return out
