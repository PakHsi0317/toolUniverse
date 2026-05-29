"""SpecGenerator: LLM call that turns NL → ToolSpec, using seeds as few-shot.

Uses llm_client.complete_structured which retries on parse failure and
caches all calls. Output is guaranteed to be a valid ToolSpec (Pydantic
validation happens inside complete_structured).
"""

from __future__ import annotations

import json

from src.llm_client import LLMClient
from src.schema import ToolSpec


_SYSTEM = """You are an expert at designing tool specifications for AI agents. \
You write specs that LLMs can consume reliably: clear names, precise descriptions \
that say WHEN to use (and not use) the tool, parameter descriptions that include \
format requirements and valid ranges, and complete return schemas."""


_USER_TEMPLATE = """Below are {n_examples} examples of well-designed tool specifications:

{examples_block}

Now design a tool specification for the following request:

REQUEST: "{nl_description}"

Requirements (critical for LLM tool-use reliability):
- Tool name: snake_case, clearly indicates the action and target (e.g. 'search_pubmed', 'compute_bmi').
- Tool description: explain WHEN to use this tool AND when NOT to use it (mention alternatives if relevant). \
This is what LLMs read first to decide whether to call this tool.
- Parameter names: snake_case, unambiguous (e.g. 'gene_symbol' beats 'name').
- Parameter descriptions MUST include: what it represents, format requirements (e.g. 'ISO 8601 date', \
'4-char PDB ID'), valid value range, and default if optional.
- Parameter types: must be one of {{string, integer, number, boolean, array, object}}.
- Mark required=true ONLY for parameters the tool cannot run without.
- Return schema: type, properties (key→type), description of overall shape.

Output a single JSON object matching the ToolSpec schema. No prose, no markdown fences."""


def _format_examples(seeds: list[ToolSpec]) -> str:
    blocks = []
    for i, spec in enumerate(seeds, start=1):
        spec_json = json.dumps(spec.model_dump(), indent=2)
        blocks.append(f"EXAMPLE {i}:\n{spec_json}")
    return "\n\n".join(blocks)


class SpecGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, nl_description: str, seeds: list[ToolSpec]) -> ToolSpec:
        prompt = _USER_TEMPLATE.format(
            n_examples=len(seeds),
            examples_block=_format_examples(seeds),
            nl_description=nl_description,
        )
        return self.llm.complete_structured(
            prompt=prompt,
            schema=ToolSpec,
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=2000,
        )
