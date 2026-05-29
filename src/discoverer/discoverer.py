"""Top-level Discoverer that wires the four components together."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.llm_client import LLMClient
from src.schema import ToolSpec

from .pattern_retriever import PatternRetriever
from .spec_generator import SpecGenerator
from .static_validator import StaticValidator, ValidationError
from .stub_generator import generate_stub


@dataclass
class DiscoveryResult:
    nl_description: str
    spec: ToolSpec
    stub_source: str
    seeds_used: list[str]  # tool names of seeds passed as few-shot
    validation_errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.validation_errors


class Discoverer:
    def __init__(self, llm: LLMClient, seed_templates: list[ToolSpec]):
        self.retriever = PatternRetriever(seed_templates)
        self.spec_gen = SpecGenerator(llm)
        self.validator = StaticValidator()

    def discover(self, nl_description: str, k_seeds: int = 3) -> DiscoveryResult:
        seeds = self.retriever.retrieve(nl_description, k=k_seeds)
        spec = self.spec_gen.generate(nl_description, seeds)
        stub_source = generate_stub(spec)
        errors = self.validator.validate(spec, stub_source)
        return DiscoveryResult(
            nl_description=nl_description,
            spec=spec,
            stub_source=stub_source,
            seeds_used=[s.name for s in seeds],
            validation_errors=errors,
        )
