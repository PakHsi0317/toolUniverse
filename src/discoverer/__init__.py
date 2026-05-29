"""Tool Discoverer: NL description → ToolSpec + Python stub.

Four components, executed in sequence:
    PatternRetriever  →  SpecGenerator  →  StubGenerator  →  StaticValidator

Top-level entry: `Discoverer.discover(nl_description) -> DiscoveryResult`
"""

from .discoverer import Discoverer, DiscoveryResult
from .pattern_retriever import PatternRetriever, load_seed_templates
from .spec_generator import SpecGenerator
from .static_validator import StaticValidator, ValidationError
from .stub_generator import generate_stub

__all__ = [
    "Discoverer",
    "DiscoveryResult",
    "PatternRetriever",
    "SpecGenerator",
    "StaticValidator",
    "ValidationError",
    "generate_stub",
    "load_seed_templates",
]
