"""SpecRewriter: applies a SuggestedRewrite to a ToolSpec, returning a NEW spec.

Pure function, no LLM call. The Diagnoser already produced the new value;
this just plumbs it into the spec via the field-path syntax in schema.py.
"""

from __future__ import annotations

from src.schema import SuggestedRewrite, ToolSpec


class SpecRewriter:
    def apply(self, spec: ToolSpec, rewrite: SuggestedRewrite) -> ToolSpec:
        """Return a new spec with one field replaced. Original is untouched."""
        return spec.set_field(rewrite.field, rewrite.new_value)
