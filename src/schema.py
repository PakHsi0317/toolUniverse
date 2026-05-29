"""Tool specification and failure tracking schemas.

This is the foundation of the project. Every component (Discoverer,
Optimizer, Evaluator) reads or writes these types. The FailureRecord
in particular is the system's central data source — all Phase 3
analyses are pandas aggregations over the jsonl log of these records.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Tool Specification
# ---------------------------------------------------------------------------

ParameterType = Literal[
    "string", "integer", "number", "boolean", "array", "object"
]


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: ParameterType
    description: str
    required: bool


class ReturnSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    description: str


class ToolSpec(BaseModel):
    """The canonical tool specification, mirroring ToolUniverse Supp. Fig. 1."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: list[Parameter]
    return_schema: ReturnSchema

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def spec_hash(self) -> str:
        """Stable 12-char content hash for version tracking across iterations."""
        canonical = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # LLM serialization
    # ------------------------------------------------------------------

    def _properties_and_required(self) -> tuple[dict, list[str]]:
        properties: dict[str, dict] = {}
        required: list[str] = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return properties, required

    def to_openai_function(self) -> dict:
        """OpenAI function-calling format (chat.completions tools=[...])."""
        properties, required = self._properties_and_required()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_tool(self) -> dict:
        """Anthropic tool-use format (messages.create tools=[...])."""
        properties, required = self._properties_and_required()
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    # ------------------------------------------------------------------
    # Field addressing (used by FailureDiagnoser & SpecRewriter)
    # ------------------------------------------------------------------

    def field_paths(self) -> list[str]:
        """Every addressable field, in the syntax used by FailureRecord.blamed_field."""
        paths = ["name", "description", "return_schema.description"]
        for i in range(len(self.parameters)):
            for field in ("name", "type", "description", "required"):
                paths.append(f"parameters[{i}].{field}")
        return paths

    def get_field(self, path: str) -> Any:
        obj: Any = self
        for part in _parse_path(path):
            if isinstance(part, tuple):
                attr, idx = part
                obj = getattr(obj, attr)[idx]
            else:
                obj = getattr(obj, part)
        return obj

    def set_field(self, path: str, value: Any) -> "ToolSpec":
        """Return a NEW spec with one field replaced. Immutable by design."""
        data = self.model_dump()
        parts = _parse_path(path)
        target: Any = data
        for part in parts[:-1]:
            if isinstance(part, tuple):
                attr, idx = part
                target = target[attr][idx]
            else:
                target = target[part]
        last = parts[-1]
        if isinstance(last, tuple):
            attr, idx = last
            target[attr][idx] = value
        else:
            target[last] = value
        return ToolSpec(**data)


_PATH_INDEX_RE = re.compile(r"^(\w+)\[(\d+)\]$")


def _parse_path(path: str) -> list:
    """Parse 'parameters[2].description' -> [('parameters', 2), 'description']."""
    out: list = []
    for segment in path.split("."):
        m = _PATH_INDEX_RE.match(segment)
        if m:
            out.append((m.group(1), int(m.group(2))))
        else:
            out.append(segment)
    return out


# ---------------------------------------------------------------------------
# Failure Record (the system's central log format)
# ---------------------------------------------------------------------------

FailureType = Literal[
    "correct",
    "wrong_tool",
    "wrong_param_name",
    "wrong_type",
    "missing_required",
    "extra_argument",
    "wrong_value_format",
    "malformed_output",
]


class SuggestedRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    new_value: Any


class FailureRecord(BaseModel):
    """One row of the system's central log. Aggregated by pandas for all analyses."""

    model_config = ConfigDict(extra="forbid")

    # --- Context: identifies which run this came from ---
    tool_name: str
    spec_version: str
    iteration: int
    model: str

    # --- Test: what was tried ---
    test_prompt: str
    expected_call: dict
    actual_call: Optional[dict] = None

    # --- Diagnosis: what went wrong (None when failure_type='correct') ---
    failure_type: FailureType
    blamed_field: Optional[str] = None
    root_cause: Optional[str] = None
    suggested_rewrite: Optional[SuggestedRewrite] = None

    def is_correct(self) -> bool:
        return self.failure_type == "correct"


# ---------------------------------------------------------------------------
# JSONL log helpers
# ---------------------------------------------------------------------------

def append_record(log_path: str | Path, record: FailureRecord) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def iter_records(log_path: str | Path) -> Iterator[FailureRecord]:
    with Path(log_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield FailureRecord.model_validate_json(line)


def load_records(log_path: str | Path) -> list[FailureRecord]:
    return list(iter_records(log_path))
