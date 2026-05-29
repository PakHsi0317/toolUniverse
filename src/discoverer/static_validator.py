"""StaticValidator: pure-rule checks on (ToolSpec, Python stub source).

No LLM calls. These are deterministic checks that catch the obvious
failure modes — schema is already enforced by Pydantic on the spec
side; this layer adds cross-cutting checks (snake_case, no duplicates,
stub ↔ spec consistency) and catches stub Python that won't parse.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from src.schema import ToolSpec


_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class ValidationError:
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.field}] {self.message}"


class StaticValidator:
    def validate(
        self, spec: ToolSpec, stub_source: str
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []
        errors.extend(self._validate_spec(spec))
        errors.extend(self._validate_stub(spec, stub_source))
        return errors

    # ------------------------------------------------------------------

    def _validate_spec(self, spec: ToolSpec) -> list[ValidationError]:
        errors: list[ValidationError] = []

        if not _SNAKE_CASE.match(spec.name):
            errors.append(ValidationError("name", f"Not snake_case: {spec.name!r}"))

        param_names = [p.name for p in spec.parameters]
        if len(param_names) != len(set(param_names)):
            dups = {n for n in param_names if param_names.count(n) > 1}
            errors.append(ValidationError(
                "parameters", f"Duplicate parameter names: {sorted(dups)}"
            ))

        for i, p in enumerate(spec.parameters):
            if not _SNAKE_CASE.match(p.name):
                errors.append(ValidationError(
                    f"parameters[{i}].name", f"Not snake_case: {p.name!r}"
                ))

        if not spec.description.strip():
            errors.append(ValidationError("description", "Empty"))
        if not spec.return_schema.description.strip():
            errors.append(ValidationError(
                "return_schema.description", "Empty"
            ))

        return errors

    # ------------------------------------------------------------------

    def _validate_stub(
        self, spec: ToolSpec, stub_source: str
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []

        try:
            tree = ast.parse(stub_source)
        except SyntaxError as e:
            errors.append(ValidationError("stub", f"Python syntax error: {e}"))
            return errors  # Can't do further checks if it won't parse

        # Find the function definition matching spec.name
        func = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == spec.name),
            None,
        )
        if func is None:
            errors.append(ValidationError(
                "stub", f"No function named {spec.name!r} found in stub"
            ))
            return errors

        # Function parameter set must match spec parameter set
        # (allowing optional params to appear after required ones — Python rule)
        stub_arg_names = [a.arg for a in func.args.args]
        spec_param_names = [p.name for p in spec.parameters]

        if set(stub_arg_names) != set(spec_param_names):
            missing = set(spec_param_names) - set(stub_arg_names)
            extra = set(stub_arg_names) - set(spec_param_names)
            msg_parts = []
            if missing:
                msg_parts.append(f"missing args: {sorted(missing)}")
            if extra:
                msg_parts.append(f"unexpected args: {sorted(extra)}")
            errors.append(ValidationError("stub", "; ".join(msg_parts)))

        # Stub should have a docstring
        if not (ast.get_docstring(func) or "").strip():
            errors.append(ValidationError("stub", "Missing docstring"))

        return errors
