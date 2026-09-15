"""Scratchpad: the single state carrier between steps (design doc §2 decision 2).

Steps communicate only through the scratchpad. ``{{var}}`` templates in a
step's input_mapping are resolved against it.

``validate_plan`` performs the topological check the Executor relies on:
every reference must resolve to an EARLIER step's output_var or an initial
variable. Forward references are rejected as invalid (they are either
undefined or circular in a sequential execution).
"""

from __future__ import annotations

import re
import threading

from core.plan import InvalidPlanError, Plan

_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class MissingVariableError(Exception):
    """Raised when rendering references a variable not present in the pool."""


def extract_refs(text: str) -> set[str]:
    """Return the set of variable names referenced by {{var}} in text."""
    return set(_TEMPLATE.findall(text))


class Scratchpad:
    """All access is lock-guarded: steps may run concurrently, and each step
    both reads its inputs and writes its output through the same pool."""

    def __init__(self, initial: dict[str, object] | None = None) -> None:
        self._vars: dict[str, object] = dict(initial or {})
        self._lock = threading.RLock()

    def set(self, name: str, value: object) -> None:
        with self._lock:
            self._vars[name] = value

    def get(self, name: str, default: object = None) -> object:
        with self._lock:
            return self._vars.get(name, default)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._vars

    def variables(self) -> dict[str, object]:
        """Snapshot of all current variables (for dependency scheduling)."""
        with self._lock:
            return dict(self._vars)

    def render(self, text: str) -> str:
        """Replace every {{var}} in text; raises MissingVariableError if any
        referenced variable is absent."""

        def _sub(m: re.Match[str]) -> str:
            name = m.group(1)
            with self._lock:
                if name not in self._vars:
                    raise MissingVariableError(f"variable '{name}' is not set")
                return str(self._vars[name])

        return _TEMPLATE.sub(_sub, text)

    def render_mapping(self, mapping: dict[str, str]) -> dict[str, str]:
        return {k: self.render(v) for k, v in mapping.items()}


def validate_plan(plan: Plan, initial_vars: set[str] | None = None) -> None:
    """Check that every input reference resolves to an earlier step output
    or an initial variable. Raises InvalidPlanError otherwise."""
    available = set(initial_vars or set())
    for step in plan.steps:
        for key, template in step.input_mapping.items():
            refs = extract_refs(template)
            unknown = refs - available
            if unknown:
                raise InvalidPlanError(
                    f"step {step.id} input '{key}' references undefined or "
                    f"forward variable(s): {', '.join(sorted(unknown))}"
                )
        available.add(step.output_var)
