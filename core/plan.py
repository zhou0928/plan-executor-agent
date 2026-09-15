"""Plan domain objects: structured plan JSON parsing and validation.

The plan is the contract between Planner and Executor (see design doc §3.2):
    {"steps": [{"id": 1, "description": "...", "tool": "x" | null,
                "input_mapping": {"k": "{{var}}"}, "output_var": "v"}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass


class InvalidPlanError(Exception):
    """Raised when a plan fails parsing or validation. Message must be
    safe to feed back to the Replanner."""


@dataclass(frozen=True)
class Step:
    id: int
    description: str
    tool: str | None  # None = pure-reasoning step
    input_mapping: dict[str, str]
    output_var: str


@dataclass(frozen=True)
class Plan:
    steps: list[Step]

    @classmethod
    def parse(cls, raw: str | dict, max_steps: int = 20) -> "Plan":
        """Parse a plan from a JSON string (or already-decoded dict).

        Raises InvalidPlanError with a reason on any violation.
        """
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as e:
                raise InvalidPlanError(f"plan is not valid JSON: {e}") from e

        if not isinstance(raw, dict):
            raise InvalidPlanError("plan must be a JSON object")

        steps_raw = raw.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise InvalidPlanError("plan must contain a non-empty 'steps' list")

        if len(steps_raw) > max_steps:
            raise InvalidPlanError(
                f"plan has {len(steps_raw)} steps, exceeding limit {max_steps}"
            )

        steps: list[Step] = []
        seen_ids: set[int] = set()
        seen_outputs: set[str] = set()
        for i, item in enumerate(steps_raw):
            steps.append(cls._parse_step(i, item, seen_ids, seen_outputs))

        return cls(steps=steps)

    @staticmethod
    def _parse_step(
        index: int,
        item: object,
        seen_ids: set[int],
        seen_outputs: set[str],
    ) -> Step:
        where = f"steps[{index}]"
        if not isinstance(item, dict):
            raise InvalidPlanError(f"{where} must be an object")

        step_id = item.get("id")
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            raise InvalidPlanError(f"{where}.id must be an integer")

        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            raise InvalidPlanError(f"{where}.description must be a non-empty string")

        tool = item.get("tool")
        if isinstance(tool, str) and tool.strip().lower() in ("", "null", "none", "nil"):
            tool = None
        if tool is not None and (not isinstance(tool, str) or not tool.strip()):
            raise InvalidPlanError(f"{where}.tool must be a string or null")

        input_mapping = item.get("input_mapping", {})
        if not isinstance(input_mapping, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in input_mapping.items()
        ):
            raise InvalidPlanError(
                f"{where}.input_mapping must be an object of string -> string"
            )

        output_var = item.get("output_var")
        if not isinstance(output_var, str) or not output_var.strip():
            raise InvalidPlanError(f"{where}.output_var must be a non-empty string")

        if step_id in seen_ids:
            raise InvalidPlanError(f"{where}.id {step_id} is duplicated")
        if output_var in seen_outputs:
            raise InvalidPlanError(f"{where}.output_var '{output_var}' is duplicated")

        seen_ids.add(step_id)
        seen_outputs.add(output_var)

        return Step(
            id=step_id,
            description=description,
            tool=tool.strip() if isinstance(tool, str) else None,
            input_mapping=input_mapping,
            output_var=output_var,
        )
