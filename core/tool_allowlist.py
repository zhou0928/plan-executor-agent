"""Parse and apply an optional tool allowlist.

Dify declares `allowed_tools` as `type: any / scope: array[string]`, so the
value arrives as anything: None, "", a real list, or a JSON-encoded string.
`None`/empty always means "no restriction" (the full attached tool list runs).
"""

from __future__ import annotations

import json
from typing import Any


def allowed_tool_names(value: Any) -> set[str] | None:
    """Normalize the allowlist to a set of names, or None when unrestricted."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = [stripped]
    if not isinstance(value, list) or not value:
        return None
    names = {item.strip() for item in value if isinstance(item, str) and item.strip()}
    return names or None


def filter_allowed_tools(tools: list[Any], allowed_tools: Any) -> list[Any]:
    """Keep only tools whose `identity.name` is in the allowlist; unrestricted when empty."""
    names = allowed_tool_names(allowed_tools)
    if names is None:
        return tools
    return [t for t in tools if getattr(getattr(t, "identity", None), "name", None) in names]
