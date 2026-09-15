"""Small pure helpers for strategy parameter parsing.

Kept dependency-free so core/ stays importable in tests without dify_plugin.
"""

from __future__ import annotations

from typing import Any


def to_int(raw: Any, default: int) -> int:
    """Best-effort cast of a node parameter to int; junk falls back to default.

    Dify sends numbers as numbers, but workflow variables can arrive as
    "3" / "3.7" / "abc" / "" / None depending on how the node was wired.
    """
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default