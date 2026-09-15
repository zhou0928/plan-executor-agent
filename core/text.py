"""Text shaping for values that travel between steps."""

from __future__ import annotations


def truncate_middle(text: str, limit: int | None) -> str:
    """Clip an oversized value, keeping both ends and marking what was dropped.

    A scraped page or a retrieved segment can run to tens of thousands of
    characters; pasted verbatim into the next prompt it dominates the token
    bill. The tail matters as much as the head (conclusions, error text), so
    the middle is what goes. ``limit`` of ``None`` or <= 0 disables clipping.
    """
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    keep = max(limit // 2, 1)
    dropped = len(text) - 2 * keep
    if dropped <= 0:
        return text
    return f"{text[:keep]}\n…[已省略 {dropped} 字符]…\n{text[-keep:]}"
