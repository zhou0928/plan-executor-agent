"""Text shaping for values that travel between steps."""

from __future__ import annotations

import re

_OPEN = "<think>"
_CLOSE = "</think>"


def strip_think(text: str) -> str:
    """Drop the reasoning block a thinking model (qwen3…) prefixes to its reply.

    Mirrors core.planner's handling of JSON replies: the answer lives after the
    last closed ``</think>``. Never touches text that has no thinking block.
    """
    if not text:
        return text
    if _CLOSE in text:
        text = text.rsplit(_CLOSE, 1)[1]
    cut = text.find(_OPEN)
    if cut >= 0:
        text = text[:cut]
    return re.sub(rf"{_OPEN}.*?{_CLOSE}", "", text, flags=re.DOTALL)


class ThinkFilter:
    """Streaming ``strip_think``: hides thinking blocks as they flow past.

    Chunks arrive on arbitrary boundaries, so a trailing fragment that might be
    the start of a tag is held back until the next chunk settles it; everything
    else is emitted immediately (streaming is not sacrificed to filtering).
    """

    def __init__(self) -> None:
        self._pending = ""
        self._inside = False

    def feed(self, piece: str) -> str:
        self._pending += piece
        out: list[str] = []
        while self._pending:
            if self._inside:
                end = self._pending.find(_CLOSE)
                if end < 0:
                    self._pending = self._pending[-(len(_CLOSE) - 1) :]  # keep a partial close tag
                    break
                self._pending = self._pending[end + len(_CLOSE) :]
                self._inside = False
                continue
            start = self._pending.find(_OPEN)
            close = self._pending.find(_CLOSE)
            if close >= 0 and (start < 0 or close < start):
                # Already-closed reasoning (the model opened before streaming).
                self._pending = self._pending[close + len(_CLOSE) :]
                continue
            if start >= 0:
                out.append(self._pending[:start])
                self._pending = self._pending[start + len(_OPEN) :]
                self._inside = True
                continue
            held = max(self._partial_tail(_OPEN), self._partial_tail(_CLOSE))
            out.append(self._pending[: len(self._pending) - held])
            self._pending = self._pending[len(self._pending) - held :]
            break
        return "".join(out)

    def flush(self) -> str:
        """Text still buffered when the stream ended (a dangling partial tag)."""
        text = "" if self._inside else self._pending
        self._pending = ""
        self._inside = False
        return text

    def _partial_tail(self, tag: str) -> int:
        for n in range(min(len(self._pending), len(tag) - 1), 0, -1):
            if tag.startswith(self._pending[-n:]):
                return n
        return 0


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


def render_partial(results: dict[str, object], skip: set[str] | None = None) -> str:
    """Render the values a stopped run did produce, skipping bookkeeping vars.

    The scratchpad always carries the inputs (``query``); printing the raw JSON
    of everything makes an unfinished run look like it returned a request echo.
    """
    skip = skip or set()
    done = {k: v for k, v in results.items() if k not in skip}
    if not done:
        return "（时间预算内未完成任何步骤）"
    return "\n\n".join(f"【{k}】\n{v}" for k, v in done.items())
