"""Wall-clock budgeting for model calls.

A model call cannot be interrupted once it is in flight: a slow local model can
spend minutes on a single reply, and the plugin daemon kills the whole
invocation when that overruns `PLUGIN_MAX_EXECUTION_TIMEOUT`. Every call
therefore gets the remaining budget as its ceiling; an overrun raises, and the
strategy's existing failure paths already know how to answer with what they
have.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


class BudgetTimeout(TimeoutError):
    """Raised instead of letting a single call run past the deadline."""


def run_with_budget(fn: Callable[[], Any], deadline: float | None, what: str) -> Any:
    """Call ``fn()``, giving it at most what is left of ``deadline``.

    ``deadline`` of ``None`` means no budget: the call runs normally. The worker
    thread is a daemon, so an overrunning call keeps burning in the background
    without holding up the answer.
    """
    if deadline is None:
        return fn()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BudgetTimeout(f"{what}未开始：执行时间预算已用尽")
    box: dict[str, Any] = {}
    errors: list[BaseException] = []

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller thread
            errors.append(e)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(remaining)
    if worker.is_alive():
        raise BudgetTimeout(f"{what}超时：超过剩余执行时间预算")
    if errors:
        raise errors[0]
    return box["value"]


class BoundedCaller:
    """Wraps an LLM caller so each call obeys the deadline.

    Only the non-streaming form is bounded; streamed calls report tokens to the
    user as they arrive and are left to the reserve logic in the strategy.
    """

    def __init__(self, inner: Any, deadline: float | None) -> None:
        self._inner = inner
        self._deadline = deadline

    def __call__(self, system: str, user: str) -> str:
        return run_with_budget(lambda: self._inner(system, user), self._deadline, "模型调用")
