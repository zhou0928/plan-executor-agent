"""Executor: drives the plan step by step (design doc §3.1).

Every step is dispatched through a SubAgentPool. Results (and failures) are
written back to the scratchpad. Progress is reported through a callback so
the strategy layer can stream it to the Dify UI.

With ``max_parallel > 1`` the plan is scheduled as a dependency DAG: a step
starts as soon as every ``{{var}}`` it references is in the scratchpad. The
sequential path stays the default because most plans are chains.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

from core.plan import Plan, Step
from core.scratchpad import MissingVariableError, Scratchpad, extract_refs
from core.subagent_pool import StepExecutionError, SubAgentPool

# progress(step_id, done, total, description)
ProgressCallback = Callable[[int, int, int, str], None]

_STEP_ERRORS = (StepExecutionError, MissingVariableError)


@dataclass
class ExecutionOutcome:
    failed_step: Step | None = None
    error: str | None = None
    completed: bool = False  # True when all steps finished


class Executor:
    def __init__(
        self,
        pool: SubAgentPool,
        on_progress: ProgressCallback | None = None,
        max_parallel: int = 1,
        step_timeout: float | None = None,
    ) -> None:
        self._pool = pool
        self._on_progress = on_progress
        self._max_parallel = max_parallel
        self._step_timeout = step_timeout

    def run(self, plan: Plan, scratchpad: Scratchpad, start_index: int = 0) -> ExecutionOutcome:
        """Execute steps[start_index:]; returns what happened.

        On step failure the failure is recorded in the scratchpad as
        ``{output_var}_error`` and execution stops for Replanner takeover.
        """
        steps = plan.steps[start_index:]
        if not steps:
            return ExecutionOutcome(completed=True)
        if self._max_parallel <= 1:
            return self._run_sequential(steps, scratchpad)
        return self._run_parallel(steps, scratchpad)

    def _execute_step(self, step: Step, scratchpad: Scratchpad, stream: bool) -> str:
        """Run one step, optionally bounded by ``step_timeout``.

        A step that exceeds the timeout raises StepExecutionError so the
        caller's normal failure path (record + Replanner) takes over; the
        hung call keeps running on a daemon thread and never blocks exit.
        """
        if self._step_timeout is None:
            return self._pool.execute(step, scratchpad, stream=stream)
        box: dict[str, str] = {}
        errors: list[BaseException] = []

        def target() -> None:
            try:
                box["result"] = self._pool.execute(step, scratchpad, stream=stream)
            except BaseException as e:  # noqa: BLE001 - re-raised on caller thread
                errors.append(e)

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(self._step_timeout)
        if worker.is_alive():
            raise StepExecutionError(f"步骤执行超时（>{self._step_timeout:g} 秒）")
        if errors:
            raise errors[0]
        return box["result"]

    def _run_sequential(self, steps: list[Step], scratchpad: Scratchpad) -> ExecutionOutcome:
        total = len(steps)
        for i, step in enumerate(steps):
            if self._on_progress:
                self._on_progress(step.id, i + 1, total, step.description)
            try:
                result = self._execute_step(step, scratchpad, stream=(i == total - 1))
            except _STEP_ERRORS as e:
                scratchpad.set(f"{step.output_var}_error", str(e))
                return ExecutionOutcome(failed_step=step, error=str(e))
            scratchpad.set(step.output_var, result)
        return ExecutionOutcome(completed=True)

    def _run_parallel(self, steps: list[Step], scratchpad: Scratchpad) -> ExecutionOutcome:
        total = len(steps)
        last_id = steps[-1].id
        pending = list(steps)
        failure: list[ExecutionOutcome] = []
        counter = {"done": 0}
        lock = threading.Lock()

        def ready(step: Step) -> bool:
            needed: set[str] = set()
            for template in step.input_mapping.values():
                needed |= extract_refs(template)
            return needed <= set(scratchpad.variables())

        def work(step: Step) -> None:
            try:
                result = self._execute_step(step, scratchpad, stream=(step.id == last_id))
            except _STEP_ERRORS as e:
                scratchpad.set(f"{step.output_var}_error", str(e))
                with lock:
                    if not failure:
                        failure.append(ExecutionOutcome(failed_step=step, error=str(e)))
                return
            scratchpad.set(step.output_var, result)
            with lock:
                counter["done"] += 1
                if self._on_progress:
                    self._on_progress(step.id, counter["done"], total, step.description)

        with ThreadPoolExecutor(max_workers=self._max_parallel) as workers:
            running: dict[Future[None], Step] = {}
            while pending or running:
                for step in list(pending):
                    if failure:
                        break
                    if ready(step):
                        pending.remove(step)
                        running[workers.submit(work, step)] = step
                if failure and not running:
                    break
                if not running:
                    blocked = pending[0]
                    return ExecutionOutcome(
                        failed_step=blocked,
                        error=f"步骤 {blocked.id} 依赖的变量永远不会产出（计划存在前向引用）",
                    )
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    running.pop(future, None)

        if failure:
            return failure[0]
        return ExecutionOutcome(completed=True)
