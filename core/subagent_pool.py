"""SubAgentPool: the seam where v2 multi-agent collaboration plugs in.

v1 ships only LocalStepExecutor. The Executor dispatches every step through
a SubAgentPool, so v2 can swap in a pool of sub-agents (running independent
steps concurrently) without touching the Executor loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from core.plan import Step
from core.scratchpad import Scratchpad

if TYPE_CHECKING:
    from core.planner import LLMCaller

# tool_invoker(tool_name, params) -> str
ToolInvoker = Callable[[str, dict[str, Any]], str]


class StepExecutionError(Exception):
    """Raised when a step fails. The message is fed back to the Replanner."""


class SubAgentPool(ABC):
    @abstractmethod
    def execute(self, step: Step, scratchpad: Scratchpad, stream: bool = False) -> str:
        """Execute one step against the scratchpad; return the step output.

        `stream=True` lets a pool emit partial output through its token
        callback; it is set for the final step so the answer reaches the user
        incrementally. Pools without streaming may ignore it.
        """


class LocalStepExecutor(SubAgentPool):
    """v1 executor: reasoning steps go to the LLM, tool steps go to the
    tool invoker with rendered input_mapping as parameters."""

    def __init__(
        self,
        llm: LLMCaller,
        tool_invoker: ToolInvoker,
        llm_stream: Callable[[str, str], Iterator[str]] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> None:
        self._llm: LLMCaller = llm
        self._llm_stream = llm_stream
        self._on_token = on_token
        self._tool_invoker = tool_invoker

    def execute(self, step: Step, scratchpad: Scratchpad, stream: bool = False) -> str:
        if step.tool is None:
            return self._reason(step, scratchpad, stream)
        params = scratchpad.render_mapping(step.input_mapping)
        return self._tool_invoker(step.tool, params)

    def _reason(self, step: Step, scratchpad: Scratchpad, stream: bool = False) -> str:
        inputs = "\n".join(
            f"- {k}: {v}" for k, v in scratchpad.render_mapping(step.input_mapping).items()
        )
        system = "你是严谨的执行者。只完成当前步骤要求的任务，输出该步骤的结果本身，不要寒暄。"
        user = f"当前步骤：{step.description}"
        if inputs:
            user += f"\n可用输入：\n{inputs}"
        if stream and self._llm_stream is not None and self._on_token is not None:
            try:
                parts: list[str] = []
                for chunk in self._llm_stream(system, user):
                    parts.append(chunk)
                    self._on_token(chunk)
                return "".join(parts)
            except Exception as e:  # noqa: BLE001 - normalised so the step fails via the replan path
                raise StepExecutionError(f"推理步骤失败：{e}") from e
        try:
            return self._llm(system, user)
        except Exception as e:  # noqa: BLE001 - normalised so the step fails via the replan path
            raise StepExecutionError(f"推理步骤失败：{e}") from e
