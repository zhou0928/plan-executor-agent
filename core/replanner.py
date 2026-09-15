"""Replanner: regenerates the remaining plan after a failure (design doc §5).

Replan budget is shared with tool failures via ``max_replan``. When the
budget is exhausted the caller falls back to a best-effort answer with an
explicit "计划失败" notice.
"""

from __future__ import annotations

from core.plan import InvalidPlanError, Plan, Step
from core.scratchpad import Scratchpad

REPLAN_SYSTEM = """你是重规划专家。原计划的某一步失败了。请基于已有成果和失败原因，为剩余目标生成新的计划。

要求：
1. 已完成的步骤不要重复。
2. 如果失败无法绕过，设计替代路径；确实无法继续时输出一个单步的纯推理计划（tool 为 null），基于已有信息尽力作答。
3. 只输出 JSON，格式与原计划相同：
{{"steps": [{{"id": 1, "description": "...", "tool": "工具名或null", "input_mapping": {{}}, "output_var": "变量名"}}]}}"""


class Replanner:
    """Budgets are separate because the failures are different: repeated
    malformed JSON must not consume the quota reserved for real step failures."""

    def __init__(self, llm, max_replan: int = 3, max_invalid_plan: int = 2) -> None:
        # llm: Planner-like callable(system, user) -> raw plan text
        self._llm = llm
        self._max_replan = max_replan
        self._max_invalid_plan = max_invalid_plan

    @property
    def budget_left(self) -> int:
        return self._max_replan

    @property
    def invalid_budget_left(self) -> int:
        return self._max_invalid_plan

    def replan(
        self,
        original: Plan,
        failed_step: Step,
        error: str,
        scratchpad: Scratchpad,
        remaining_goal: str,
    ) -> Plan:
        """Produce a new plan covering the remaining work.

        Raises BudgetExhaustedError when a budget is used up; InvalidPlanError
        when the model returned an unusable plan (after drawing on the
        invalid-plan budget).
        """
        if self._max_replan <= 0:
            raise BudgetExhaustedError("重规划次数已耗尽")

        done_lines = []
        for s in original.steps:
            if scratchpad.has(s.output_var):
                done_lines.append(f"✓ 步骤{s.id} {s.description} → {s.output_var}={scratchpad.get(s.output_var)!r}")
            elif s.id == failed_step.id:
                done_lines.append(f"✗ 步骤{s.id} {s.description} 失败：{error}")

        user = (
            f"原始目标：{remaining_goal}\n\n"
            f"当前进度：\n" + ("\n".join(done_lines) or "（尚无已完成步骤）") + "\n\n"
            f"失败步骤：步骤{failed_step.id}「{failed_step.description}」，失败原因：{error}\n\n"
            "请生成从当前状态继续的新计划。"
        )
        raw = self._llm(REPLAN_SYSTEM, user)
        from core.planner import extract_json  # noqa: PLC0415

        try:
            new_plan = Plan.parse(extract_json(raw))
        except InvalidPlanError:
            self._max_invalid_plan -= 1
            if self._max_invalid_plan < 0:
                raise BudgetExhaustedError("重规划始终无法产出有效计划") from None
            raise

        # Only a usable replan consumes the tool-failure budget.
        self._max_replan -= 1
        return new_plan


class BudgetExhaustedError(Exception):
    """No replanning budget left; caller should fall back to best-effort answer."""
