"""Planner: turns a goal + tool list into a structured Plan (design doc §3.1).

The LLM caller is injected as a callable so unit tests can mock it:
    llm(system: str, user: str) -> str
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from core.plan import Plan

LLMCaller = Callable[[str, str], str]

DEFAULT_PLANNING_PROMPT = """你是任务规划专家。把用户目标拆解为可直接执行的步骤，最终产出一个交付物。

可用工具：
{tools}

要求：
1. 最多 {max_steps} 步，越少越好；每一步都要向最终交付物推进。
2. 禁止元步骤：不要写"解析需求/分析信息/判断完整度/确定方案/设置默认值"这类步骤。
   信息缺失时，在同一步内用合理默认值直接产出，并在 description 里注明假设。
3. 最后一步必须产出用户要的东西本身（脚本/答案/文档），而不是对它的描述。
4. 优先使用工具获取事实；纯总结、推理类步骤把 tool 设为 null。
5. input_mapping 的值必须是字符串：用 "{{变量名}}" 引用前面步骤的 output_var 或初始变量，
   例如 {"city": "{{query}}"}。
6. 只输出 JSON，不要输出任何其他文字。格式：
{"steps": [{"id": 1, "description": "...", "tool": "工具名或null", "input_mapping": {"参数": "{{变量}}"}, "output_var": "变量名"}]}"""


def tools_description(tools: list[Any]) -> str:
    """Render the tool manifest injected into the planning prompt.

    Tools are SDK ToolEntity objects (the strategy layer owns that import, so
    this module stays dependency-free); fields are read by attribute, not by
    dict key: identity.name, description.llm, parameters[].name.
    """
    if not tools:
        return "（无可用工具，请全部使用纯推理步骤）"
    lines = []
    for t in tools:
        params = ", ".join(p.name for p in (t.parameters or [])) or "无参数"
        desc = t.description.llm if t.description else ""
        lines.append(f"- {t.identity.name}: {desc}（参数: {params}）")
    return "\n".join(lines)


def extract_json(text: str) -> str:
    """Extract the first JSON object from an LLM reply that may contain
    markdown fences, thinking blocks, or chatter."""
    # Thinking models (qwen3 etc.): the final answer lives after the last
    # closed </think>; everything before it is reasoning.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    # Strip any remaining (possibly nested) closed thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return brace.group(0)
    return text


class Planner:
    def __init__(
        self,
        llm: LLMCaller,
        planning_prompt: str | None = None,
        max_steps: int = 20,
    ) -> None:
        self._llm = llm
        self._template = planning_prompt or DEFAULT_PLANNING_PROMPT
        self._max_steps = max_steps

    def plan(
        self,
        goal: str,
        tools: list[Any],
        instruction: str = "",
        initial_vars: list[str] | None = None,
    ) -> Plan:
        # .format() would crash on literal braces in a custom planning_prompt
        # (e.g. a JSON example {"steps": ...}); .replace() only touches the
        # two placeholders and leaves everything else untouched.
        system = (
            self._template.replace("{tools}", tools_description(tools))
            .replace("{max_steps}", str(self._max_steps))
        )
        user_parts = [f"目标：{goal}"]
        if instruction:
            user_parts.append(f"额外指令：{instruction}")
        if initial_vars:
            user_parts.append(f"初始可用变量：{', '.join(initial_vars)}")
        raw = self._llm(system, "\n\n".join(user_parts))
        return Plan.parse(extract_json(raw), max_steps=self._max_steps)
