"""Runnable check for the tool-wiring contract, run with the SDK present.

Lives outside pytest's collection (no ``test_`` prefix) because it needs
``dify_plugin`` and the dash-named strategy module loaded by path:

    python3 tests/verify_tools_contract.py

Guards the three bugs found end-to-end:
  1. tools arrive as plain dicts   -> ``_prepare_tools`` must yield ToolEntity
  2. planner reads tools as dicts  -> ``tools_description`` must not call .get
  3. invocation must forward provider/provider_type, the tool's runtime
     settings and its credential (only ``parameters`` reaches the provider)
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dify_plugin.entities.agent import AgentInvokeMessage  # noqa: E402
from dify_plugin.entities.tool import ToolInvokeMessage, ToolProviderType  # noqa: E402
from dify_plugin.interfaces.agent import ToolEntity  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "pe_strategy", ROOT / "strategies" / "plan-executor-agent.py"
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

I18N = {"en_us": "x", "zh_hans": "x"}

# Exactly what api/core/workflow/nodes/agent/runtime_support.py sends:
# tool_runtime.entity.model_dump(mode="json") + runtime_parameters +
# credential_id + provider_type.
TOOL_DICT = {
    "identity": {
        "author": "",
        "name": "webscraper",
        "label": I18N,
        "provider": "webscraper",
    },
    "parameters": [
        {
            "name": "url",
            "label": I18N,
            "human_description": I18N,
            "type": "string",
            "form": "llm",
            "required": True,
        }
    ],
    "description": {"human": I18N, "llm": "一个用于爬取网页的工具。"},
    "output_schema": {},
    "credential_id": None,
    "credential_type": None,
    "has_runtime_parameters": True,
    "provider_type": "builtin",
    "runtime_parameters": {"user_agent": "Mozilla/5.0", "generate_summary": "false"},
}


def _text_result() -> list:
    return [
        ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.TEXT,
            message=ToolInvokeMessage.TextMessage(text="ok"),
        )
    ]


def _blob_result() -> list:
    return [
        ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.BLOB,
            message=ToolInvokeMessage.BlobMessage(blob=b"x" * 2048),
            meta={"filename": "map.png", "mime_type": "image/png"},
        )
    ]


def _image_link_result() -> list:
    return [
        ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.IMAGE_LINK,
            message=ToolInvokeMessage.TextMessage(text="https://host/files/tools/abc.png"),
            meta={"filename": "map.png", "mime_type": "image/png"},
        )
    ]


def _huge_text_result(size: int = 20000) -> list:
    return [
        ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.TEXT,
            message=ToolInvokeMessage.TextMessage(text="A" * size),
        )
    ]


def main() -> int:
    from core.planner import tools_description

    strat = object.__new__(mod.PlanExecutorAgentAgentStrategy)
    captured: dict = {}

    def invoke(**kwargs):
        captured.update(kwargs)
        return _text_result()

    strat.response_type = AgentInvokeMessage  # set by AgentStrategy.__init__, skipped via __new__
    strat._session = SimpleNamespace(tool=SimpleNamespace(invoke=invoke))

    # 1. dict -> ToolEntity (bug 1: a raw dict made the SDK drop every tool)
    tools = strat._prepare_tools([TOOL_DICT])
    assert isinstance(tools[0], ToolEntity), f"not a ToolEntity: {type(tools[0])}"

    # 2. the planner renders the manifest by attribute (bug 2: it did t.get)
    manifest = tools_description(tools)
    assert "webscraper" in manifest and "爬取网页" in manifest, manifest
    assert "url" in manifest, manifest
    # the allowlist filters on identity.name (bug: ToolEntity has no .name)
    assert mod.filter_allowed_tools(tools, ["webscraper"]) == tools
    assert mod.filter_allowed_tools(tools, ["nope"]) == []

    # 3. invocation forwards what the plan cannot supply (bug 3)
    out = strat._invoke_tool("webscraper", {"url": "https://example.com"}, tools)
    assert captured["provider"] == "webscraper", captured
    assert captured["provider_type"] == ToolProviderType.BUILT_IN, captured
    assert captured["tool_name"] == "webscraper", captured
    assert captured["credential_id"] is None, captured
    params = captured["parameters"]
    assert params["url"] == "https://example.com", params
    assert params["user_agent"] == "Mozilla/5.0", params
    assert params["generate_summary"] == "false", params
    assert out == "ok", out

    # plan params win over the tool's configured runtime values
    strat._invoke_tool("webscraper", {"user_agent": "plan-agent"}, tools)
    assert captured["parameters"]["user_agent"] == "plan-agent", captured

    # unknown tool name still fails the step instead of silently skipping
    try:
        strat._invoke_tool("ghost", {}, tools)
    except mod.StepExecutionError as e:
        assert "ghost" in str(e)
    else:
        raise AssertionError("unknown tool did not raise StepExecutionError")

    # a blob tool result must never be repr()'d into the answer (base64 blowup),
    # and it must reach the node output as a real file
    artifacts: list = []
    strat._session = SimpleNamespace(tool=SimpleNamespace(invoke=lambda **kw: _blob_result()))
    assert strat._invoke_tool(
        "webscraper", {}, tools, on_artifact=artifacts.append
    ) == "[blob] map.png (2 KB)"
    assert len(artifacts) == 1, artifacts
    assert artifacts[0].type == AgentInvokeMessage.MessageType.BLOB, artifacts[0].type
    assert artifacts[0].message.blob == b"x" * 2048

    # link-ish results are forwarded too (image inline, other links as links)
    artifacts.clear()
    strat._session = SimpleNamespace(tool=SimpleNamespace(invoke=lambda **kw: _image_link_result()))
    strat._invoke_tool("webscraper", {}, tools, on_artifact=artifacts.append)
    assert len(artifacts) == 1, artifacts
    assert artifacts[0].type == AgentInvokeMessage.MessageType.IMAGE, artifacts[0].type
    assert artifacts[0].message.text.endswith("abc.png"), artifacts[0].message

    # oversized output is clipped with both ends kept
    strat._session = SimpleNamespace(tool=SimpleNamespace(invoke=lambda **kw: _huge_text_result()))
    clipped = strat._invoke_tool("webscraper", {}, tools, limit=1000)
    assert "已省略" in clipped, clipped[:80]
    assert len(clipped) < 1200, len(clipped)
    assert clipped.startswith("A") and clipped.endswith("A")
    full = strat._invoke_tool("webscraper", {}, tools, limit=None)
    assert len(full) == 20000, len(full)

    # history rides on the planning/final-answer funnel only, and both share one
    # usage sink so the node's usage panel still sees every call
    seen: dict = {}

    def capture(**kwargs):
        seen["messages"] = kwargs["prompt_messages"]
        return SimpleNamespace(message=SimpleNamespace(content="ok"), usage=None)

    strat._session = SimpleNamespace(model=SimpleNamespace(llm=SimpleNamespace(invoke=capture)))
    usage: dict = {"usage": None}
    lock = threading.Lock()
    history = [{"role": "user", "content": "上一轮问题"}]
    plan_llm = strat._make_llm_caller({}, history=history, usage=usage, usage_lock=lock)
    step_llm = strat._make_llm_caller({}, usage=usage, usage_lock=lock)
    plan_llm("sys", "现在的问题")
    plan_msgs = seen["messages"]
    step_llm("sys", "现在的问题")
    step_msgs = seen["messages"]
    assert any(getattr(m, "content", None) == "上一轮问题" for m in plan_msgs), plan_msgs
    assert not any(getattr(m, "content", None) == "上一轮问题" for m in step_msgs), step_msgs
    assert plan_llm.usage is step_llm.usage is usage

    print("verify_tools_contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
