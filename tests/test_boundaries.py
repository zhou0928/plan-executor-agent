"""Boundary regression tests for the plan-executor review fixes.

Two tiers, same file:
- core tests run everywhere (dify_plugin not installed locally);
- strategy tests are gated by ``pytest.importorskip("dify_plugin")`` and load
  the dash-named strategy module through importlib (cannot be a plain import).
"""

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import dify_plugin  # noqa: F401 -- ponytail: import at collection (single-threaded) so its gevent patch never races executor-test daemon threads -> CPython 3.14 importlib _ModuleLock stays intact
except ImportError:
    pass  # no SDK locally: strategy tests skip via _load_strategy()

from core.executor import Executor
from core.params import to_int
from core.plan import InvalidPlanError, Plan
from core.planner import Planner
from core.scratchpad import Scratchpad, validate_plan

STRATEGY_PATH = Path(__file__).resolve().parent.parent / "strategies" / "plan-executor-agent.py"


def make_plan(raw: str) -> Plan:
    return Plan.parse(raw)


PLAN_1STEP = """
{"steps": [
  {"id": 1, "description": "查天气", "tool": "weather", "input_mapping": {"city": "{{city}}"}, "output_var": "w"}
]}
"""


class HangingPool:
    """SubAgentPool whose execute() never returns in time (simulated hang)."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def execute(self, step, scratchpad, stream=False):
        self.calls.append(step.id)
        time.sleep(60)
        return "late"


class PromptCapturePool:
    """Instant pool whose result is captured by the caller."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list = []

    def execute(self, step, scratchpad, stream=False):
        self.calls.append(step)
        return self.result


# ---------- Planner: literal braces in custom planning_prompt ----------

class TestPlannerLiteralBraces:
    def test_custom_prompt_with_literal_braces_survives(self):
        # .format() would crash on these braces (KeyError); .replace() must not.
        prompt = (
            "根据目标规划。输出 JSON 示例：{\"steps\": [{\"id\": 1}]}。"
            "可用工具：\n{tools}\n最多 {max_steps} 步。"
        )
        captured: dict[str, str] = {}

        def llm(system, user):
            captured["system"] = system
            return '{"steps": [{"id": 1, "description": "d", "tool": null, "input_mapping": {}, "output_var": "o"}]}'

        p = Planner(llm=llm, planning_prompt=prompt, max_steps=3)
        plan = p.plan(
            "目标",
            tools=[
                SimpleNamespace(
                    identity=SimpleNamespace(name="weather"),
                    description=SimpleNamespace(llm="查"),
                    parameters=[],
                )
            ],
        )
        assert len(plan.steps) == 1
        assert '{"steps": [{"id": 1}]}' in captured["system"], "literal braces must survive"
        assert "weather" in captured["system"], "{tools} must still be substituted"
        assert "3" in captured["system"], "{max_steps} must still be substituted"

    def test_validate_plan_flags_forward_reference(self):
        plan = make_plan(
            '{"steps": [{"id": 1, "description": "d", "tool": null, "input_mapping": {"x": "{{nope}}"}, "output_var": "o"}]}'
        )
        with pytest.raises(InvalidPlanError):
            validate_plan(plan, set())


# ---------- Executor: step timeout (sequential + parallel) ----------

class TestExecutorTimeout:
    def test_sequential_timeout_records_failure(self):
        sp = Scratchpad({"city": "北京"})
        outcome = Executor(HangingPool(), step_timeout=0.1).run(make_plan(PLAN_1STEP), sp)
        assert outcome.failed_step is not None
        assert outcome.failed_step.id == 1
        assert "超时" in outcome.error
        assert sp.get("w_error") is not None
        assert not sp.has("w")

    def test_parallel_timeout_records_failure_and_pool_exits(self):
        sp = Scratchpad()
        plan = make_plan(
            '{"steps": ['
            '{"id": 1, "description": "甲", "tool": "ta", "input_mapping": {}, "output_var": "a"},'
            '{"id": 2, "description": "乙", "tool": "tb", "input_mapping": {}, "output_var": "b"}'
            "]}"
        )
        outcome = Executor(HangingPool(), max_parallel=4, step_timeout=0.1).run(plan, sp)
        assert outcome.failed_step is not None
        assert "超时" in outcome.error
        assert sp.get("a_error") is not None or sp.get("b_error") is not None

    def test_fast_step_within_timeout_completes(self):
        sp = Scratchpad({"city": "北京"})
        outcome = Executor(PromptCapturePool("ok"), step_timeout=0.1).run(make_plan(PLAN_1STEP), sp)
        assert outcome.completed
        assert sp.get("w") == "ok"


# ---------- Executor: wall-clock deadline (daemon kills the run past its cap) ----------

class TestExecutorDeadline:
    def test_past_deadline_skips_every_step(self):
        sp = Scratchpad({"city": "北京"})
        pool = PromptCapturePool("ok")
        outcome = Executor(pool, deadline=time.monotonic() - 1).run(make_plan(PLAN_1STEP), sp)
        assert outcome.timed_out
        assert not outcome.completed
        assert not sp.has("w")
        assert pool.calls == []  # no work started at all

    def test_future_deadline_runs_normally(self):
        sp = Scratchpad({"city": "北京"})
        outcome = Executor(PromptCapturePool("ok"), deadline=time.monotonic() + 60).run(
            make_plan(PLAN_1STEP), sp
        )
        assert outcome.completed
        assert not outcome.timed_out
        assert sp.get("w") == "ok"

    def test_step_timeout_capped_by_remaining_budget(self):
        sp = Scratchpad({"city": "北京"})
        started = time.monotonic()
        outcome = Executor(HangingPool(), step_timeout=30, deadline=started + 0.15).run(
            make_plan(PLAN_1STEP), sp
        )
        elapsed = time.monotonic() - started
        assert outcome.failed_step is not None, "a step that outlives the budget must fail, not hang"
        assert elapsed < 5, f"budget must cap the step timeout, took {elapsed:.1f}s"

    def test_parallel_past_deadline_reports_timed_out(self):
        sp = Scratchpad()
        plan = make_plan(
            '{"steps": ['
            '{"id": 1, "description": "甲", "tool": "ta", "input_mapping": {}, "output_var": "a"},'
            '{"id": 2, "description": "乙", "tool": "tb", "input_mapping": {}, "output_var": "b"}'
            "]}"
        )
        outcome = Executor(
            PromptCapturePool("ok"), max_parallel=4, deadline=time.monotonic() - 1
        ).run(plan, sp)
        assert outcome.timed_out
        assert not sp.has("a") and not sp.has("b")


# ---------- to_int ----------

class TestToInt:
    def test_junk_falls_back_to_default(self):
        assert to_int(None, 20) == 20
        assert to_int("", 20) == 20
        assert to_int("abc", 20) == 20
        assert to_int("3.7", 20) == 20  # int("3.7") raises ValueError

    def test_numbers_and_numeric_strings_parse(self):
        assert to_int(5, 20) == 5
        assert to_int("5", 20) == 5
        assert to_int(3.7, 20) == 3  # int(float) truncates


# ---------- strategy-level (needs dify_plugin; skips locally) ----------

def _load_strategy():
    dify_plugin = pytest.importorskip("dify_plugin")  # noqa: F841 - proves SDK present
    spec = importlib.util.spec_from_file_location("plan_executor_agent_strategy", STRATEGY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INVALID_PLAN = (
    '{"steps": [{"id": 1, "description": "d", "tool": null,'
    ' "input_mapping": {"x": "{{nope}}"}, "output_var": "o"}]}'
)
GOOD_PLAN = (
    '{"steps": [{"id": 1, "description": "d", "tool": null,'
    ' "input_mapping": {}, "output_var": "o"}]}'
)


class TestRenderToolMessage:
    def test_variants_render_without_repr(self):
        mod = _load_strategy()
        TYPE = mod.ToolInvokeMessage.MessageType
        cases = [
            # (message, expected) -- the SDK wrapper: type on the outside,
            # payload inside .message (text/json_object/blob)
            (SimpleNamespace(type=TYPE.TEXT, message=SimpleNamespace(text="你好")), "你好"),
            (SimpleNamespace(type=TYPE.TEXT, message=SimpleNamespace(text=None)), ""),
            (
                SimpleNamespace(type=TYPE.JSON, message=SimpleNamespace(json_object={"a": 1})),
                '{"a": 1}',
            ),
            (SimpleNamespace(type=TYPE.JSON, message=SimpleNamespace(json_object=None)), ""),
            # non-TEXT/JSON: terse [label] meta, never repr() (could be base64)
            (
                SimpleNamespace(
                    type=TYPE.BLOB,
                    message=SimpleNamespace(blob=b"x" * 2048),
                    meta={"filename": "map.png", "mime_type": "image/png"},
                ),
                "[blob] map.png (2 KB)",
            ),
            (SimpleNamespace(type=TYPE.IMAGE, message=None, meta=None), "[image]"),
            (SimpleNamespace(type=None, message=None), "[result]"),
        ]
        for msg, expected in cases:
            assert mod._render_tool_message(msg) == expected


class TestPlanWithValidation:
    def test_retries_then_succeeds(self):
        mod = _load_strategy()
        calls = {"n": 0}

        def llm(system, user):
            calls["n"] += 1
            return INVALID_PLAN if calls["n"] == 1 else GOOD_PLAN

        plan = mod._plan_with_validation(Planner(llm=llm), "目标", [], "", set(), budget=2)
        assert len(plan.steps) == 1
        assert calls["n"] == 2, "one invalid plan + one retry"

    def test_past_deadline_stops_retrying(self):
        mod = _load_strategy()
        calls = {"n": 0}

        def llm(system, user):
            calls["n"] += 1
            return INVALID_PLAN

        with pytest.raises(InvalidPlanError):
            mod._plan_with_validation(
                Planner(llm=llm),
                "目标",
                [],
                "",
                set(),
                budget=5,
                deadline=time.monotonic() - 1,
            )
        assert calls["n"] == 1, "first attempt always runs, none after the deadline"

    def test_exhausts_budget_and_raises(self):
        mod = _load_strategy()
        calls = {"n": 0}

        def llm(system, user):
            calls["n"] += 1
            return INVALID_PLAN

        with pytest.raises(InvalidPlanError):
            mod._plan_with_validation(Planner(llm=llm), "目标", [], "", set(), budget=2)
        assert calls["n"] == 3, "budget + 1 attempts before giving up"


class TestLLMCallerRetry:
    def test_retries_transient_then_succeeds(self):
        mod = _load_strategy()
        attempts = mod._LLM_RETRY_ATTEMPTS
        calls = {"n": 0}

        def flaky(system, user):
            calls["n"] += 1
            if calls["n"] < attempts:
                raise RuntimeError("429 Too Many Requests")
            return "ok"

        llm = mod._LLMCaller(invoke=flaky, stream=lambda s, u: iter([]), usage={})
        assert llm("s", "u") == "ok"
        assert calls["n"] == attempts, "fails until the last of N attempts, then succeeds"

    def test_gives_up_after_attempts(self):
        mod = _load_strategy()
        calls = {"n": 0}

        def always(system, user):
            calls["n"] += 1
            raise RuntimeError("boom")

        llm = mod._LLMCaller(invoke=always, stream=lambda s, u: iter([]), usage={})
        with pytest.raises(RuntimeError, match="boom"):
            llm("s", "u")
        assert calls["n"] == mod._LLM_RETRY_ATTEMPTS

    def test_stream_never_retried(self):
        mod = _load_strategy()
        calls = {"n": 0}

        def bad_stream(system, user):
            calls["n"] += 1
            raise RuntimeError("boom")

        llm = mod._LLMCaller(invoke=lambda s, u: "x", stream=bad_stream, usage={})
        with pytest.raises(RuntimeError, match="boom"):
            list(llm.stream("s", "u"))
        assert calls["n"] == 1, "stream path must not retry after partial output"


class TestCollectToolParts:
    def test_blank_messages_dropped(self):
        mod = _load_strategy()
        TYPE = mod.ToolInvokeMessage.MessageType
        msgs = [
            SimpleNamespace(type=TYPE.TEXT, message=SimpleNamespace(text="ok")),
            SimpleNamespace(type=TYPE.TEXT, message=SimpleNamespace(text="")),
            SimpleNamespace(type=TYPE.JSON, message=SimpleNamespace(json_object=None)),
        ]
        assert mod._collect_tool_parts(msgs) == ["ok"]

    def test_all_blank_is_failure_signal(self):
        mod = _load_strategy()
        TYPE = mod.ToolInvokeMessage.MessageType
        all_blank = [
            SimpleNamespace(type=TYPE.JSON, message=SimpleNamespace(json_object=None)),
            SimpleNamespace(type=TYPE.TEXT, message=SimpleNamespace(text="")),
        ]
        assert mod._collect_tool_parts(all_blank) == []


class TestInvokeToolNormalization:
    def test_tool_invoke_exception_becomes_step_error(self):
        mod = _load_strategy()

        def boom(**kwargs):
            raise RuntimeError("provider down")

        strat = object.__new__(mod.PlanExecutorAgentAgentStrategy)
        strat._session = SimpleNamespace(tool=SimpleNamespace(invoke=boom))
        tools = [
            SimpleNamespace(
                identity=SimpleNamespace(name="weather", provider=""),
                provider_type="builtin",
                runtime_parameters={},
                credential_id=None,
            )
        ]
        with pytest.raises(mod.StepExecutionError, match=r"工具 weather 调用失败.*provider down"):
            strat._invoke_tool("weather", {}, tools)


class TestFastPathNoTools:
    """No mounted tools -> answer directly through one streamed call.

    Regression: this path used to forward raw chunks, so a thinking model's
    ``<think>`` block was rendered into the answer (every other path runs the
    text through ThinkFilter), and it ignored the wall-clock budget entirely.
    """

    @staticmethod
    def _strategy(mod, chunks, seconds=480):
        strat = object.__new__(mod.PlanExecutorAgentAgentStrategy)
        strat.response_type = mod.AgentInvokeMessage

        def invoke(**kwargs):
            assert kwargs["stream"] is True
            for piece in chunks:
                yield SimpleNamespace(
                    delta=SimpleNamespace(message=SimpleNamespace(content=piece), usage=None)
                )

        strat._session = SimpleNamespace(model=SimpleNamespace(llm=SimpleNamespace(invoke=invoke)))
        return strat, {
            "model": {"provider": "p", "model": "m"},
            "query": "目标",
            "tools": [],
            "max_execution_seconds": seconds,
        }

    @staticmethod
    def _text(mod, msgs) -> str:
        TYPE = mod.AgentInvokeMessage.MessageType
        return "".join(m.message.text for m in msgs if m.type == TYPE.TEXT)

    def test_thinking_block_never_reaches_the_answer(self):
        mod = _load_strategy()
        # split across chunks on purpose: the tag arrives in fragments
        strat, params = self._strategy(mod, ["<thi", "nk>内心戏</thi", "nk>答案"])
        text = self._text(mod, list(strat._invoke(params)))
        assert text == "答案", text

    def test_past_deadline_stops_forwarding(self):
        mod = _load_strategy()

        def chunks():
            yield "A"
            yield "B"
            time.sleep(1.05)  # burns the whole 1s budget
            yield "C"  # forwarded, then the budget check breaks the loop
            yield "D"  # never reached

        strat, params = self._strategy(mod, chunks(), seconds=1)
        text = self._text(mod, list(strat._invoke(params)))
        assert "D" not in text, text
        assert "执行时间预算" in text, text