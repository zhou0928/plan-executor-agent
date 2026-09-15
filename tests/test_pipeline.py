import pytest
from types import SimpleNamespace

from core.executor import Executor
from core.plan import Plan
from core.planner import DEFAULT_PLANNING_PROMPT, Planner, extract_json, tools_description
from core.replanner import BudgetExhaustedError, Replanner
from core.scratchpad import Scratchpad
from core.subagent_pool import LocalStepExecutor, StepExecutionError


# ---------- helpers ----------

def fake_tool(name: str, description: str = "", params: tuple[str, ...] = ()) -> SimpleNamespace:
    """Minimal stand-in for the SDK ToolEntity the strategy hands to the planner."""
    return SimpleNamespace(
        identity=SimpleNamespace(name=name),
        description=SimpleNamespace(llm=description),
        parameters=[SimpleNamespace(name=p) for p in params],
    )

def make_plan(raw: str) -> Plan:
    return Plan.parse(raw)


PLAN_2STEPS = """
{"steps": [
  {"id": 1, "description": "查天气", "tool": "weather", "input_mapping": {"city": "{{city}}"}, "output_var": "w"},
  {"id": 2, "description": "给建议", "tool": null, "input_mapping": {"w": "{{w}}"}, "output_var": "advice"}
]}
"""

PLAN_INDEPENDENT = """
{"steps": [
  {"id": 1, "description": "甲", "tool": "ta", "input_mapping": {}, "output_var": "a"},
  {"id": 2, "description": "乙", "tool": "tb", "input_mapping": {}, "output_var": "b"}
]}
"""


class RecordingPool:
    """Fake SubAgentPool: scripted results, optional failure on step description."""

    def __init__(self, results: dict[str, str], fail_on: str | None = None):
        self.results = results
        self.fail_on = fail_on
        self.calls: list[tuple[int, Scratchpad]] = []
        self.stream_flags: list[tuple[int, bool]] = []

    def execute(self, step, scratchpad, stream=False):
        self.calls.append((step.id, scratchpad))
        self.stream_flags.append((step.id, stream))
        if self.fail_on and self.fail_on in step.description:
            raise StepExecutionError(f"tool {step.tool} exploded")
        return self.results.get(step.description, "ok")


# ---------- Planner ----------

class TestPlanner:
    def test_plan_happy_path(self):
        reply = '{"steps": [{"id": 1, "description": "d", "tool": null, "input_mapping": {}, "output_var": "o"}]}'
        p = Planner(llm=lambda s, u: reply)
        plan = p.plan("目标", tools=[], instruction="")
        assert len(plan.steps) == 1

    def test_plan_with_markdown_fence(self):
        reply = '好的，计划如下：\n```json\n{"steps": [{"id": 1, "description": "d", "tool": null, "input_mapping": {}, "output_var": "o"}]}\n```'
        p = Planner(llm=lambda s, u: reply)
        assert len(p.plan("g", [], "").steps) == 1

    def test_plan_prompt_contains_tools_and_goal(self):
        captured = {}

        def llm(system, user):
            captured["system"], captured["user"] = system, user
            return '{"steps": [{"id": 1, "description": "d", "tool": null, "input_mapping": {}, "output_var": "o"}]}'

        Planner(llm=llm).plan("北京天气", tools=[fake_tool("weather", "查天气", ("city",))], instruction="用中文")
        assert "weather" in captured["system"]
        assert "北京天气" in captured["user"]
        assert "用中文" in captured["user"]

    def test_plan_invalid_reply_raises(self):
        p = Planner(llm=lambda s, u: "我无法规划")
        with pytest.raises(Exception):
            p.plan("g", [], "")

    def test_extract_json_fenced(self):
        assert extract_json('```json\n{"a":1}\n```') == '{"a":1}'

    def test_extract_json_embedded(self):
        assert extract_json('前缀 {"a":1} 后缀') == '{"a":1}'

    def test_tools_description_empty(self):
        assert "无可用工具" in tools_description([])

    def test_tools_description_lists(self):
        d = tools_description([fake_tool("t1", "desc", ("a", "b"))])
        assert "t1" in d and "desc" in d and "a" in d

    def test_default_prompt_shows_unescaped_json_example(self):
        # plan() substitutes with .replace(), so brace escaping would reach the
        # model verbatim: a doubled {{"steps" ...}} example taught it to emit
        # non-string input_mapping values, which Plan.parse rejects.
        assert '{"steps": [{"id": 1' in DEFAULT_PLANNING_PROMPT
        assert '"{{变量}}"' in DEFAULT_PLANNING_PROMPT
        assert '{{"steps"' not in DEFAULT_PLANNING_PROMPT


# ---------- Executor ----------

class TestExecutor:
    def test_runs_all_steps_in_order(self):
        pool = RecordingPool({"查天气": "晴", "给建议": "穿短袖"})
        events = []
        outcome = Executor(pool, on_progress=lambda sid, d, t, desc: events.append((sid, d, t))).run(
            make_plan(PLAN_2STEPS), Scratchpad({"city": "北京"})
        )
        assert outcome.completed
        assert [c[0] for c in pool.calls] == [1, 2]
        assert events == [(1, 1, 2), (2, 2, 2)]

    def test_results_written_to_scratchpad(self):
        pool = RecordingPool({"查天气": "晴", "给建议": "穿短袖"})
        sp = Scratchpad({"city": "北京"})
        Executor(pool).run(make_plan(PLAN_2STEPS), sp)
        assert sp.get("w") == "晴"
        assert sp.get("advice") == "穿短袖"

    def test_failure_stops_and_records_error(self):
        pool = RecordingPool({}, fail_on="查天气")
        sp = Scratchpad({"city": "北京"})
        outcome = Executor(pool).run(make_plan(PLAN_2STEPS), sp)
        assert outcome.failed_step is not None
        assert outcome.failed_step.id == 1
        assert "exploded" in outcome.error
        assert sp.get("w_error") is not None
        assert not sp.has("w")

    def test_start_index_resumes(self):
        pool = RecordingPool({"给建议": "穿短袖"})
        sp = Scratchpad({"w": "晴"})
        outcome = Executor(pool).run(make_plan(PLAN_2STEPS), sp, start_index=1)
        assert outcome.completed
        assert [c[0] for c in pool.calls] == [2]  # step 1 not re-run

    def test_only_the_last_step_is_asked_to_stream(self):
        pool = RecordingPool({"查天气": "晴", "给建议": "穿短袖"})
        Executor(pool).run(make_plan(PLAN_2STEPS), Scratchpad({"city": "北京"}))
        assert pool.stream_flags == [(1, False), (2, True)]


class TestParallelExecutor:
    def test_independent_steps_all_run(self):
        pool = RecordingPool({"甲": "a", "乙": "b"})
        sp = Scratchpad()
        outcome = Executor(pool, max_parallel=3).run(make_plan(PLAN_INDEPENDENT), sp)
        assert outcome.completed
        assert sorted(c[0] for c in pool.calls) == [1, 2]
        assert sp.get("a") == "a"
        assert sp.get("b") == "b"

    def test_dependent_step_waits_for_its_input(self):
        pool = RecordingPool({"查天气": "晴", "给建议": "穿短袖"})
        sp = Scratchpad({"city": "北京"})
        outcome = Executor(pool, max_parallel=4).run(make_plan(PLAN_2STEPS), sp)
        assert outcome.completed
        assert sp.get("advice") == "穿短袖", "step 2 must see step 1's output"

    def test_parallel_failure_is_reported(self):
        pool = RecordingPool({}, fail_on="查天气")
        sp = Scratchpad({"city": "北京"})
        outcome = Executor(pool, max_parallel=4).run(make_plan(PLAN_2STEPS), sp)
        assert outcome.failed_step is not None
        assert outcome.failed_step.id == 1
        assert sp.get("w_error") is not None

    def test_progress_is_reported_for_every_step(self):
        pool = RecordingPool({"甲": "a", "乙": "b"})
        events = []
        Executor(
            pool,
            on_progress=lambda sid, d, t, desc: events.append((sid, d, t)),
            max_parallel=3,
        ).run(make_plan(PLAN_INDEPENDENT), Scratchpad())
        assert sorted(events) == [(1, 1, 2), (2, 2, 2)]


# ---------- LocalStepExecutor ----------

class TestLocalStepExecutor:
    def test_reasoning_step_uses_llm(self):
        pool = LocalStepExecutor(llm=lambda s, u: "推理结果", tool_invoker=lambda t, p: pytest.fail("should not call tool"))
        step = make_plan('{"steps": [{"id": 1, "description": "想", "tool": null, "input_mapping": {"x": "{{v}}"}, "output_var": "o"}]}').steps[0]
        assert pool.execute(step, Scratchpad({"v": "1"})) == "推理结果"

    def test_tool_step_renders_params(self):
        captured = {}
        pool = LocalStepExecutor(
            llm=lambda s, u: pytest.fail("should not call llm"),
            tool_invoker=lambda t, p: (captured.update(tool=t, params=p), "工具结果")[1],
        )
        step = make_plan('{"steps": [{"id": 1, "description": "查", "tool": "weather", "input_mapping": {"city": "{{c}}"}, "output_var": "o"}]}').steps[0]
        assert pool.execute(step, Scratchpad({"c": "北京"})) == "工具结果"
        assert captured == {"tool": "weather", "params": {"city": "北京"}}

    def test_missing_variable_raises(self):
        from core.scratchpad import MissingVariableError
        pool = LocalStepExecutor(llm=lambda s, u: "x", tool_invoker=lambda t, p: "y")
        step = make_plan('{"steps": [{"id": 1, "description": "查", "tool": "weather", "input_mapping": {"city": "{{nope}}"}, "output_var": "o"}]}').steps[0]
        with pytest.raises(MissingVariableError):
            pool.execute(step, Scratchpad())

    def test_reasoning_llm_failure_normalised_to_step_error(self):
        def boom(system, user):
            raise RuntimeError("model api down")

        pool = LocalStepExecutor(llm=boom, tool_invoker=lambda t, p: pytest.fail("should not call tool"))
        step = make_plan('{"steps": [{"id": 1, "description": "想", "tool": null, "input_mapping": {"x": "{{v}}"}, "output_var": "o"}]}').steps[0]
        with pytest.raises(StepExecutionError, match=r"推理步骤失败.*model api down"):
            pool.execute(step, Scratchpad({"v": "1"}))

    def test_reasoning_stream_failure_normalised_to_step_error(self):
        def bad_stream(system, user):
            raise RuntimeError("stream died")

        pool = LocalStepExecutor(
            llm=lambda s, u: pytest.fail("should not use plain llm"),
            tool_invoker=lambda t, p: pytest.fail("should not call tool"),
            llm_stream=bad_stream,
            on_token=lambda token: None,
        )
        step = make_plan('{"steps": [{"id": 1, "description": "想", "tool": null, "input_mapping": {}, "output_var": "o"}]}').steps[0]
        with pytest.raises(StepExecutionError, match=r"推理步骤失败.*stream died"):
            pool.execute(step, Scratchpad(), stream=True)


# ---------- Replanner ----------

class TestReplanner:
    REPLAN_REPLY = '{"steps": [{"id": 10, "description": "备用方案", "tool": null, "input_mapping": {"w": "{{w}}"}, "output_var": "advice"}]}'

    def test_replan_consumes_budget(self):
        r = Replanner(llm=lambda s, u: self.REPLAN_REPLY, max_replan=2)
        plan = make_plan(PLAN_2STEPS)
        sp = Scratchpad({"w": "晴"})
        failed = plan.steps[0]
        new = r.replan(plan, failed, "boom", sp, "目标")
        assert r.budget_left == 1
        assert new.steps[0].id == 10

    def test_replan_prompt_contains_failure_context(self):
        captured = {}

        def llm(system, user):
            captured["user"] = user
            return TestReplanner.REPLAN_REPLY

        plan = make_plan(PLAN_2STEPS)
        sp = Scratchpad({"w": "晴"})
        Replanner(llm=llm, max_replan=1).replan(plan, plan.steps[0], "超时", sp, "穿什么")
        assert "超时" in captured["user"]
        assert "步骤1" in captured["user"]

    def test_budget_exhausted(self):
        r = Replanner(llm=lambda s, u: TestReplanner.REPLAN_REPLY, max_replan=0)
        plan = make_plan(PLAN_2STEPS)
        with pytest.raises(BudgetExhaustedError):
            r.replan(plan, plan.steps[0], "e", Scratchpad(), "g")

    def test_budget_runs_out_after_repeated_use(self):
        r = Replanner(llm=lambda s, u: TestReplanner.REPLAN_REPLY, max_replan=1)
        plan = make_plan(PLAN_2STEPS)
        r.replan(plan, plan.steps[0], "e", Scratchpad(), "g")
        with pytest.raises(BudgetExhaustedError):
            r.replan(plan, plan.steps[0], "e", Scratchpad(), "g")
