"""ExecutionMetadata shape, and the split replan budgets."""

import pytest

from core.execution_metadata import ExecutionMetadata
from core.plan import InvalidPlanError, Plan
from core.replanner import BudgetExhaustedError, Replanner
from core.scratchpad import Scratchpad

VALID_PLAN = '{"steps": [{"id": 1, "description": "答", "tool": null, "input_mapping": {}, "output_var": "out"}]}'
ORIGINAL_PLAN = '{"steps": [{"id": 1, "description": "查", "tool": "t", "input_mapping": {}, "output_var": "v"}]}'


class FakeUsage:
    total_price = "0.5"
    currency = "USD"
    total_tokens = 120
    prompt_tokens = 100
    prompt_unit_price = "0.001"
    prompt_price_unit = "1000"
    prompt_price = "0.1"
    completion_tokens = 20
    completion_unit_price = "0.002"
    completion_price_unit = "1000"
    completion_price = "0.4"
    latency = 1.5


class TestExecutionMetadata:
    def test_none_usage_yields_defaults(self):
        meta = ExecutionMetadata.from_llm_usage(None).to_dict()
        assert meta["total_tokens"] == 0
        assert meta["currency"] == ""
        assert meta["total_price"] == 0.0

    def test_maps_and_casts_sdk_usage(self):
        meta = ExecutionMetadata.from_llm_usage(FakeUsage()).to_dict()
        assert meta["total_tokens"] == 120
        assert meta["prompt_tokens"] == 100
        assert meta["completion_tokens"] == 20
        assert meta["total_price"] == 0.5
        assert meta["completion_price"] == 0.4
        assert meta["currency"] == "USD"
        assert meta["latency"] == 1.5


class TestReplanBudgets:
    def test_invalid_plan_draws_only_on_the_invalid_budget(self):
        rp = Replanner(llm=lambda s, u: "not json at all", max_replan=3, max_invalid_plan=1)
        original = Plan.parse(ORIGINAL_PLAN)

        with pytest.raises(InvalidPlanError):
            rp.replan(original, original.steps[0], "boom", Scratchpad(), "goal")

        assert rp.budget_left == 3, "tool-failure budget must be untouched"
        assert rp.invalid_budget_left == 0

        with pytest.raises(BudgetExhaustedError):
            rp.replan(original, original.steps[0], "boom", Scratchpad(), "goal")

    def test_usable_replan_consumes_the_tool_budget(self):
        rp = Replanner(llm=lambda s, u: VALID_PLAN, max_replan=2, max_invalid_plan=1)
        original = Plan.parse(ORIGINAL_PLAN)

        rp.replan(original, original.steps[0], "boom", Scratchpad(), "goal")

        assert rp.budget_left == 1
        assert rp.invalid_budget_left == 1

    def test_exhausted_tool_budget_stops_before_calling_the_model(self):
        def explode(system, user):
            raise AssertionError("model must not be called with no budget left")

        rp = Replanner(llm=explode, max_replan=0, max_invalid_plan=5)
        original = Plan.parse(ORIGINAL_PLAN)

        with pytest.raises(BudgetExhaustedError):
            rp.replan(original, original.steps[0], "boom", Scratchpad(), "goal")
