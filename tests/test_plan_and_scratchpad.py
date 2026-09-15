import pytest

from core.plan import InvalidPlanError, Plan
from core.scratchpad import MissingVariableError, Scratchpad, validate_plan

VALID = """
{"steps": [
  {"id": 1, "description": "查天气", "tool": "weather_query",
   "input_mapping": {"city": "{{city}}"}, "output_var": "weather"},
  {"id": 2, "description": "给建议", "tool": null,
   "input_mapping": {"w": "{{weather}}"}, "output_var": "advice"}
]}
"""


class TestPlanParse:
    def test_valid_plan(self):
        plan = Plan.parse(VALID)
        assert len(plan.steps) == 2
        assert plan.steps[0].tool == "weather_query"
        assert plan.steps[1].tool is None  # pure reasoning step
        assert plan.steps[1].input_mapping == {"w": "{{weather}}"}

    def test_parse_from_dict(self):
        plan = Plan.parse({"steps": [{"id": 1, "description": "d", "tool": None,
                                      "input_mapping": {}, "output_var": "o"}]})
        assert plan.steps[0].id == 1

    def test_invalid_json(self):
        with pytest.raises(InvalidPlanError, match="not valid JSON"):
            Plan.parse("{not json")

    def test_not_an_object(self):
        with pytest.raises(InvalidPlanError, match="JSON object"):
            Plan.parse("[1,2]")

    def test_missing_steps(self):
        with pytest.raises(InvalidPlanError, match="non-empty 'steps'"):
            Plan.parse({"foo": 1})

    def test_empty_steps(self):
        with pytest.raises(InvalidPlanError, match="non-empty 'steps'"):
            Plan.parse({"steps": []})

    def test_step_over_limit(self):
        steps = [{"id": i, "description": "d", "tool": None,
                  "input_mapping": {}, "output_var": f"o{i}"} for i in range(21)]
        with pytest.raises(InvalidPlanError, match="exceeding limit"):
            Plan.parse({"steps": steps}, max_steps=20)

    def test_duplicate_step_id(self):
        with pytest.raises(InvalidPlanError, match="duplicated"):
            Plan.parse({"steps": [
                {"id": 1, "description": "a", "tool": None, "input_mapping": {}, "output_var": "o1"},
                {"id": 1, "description": "b", "tool": None, "input_mapping": {}, "output_var": "o2"},
            ]})

    def test_duplicate_output_var(self):
        with pytest.raises(InvalidPlanError, match="duplicated"):
            Plan.parse({"steps": [
                {"id": 1, "description": "a", "tool": None, "input_mapping": {}, "output_var": "o"},
                {"id": 2, "description": "b", "tool": None, "input_mapping": {}, "output_var": "o"},
            ]})

    def test_bad_id_type(self):
        with pytest.raises(InvalidPlanError, match="integer"):
            Plan.parse({"steps": [{"id": "x", "description": "d", "tool": None,
                                   "input_mapping": {}, "output_var": "o"}]})

    def test_bad_output_var(self):
        with pytest.raises(InvalidPlanError, match="output_var"):
            Plan.parse({"steps": [{"id": 1, "description": "d", "tool": None,
                                   "input_mapping": {}, "output_var": ""}]})

    def test_bad_input_mapping(self):
        with pytest.raises(InvalidPlanError, match="input_mapping"):
            Plan.parse({"steps": [{"id": 1, "description": "d", "tool": None,
                                   "input_mapping": {"k": 1}, "output_var": "o"}]})

    @pytest.mark.parametrize("raw_tool", ["null", "None", "nil", ""])
    def test_string_null_tool_becomes_none(self, raw_tool):
        plan = Plan.parse({"steps": [{"id": 1, "description": "d", "tool": raw_tool,
                                      "input_mapping": {}, "output_var": "o"}]})
        assert plan.steps[0].tool is None

    def test_boolean_id_rejected(self):
        # bool is a subclass of int in Python; must be rejected explicitly
        with pytest.raises(InvalidPlanError, match="integer"):
            Plan.parse({"steps": [{"id": True, "description": "d", "tool": None,
                                   "input_mapping": {}, "output_var": "o"}]})


class TestScratchpad:
    def test_set_get_has(self):
        s = Scratchpad({"a": 1})
        assert s.get("a") == 1
        assert s.has("a")
        assert not s.has("b")
        assert s.get("b", "dflt") == "dflt"
        s.set("b", 2)
        assert s.get("b") == 2

    def test_render(self):
        s = Scratchpad({"city": "北京"})
        assert s.render("今天{{city}}天气如何") == "今天北京天气如何"

    def test_render_missing_raises(self):
        s = Scratchpad()
        with pytest.raises(MissingVariableError, match="not set"):
            s.render("{{nope}}")

    def test_render_mapping(self):
        s = Scratchpad({"w": "晴"})
        assert s.render_mapping({"a": "{{w}}", "b": "常量"}) == {"a": "晴", "b": "常量"}

    def test_template_with_spaces(self):
        s = Scratchpad({"v": "x"})
        assert s.render("{{ v }}") == "x"


class TestValidatePlan:
    def test_valid_chain(self):
        plan = Plan.parse(VALID)
        validate_plan(plan, initial_vars={"city"})

    def test_undefined_ref(self):
        # {{weather}} referenced by step 2 exists, but {{city}} does not
        plan = Plan.parse(VALID)
        with pytest.raises(InvalidPlanError, match="city"):
            validate_plan(plan, initial_vars=set())

    def test_forward_reference_rejected(self):
        raw = """
        {"steps": [
          {"id": 1, "description": "a", "tool": null,
           "input_mapping": {"x": "{{later}}"}, "output_var": "o1"},
          {"id": 2, "description": "b", "tool": null,
           "input_mapping": {}, "output_var": "later"}
        ]}
        """
        with pytest.raises(InvalidPlanError, match="forward"):
            validate_plan(Plan.parse(raw))

    def test_circular_dependency_rejected(self):
        raw = """
        {"steps": [
          {"id": 1, "description": "a", "tool": null,
           "input_mapping": {"x": "{{o2}}"}, "output_var": "o1"},
          {"id": 2, "description": "b", "tool": null,
           "input_mapping": {"y": "{{o1}}"}, "output_var": "o2"}
        ]}
        """
        with pytest.raises(InvalidPlanError, match="forward"):
            validate_plan(Plan.parse(raw))
