"""allowlist parsing: Dify passes `allowed_tools` as `type: any`, so it can
arrive as None, "", a list, or a JSON-encoded string. Empty always = no filter."""

from types import SimpleNamespace

from core.tool_allowlist import allowed_tool_names, filter_allowed_tools


class FakeTool:
    """Stands in for the SDK ToolEntity: the name lives under identity."""

    def __init__(self, name: str):
        self.identity = SimpleNamespace(name=name)


TOOLS = [FakeTool("search"), FakeTool("read_file"), FakeTool("http")]


def names_of(tools) -> set[str]:
    return {t.identity.name for t in tools}


# ---------- allowed_tool_names ----------

def test_none_and_blank_mean_unrestricted():
    assert allowed_tool_names(None) is None
    assert allowed_tool_names("") is None
    assert allowed_tool_names("   ") is None
    assert allowed_tool_names([]) is None


def test_accepts_list_json_string_and_bare_string():
    assert allowed_tool_names(["a", "b"]) == {"a", "b"}
    assert allowed_tool_names('["a", "b"]') == {"a", "b"}
    assert allowed_tool_names("a") == {"a"}


def test_drops_non_strings_and_blanks():
    assert allowed_tool_names([None, "", "  ", "a", 3]) == {"a"}


def test_invalid_types_are_unrestricted():
    assert allowed_tool_names(123) is None
    assert allowed_tool_names({"a": 1}) is None


# ---------- filter_allowed_tools ----------

def test_empty_allowlist_keeps_every_tool():
    assert filter_allowed_tools(TOOLS, None) == TOOLS
    assert filter_allowed_tools(TOOLS, []) == TOOLS


def test_filters_to_named_tools_only():
    assert names_of(filter_allowed_tools(TOOLS, ["search", "http"])) == {"search", "http"}
    assert names_of(filter_allowed_tools(TOOLS, '["read_file"]')) == {"read_file"}


def test_unknown_names_yield_empty_result():
    assert filter_allowed_tools(TOOLS, ["nope"]) == []


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
