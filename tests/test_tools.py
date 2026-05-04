import pytest
from contract_intel_mvp.agent.tools import ToolRegistry, ToolError


def test_registry_dispatch_calls_registered_tool():
    reg = ToolRegistry()

    @reg.register("ping")
    def _ping(*, msg: str) -> dict:
        return {"echo": msg}

    out = reg.call("ping", {"msg": "hi"})
    assert out == {"echo": "hi"}


def test_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="unknown tool"):
        reg.call("nope", {})


def test_arg_validation_failure_is_tool_error():
    reg = ToolRegistry()

    @reg.register("strict")
    def _strict(*, n: int) -> dict:
        return {"n": n}

    with pytest.raises(ToolError):
        reg.call("strict", {"wrong": 1})


def test_list_tools_returns_signatures():
    reg = ToolRegistry()
    @reg.register("foo", description="foo doc")
    def _foo(*, a: str) -> dict: return {}
    listing = reg.list_tools()
    assert listing[0]["name"] == "foo"
    assert listing[0]["description"] == "foo doc"
    assert "a" in listing[0]["parameters"]
