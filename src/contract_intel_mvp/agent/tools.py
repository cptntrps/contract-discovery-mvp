"""Tool registry for the planner loop."""
from __future__ import annotations
import inspect
from typing import Any, Callable


class ToolError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, *, description: str = ""):
        def deco(fn: Callable[..., dict]):
            sig = inspect.signature(fn)
            params = {p.name: str(p.annotation) for p in sig.parameters.values()
                      if p.kind == p.KEYWORD_ONLY}
            self._tools[name] = {"fn": fn, "description": description,
                                 "parameters": params}
            return fn
        return deco

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self._tools.get(name)
        if not spec:
            raise ToolError(f"unknown tool: {name}")
        try:
            return spec["fn"](**args)
        except TypeError as e:
            raise ToolError(f"bad args for {name}: {e}") from e

    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": n, "description": s["description"], "parameters": s["parameters"]}
                for n, s in self._tools.items()]
