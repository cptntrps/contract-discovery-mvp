"""Parallel primary + shadow model extraction."""
from __future__ import annotations
import asyncio
import json as _json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ShadowPair:
    primary: dict[str, Any]
    shadow: dict[str, Any]
    primary_engine: str
    shadow_engine: str


async def _async_call_ollama_json(*, model: str, prompt: str,
                                  base_url: str = "http://127.0.0.1:11434"
                                  ) -> dict[str, Any] | None:
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False,
               "options": {"temperature": 0.1}}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{base_url}/api/generate", json=payload)
            r.raise_for_status()
            envelope = r.json()
    except Exception:
        return None
    text = envelope.get("response", "") or envelope.get("thinking", "")
    if not text:
        return None
    try:
        parsed = _json.loads(text)
    except _json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _heuristic_extraction_stub() -> dict[str, Any]:
    return {"contract_type": "unknown", "coversheet": {}, "key_clauses": [],
            "rationale": "heuristic fallback (shadow path)"}


async def run_shadow_pair(*, prompt: str, primary_model: str, shadow_model: str
                          ) -> ShadowPair:
    primary_task = _async_call_ollama_json(model=primary_model, prompt=prompt)
    shadow_task = _async_call_ollama_json(model=shadow_model, prompt=prompt)
    primary_raw, shadow_raw = await asyncio.gather(primary_task, shadow_task)
    return ShadowPair(
        primary=primary_raw or _heuristic_extraction_stub(),
        shadow=shadow_raw or _heuristic_extraction_stub(),
        primary_engine="ollama" if primary_raw else "heuristic_fallback",
        shadow_engine="ollama" if shadow_raw else "heuristic_fallback",
    )
