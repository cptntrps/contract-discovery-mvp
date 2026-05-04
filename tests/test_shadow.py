import asyncio
from contract_intel_mvp.agent import shadow


def test_pair_extraction_calls_both_models(monkeypatch):
    calls = []

    async def fake_call(*, model, prompt):
        calls.append(model)
        return {"contract_type": f"type-from-{model}", "key_clauses": [], "coversheet": {}}

    monkeypatch.setattr(shadow, "_async_call_ollama_json", fake_call)
    result = asyncio.run(shadow.run_shadow_pair(
        prompt="extract this", primary_model="qwen2.5:14b", shadow_model="qwen3:4b"))

    assert sorted(calls) == ["qwen2.5:14b", "qwen3:4b"]
    assert result.primary["contract_type"] == "type-from-qwen2.5:14b"
    assert result.shadow["contract_type"] == "type-from-qwen3:4b"
    assert result.primary_engine == "ollama"
    assert result.shadow_engine == "ollama"


def test_pair_records_fallback_when_shadow_returns_none(monkeypatch):
    async def fake_call(*, model, prompt):
        if "qwen3" in model:
            return None
        return {"contract_type": "License", "key_clauses": [], "coversheet": {}}

    monkeypatch.setattr(shadow, "_async_call_ollama_json", fake_call)
    result = asyncio.run(shadow.run_shadow_pair(
        prompt="x", primary_model="qwen2.5:14b", shadow_model="qwen3:4b"))
    assert result.primary_engine == "ollama"
    assert result.shadow_engine == "heuristic_fallback"
    assert result.shadow is not None
