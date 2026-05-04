import pytest
from contract_intel_mvp.engine_gate import check_engine_integrity, EngineContamination


def test_all_ollama_passes():
    rows = [{"doc_id": "a", "engine": "ollama"},
            {"doc_id": "b", "engine": "ollama"}]
    report = check_engine_integrity(rows)
    assert report["ok"] is True
    assert report["fallback_count"] == 0


def test_any_fallback_fails():
    rows = [{"doc_id": "a", "engine": "ollama"},
            {"doc_id": "b", "engine": "heuristic_fallback"}]
    with pytest.raises(EngineContamination) as exc:
        check_engine_integrity(rows)
    assert "1 of 2" in str(exc.value)


def test_allow_fallback_returns_tagged_report():
    rows = [{"doc_id": "a", "engine": "heuristic_fallback"}]
    report = check_engine_integrity(rows, allow_fallback=True)
    assert report["ok"] is False
    assert report["fallback_count"] == 1
    assert report["fallback_doc_ids"] == ["a"]


def test_empty_rows_is_explicit_failure():
    with pytest.raises(EngineContamination, match="no rows"):
        check_engine_integrity([])
