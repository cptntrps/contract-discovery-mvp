from pathlib import Path
from contract_intel_mvp.discovery.classifier import classify_candidates
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature


def _seed(tmp_root):
    init_signature(tmp_root, interview={
        "target_class": "License Agreement",
        "target_description": "primary IP grant",
        "clause_types": [
            {"type": "license_grant", "description": "right to use IP",
             "is_must_have": True, "seed_variations": ["Licensor grants Licensee a license"]},
            {"type": "primary_distribution", "description": "appoints distributor",
             "is_must_have": False, "seed_variations": ["appoints Distributor"]},
        ],
    })
    init_library_from_signature(tmp_root)


def test_classifier_extracts_per_clause_evidence(tmp_root, discovery_corpus, monkeypatch):
    _seed(tmp_root)
    import contract_intel_mvp.discovery.classifier as c
    def stub(*, model, prompt):
        is_lic = "TRADEMARK LICENSE" in prompt
        return {
            "verdict": "yes" if is_lic else "no",
            "confidence": 0.9,
            "evidence_per_clause_type": {
                "license_grant": "Licensor hereby grants" if is_lic else "",
                "primary_distribution": "" if is_lic else "appoints Distributor",
            },
            "rationale": "stub",
        }
    monkeypatch.setattr(c, "_call_ollama_json", stub)
    cands = [{"doc_id": "doc_lic_0", "score": 0.9},
             {"doc_id": "doc_dis_0", "score": 0.5}]
    out = classify_candidates(tmp_root, candidates=cands, model="qwen3:4b")
    by_id = {r["doc_id"]: r for r in out}
    assert by_id["doc_lic_0"]["verdict"] == "yes"
    assert "license_grant" in by_id["doc_lic_0"]["evidence_per_clause_type"]
    assert by_id["doc_lic_0"]["evidence_per_clause_type"]["license_grant"] == "Licensor hereby grants"
    assert by_id["doc_dis_0"]["verdict"] == "no"
    assert all(r["engine"] == "ollama" for r in out)


def test_classifier_falls_back_when_model_returns_none(tmp_root, discovery_corpus, monkeypatch):
    _seed(tmp_root)
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **_: None)
    out = classify_candidates(tmp_root, candidates=[{"doc_id": "doc_lic_0", "score": 0.9}],
                              model="qwen3:4b")
    assert out[0]["engine"] == "heuristic_fallback"
    assert out[0]["verdict"] in {"yes", "no"}
    assert "evidence_per_clause_type" in out[0]
