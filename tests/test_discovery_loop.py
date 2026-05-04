import json
from pathlib import Path
from contract_intel_mvp.discovery.loop import run_round, submit_labels, finalize
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature
from contract_intel_mvp.discovery.embeddings import embed_corpus


def _fake_embed(text, **_):
    if "TRADEMARK LICENSE" in text or "Licensor" in text:
        return [1.0, 0.0, 0.0, 0.1]
    if "DISTRIBUTOR" in text: return [0.0, 1.0, 0.0, 0.1]
    if "STRATEGIC" in text:   return [0.0, 0.0, 1.0, 0.1]
    return [0.0, 0.0, 0.0, 1.0]


def test_run_round_writes_artifacts(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text")
    init_signature(tmp_root, interview={
        "target_class": "License Agreement", "target_description": "TRADEMARK LICENSE",
        "clause_types": [
            {"type": "license_grant", "description": "x", "is_must_have": True, "seed_variations": []},
            {"type": "primary_distribution", "description": "x", "is_must_have": False, "seed_variations": []},
        ],
    })
    init_library_from_signature(tmp_root)
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **kw: {
        "verdict": "yes" if "TRADEMARK LICENSE" in kw["prompt"] else "no",
        "confidence": 0.85,
        "evidence_per_clause_type": {"license_grant": "Licensor hereby grants" if "TRADEMARK" in kw["prompt"] else "",
                                       "primary_distribution": "" if "TRADEMARK" in kw["prompt"] else "appoints"},
        "rationale": "x"
    })
    out = run_round(tmp_root, classifier_model="qwen3:4b",
                    top_k=15, batch_size=8, round_index=0, seed=1)
    assert out["round_index"] == 0
    assert out["classifications_count"] == 15
    assert (tmp_root / "data" / "discovery" / "review_queue_round_0.json").exists()


def test_submit_labels_grows_library(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text")
    init_signature(tmp_root, interview={
        "target_class": "License", "target_description": "TRADEMARK LICENSE",
        "clause_types": [
            {"type": "license_grant", "description": "x", "is_must_have": True, "seed_variations": []},
        ],
    })
    init_library_from_signature(tmp_root)
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **kw: {
        "verdict": "yes", "confidence": 0.9,
        "evidence_per_clause_type": {"license_grant": "Licensor hereby grants Licensee a license"},
        "rationale": "x"
    })
    run_round(tmp_root, classifier_model="qwen3:4b", top_k=5, batch_size=3, round_index=0, seed=1)
    queue = json.loads((tmp_root / "data" / "discovery" / "review_queue_round_0.json").read_text())
    first = queue["items"][0]
    res = submit_labels(tmp_root, round_index=0, labels=[
        {"doc_id": first["doc_id"], "verdict": "yes"}
    ])
    assert res["library_growth"] >= 1
    from contract_intel_mvp.discovery.library import load_library
    lib = load_library(tmp_root)
    lg = next(ct for ct in lib["clause_types"] if ct["type"] == "license_grant")
    assert any(v["confirmed_by"] == "auto_from_sme_yes" for v in lg["variations"])


def test_finalize_emits_positives_and_borderline(tmp_root):
    init_signature(tmp_root, interview={"target_class": "X", "target_description": "x",
                                        "clause_types": []})
    init_library_from_signature(tmp_root)
    classifications = [
        {"doc_id": "a", "verdict": "yes", "confidence": 0.95, "engine": "ollama",
         "evidence_per_clause_type": {}},
        {"doc_id": "b", "verdict": "yes", "confidence": 0.55, "engine": "ollama",
         "evidence_per_clause_type": {}},
        {"doc_id": "c", "verdict": "no",  "confidence": 0.9,  "engine": "ollama",
         "evidence_per_clause_type": {}},
    ]
    (tmp_root / "data" / "discovery").mkdir(parents=True, exist_ok=True)
    (tmp_root / "data" / "discovery" / "classifications_round_0.json").write_text(json.dumps(classifications))
    out = finalize(tmp_root, round_index=0, borderline_threshold=0.7)
    assert out["positives_count"] == 2
    assert out["borderline_count"] == 1
    final = json.loads((tmp_root / "data" / "discovery" / "final.json").read_text())
    assert {p["doc_id"] for p in final["positives"]} == {"a", "b"}
