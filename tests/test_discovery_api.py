import json
from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def _seed_minimal(tmp_root, monkeypatch, discovery_corpus):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed",
                        lambda t, **_: [1.0, 0.0, 0.0, 0.0] if "TRADEMARK" in t else [0.0]*3 + [1.0])
    from contract_intel_mvp.discovery.signature import init_signature
    from contract_intel_mvp.discovery.library import init_library_from_signature
    init_signature(tmp_root, interview={
        "target_class": "License Agreement",
        "target_description": "TRADEMARK LICENSE",
        "clause_types": [{"type": "license_grant", "description": "x",
                          "is_must_have": True, "seed_variations": []}],
    })
    init_library_from_signature(tmp_root)
    from contract_intel_mvp.discovery.embeddings import embed_corpus
    embed_corpus(tmp_root, model="nomic-embed-text")


def test_state_endpoint(tmp_root, discovery_corpus, monkeypatch):
    _seed_minimal(tmp_root, monkeypatch, discovery_corpus)
    client = TestClient(build_app(root=tmp_root))
    state = client.get("/api/discovery/state").json()
    assert state["embedded_count"] == 20
    assert state["target_class"] == "License Agreement"
    assert state["finalized"] is False
    assert state["library_size"] >= 1


def test_library_endpoint_returns_full_library(tmp_root, discovery_corpus, monkeypatch):
    _seed_minimal(tmp_root, monkeypatch, discovery_corpus)
    client = TestClient(build_app(root=tmp_root))
    lib = client.get("/api/discovery/library").json()
    assert lib["target_class"] == "License Agreement"
    assert any(ct["type"] == "license_grant" for ct in lib["clause_types"])


def test_submit_labels_grows_library_via_api(tmp_root, discovery_corpus, monkeypatch):
    _seed_minimal(tmp_root, monkeypatch, discovery_corpus)
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **kw: {
        "verdict": "yes", "confidence": 0.9,
        "evidence_per_clause_type": {"license_grant": "Licensor hereby grants Licensee a license to use the Marks"},
        "rationale": "x"
    })
    client = TestClient(build_app(root=tmp_root))
    rr = client.post("/api/discovery/run-round",
                     json={"classifier_model": "qwen3:4b", "top_k": 5,
                           "batch_size": 3, "round_index": 0}).json()
    queue = json.loads((tmp_root / "data" / "discovery" / "review_queue_round_0.json").read_text())
    first = queue["items"][0]
    sub = client.post("/api/discovery/submit-labels", json={
        "round_index": 0,
        "labels": [{"doc_id": first["doc_id"], "verdict": "yes"}],
    }).json()
    assert sub["library_growth"] >= 1
