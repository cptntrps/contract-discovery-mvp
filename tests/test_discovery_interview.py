from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def test_discovery_chat_initial_returns_monologue(tmp_root, monkeypatch):
    monkeypatch.delenv("OPENAI_INTERVIEW", raising=False)
    app = build_app(root=tmp_root)
    client = TestClient(app)
    resp = client.post("/api/interview/discovery-chat", json={
        "signature": {"target_class": "", "target_description": "", "clause_types": []},
        "message": "",
        "initial": True,
    }).json()
    assert resp["engine"] == "scripted_opening"
    assert "discovery agent" in resp["assistant"].lower()


def test_discovery_chat_save_initializes_library(tmp_root, monkeypatch):
    monkeypatch.delenv("OPENAI_INTERVIEW", raising=False)
    app = build_app(root=tmp_root)
    client = TestClient(app)
    resp = client.post("/api/interview/discovery-chat", json={
        "signature": {
            "target_class": "License Agreement",
            "target_description": "Primary IP grant",
            "clause_types": [
                {"type": "license_grant", "description": "right to use", "is_must_have": True,
                 "seed_variations": ["Licensor grants Licensee a license"]},
                {"type": "primary_distribution", "description": "appoints distributor",
                 "is_must_have": False,
                 "seed_variations": ["appoints Distributor as the exclusive distributor"]},
            ],
        },
        "message": "save",
        "save": True,
    }).json()
    assert resp["saved"] is True
    assert (tmp_root / "data" / "discovery" / "signature.json").exists()
    assert (tmp_root / "data" / "discovery" / "clause_library.json").exists()
