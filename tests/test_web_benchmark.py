import json
from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def test_benchmark_endpoint_returns_three_way(tmp_root: Path):
    (tmp_root / "data" / "runs" / "benchmark.json").write_text(json.dumps({
        "engine_integrity": "ok",
        "n_docs": 30,
        "models": {"large": "qwen2.5:14b", "small_cold": "qwen3:4b", "small_reviewed": "qwen3:4b"},
        "metrics": {
            "contract_type_accuracy": {"large": 0.97, "small_cold": 0.83, "small_reviewed": 0.93},
            "clause_family_f1":        {"large": 0.69, "small_cold": 0.41, "small_reviewed": 0.62},
        },
    }))
    app = build_app(root=tmp_root)
    client = TestClient(app)
    out = client.get("/api/benchmark/three-way").json()
    assert out["engine_integrity"] == "ok"
    assert out["metrics"]["clause_family_f1"]["small_reviewed"] == 0.62


def test_counterfactual_endpoint_recomputes(tmp_root: Path, monkeypatch):
    import contract_intel_mvp.web as w
    monkeypatch.setattr(w, "recompute_without_verification",
                        lambda root, model: {"f1_with_verifier_on": 0.62,
                                              "f1_with_verifier_off": 0.51, "delta": 0.11})
    monkeypatch.setattr(w, "recompute_without_reviewed_context",
                        lambda root: {"clause_family_f1": 0.41})
    app = build_app(root=tmp_root)
    client = TestClient(app)
    a = client.post("/api/benchmark/counterfactual",
                    json={"toggle": "verifier_off", "model": "qwen3:4b"}).json()
    assert a["f1_with_verifier_off"] == 0.51
    b = client.post("/api/benchmark/counterfactual",
                    json={"toggle": "context_off"}).json()
    assert b["clause_family_f1"] == 0.41
