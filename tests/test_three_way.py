import json
from pathlib import Path
from contract_intel_mvp.benchmark.three_way import run_three_way


def _seed_runs(root: Path):
    (root / "data" / "corpus" / "splits.json").write_text(json.dumps({
        "review_set": ["doc_001"],
        "holdout_set": ["doc_002", "doc_003"],
    }))
    (root / "data" / "runs" / "second_run_primary_holdout.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "License",
         "key_clauses": [{"family": "termination"}], "coversheet": {}},
        {"doc_id": "doc_003", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [{"family": "law"}], "coversheet": {}},
    ]))
    (root / "data" / "runs" / "shadow_holdout_cold_results.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [], "coversheet": {}},
        {"doc_id": "doc_003", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [{"family": "law"}], "coversheet": {}},
    ]))
    (root / "data" / "runs" / "second_run_results.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "License",
         "key_clauses": [{"family": "termination"}], "coversheet": {}},
        {"doc_id": "doc_003", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [{"family": "law"}], "coversheet": {}},
    ]))
    (root / "data" / "reviews" / "holdout_gold.json").write_text(json.dumps([
        {"doc_id": "doc_002", "accepted_contract_type": "License",
         "accepted_key_clauses": [{"family": "termination"}], "accepted_coversheet": {}},
        {"doc_id": "doc_003", "accepted_contract_type": "Service",
         "accepted_key_clauses": [{"family": "law"}], "accepted_coversheet": {}},
    ]))


def test_three_way_engine_gate_passes(tmp_root: Path):
    _seed_runs(tmp_root)
    out = run_three_way(tmp_root, large="qwen2.5:14b", small="qwen3:4b")
    assert out["engine_integrity"] == "ok"
    assert out["n_docs"] == 2
    m = out["metrics"]["contract_type_accuracy"]
    assert m["large"] == 1.0
    assert m["small_cold"] == 0.5
    assert m["small_reviewed"] == 1.0


def test_three_way_engine_gate_fails_on_fallback(tmp_root: Path):
    _seed_runs(tmp_root)
    runs = json.loads((tmp_root / "data" / "runs" / "second_run_results.json").read_text())
    runs[0]["engine"] = "heuristic_fallback"
    (tmp_root / "data" / "runs" / "second_run_results.json").write_text(json.dumps(runs))
    out = run_three_way(tmp_root, large="qwen2.5:14b", small="qwen3:4b",
                        allow_fallback=False)
    assert out["engine_integrity"] == "contaminated"
    assert "metrics" not in out
