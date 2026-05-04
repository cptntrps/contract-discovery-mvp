import json
from pathlib import Path
from contract_intel_mvp.benchmark.counterfactual import (
    recompute_without_verification, recompute_without_reviewed_context
)


def test_recompute_without_verification_drops_unverified_clauses(tmp_root: Path):
    (tmp_root / "data" / "runs" / "second_run_results.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "License",
         "key_clauses": [
             {"family": "termination", "evidence_snippet": "verified"},
         ],
         "coversheet": {},
         "evidence_verification": {"final_missing": 1, "rejected_families": ["ip"]}},
    ]))
    (tmp_root / "data" / "corpus" / "splits.json").write_text(
        json.dumps({"review_set": [], "holdout_set": ["doc_002"]}))
    (tmp_root / "data" / "reviews" / "holdout_gold.json").write_text(json.dumps([
        {"doc_id": "doc_002", "accepted_contract_type": "License",
         "accepted_key_clauses": [{"family": "termination"}, {"family": "ip"}],
         "accepted_coversheet": {}},
    ]))
    out = recompute_without_verification(tmp_root, model="qwen3:4b")
    assert "f1_with_verifier_on" in out
    assert "f1_with_verifier_off" in out
    assert out["f1_with_verifier_off"] != out["f1_with_verifier_on"]


def test_recompute_without_reviewed_context_uses_cold(tmp_root: Path):
    (tmp_root / "data" / "runs" / "shadow_holdout_cold_results.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [], "coversheet": {}},
    ]))
    (tmp_root / "data" / "corpus" / "splits.json").write_text(
        json.dumps({"review_set": [], "holdout_set": ["doc_002"]}))
    (tmp_root / "data" / "reviews" / "holdout_gold.json").write_text(json.dumps([
        {"doc_id": "doc_002", "accepted_contract_type": "License",
         "accepted_key_clauses": [], "accepted_coversheet": {}},
    ]))
    out = recompute_without_reviewed_context(tmp_root)
    assert out["contract_type_accuracy"] == 0.0
