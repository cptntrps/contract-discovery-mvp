from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.benchmark.three_way import _load, _accuracy, _clause_f1, _by_id


def recompute_without_verification(root: Path, *, model: str) -> dict[str, Any]:
    """Score the small_reviewed column with rejected clauses re-included to show verifier value."""
    splits = _load(root / "data" / "corpus" / "splits.json", {"holdout_set": []})
    holdout_ids = set(splits["holdout_set"])
    revd = [r for r in _load(root / "data" / "runs" / "second_run_results.json", []) if r["doc_id"] in holdout_ids]
    gold = _by_id(_load(root / "data" / "reviews" / "holdout_gold.json", []))

    def _f1_over(rows, with_verifier: bool) -> float:
        f1s = []
        for r in rows:
            if r["doc_id"] not in gold:
                continue
            preds = r.get("key_clauses", [])
            if not with_verifier:
                rejected = r.get("evidence_verification", {}).get("rejected_families", [])
                preds = preds + [{"family": fam} for fam in rejected]
            _, _, f1 = _clause_f1(preds, gold[r["doc_id"]].get("accepted_key_clauses", []))
            f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0

    on = _f1_over(revd, True)
    off = _f1_over(revd, False)
    out = {
        "f1_with_verifier_on":  on,
        "f1_with_verifier_off": off,
        "delta": on - off,
        "n_docs": len(revd),
        "model": model,
    }
    (root / "data" / "runs" / "counterfactual_verifier.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out


def recompute_without_reviewed_context(root: Path) -> dict[str, Any]:
    splits = _load(root / "data" / "corpus" / "splits.json", {"holdout_set": []})
    holdout_ids = set(splits["holdout_set"])
    cold = [r for r in _load(root / "data" / "runs" / "shadow_holdout_cold_results.json", []) if r["doc_id"] in holdout_ids]
    gold = _by_id(_load(root / "data" / "reviews" / "holdout_gold.json", []))
    aligned = [{"pred_type": r["contract_type"],
                "gold_type": gold[r["doc_id"]]["accepted_contract_type"]}
               for r in cold if r["doc_id"] in gold]
    f1s = []
    for r in cold:
        if r["doc_id"] not in gold:
            continue
        _, _, f1 = _clause_f1(r.get("key_clauses", []),
                              gold[r["doc_id"]].get("accepted_key_clauses", []))
        f1s.append(f1)
    out = {
        "contract_type_accuracy": _accuracy(aligned, "pred_type", "gold_type"),
        "clause_family_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "n_docs": len(aligned),
    }
    (root / "data" / "runs" / "counterfactual_context.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out
