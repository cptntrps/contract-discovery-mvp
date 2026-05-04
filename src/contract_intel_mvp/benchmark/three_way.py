from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.engine_gate import check_engine_integrity, EngineContamination


def _load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["doc_id"]: r for r in rows}


def _accuracy(rows, key_pred, key_gold) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r[key_pred] == r[key_gold]) / len(rows)


def _clause_f1(pred: list[dict], gold: list[dict]) -> tuple[float, float, float]:
    p = {c.get("family") for c in pred if c.get("family")}
    g = {c.get("family") for c in gold if c.get("family")}
    if not p and not g:
        return 1.0, 1.0, 1.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def run_three_way(root: Path, *, large: str, small: str,
                  allow_fallback: bool = False) -> dict[str, Any]:
    splits = _load(root / "data" / "corpus" / "splits.json", {"holdout_set": []})
    holdout_ids = set(splits["holdout_set"])
    large_rows  = [r for r in _load(root / "data" / "runs" / "second_run_primary_holdout.json", []) if r["doc_id"] in holdout_ids]
    cold_rows   = [r for r in _load(root / "data" / "runs" / "shadow_holdout_cold_results.json", []) if r["doc_id"] in holdout_ids]
    revd_rows   = [r for r in _load(root / "data" / "runs" / "second_run_results.json", []) if r["doc_id"] in holdout_ids]
    gold        = _by_id(_load(root / "data" / "reviews" / "holdout_gold.json", []))

    try:
        for label, rows in [("large", large_rows), ("small_cold", cold_rows),
                            ("small_reviewed", revd_rows)]:
            check_engine_integrity(rows, allow_fallback=allow_fallback)
        integrity = "ok"
    except EngineContamination as e:
        out = {"engine_integrity": "contaminated", "error": str(e),
               "n_docs": len(holdout_ids)}
        (root / "data" / "runs" / "benchmark.json").write_text(json.dumps(out, indent=2))
        return out

    def _aligned(rows):
        return [{
            "doc_id": r["doc_id"],
            "pred_type": r["contract_type"],
            "gold_type": gold[r["doc_id"]]["accepted_contract_type"],
            "pred_clauses": r.get("key_clauses", []),
            "gold_clauses": gold[r["doc_id"]].get("accepted_key_clauses", []),
        } for r in rows if r["doc_id"] in gold]

    metrics: dict[str, dict[str, float]] = {}
    for label, rows in [("large", large_rows), ("small_cold", cold_rows),
                        ("small_reviewed", revd_rows)]:
        a = _aligned(rows)
        type_acc = _accuracy(a, "pred_type", "gold_type")
        f1s = [_clause_f1(r["pred_clauses"], r["gold_clauses"])[2] for r in a]
        metrics.setdefault("contract_type_accuracy", {})[label] = type_acc
        metrics.setdefault("clause_family_f1", {})[label] = sum(f1s) / len(f1s) if f1s else 0.0

    out = {
        "engine_integrity": integrity,
        "n_docs": len(holdout_ids),
        "models": {"large": large, "small_cold": small, "small_reviewed": small},
        "metrics": metrics,
    }
    (root / "data" / "runs" / "benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
