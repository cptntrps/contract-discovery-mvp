from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.signature import load_signature


def _path(root): return root / "data" / "discovery" / "rounds.json"


def record_round(root: Path, *, round_index: int, corrections: int,
                 library_growth: int, batch_size: int) -> None:
    p = _path(root); p.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(p.read_text()) if p.exists() else {"rounds": []}
    state["rounds"].append({
        "round_index": round_index, "corrections": corrections,
        "library_growth": library_growth, "batch_size": batch_size,
    })
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_stop(root: Path, *, threshold: int = 3, max_rounds: int = 5) -> bool:
    p = _path(root)
    if not p.exists(): return False
    rounds = json.loads(p.read_text()).get("rounds", [])
    if len(rounds) >= max_rounds: return True
    if rounds and rounds[-1]["corrections"] < threshold and rounds[-1]["library_growth"] == 0:
        return True
    return False


def current_metrics(root: Path, *, classifications: list[dict[str, Any]]) -> dict[str, Any]:
    sig = load_signature(root)
    pos_gold = set(sig.confirmed_positive_doc_ids)
    neg_gold = set(sig.confirmed_negative_doc_ids)
    by_id = {c["doc_id"]: c for c in classifications}
    tp = sum(1 for d in pos_gold if by_id.get(d, {}).get("verdict") == "yes")
    fn = sum(1 for d in pos_gold if by_id.get(d, {}).get("verdict") == "no")
    fp = sum(1 for d in neg_gold if by_id.get(d, {}).get("verdict") == "yes")
    tn = sum(1 for d in neg_gold if by_id.get(d, {}).get("verdict") == "no")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"true_positives": tp, "true_negatives": tn,
            "false_positives": fp, "false_negatives": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "labeled_count": len(pos_gold) + len(neg_gold)}
