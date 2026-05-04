from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.ranker import rank_corpus
from contract_intel_mvp.discovery.classifier import classify_candidates
from contract_intel_mvp.discovery.sampler import sample_for_review
from contract_intel_mvp.discovery.harvest import harvest_from_label
from contract_intel_mvp.discovery.convergence import current_metrics, record_round
from contract_intel_mvp.discovery.signature import load_signature


def run_round(root: Path, *, classifier_model: str, top_k: int = 200,
              batch_size: int = 20, round_index: int, seed: int = 0,
              progress_cb=None) -> dict[str, Any]:
    if progress_cb is not None:
        progress_cb(0, top_k, "ranking corpus")
    ranked = rank_corpus(root, top_k=top_k)
    classifications = classify_candidates(root, candidates=ranked,
                                          model=classifier_model,
                                          progress_cb=progress_cb)
    queue = sample_for_review(root, classifications=classifications,
                              batch_size=batch_size, seed=seed)
    metrics = current_metrics(root, classifications=classifications)
    out_dir = root / "data" / "discovery"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"classifications_round_{round_index}.json").write_text(
        json.dumps(classifications, indent=2), encoding="utf-8")
    (out_dir / f"review_queue_round_{round_index}.json").write_text(
        json.dumps({"round_index": round_index, "items": queue, "metrics": metrics}, indent=2),
        encoding="utf-8")
    return {"round_index": round_index,
            "classifications_count": len(classifications),
            "review_queue_size": len(queue),
            "metrics": metrics}


def submit_labels(root: Path, *, round_index: int,
                  labels: list[dict[str, Any]]) -> dict[str, Any]:
    cls_path = root / "data" / "discovery" / f"classifications_round_{round_index}.json"
    classifications = json.loads(cls_path.read_text()) if cls_path.exists() else []
    by_id = {c["doc_id"]: c for c in classifications}
    corrections = 0
    library_growth = 0
    for lbl in labels:
        doc_id = lbl["doc_id"]; sme = lbl["verdict"]
        if sme not in {"yes", "no", "borderline"}:
            continue
        cls = by_id.get(doc_id)
        if cls is None:
            continue
        if sme != cls.get("verdict") and sme != "borderline":
            corrections += 1
        result = harvest_from_label(root, classification=cls, sme_verdict=sme)
        library_growth += result.get("library_growth", 0)
    record_round(root, round_index=round_index, corrections=corrections,
                 library_growth=library_growth, batch_size=len(labels))
    return {"round_index": round_index, "labels_received": len(labels),
            "corrections": corrections, "library_growth": library_growth}


def finalize(root: Path, *, round_index: int,
             borderline_threshold: float = 0.7) -> dict[str, Any]:
    cls = json.loads((root / "data" / "discovery" / f"classifications_round_{round_index}.json").read_text())
    positives = [c for c in cls if c["verdict"] == "yes"]
    borderline = [c for c in positives if c.get("confidence", 0) < borderline_threshold]
    metrics = current_metrics(root, classifications=cls)
    sig = load_signature(root)
    out = {
        "target_class": sig.target_class,
        "round_index": round_index,
        "positives_count": len(positives),
        "borderline_count": len(borderline),
        "metrics": metrics,
        "positives": [{"doc_id": c["doc_id"], "confidence": c["confidence"],
                       "evidence_per_clause_type": c.get("evidence_per_clause_type", {})}
                      for c in positives],
    }
    out_dir = root / "data" / "discovery"
    (out_dir / "final.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (out_dir / "borderline.json").write_text(
        json.dumps({"items": borderline, "threshold": borderline_threshold}, indent=2),
        encoding="utf-8")
    return {"positives_count": len(positives),
            "borderline_count": len(borderline),
            "final_path": str((out_dir / "final.json").relative_to(root))}
