"""Score a discovery final.json against CUAD's filename-derived contract types.

CUAD doesn't have an explicit "contract type" column; the type is encoded in
the filename and `pipeline._contract_type_from_cuad_filename` is the canonical
parser the project already uses elsewhere.

Usage:
    python tools/score_against_cuad.py [target_substring]

Example:
    python tools/score_against_cuad.py license
        → counts a doc as gold-positive if its CUAD-derived type contains
          'license' (case-insensitive). Reports precision / recall / F1
          against final.json's predicted positives.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Make the package importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from contract_intel_mvp.pipeline import _contract_type_from_cuad_filename


def score(root: Path, target_substring: str) -> dict:
    needle = target_substring.lower()

    # Predicted positives from discovery final.
    final_path = root / "data" / "discovery" / "final.json"
    if not final_path.exists():
        raise SystemExit(f"no final.json at {final_path}; run discovery finalize first")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    predicted_ids = {p["doc_id"] for p in final.get("positives", [])}

    # Ground truth: every ingested doc → CUAD-derived type → check if matches needle.
    docs_path = root / "data" / "corpus" / "documents.jsonl"
    if not docs_path.exists():
        raise SystemExit(f"no documents.jsonl at {docs_path}")
    gold_ids: set[str] = set()
    type_counts: dict[str, int] = {}
    for line in docs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        # CUAD's filename parser expects the original CUAD-style filename.
        # We stored the title (first-line heuristic) and source filename in different paths,
        # so try both: prefer 'source_path' or 'filename' if present, else 'title'.
        candidate = (doc.get("source_path") or doc.get("filename") or doc.get("title") or "")
        cuad_type = _contract_type_from_cuad_filename(Path(candidate).name if candidate else "")
        type_counts[cuad_type] = type_counts.get(cuad_type, 0) + 1
        if needle in cuad_type.lower():
            gold_ids.add(doc["doc_id"])

    tp = predicted_ids & gold_ids
    fp = predicted_ids - gold_ids
    fn = gold_ids - predicted_ids

    prec = len(tp) / len(predicted_ids) if predicted_ids else 0.0
    rec = len(tp) / len(gold_ids) if gold_ids else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "target_substring": target_substring,
        "target_class_in_signature": final.get("target_class"),
        "corpus_size": sum(type_counts.values()),
        "gold_positives": len(gold_ids),
        "predicted_positives": len(predicted_ids),
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "false_positive_doc_ids": sorted(fp),
        "false_negative_doc_ids": sorted(fn),
        "type_distribution_top10": dict(sorted(type_counts.items(),
                                                key=lambda kv: -kv[1])[:10]),
    }


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "license"
    out = score(ROOT, target)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
