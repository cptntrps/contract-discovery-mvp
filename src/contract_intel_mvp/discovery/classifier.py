from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.pipeline import _call_ollama_json
from contract_intel_mvp.prompts import DISCOVERY_CLASSIFY_PROMPT
from contract_intel_mvp.discovery.library import render_library_text, load_library


def _load_docs_by_id(root: Path) -> dict[str, dict]:
    p = root / "data" / "corpus" / "documents.jsonl"
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def _heuristic_classify(doc_text: str, lib: dict) -> dict[str, Any]:
    text = doc_text.lower()
    have = {}
    pos = neg = 0
    for ct in lib.get("clause_types", []):
        hit = any(v["text"].lower() in text for v in ct["variations"])
        have[ct["type"]] = doc_text[:160] if hit else ""
        if hit and ct["is_must_have"]:
            pos += 1
        if hit and not ct["is_must_have"]:
            neg += 1
    must = sum(1 for ct in lib.get("clause_types", []) if ct["is_must_have"]) or 1
    score = pos / must - 0.5 * neg
    return {
        "verdict": "yes" if score > 0.5 else "no",
        "confidence": max(0.0, min(1.0, abs(score - 0.5) + 0.4)),
        "evidence_per_clause_type": have,
        "rationale": "heuristic fallback",
    }


def classify_candidates(root: Path, *, candidates: list[dict[str, Any]],
                        model: str,
                        progress_cb=None) -> list[dict[str, Any]]:
    lib = load_library(root)
    library_block = render_library_text(root, max_per_type=5)
    docs_by_id = _load_docs_by_id(root)
    results = []
    total = len(candidates)
    for i, cand in enumerate(candidates):
        doc = docs_by_id.get(cand["doc_id"])
        if doc is None:
            continue
        doc_text = doc.get("text") or ""
        prompt = DISCOVERY_CLASSIFY_PROMPT.format(
            library_block=library_block,
            doc_text=doc_text[:6000],
        )
        parsed = _call_ollama_json(model=model, prompt=prompt)
        valid = (
            parsed
            and isinstance(parsed.get("verdict"), str)
            and parsed["verdict"] in {"yes", "no"}
            and isinstance(parsed.get("evidence_per_clause_type"), dict)
        )
        if valid:
            try:
                parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            except (TypeError, ValueError):
                parsed["confidence"] = 0.5
            engine = "ollama"
        else:
            parsed = _heuristic_classify(doc_text, lib)
            engine = "heuristic_fallback"
        parsed["doc_id"] = cand["doc_id"]
        parsed["screen_score"] = cand.get("score")
        parsed["engine"] = engine
        results.append(parsed)
        if progress_cb is not None:
            try:
                progress_cb(i + 1, total, f"classifying {cand['doc_id']}")
            except Exception:
                pass
    return results
