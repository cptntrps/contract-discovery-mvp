from __future__ import annotations
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.signature import load_signature, save_signature
from contract_intel_mvp.discovery.library import append_variations, load_library


def harvest_from_label(root: Path, *, classification: dict[str, Any],
                       sme_verdict: str) -> dict[str, Any]:
    """Update library and signature from one SME verdict."""
    if sme_verdict not in {"yes", "no", "borderline"}:
        raise ValueError("sme_verdict must be yes/no/borderline")
    if sme_verdict == "borderline":
        return {"updated_clause_types": [], "library_growth": 0}

    sig = load_signature(root)
    doc_id = classification["doc_id"]
    if sme_verdict == "yes":
        if doc_id not in sig.confirmed_positive_doc_ids:
            sig.confirmed_positive_doc_ids.append(doc_id)
    else:
        if doc_id not in sig.confirmed_negative_doc_ids:
            sig.confirmed_negative_doc_ids.append(doc_id)
    save_signature(root, sig)

    lib = load_library(root)
    must_have = {ct["type"] for ct in lib["clause_types"] if ct["is_must_have"]}
    must_not  = {ct["type"] for ct in lib["clause_types"] if not ct["is_must_have"]}
    evidence = classification.get("evidence_per_clause_type", {}) or {}
    growth = 0
    updated_types: list[str] = []
    target_set = must_have if sme_verdict == "yes" else must_not
    confirmed_by = "auto_from_sme_yes" if sme_verdict == "yes" else "auto_from_sme_no"

    for clause_type, text in evidence.items():
        if clause_type in target_set and (text or "").strip():
            before = len(next((ct["variations"] for ct in lib["clause_types"]
                                if ct["type"] == clause_type), []))
            append_variations(root, clause_type=clause_type, variations=[
                {"text": text, "source_doc_id": doc_id, "confirmed_by": confirmed_by}
            ])
            lib_after = load_library(root)
            after = len(next((ct["variations"] for ct in lib_after["clause_types"]
                               if ct["type"] == clause_type), []))
            if after > before:
                growth += after - before
                updated_types.append(clause_type)

    return {"updated_clause_types": updated_types, "library_growth": growth}
