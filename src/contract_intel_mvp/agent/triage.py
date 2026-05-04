"""Uncertainty scoring -> review queue."""
from __future__ import annotations
from typing import Any


def score_document(extraction: dict[str, Any], interview: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    score = 0.0

    ver = extraction.get("evidence_verification", {})
    n_missing = ver.get("final_missing", 0)
    if n_missing > 0:
        reasons.append("unverifiable_spans")
        score += min(0.5, 0.15 * n_missing)

    alts = extraction.get("type_alternatives", [])
    if alts and (alts[0].get("score") or 0) >= 0.4:
        reasons.append("close_type_alternative")
        score += 0.25

    expected = set(interview.get("key_clause_families", []) or [])
    found = {c.get("family") for c in extraction.get("key_clauses", [])}
    missing = expected - found
    if missing:
        reasons.append("missing_expected_clauses")
        score += min(0.3, 0.1 * len(missing))

    if ver.get("attempts", 1) > 1:
        reasons.append("required_retries")
        score += 0.1 * (ver["attempts"] - 1)

    return {
        "doc_id": extraction["doc_id"],
        "uncertainty": round(min(1.0, score), 3),
        "reasons": reasons,
    }


def build_review_queue(extractions: list[dict[str, Any]],
                       interview: dict[str, Any],
                       *, threshold: float = 0.3) -> list[dict[str, Any]]:
    scored = [score_document(e, interview) for e in extractions]
    flagged = [s for s in scored if s["uncertainty"] >= threshold]
    flagged.sort(key=lambda s: s["uncertainty"], reverse=True)
    return flagged
