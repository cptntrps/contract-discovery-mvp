"""Evidence-span verification against source text."""
from __future__ import annotations
import re
from dataclasses import dataclass

from contract_intel_mvp.prompts import REGROUND_PROMPT
from contract_intel_mvp.pipeline import _call_ollama_json


@dataclass
class EvidenceReport:
    verified: int
    missing: int
    missing_families: list[str]
    verified_families: list[str]

    @property
    def is_clean(self) -> bool:
        return self.missing == 0


_WS = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


def verify_spans(clauses: list[dict], source_text: str) -> EvidenceReport:
    haystack = _normalize_for_match(source_text)
    verified, missing = [], []
    for c in clauses:
        snippet = (c.get("evidence_snippet") or "").strip()
        if not snippet:
            missing.append(c.get("family", "unknown"))
            continue
        needle = _normalize_for_match(snippet)
        if needle and needle in haystack:
            verified.append(c.get("family", "unknown"))
        else:
            missing.append(c.get("family", "unknown"))
    return EvidenceReport(
        verified=len(verified),
        missing=len(missing),
        missing_families=missing,
        verified_families=verified,
    )


def extract_with_verification(*, source_text: str, model: str,
                              initial_extraction: dict, max_retries: int = 2
                              ) -> tuple[dict, list[EvidenceReport]]:
    """Verify spans on an initial extraction; re-prompt on misses up to max_retries."""
    extraction = dict(initial_extraction)
    reports: list[EvidenceReport] = []
    for _attempt in range(max_retries + 1):
        report = verify_spans(extraction.get("key_clauses", []), source_text)
        reports.append(report)
        if report.is_clean:
            break
        prompt = REGROUND_PROMPT.format(
            missing_families=report.missing_families,
            source_text=source_text[:8000],
        )
        regrounded = _call_ollama_json(model=model, prompt=prompt) or {}
        new_clauses = regrounded.get("key_clauses", [])
        existing = {c["family"]: c for c in extraction.get("key_clauses", [])
                    if c.get("family") in report.verified_families}
        for c in new_clauses:
            existing[c.get("family")] = c
        extraction["key_clauses"] = list(existing.values())
    final_report = verify_spans(extraction.get("key_clauses", []), source_text)
    extraction["key_clauses"] = [
        c for c in extraction.get("key_clauses", [])
        if c.get("family") in final_report.verified_families
    ]
    extraction["evidence_verification"] = {
        "attempts": len(reports),
        "final_verified": final_report.verified,
        "final_missing": final_report.missing,
        "rejected_families": final_report.missing_families,
    }
    return extraction, reports
