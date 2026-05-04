from contract_intel_mvp.agent.verifier import (
    verify_spans, EvidenceReport, _normalize_for_match
)


def test_all_spans_verified():
    text = "This Agreement may be terminated upon 30 days written notice."
    clauses = [
        {"family": "termination", "evidence_snippet": "terminated upon 30 days written notice"}
    ]
    report = verify_spans(clauses, text)
    assert report.verified == 1
    assert report.missing == 0
    assert report.is_clean is True


def test_some_spans_missing():
    text = "Termination requires 30 days notice."
    clauses = [
        {"family": "termination", "evidence_snippet": "30 days notice"},
        {"family": "indemnity",   "evidence_snippet": "Acme indemnifies Buyer for losses"},
    ]
    report = verify_spans(clauses, text)
    assert report.verified == 1
    assert report.missing == 1
    assert report.missing_families == ["indemnity"]
    assert report.is_clean is False


def test_normalization_handles_whitespace_and_case():
    text = "Governing\nLaw:  Delaware."
    clauses = [{"family": "law", "evidence_snippet": "governing law: delaware"}]
    report = verify_spans(clauses, text)
    assert report.is_clean is True


def test_empty_clauses_is_clean():
    report = verify_spans([], "any text")
    assert report.is_clean is True
    assert report.verified == 0


def test_extract_with_verification_retries_then_drops(monkeypatch):
    from contract_intel_mvp.agent import verifier as v
    text = "This Agreement may be terminated upon 30 days written notice."
    initial = {"key_clauses": [
        {"family": "termination", "evidence_snippet": "30 days written notice"},
        {"family": "indemnity",   "evidence_snippet": "fabricated quote not in source"},
    ]}
    monkeypatch.setattr(v, "_call_ollama_json", lambda **_: {
        "key_clauses": [{"family": "indemnity",
                         "evidence_snippet": "still fabricated"}]
    })
    final, reports = v.extract_with_verification(
        source_text=text, model="qwen3:4b", initial_extraction=initial, max_retries=2)
    families = [c["family"] for c in final["key_clauses"]]
    assert "termination" in families
    assert "indemnity" not in families
    assert final["evidence_verification"]["rejected_families"] == ["indemnity"]
