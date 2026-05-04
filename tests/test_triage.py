from contract_intel_mvp.agent.triage import score_document, build_review_queue


def test_clean_extraction_low_uncertainty():
    extraction = {
        "doc_id": "doc_001",
        "contract_type": "License",
        "type_alternatives": [{"type": "Service", "score": 0.05}],
        "evidence_verification": {"attempts": 1, "final_missing": 0, "rejected_families": []},
        "key_clauses": [{"family": "termination"}, {"family": "law"}],
    }
    interview = {"key_clause_families": ["termination", "law"]}
    score = score_document(extraction, interview)
    assert score["uncertainty"] < 0.3
    assert score["reasons"] == []


def test_unverifiable_spans_raise_uncertainty():
    extraction = {
        "doc_id": "doc_002",
        "contract_type": "License",
        "type_alternatives": [],
        "evidence_verification": {"attempts": 3, "final_missing": 2,
                                  "rejected_families": ["indemnity", "ip"]},
        "key_clauses": [],
    }
    score = score_document(extraction, {"key_clause_families": []})
    assert "unverifiable_spans" in score["reasons"]
    assert score["uncertainty"] >= 0.4


def test_missing_expected_clauses_flagged():
    extraction = {
        "doc_id": "doc_003",
        "contract_type": "License",
        "type_alternatives": [],
        "evidence_verification": {"attempts": 1, "final_missing": 0, "rejected_families": []},
        "key_clauses": [{"family": "termination"}],
    }
    interview = {"key_clause_families": ["termination", "indemnity", "ip"]}
    score = score_document(extraction, interview)
    assert "missing_expected_clauses" in score["reasons"]


def test_close_type_alternatives_flagged():
    extraction = {
        "doc_id": "doc_004",
        "contract_type": "License",
        "type_alternatives": [{"type": "Service", "score": 0.85}],
        "evidence_verification": {"attempts": 1, "final_missing": 0, "rejected_families": []},
        "key_clauses": [],
    }
    score = score_document(extraction, {"key_clause_families": []})
    assert "close_type_alternative" in score["reasons"]


def test_build_review_queue_sorts_descending(tmp_root):
    extractions = [
        {"doc_id": "a", "contract_type": "X", "type_alternatives": [],
         "evidence_verification": {"attempts": 1, "final_missing": 0, "rejected_families": []},
         "key_clauses": [{"family": "t"}]},
        {"doc_id": "b", "contract_type": "Y", "type_alternatives": [],
         "evidence_verification": {"attempts": 3, "final_missing": 5,
                                   "rejected_families": ["x", "y", "z", "w", "v"]},
         "key_clauses": []},
    ]
    queue = build_review_queue(extractions, {"key_clause_families": ["t"]}, threshold=0.3)
    assert queue[0]["doc_id"] == "b"
    assert all(item["uncertainty"] >= 0.3 for item in queue)
