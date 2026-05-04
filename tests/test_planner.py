from contract_intel_mvp.agent.planner import deterministic_next_action


def test_first_action_is_ingest_when_no_corpus():
    state = {"docs_extracted": 0, "docs_total": 0, "review_pending": 0,
             "holdout_remaining": 0, "ingested": False}
    assert deterministic_next_action(state)["action"] == "ingest_corpus"


def test_after_ingest_extract_review_set():
    state = {"docs_extracted": 0, "docs_total": 40, "review_pending": 0,
             "holdout_remaining": 30, "ingested": True, "phase": "review"}
    assert deterministic_next_action(state)["action"] == "extract_review_batch"


def test_after_extraction_runs_triage():
    state = {"docs_extracted": 40, "docs_total": 40, "review_pending": 0,
             "holdout_remaining": 30, "ingested": True, "phase": "review",
             "triage_done": False}
    assert deterministic_next_action(state)["action"] == "triage"


def test_after_triage_awaits_human():
    state = {"docs_extracted": 40, "docs_total": 40, "review_pending": 12,
             "holdout_remaining": 30, "ingested": True, "phase": "review",
             "triage_done": True, "review_completed": False}
    assert deterministic_next_action(state)["action"] == "await_human"


def test_after_review_extracts_holdout():
    state = {"docs_extracted": 40, "docs_total": 40, "review_pending": 0,
             "holdout_remaining": 30, "ingested": True, "phase": "holdout",
             "triage_done": True, "review_completed": True, "cold_done": False}
    assert deterministic_next_action(state)["action"] == "extract_holdout_batch"


def test_after_holdout_runs_cold_small():
    state = {"docs_extracted": 30, "docs_total": 30, "review_pending": 0,
             "holdout_remaining": 0, "ingested": True, "phase": "holdout",
             "triage_done": True, "review_completed": True,
             "cold_done": False, "benchmarked": False}
    assert deterministic_next_action(state)["action"] == "extract_holdout_cold_small"


def test_terminates_when_done():
    state = {"docs_extracted": 70, "docs_total": 70, "review_pending": 0,
             "holdout_remaining": 0, "ingested": True, "phase": "holdout",
             "triage_done": True, "review_completed": True,
             "cold_done": True, "benchmarked": True}
    assert deterministic_next_action(state)["action"] == "stop"
