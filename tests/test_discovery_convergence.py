from pathlib import Path
import json
from contract_intel_mvp.discovery.convergence import (
    record_round, current_metrics, should_stop
)
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature
from contract_intel_mvp.discovery.harvest import harvest_from_label


def _seed(tmp_root):
    init_signature(tmp_root, interview={"target_class": "X", "target_description": "x",
                                        "clause_types": []})
    init_library_from_signature(tmp_root)


def test_record_round_persists(tmp_root):
    _seed(tmp_root)
    record_round(tmp_root, round_index=0, corrections=8, library_growth=12, batch_size=20)
    record_round(tmp_root, round_index=1, corrections=4, library_growth=3, batch_size=20)
    state = json.loads((tmp_root / "data" / "discovery" / "rounds.json").read_text())
    assert len(state["rounds"]) == 2
    assert state["rounds"][1]["library_growth"] == 3


def test_should_stop_when_corrections_low_AND_no_growth(tmp_root):
    _seed(tmp_root)
    record_round(tmp_root, round_index=0, corrections=10, library_growth=8, batch_size=20)
    record_round(tmp_root, round_index=1, corrections=2, library_growth=0, batch_size=20)
    assert should_stop(tmp_root, threshold=3, max_rounds=5) is True


def test_should_keep_going_if_library_still_growing(tmp_root):
    _seed(tmp_root)
    record_round(tmp_root, round_index=0, corrections=10, library_growth=8, batch_size=20)
    record_round(tmp_root, round_index=1, corrections=2, library_growth=5, batch_size=20)
    assert should_stop(tmp_root, threshold=3, max_rounds=5) is False


def test_should_stop_after_max_rounds(tmp_root):
    _seed(tmp_root)
    for i in range(5):
        record_round(tmp_root, round_index=i, corrections=10, library_growth=10, batch_size=20)
    assert should_stop(tmp_root, threshold=3, max_rounds=5) is True


def test_metrics_on_partial_gold(tmp_root):
    _seed(tmp_root)
    cls = [
        {"doc_id": "a", "verdict": "yes", "evidence_per_clause_type": {}},
        {"doc_id": "b", "verdict": "no",  "evidence_per_clause_type": {}},
        {"doc_id": "c", "verdict": "yes", "evidence_per_clause_type": {}},
    ]
    harvest_from_label(tmp_root, classification=cls[0], sme_verdict="yes")
    harvest_from_label(tmp_root, classification=cls[1], sme_verdict="yes")  # FN: agent said no
    harvest_from_label(tmp_root, classification=cls[2], sme_verdict="no")   # FP: agent said yes
    m = current_metrics(tmp_root, classifications=cls)
    assert m["true_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["false_positives"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
