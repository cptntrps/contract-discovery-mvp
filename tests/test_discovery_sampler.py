from contract_intel_mvp.discovery.sampler import sample_for_review
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature
from contract_intel_mvp.discovery.harvest import harvest_from_label


def _classified(verdict, confidence, doc_id):
    return {"doc_id": doc_id, "verdict": verdict, "confidence": confidence,
            "engine": "ollama", "evidence_per_clause_type": {}}


def _seed(tmp_root):
    init_signature(tmp_root, interview={"target_class": "X", "target_description": "x",
                                        "clause_types": []})
    init_library_from_signature(tmp_root)


def test_sampler_returns_mix_of_high_low_borderline(tmp_root):
    _seed(tmp_root)
    rows = (
        [_classified("yes", 0.95 - i*0.02, f"yes_high_{i}") for i in range(10)] +
        [_classified("yes", 0.55 + i*0.01, f"yes_low_{i}") for i in range(10)] +
        [_classified("no", 0.55 + i*0.01, f"no_near_{i}") for i in range(10)] +
        [_classified("no", 0.95 - i*0.02, f"no_high_{i}") for i in range(10)]
    )
    sample = sample_for_review(tmp_root, classifications=rows, batch_size=20, seed=1)
    assert len(sample) == 20
    ids = {s["doc_id"] for s in sample}
    assert any(i.startswith("yes_high_") for i in ids)
    assert any(i.startswith("yes_low_") for i in ids)
    assert any(i.startswith("no_near_") for i in ids)


def test_sampler_skips_already_labeled(tmp_root):
    _seed(tmp_root)
    cls = {"doc_id": "yes_high_0", "verdict": "yes",
           "evidence_per_clause_type": {}}
    harvest_from_label(tmp_root, classification=cls, sme_verdict="yes")
    rows = [_classified("yes", 0.95, "yes_high_0"),
            _classified("yes", 0.94, "yes_high_1")]
    sample = sample_for_review(tmp_root, classifications=rows, batch_size=2, seed=1)
    ids = {s["doc_id"] for s in sample}
    assert "yes_high_0" not in ids


def test_sampler_attaches_reason_codes(tmp_root):
    _seed(tmp_root)
    rows = [_classified("yes", 0.95, "a"), _classified("yes", 0.55, "b"),
            _classified("no", 0.55, "c"), _classified("no", 0.95, "d")]
    sample = sample_for_review(tmp_root, classifications=rows, batch_size=4, seed=1)
    reasons = {s["doc_id"]: s["reason"] for s in sample}
    assert reasons["a"] == "high_confidence_positive"
    assert reasons["b"] == "low_confidence_positive"
    assert reasons["c"] == "borderline_negative"
