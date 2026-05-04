from pathlib import Path
from contract_intel_mvp.discovery.harvest import harvest_from_label
from contract_intel_mvp.discovery.signature import init_signature, save_signature, load_signature
from contract_intel_mvp.discovery.library import init_library_from_signature, load_library


def _seed(tmp_root):
    init_signature(tmp_root, interview={
        "target_class": "License Agreement", "target_description": "x",
        "clause_types": [
            {"type": "license_grant", "description": "x", "is_must_have": True, "seed_variations": []},
            {"type": "primary_distribution", "description": "x", "is_must_have": False, "seed_variations": []},
        ],
    })
    init_library_from_signature(tmp_root)


def test_harvest_yes_appends_must_have_evidence(tmp_root):
    _seed(tmp_root)
    classification = {
        "doc_id": "doc_lic_0", "verdict": "yes",
        "evidence_per_clause_type": {
            "license_grant": "Licensor hereby grants to Licensee a non-exclusive license to use the Marks",
            "primary_distribution": "",
        },
    }
    harvest_from_label(tmp_root, classification=classification, sme_verdict="yes")
    lib = load_library(tmp_root)
    lg = next(ct for ct in lib["clause_types"] if ct["type"] == "license_grant")
    assert any(v["text"].startswith("Licensor hereby grants") for v in lg["variations"])
    assert any(v["confirmed_by"] == "auto_from_sme_yes" for v in lg["variations"])
    sig = load_signature(tmp_root)
    assert "doc_lic_0" in sig.confirmed_positive_doc_ids


def test_harvest_no_appends_must_not_have_evidence(tmp_root):
    _seed(tmp_root)
    classification = {
        "doc_id": "doc_dis_0", "verdict": "yes",  # agent thought yes
        "evidence_per_clause_type": {
            "license_grant": "license to use the Marks",
            "primary_distribution": "Company hereby appoints Distributor as the exclusive distributor",
        },
    }
    harvest_from_label(tmp_root, classification=classification, sme_verdict="no")
    lib = load_library(tmp_root)
    pd = next(ct for ct in lib["clause_types"] if ct["type"] == "primary_distribution")
    assert any("appoints Distributor" in v["text"] for v in pd["variations"])
    # license_grant should NOT have absorbed the false-positive evidence
    lg = next(ct for ct in lib["clause_types"] if ct["type"] == "license_grant")
    assert not any("license to use the Marks" in v["text"] for v in lg["variations"])
    sig = load_signature(tmp_root)
    assert "doc_dis_0" in sig.confirmed_negative_doc_ids


def test_harvest_borderline_does_not_change_library_or_signature(tmp_root):
    _seed(tmp_root)
    classification = {"doc_id": "doc_sa_0", "verdict": "yes",
                      "evidence_per_clause_type": {"license_grant": "x", "primary_distribution": ""}}
    harvest_from_label(tmp_root, classification=classification, sme_verdict="borderline")
    sig = load_signature(tmp_root)
    assert "doc_sa_0" not in sig.confirmed_positive_doc_ids
    assert "doc_sa_0" not in sig.confirmed_negative_doc_ids
