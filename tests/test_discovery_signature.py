from pathlib import Path
import json
from contract_intel_mvp.discovery.signature import (
    init_signature, load_signature, save_signature, ClassSignature, ClauseType,
)


def test_init_with_clause_types(tmp_root):
    interview = {
        "target_class": "License Agreement",
        "target_description": "Primary purpose is granting IP rights",
        "clause_types": [
            {"type": "license_grant", "description": "Grantor gives Grantee right to use IP",
             "is_must_have": True,
             "seed_variations": ["Licensor hereby grants to Licensee a non-exclusive license"]},
            {"type": "primary_distribution_appointment",
             "description": "Appoints distributor as primary purpose",
             "is_must_have": False,
             "seed_variations": ["Company hereby appoints Distributor as the exclusive distributor"]},
        ],
    }
    sig = init_signature(tmp_root, interview=interview)
    assert sig.target_class == "License Agreement"
    assert len(sig.clause_types) == 2
    assert sig.clause_types[0].type == "license_grant"
    assert sig.clause_types[0].is_must_have is True
    assert sig.clause_types[1].is_must_have is False
    loaded = load_signature(tmp_root)
    assert loaded.target_class == "License Agreement"
    assert loaded.clause_types[0].type == "license_grant"


def test_signature_has_confirmed_doc_lists(tmp_root):
    sig = init_signature(tmp_root, interview={
        "target_class": "X", "target_description": "x", "clause_types": [],
    })
    assert sig.confirmed_positive_doc_ids == []
    assert sig.confirmed_negative_doc_ids == []
