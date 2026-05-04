from pathlib import Path
import json
from contract_intel_mvp.discovery.library import (
    init_library_from_signature, append_variations, load_library, render_library_text
)
from contract_intel_mvp.discovery.signature import init_signature


def _seed_sig(tmp_root):
    return init_signature(tmp_root, interview={
        "target_class": "License Agreement",
        "target_description": "primary IP grant",
        "clause_types": [
            {"type": "license_grant", "description": "right to use IP",
             "is_must_have": True,
             "seed_variations": ["Licensor hereby grants to Licensee a license"]},
            {"type": "primary_distribution", "description": "appoints distributor",
             "is_must_have": False, "seed_variations": ["appoints Distributor"]},
        ],
    })


def test_init_library_seeds_from_signature(tmp_root):
    _seed_sig(tmp_root)
    lib = init_library_from_signature(tmp_root)
    assert lib["target_class"] == "License Agreement"
    assert len(lib["clause_types"]) == 2
    license_grant = next(ct for ct in lib["clause_types"] if ct["type"] == "license_grant")
    assert len(license_grant["variations"]) == 1
    v = license_grant["variations"][0]
    assert v["text"] == "Licensor hereby grants to Licensee a license"
    assert v["source_doc_id"] == "seed"
    assert v["confirmed_by"] == "interview_seed"
    assert "added_at" in v
    assert v["embedding_id"] is None


def test_append_variations_records_provenance(tmp_root):
    _seed_sig(tmp_root); init_library_from_signature(tmp_root)
    append_variations(tmp_root, clause_type="license_grant", variations=[
        {"text": "Owner shall and hereby does grant to User a limited license",
         "source_doc_id": "doc_lic_0", "confirmed_by": "auto_from_sme_yes"},
    ])
    lib = load_library(tmp_root)
    license_grant = next(ct for ct in lib["clause_types"] if ct["type"] == "license_grant")
    assert len(license_grant["variations"]) == 2
    new_v = license_grant["variations"][1]
    assert new_v["source_doc_id"] == "doc_lic_0"
    assert new_v["confirmed_by"] == "auto_from_sme_yes"
    assert "added_at" in new_v
    assert new_v["embedding_id"] is None


def test_append_variations_dedupes_exact_text(tmp_root):
    _seed_sig(tmp_root); init_library_from_signature(tmp_root)
    text = "Owner grants User a license"
    append_variations(tmp_root, clause_type="license_grant", variations=[
        {"text": text, "source_doc_id": "doc_a", "confirmed_by": "auto"},
        {"text": text, "source_doc_id": "doc_b", "confirmed_by": "auto"},
    ])
    lib = load_library(tmp_root)
    license_grant = next(ct for ct in lib["clause_types"] if ct["type"] == "license_grant")
    # 1 seed + 1 new (second occurrence deduped)
    assert len(license_grant["variations"]) == 2


def test_render_library_text_for_few_shot_prompt(tmp_root):
    _seed_sig(tmp_root); init_library_from_signature(tmp_root)
    text = render_library_text(tmp_root, max_per_type=5)
    assert "license_grant" in text
    assert "Licensor hereby grants" in text
    assert "primary_distribution" in text
