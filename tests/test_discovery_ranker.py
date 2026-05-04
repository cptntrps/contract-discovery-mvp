from pathlib import Path
from contract_intel_mvp.discovery.ranker import rank_corpus
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature
from contract_intel_mvp.discovery.embeddings import embed_corpus


def _fake_embed(text, **_):
    if "TRADEMARK LICENSE" in text or "license agreement" in text.lower():
        return [1.0, 0.0, 0.0, 0.1]
    if "DISTRIBUTOR" in text: return [0.3, 1.0, 0.0, 0.1]
    if "STRATEGIC ALLIANCE" in text: return [0.2, 0.3, 1.0, 0.1]
    return [0.0, 0.0, 0.0, 1.0]


def _seed(tmp_root):
    init_signature(tmp_root, interview={
        "target_class": "License Agreement",
        "target_description": "TRADEMARK LICENSE AGREEMENT royalty grant.",
        "clause_types": [{"type": "license_grant", "description": "x",
                          "is_must_have": True, "seed_variations": []}],
    })
    init_library_from_signature(tmp_root)


def test_rank_returns_license_first(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text"); _seed(tmp_root)
    ranked = rank_corpus(tmp_root, top_k=10)
    top5 = [r["doc_id"] for r in ranked[:5]]
    assert all(t.startswith("doc_lic_") for t in top5)


def test_rank_filename_rule_boosts_matching_titles(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    # Make all embeddings identical so only the filename rule discriminates
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t, **_: [0.5, 0.5, 0.5, 0.5])
    embed_corpus(tmp_root, model="nomic-embed-text"); _seed(tmp_root)
    ranked = rank_corpus(tmp_root, top_k=20)
    # LIC titles should outrank DIS/SA/UNK due to filename rule
    lic_positions = [i for i, r in enumerate(ranked) if r["doc_id"].startswith("doc_lic_")]
    other_positions = [i for i, r in enumerate(ranked) if not r["doc_id"].startswith("doc_lic_")]
    assert max(lic_positions) < min(other_positions)


def test_rank_demotes_confirmed_negatives(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text"); _seed(tmp_root)
    from contract_intel_mvp.discovery.signature import load_signature, save_signature
    sig = load_signature(tmp_root); sig.confirmed_negative_doc_ids = ["doc_lic_0"]
    save_signature(tmp_root, sig)
    ranked = rank_corpus(tmp_root, top_k=20)
    pos = next(i for i, r in enumerate(ranked) if r["doc_id"] == "doc_lic_0")
    assert pos >= 5
