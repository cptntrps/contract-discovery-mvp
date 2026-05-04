from pathlib import Path
import json
from contract_intel_mvp.discovery.embeddings import (
    embed_corpus, load_embeddings, EmbeddingsStore
)


def test_embed_corpus_writes_jsonl(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", lambda text, **_: [float(len(text) % 7), 0.1, 0.2, 0.3])
    out = embed_corpus(tmp_root, model="nomic-embed-text")
    assert out["embedded"] == 20
    rows = [json.loads(l) for l in (tmp_root / "data" / "discovery" / "embeddings.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    assert all(len(r["embedding"]) == 4 for r in rows)


def test_embed_corpus_skips_existing(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    calls = []
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t, **_: calls.append(t) or [0.0]*4)
    embed_corpus(tmp_root, model="nomic-embed-text")
    n = len(calls)
    embed_corpus(tmp_root, model="nomic-embed-text")
    assert len(calls) == n


def test_load_embeddings_returns_store(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t, **_: [1.0, 0.0, 0.0, 0.0])
    embed_corpus(tmp_root, model="nomic-embed-text")
    store = load_embeddings(tmp_root)
    assert isinstance(store, EmbeddingsStore)
    assert len(store.doc_ids) == 20
    assert store.matrix.shape == (20, 4)
