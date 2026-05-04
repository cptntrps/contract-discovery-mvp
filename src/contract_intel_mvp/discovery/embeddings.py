from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.request, urllib.error
import numpy as np


def _call_ollama_embed(text: str, *, model: str = "nomic-embed-text",
                       base_url: str = "http://127.0.0.1:11434") -> list[float] | None:
    body = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/embeddings", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8")).get("embedding")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


@dataclass
class EmbeddingsStore:
    doc_ids: list[str]
    matrix: np.ndarray
    model: str


def _docs_path(root): return root / "data" / "corpus" / "documents.jsonl"
def _emb_path(root): return root / "data" / "discovery" / "embeddings.jsonl"


def _read_existing(path):
    out = {}
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line); out[row["doc_id"]] = row
    return out


def embed_corpus(root: Path, *, model: str = "nomic-embed-text",
                 max_chars: int = 8000) -> dict[str, Any]:
    docs_path = _docs_path(root)
    if not docs_path.exists():
        raise ValueError("no documents.jsonl - run ingest first")
    out_path = _emb_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(out_path)
    embedded = skipped = 0
    failed_doc_ids: list[dict[str, str]] = []
    with out_path.open("a", encoding="utf-8") as f:
        for line in docs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            doc = json.loads(line); doc_id = doc["doc_id"]
            if doc_id in existing and existing[doc_id].get("model") == model:
                skipped += 1; continue
            text = (doc.get("text") or "")[:max_chars]
            if not text.strip():
                failed_doc_ids.append({"doc_id": doc_id, "reason": "empty_text"})
                continue
            vec = _call_ollama_embed(text, model=model)
            if vec is None:
                # Retry once with a smaller chunk in case of length / encoding edge cases.
                vec = _call_ollama_embed(text[:2000], model=model)
            if vec is None:
                failed_doc_ids.append({"doc_id": doc_id, "reason": "ollama_none",
                                        "title": (doc.get("title") or "")[:80],
                                        "text_len": len(text)})
                continue
            f.write(json.dumps({"doc_id": doc_id, "model": model, "embedding": vec}) + "\n")
            embedded += 1
    return {"embedded": embedded, "skipped": skipped, "failed": len(failed_doc_ids),
            "failed_doc_ids": failed_doc_ids,
            "path": str(out_path.relative_to(root))}


def load_embeddings(root: Path) -> EmbeddingsStore:
    p = _emb_path(root)
    if not p.exists():
        raise ValueError(f"no embeddings at {p}")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows: raise ValueError("empty embeddings file")
    return EmbeddingsStore(doc_ids=[r["doc_id"] for r in rows],
                           matrix=np.array([r["embedding"] for r in rows], dtype=np.float32),
                           model=rows[0]["model"])
