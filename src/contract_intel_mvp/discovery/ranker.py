from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

from contract_intel_mvp.discovery import embeddings as _embeddings_mod
from contract_intel_mvp.discovery.embeddings import load_embeddings
from contract_intel_mvp.discovery.signature import load_signature
from contract_intel_mvp.discovery.library import load_library


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def _doc_titles(root: Path) -> dict[str, str]:
    p = root / "data" / "corpus" / "documents.jsonl"
    out = {}
    if not p.exists(): return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line); out[d["doc_id"]] = (d.get("title") or "")
    return out


def _filename_score(title: str, target_class: str, anti_keywords: list[str]) -> float:
    t = title.lower(); cls = target_class.lower()
    score = 0.0
    if cls and cls in t: score += 0.25
    for w in cls.split():
        if len(w) >= 5 and w in t: score += 0.05
    for a in anti_keywords:
        if a.lower() in t: score -= 0.30
    return score


def rank_corpus(root: Path, *, top_k: int | None = None,
                positive_boost: float = 0.20, negative_demote: float = 1.0) -> list[dict[str, Any]]:
    sig = load_signature(root)
    try:
        lib = load_library(root)
    except ValueError:
        lib = {"clause_types": []}
    store = load_embeddings(root)

    query_text = sig.target_description
    for ct in lib.get("clause_types", []):
        if ct.get("is_must_have"):
            for v in ct["variations"][:3]:
                query_text += "\n" + v["text"]
    qvec = _embeddings_mod._call_ollama_embed(query_text, model=store.model)
    if qvec is None:
        raise RuntimeError("could not embed signature query")
    q = np.array(qvec, dtype=np.float32)
    M = _normalize(store.matrix); qn = _normalize(q[None, :])[0]
    sims = (M @ qn).tolist()

    anti_keywords: list[str] = []
    for ct in lib.get("clause_types", []):
        if not ct.get("is_must_have"):
            for word in ct["type"].split("_"):
                if len(word) >= 4: anti_keywords.append(word)

    titles = _doc_titles(root)
    pos_set = set(sig.confirmed_positive_doc_ids)
    neg_set = set(sig.confirmed_negative_doc_ids)

    scored = []
    for doc_id, sim in zip(store.doc_ids, sims):
        title_bonus = _filename_score(titles.get(doc_id, ""), sig.target_class, anti_keywords)
        score = float(sim) + title_bonus
        if doc_id in pos_set: score = min(2.0, score + positive_boost)
        if doc_id in neg_set: score = max(-2.0, score - negative_demote)
        scored.append({"doc_id": doc_id, "score": score, "similarity": float(sim),
                       "title_bonus": title_bonus,
                       "label": "positive" if doc_id in pos_set else
                                "negative" if doc_id in neg_set else "unlabeled"})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k] if top_k else scored
