# Discovery Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pivot the MVP from "extract structured fields from every contract" to "find the contracts of one specific type in a haystack." Drop a folder of N contracts, describe the target type to an agent, and get back a clean labelled list with precision/recall, after the agent has shown you ~60 borderline cases for yes/no judgments across 3 rounds of active learning.

**Architecture:** Add a `discovery/` module that embeds every contract once with Ollama's `nomic-embed-text`, builds a target-class signature from the interview output, ranks all contracts by similarity to the signature, classifies the top candidates with the small Ollama model (`qwen3:4b`) producing yes/no + confidence, then runs an active-learning loop: an uncertainty sampler picks 20 docs per round for the SME, the agent updates the signature from the corrections, re-ranks and re-classifies. Stops when corrections-per-round drops below threshold. The existing `agent/` planner is extended with a new state machine for discovery mode; existing extraction code stays untouched (kept as the secondary "extract details on the confirmed positives" path).

**Tech Stack:** Python 3.10+, pytest, FastAPI/http.server (existing `web.py`), Ollama HTTP API at `127.0.0.1:11434` (`nomic-embed-text` for embeddings, `qwen3:4b` for classification, `qwen2.5:14b` for arbitration on borderline cases), `numpy` for cosine similarity, no SQLite (state stays in JSON under `data/discovery/`).

---

## Pre-flight (Task 0): Confirm fixture + numpy dep

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add numpy to dev deps**

Append to `pyproject.toml [project.optional-dependencies] dev`:
```toml
dev = ["pytest", "pytest-asyncio", "httpx", "fastapi", "numpy"]
```

Install:
```bash
. .venv/bin/activate && pip install -q numpy
```

- [ ] **Step 2: Add a `corpus_fixture` to `tests/conftest.py`**

```python
@pytest.fixture
def discovery_corpus(tmp_root: Path) -> Path:
    """20 fake contracts: 5 publishing agreements, 5 licenses, 5 NDAs, 5 unknown."""
    import json
    docs = []
    samples = [
        ("pub", "PUBLISHING AGREEMENT. Author grants Publisher exclusive rights to publish the Work in print and digital editions. Royalties paid quarterly at 12% of net receipts. Term 7 years."),
        ("lic", "SOFTWARE LICENSE AGREEMENT. Licensor grants Licensee a non-exclusive license to use the Software. Annual fee $50,000. Term 1 year, auto-renew."),
        ("nda", "MUTUAL NON-DISCLOSURE AGREEMENT. Each party agrees to keep Confidential Information secret for 5 years. No fees. No license granted."),
        ("unk", "EQUIPMENT LEASE. Lessor leases the Equipment to Lessee for monthly rent of $2,000. Term 36 months."),
    ]
    for cls, text in samples:
        for i in range(5):
            doc_id = f"doc_{cls}_{i}"
            docs.append({"doc_id": doc_id, "title": f"{cls.upper()} {i}",
                         "text": f"{text} (instance {i})", "source": "fixture"})
    path = tmp_root / "data" / "corpus" / "documents.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    return path
```

- [ ] **Step 3: Verify**

```bash
pytest tests/ -q
```
Expected: 42 passed (no new tests yet; just confirming nothing broke).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/conftest.py
git commit -m "chore: add numpy dep + discovery_corpus fixture"
```

---

## Task 1: Embeddings store

Embed every contract once with `nomic-embed-text`, persist to `data/discovery/embeddings.jsonl`. Idempotent: skips docs already embedded.

**Files:**
- Create: `src/contract_intel_mvp/discovery/__init__.py`
- Create: `src/contract_intel_mvp/discovery/embeddings.py`
- Create: `tests/test_discovery_embeddings.py`

- [ ] **Step 1: Write failing test**

`tests/test_discovery_embeddings.py`:
```python
from pathlib import Path
import json
from contract_intel_mvp.discovery.embeddings import (
    embed_corpus, load_embeddings, EmbeddingsStore
)


def test_embed_corpus_writes_jsonl(tmp_root: Path, discovery_corpus: Path, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    # Stub Ollama: return a deterministic 4-d vector seeded by text length.
    monkeypatch.setattr(e, "_call_ollama_embed",
                        lambda text: [float(len(text) % 7), 0.1, 0.2, 0.3])
    out = embed_corpus(tmp_root, model="nomic-embed-text")
    assert out["embedded"] == 20
    path = tmp_root / "data" / "discovery" / "embeddings.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 20
    assert {r["doc_id"] for r in rows} == {f"doc_{c}_{i}" for c in ("pub","lic","nda","unk") for i in range(5)}
    assert all(len(r["embedding"]) == 4 for r in rows)
    assert all(r["model"] == "nomic-embed-text" for r in rows)


def test_embed_corpus_skips_existing(tmp_root: Path, discovery_corpus: Path, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    calls = []
    def stub(text):
        calls.append(text)
        return [0.0, 0.0, 0.0, 0.0]
    monkeypatch.setattr(e, "_call_ollama_embed", stub)
    embed_corpus(tmp_root, model="nomic-embed-text")
    n_first = len(calls)
    embed_corpus(tmp_root, model="nomic-embed-text")
    assert len(calls) == n_first  # second run skipped all 20


def test_load_embeddings_returns_store(tmp_root: Path, discovery_corpus: Path, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t: [1.0, 0.0, 0.0, 0.0])
    embed_corpus(tmp_root, model="nomic-embed-text")
    store = load_embeddings(tmp_root)
    assert isinstance(store, EmbeddingsStore)
    assert len(store.doc_ids) == 20
    assert store.matrix.shape == (20, 4)
```

- [ ] **Step 2: Run, expect FAIL (module missing)**

```bash
pytest tests/test_discovery_embeddings.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/contract_intel_mvp/discovery/__init__.py`:
```python
"""Discovery: find contracts of one target class in a haystack."""
```

`src/contract_intel_mvp/discovery/embeddings.py`:
```python
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
            payload = json.loads(r.read().decode("utf-8"))
        return payload.get("embedding")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


@dataclass
class EmbeddingsStore:
    doc_ids: list[str]
    matrix: np.ndarray  # shape (N, D)
    model: str


def _docs_path(root: Path) -> Path:
    return root / "data" / "corpus" / "documents.jsonl"


def _emb_path(root: Path) -> Path:
    return root / "data" / "discovery" / "embeddings.jsonl"


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["doc_id"]] = row
    return out


def embed_corpus(root: Path, *, model: str = "nomic-embed-text",
                 max_chars: int = 8000) -> dict[str, Any]:
    docs_path = _docs_path(root)
    if not docs_path.exists():
        raise ValueError("no documents.jsonl - run ingest first")
    out_path = _emb_path(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(out_path)
    embedded = 0
    skipped = 0
    failed = 0
    with out_path.open("a", encoding="utf-8") as f:
        for line in docs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            if doc_id in existing and existing[doc_id].get("model") == model:
                skipped += 1
                continue
            text = (doc.get("text") or "")[:max_chars]
            vec = _call_ollama_embed(text, model=model)
            if vec is None:
                failed += 1
                continue
            row = {"doc_id": doc_id, "model": model, "embedding": vec}
            f.write(json.dumps(row) + "\n")
            embedded += 1
    return {"embedded": embedded, "skipped": skipped, "failed": failed,
            "path": str(out_path.relative_to(root))}


def load_embeddings(root: Path) -> EmbeddingsStore:
    path = _emb_path(root)
    if not path.exists():
        raise ValueError(f"no embeddings at {path}; run embed_corpus first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty embeddings file")
    doc_ids = [r["doc_id"] for r in rows]
    matrix = np.array([r["embedding"] for r in rows], dtype=np.float32)
    return EmbeddingsStore(doc_ids=doc_ids, matrix=matrix, model=rows[0]["model"])
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_discovery_embeddings.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/discovery/ tests/test_discovery_embeddings.py
git commit -m "feat(discovery): per-doc embeddings store via ollama nomic-embed-text"
```

---

## Task 2: Class signature

A target-class signature has three parts: a free-text description (used to embed and to prompt the classifier), structured "must contain" / "must not contain" terms, and an evolving list of confirmed positive doc_ids and rejected doc_ids.

**Files:**
- Create: `src/contract_intel_mvp/discovery/signature.py`
- Create: `tests/test_discovery_signature.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_signature.py`:
```python
from pathlib import Path
import json
from contract_intel_mvp.discovery.signature import (
    init_signature, load_signature, save_signature,
    update_signature_from_label, ClassSignature,
)


def test_init_signature_seeds_from_interview(tmp_root: Path):
    interview = {
        "target_class": "Publishing Agreement",
        "target_description": "Author grants Publisher rights to publish a Work; royalties stated.",
        "must_contain_terms": ["author", "publisher", "royalties"],
        "must_not_contain_terms": ["non-disclosure"],
    }
    sig = init_signature(tmp_root, interview=interview)
    assert sig.target_class == "Publishing Agreement"
    assert "royalties" in sig.must_contain_terms
    assert sig.confirmed_positive_doc_ids == []
    assert sig.confirmed_negative_doc_ids == []
    # Persisted
    loaded = load_signature(tmp_root)
    assert loaded.target_class == "Publishing Agreement"


def test_update_signature_records_yes_label(tmp_root: Path):
    sig = init_signature(tmp_root, interview={
        "target_class": "Publishing Agreement",
        "target_description": "x", "must_contain_terms": [], "must_not_contain_terms": [],
    })
    sig = update_signature_from_label(tmp_root, doc_id="doc_pub_0", verdict="yes",
                                       evidence_excerpt="Author grants Publisher exclusive rights")
    assert "doc_pub_0" in sig.confirmed_positive_doc_ids
    assert "Author grants Publisher" in sig.confirmed_examples_excerpts[0]


def test_update_signature_records_no_label(tmp_root: Path):
    init_signature(tmp_root, interview={
        "target_class": "Publishing Agreement",
        "target_description": "x", "must_contain_terms": [], "must_not_contain_terms": [],
    })
    sig = update_signature_from_label(tmp_root, doc_id="doc_lic_0", verdict="no",
                                       evidence_excerpt="Software license, not publishing")
    assert "doc_lic_0" in sig.confirmed_negative_doc_ids
    assert "Software license, not publishing" in sig.rejected_examples_excerpts[0]


def test_update_signature_idempotent_on_same_doc(tmp_root: Path):
    init_signature(tmp_root, interview={
        "target_class": "X", "target_description": "x",
        "must_contain_terms": [], "must_not_contain_terms": [],
    })
    update_signature_from_label(tmp_root, doc_id="doc_a", verdict="yes", evidence_excerpt="e1")
    sig = update_signature_from_label(tmp_root, doc_id="doc_a", verdict="yes", evidence_excerpt="e1-again")
    assert sig.confirmed_positive_doc_ids == ["doc_a"]  # not duplicated
```

- [ ] **Step 2: Run, expect FAIL**

```bash
pytest tests/test_discovery_signature.py -v
```

- [ ] **Step 3: Implement**

`src/contract_intel_mvp/discovery/signature.py`:
```python
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ClassSignature:
    target_class: str
    target_description: str
    must_contain_terms: list[str] = field(default_factory=list)
    must_not_contain_terms: list[str] = field(default_factory=list)
    confirmed_positive_doc_ids: list[str] = field(default_factory=list)
    confirmed_negative_doc_ids: list[str] = field(default_factory=list)
    confirmed_examples_excerpts: list[str] = field(default_factory=list)
    rejected_examples_excerpts: list[str] = field(default_factory=list)


def _path(root: Path) -> Path:
    return root / "data" / "discovery" / "signature.json"


def init_signature(root: Path, *, interview: dict[str, Any]) -> ClassSignature:
    sig = ClassSignature(
        target_class=str(interview.get("target_class", "")).strip() or "Target Class",
        target_description=str(interview.get("target_description", "")).strip(),
        must_contain_terms=list(interview.get("must_contain_terms") or []),
        must_not_contain_terms=list(interview.get("must_not_contain_terms") or []),
    )
    save_signature(root, sig)
    return sig


def save_signature(root: Path, sig: ClassSignature) -> None:
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(sig), indent=2), encoding="utf-8")


def load_signature(root: Path) -> ClassSignature:
    p = _path(root)
    if not p.exists():
        raise ValueError(f"no signature at {p}")
    return ClassSignature(**json.loads(p.read_text(encoding="utf-8")))


def update_signature_from_label(root: Path, *, doc_id: str, verdict: str,
                                evidence_excerpt: str) -> ClassSignature:
    if verdict not in {"yes", "no", "borderline"}:
        raise ValueError(f"verdict must be yes/no/borderline, got {verdict!r}")
    sig = load_signature(root)
    if verdict == "yes":
        if doc_id not in sig.confirmed_positive_doc_ids:
            sig.confirmed_positive_doc_ids.append(doc_id)
            sig.confirmed_examples_excerpts.append(evidence_excerpt[:400])
    elif verdict == "no":
        if doc_id not in sig.confirmed_negative_doc_ids:
            sig.confirmed_negative_doc_ids.append(doc_id)
            sig.rejected_examples_excerpts.append(evidence_excerpt[:400])
    save_signature(root, sig)
    return sig
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_discovery_signature.py -v
git add src/contract_intel_mvp/discovery/signature.py tests/test_discovery_signature.py
git commit -m "feat(discovery): class signature schema with positive/negative examples"
```

---

## Task 3: Pre-screen ranker

Cosine similarity between every doc embedding and the signature description's embedding (computed on demand). Output a ranked list of (doc_id, score). Boost confirmed positives, demote confirmed negatives.

**Files:**
- Create: `src/contract_intel_mvp/discovery/ranker.py`
- Create: `tests/test_discovery_ranker.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_ranker.py`:
```python
from pathlib import Path
import json
import numpy as np
from contract_intel_mvp.discovery.ranker import rank_corpus
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.embeddings import embed_corpus


def _fake_embed(text: str):
    """Deterministic embedding: pub docs get vector close to (1,0,0,0), lic close to (0,1,0,0)."""
    if "PUBLISHING" in text or "publishing agreement" in text.lower() or "Author grants Publisher" in text:
        return [1.0, 0.0, 0.0, 0.1]
    if "LICENSE" in text or "license" in text.lower():
        return [0.0, 1.0, 0.0, 0.1]
    if "NON-DISCLOSURE" in text:
        return [0.0, 0.0, 1.0, 0.1]
    return [0.0, 0.0, 0.0, 1.0]


def test_rank_corpus_pub_signature_returns_pub_docs_first(tmp_root: Path, discovery_corpus: Path, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text")
    init_signature(tmp_root, interview={
        "target_class": "Publishing Agreement",
        "target_description": "PUBLISHING AGREEMENT between Author and Publisher with royalties.",
        "must_contain_terms": [], "must_not_contain_terms": [],
    })
    ranked = rank_corpus(tmp_root, top_k=10)
    top_ids = [r["doc_id"] for r in ranked[:5]]
    assert all(tid.startswith("doc_pub_") for tid in top_ids)


def test_rank_corpus_demotes_confirmed_negatives(tmp_root: Path, discovery_corpus: Path, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text")
    init_signature(tmp_root, interview={
        "target_class": "Publishing Agreement",
        "target_description": "PUBLISHING AGREEMENT between Author and Publisher with royalties.",
        "must_contain_terms": [], "must_not_contain_terms": [],
    })
    from contract_intel_mvp.discovery.signature import update_signature_from_label
    update_signature_from_label(tmp_root, doc_id="doc_pub_0", verdict="no", evidence_excerpt="not actually pub")
    ranked = rank_corpus(tmp_root, top_k=20)
    pos = next(i for i, r in enumerate(ranked) if r["doc_id"] == "doc_pub_0")
    assert pos >= 5  # demoted
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement**

`src/contract_intel_mvp/discovery/ranker.py`:
```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

from contract_intel_mvp.discovery.embeddings import (
    load_embeddings, _call_ollama_embed
)
from contract_intel_mvp.discovery.signature import load_signature


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def rank_corpus(root: Path, *, top_k: int | None = None,
                positive_boost: float = 0.2, negative_demote: float = 0.3) -> list[dict[str, Any]]:
    sig = load_signature(root)
    store = load_embeddings(root)
    query_text = sig.target_description
    if sig.confirmed_examples_excerpts:
        query_text += "\n\nConfirmed examples:\n" + "\n".join(sig.confirmed_examples_excerpts[:5])
    qvec = _call_ollama_embed(query_text, model=store.model)
    if qvec is None:
        raise RuntimeError("could not embed signature query")
    q = np.array(qvec, dtype=np.float32)
    M = _normalize(store.matrix)
    qn = _normalize(q[None, :])[0]
    sims = (M @ qn).tolist()
    pos_set = set(sig.confirmed_positive_doc_ids)
    neg_set = set(sig.confirmed_negative_doc_ids)
    scored: list[dict[str, Any]] = []
    for doc_id, sim in zip(store.doc_ids, sims):
        score = float(sim)
        if doc_id in pos_set:
            score = min(1.0, score + positive_boost)
        if doc_id in neg_set:
            score = max(-1.0, score - negative_demote)
        scored.append({"doc_id": doc_id, "score": score, "similarity": float(sim),
                       "label": "positive" if doc_id in pos_set else
                                "negative" if doc_id in neg_set else "unlabeled"})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k] if top_k else scored
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_discovery_ranker.py -v
git add src/contract_intel_mvp/discovery/ranker.py tests/test_discovery_ranker.py
git commit -m "feat(discovery): cosine pre-screen ranker with positive/negative example boost"
```

---

## Task 4: Classifier (yes/no + confidence)

For the top-K ranked candidates, prompt the small Ollama model with the signature + the doc text. Parse `{verdict, confidence, evidence_excerpt}`. Tag rows with `engine: ollama|heuristic_fallback`.

**Files:**
- Create: `src/contract_intel_mvp/discovery/classifier.py`
- Create: `src/contract_intel_mvp/prompts.py` *(append CLASSIFY_PROMPT)*
- Create: `tests/test_discovery_classifier.py`

- [ ] **Step 1: Append prompt**

In `src/contract_intel_mvp/prompts.py`, append:
```python
CLASSIFY_PROMPT = """You are deciding whether a single contract belongs to the target class.

Target class: {target_class}
Target description: {target_description}
Must contain (any): {must_contain_terms}
Must NOT contain: {must_not_contain_terms}

Confirmed positive examples (excerpts):
{positive_examples}

Confirmed negative examples (excerpts):
{negative_examples}

Contract under review (first 6000 chars):
{doc_text}

Answer in JSON only. Schema:
{{
  "verdict": "yes" | "no",
  "confidence": <float 0..1>,
  "evidence_excerpt": "<verbatim substring from the contract that supports your verdict, max 300 chars>",
  "rationale": "<one sentence>"
}}
"""
```

- [ ] **Step 2: Failing test**

`tests/test_discovery_classifier.py`:
```python
from pathlib import Path
from contract_intel_mvp.discovery.classifier import classify_candidates
from contract_intel_mvp.discovery.signature import init_signature


def test_classifier_calls_model_for_each_candidate(tmp_root: Path, discovery_corpus: Path, monkeypatch):
    init_signature(tmp_root, interview={
        "target_class": "Publishing Agreement",
        "target_description": "x", "must_contain_terms": [], "must_not_contain_terms": [],
    })
    calls = []
    def stub(*, model, prompt):
        calls.append(model)
        # "yes" for any doc that mentions PUBLISHING
        verdict = "yes" if "PUBLISHING" in prompt else "no"
        return {"verdict": verdict, "confidence": 0.8,
                "evidence_excerpt": "stub", "rationale": "stub"}
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", stub)
    candidates = [{"doc_id": f"doc_pub_{i}", "score": 0.9} for i in range(3)] + \
                 [{"doc_id": f"doc_lic_{i}", "score": 0.5} for i in range(3)]
    out = classify_candidates(tmp_root, candidates=candidates, model="qwen3:4b")
    by_id = {r["doc_id"]: r for r in out}
    assert by_id["doc_pub_0"]["verdict"] == "yes"
    assert by_id["doc_lic_0"]["verdict"] == "no"
    assert all(r["engine"] == "ollama" for r in out)


def test_classifier_falls_back_when_model_returns_none(tmp_root: Path, discovery_corpus: Path, monkeypatch):
    init_signature(tmp_root, interview={
        "target_class": "X", "target_description": "x",
        "must_contain_terms": [], "must_not_contain_terms": [],
    })
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **_: None)
    out = classify_candidates(tmp_root, candidates=[{"doc_id": "doc_pub_0", "score": 0.9}],
                              model="qwen3:4b")
    assert out[0]["engine"] == "heuristic_fallback"
    assert out[0]["verdict"] in {"yes", "no"}
    assert 0.0 <= out[0]["confidence"] <= 1.0
```

- [ ] **Step 3: Run, FAIL**

- [ ] **Step 4: Implement**

`src/contract_intel_mvp/discovery/classifier.py`:
```python
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

from contract_intel_mvp.pipeline import _call_ollama_json, _load_documents
from contract_intel_mvp.prompts import CLASSIFY_PROMPT
from contract_intel_mvp.discovery.signature import load_signature


def _heuristic_classify(doc_text: str, sig) -> dict[str, Any]:
    text = doc_text.lower()
    pos_hits = sum(1 for t in sig.must_contain_terms if t.lower() in text)
    neg_hits = sum(1 for t in sig.must_not_contain_terms if t.lower() in text)
    must_terms = max(1, len(sig.must_contain_terms))
    score = pos_hits / must_terms - 0.5 * neg_hits
    verdict = "yes" if score > 0.4 else "no"
    confidence = max(0.0, min(1.0, abs(score - 0.4) + 0.4))
    return {"verdict": verdict, "confidence": confidence,
            "evidence_excerpt": doc_text[:200], "rationale": "heuristic fallback"}


def classify_candidates(root: Path, *, candidates: list[dict[str, Any]],
                        model: str) -> list[dict[str, Any]]:
    sig = load_signature(root)
    docs_by_id = {d.doc_id: d for d in _load_documents(root)}
    results: list[dict[str, Any]] = []
    for cand in candidates:
        doc = docs_by_id.get(cand["doc_id"])
        if doc is None:
            continue
        prompt = CLASSIFY_PROMPT.format(
            target_class=sig.target_class,
            target_description=sig.target_description,
            must_contain_terms=", ".join(sig.must_contain_terms) or "(none)",
            must_not_contain_terms=", ".join(sig.must_not_contain_terms) or "(none)",
            positive_examples="\n---\n".join(sig.confirmed_examples_excerpts[:3]) or "(none yet)",
            negative_examples="\n---\n".join(sig.rejected_examples_excerpts[:3]) or "(none yet)",
            doc_text=(doc.text or "")[:6000],
        )
        parsed = _call_ollama_json(model=model, prompt=prompt)
        if parsed and isinstance(parsed.get("verdict"), str) and \
                parsed["verdict"] in {"yes", "no"}:
            engine = "ollama"
            try:
                parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            except (TypeError, ValueError):
                parsed["confidence"] = 0.5
            parsed.setdefault("evidence_excerpt", "")
            parsed.setdefault("rationale", "")
        else:
            parsed = _heuristic_classify(doc.text or "", sig)
            engine = "heuristic_fallback"
        parsed["doc_id"] = cand["doc_id"]
        parsed["screen_score"] = cand.get("score")
        parsed["engine"] = engine
        results.append(parsed)
    return results
```

- [ ] **Step 5: Pass + commit**

```bash
pytest tests/test_discovery_classifier.py -v
git add src/contract_intel_mvp/discovery/classifier.py src/contract_intel_mvp/prompts.py tests/test_discovery_classifier.py
git commit -m "feat(discovery): yes/no/confidence classifier with signature-aware prompt"
```

---

## Task 5: Uncertainty sampler

Pick 20 docs per round to show the SME: 5 highest-confidence positives, 5 lowest-confidence positives, 5 highest-confidence negatives near threshold, 5 random borderline. Skip docs already in `confirmed_positive_doc_ids` or `confirmed_negative_doc_ids`.

**Files:**
- Create: `src/contract_intel_mvp/discovery/sampler.py`
- Create: `tests/test_discovery_sampler.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_sampler.py`:
```python
from pathlib import Path
from contract_intel_mvp.discovery.sampler import sample_for_review
from contract_intel_mvp.discovery.signature import init_signature, update_signature_from_label


def _classified(verdict, confidence, doc_id):
    return {"doc_id": doc_id, "verdict": verdict, "confidence": confidence,
            "engine": "ollama", "evidence_excerpt": "x"}


def test_sampler_returns_mix_of_high_low_borderline(tmp_root: Path):
    init_signature(tmp_root, interview={"target_class": "X", "target_description": "x",
                                        "must_contain_terms": [], "must_not_contain_terms": []})
    rows = (
        [_classified("yes", 0.95 - i*0.02, f"yes_high_{i}") for i in range(10)] +
        [_classified("yes", 0.55 + i*0.01, f"yes_low_{i}") for i in range(10)] +
        [_classified("no", 0.55 + i*0.01, f"no_near_{i}") for i in range(10)] +
        [_classified("no", 0.95 - i*0.02, f"no_high_{i}") for i in range(10)]
    )
    sample = sample_for_review(tmp_root, classifications=rows, batch_size=20, seed=1)
    assert len(sample) == 20
    ids = {s["doc_id"] for s in sample}
    # Should pull from each bucket
    assert any(i.startswith("yes_high_") for i in ids)
    assert any(i.startswith("yes_low_") for i in ids)
    assert any(i.startswith("no_near_") for i in ids)


def test_sampler_skips_already_labeled(tmp_root: Path):
    init_signature(tmp_root, interview={"target_class": "X", "target_description": "x",
                                        "must_contain_terms": [], "must_not_contain_terms": []})
    update_signature_from_label(tmp_root, doc_id="yes_high_0", verdict="yes", evidence_excerpt="e")
    rows = [_classified("yes", 0.95, "yes_high_0"),
            _classified("yes", 0.94, "yes_high_1")]
    sample = sample_for_review(tmp_root, classifications=rows, batch_size=2, seed=1)
    ids = {s["doc_id"] for s in sample}
    assert "yes_high_0" not in ids


def test_sampler_attaches_reason_codes(tmp_root: Path):
    init_signature(tmp_root, interview={"target_class": "X", "target_description": "x",
                                        "must_contain_terms": [], "must_not_contain_terms": []})
    rows = [_classified("yes", 0.95, "a"), _classified("yes", 0.55, "b"),
            _classified("no", 0.55, "c"), _classified("no", 0.95, "d")]
    sample = sample_for_review(tmp_root, classifications=rows, batch_size=4, seed=1)
    reasons = {s["doc_id"]: s["reason"] for s in sample}
    assert reasons["a"] == "high_confidence_positive"
    assert reasons["b"] == "low_confidence_positive"
    assert reasons["c"] == "borderline_negative"
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement**

`src/contract_intel_mvp/discovery/sampler.py`:
```python
from __future__ import annotations
import random
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.signature import load_signature


def _strip_labeled(rows: list[dict[str, Any]], labeled: set[str]) -> list[dict[str, Any]]:
    return [r for r in rows if r["doc_id"] not in labeled]


def sample_for_review(root: Path, *, classifications: list[dict[str, Any]],
                      batch_size: int = 20, seed: int = 0) -> list[dict[str, Any]]:
    sig = load_signature(root)
    labeled = set(sig.confirmed_positive_doc_ids) | set(sig.confirmed_negative_doc_ids)
    rows = _strip_labeled(classifications, labeled)
    yes_rows = sorted([r for r in rows if r["verdict"] == "yes"],
                      key=lambda r: r["confidence"], reverse=True)
    no_rows = sorted([r for r in rows if r["verdict"] == "no"],
                     key=lambda r: r["confidence"])
    quota = max(1, batch_size // 4)
    picks: list[dict[str, Any]] = []
    seen: set[str] = set()
    def take(bucket, reason, n):
        for r in bucket:
            if len(picks) >= batch_size: return
            if r["doc_id"] in seen: continue
            entry = dict(r); entry["reason"] = reason
            picks.append(entry); seen.add(r["doc_id"])
            if sum(1 for p in picks if p["reason"] == reason) >= n: return
    take(yes_rows, "high_confidence_positive", quota)
    take(yes_rows[::-1], "low_confidence_positive", quota)
    take(no_rows, "borderline_negative", quota)
    rng = random.Random(seed)
    remaining = [r for r in rows if r["doc_id"] not in seen]
    rng.shuffle(remaining)
    take(remaining, "borderline_random", batch_size - len(picks))
    return picks[:batch_size]
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_discovery_sampler.py -v
git add src/contract_intel_mvp/discovery/sampler.py tests/test_discovery_sampler.py
git commit -m "feat(discovery): uncertainty sampler with reason codes"
```

---

## Task 6: Convergence detector + precision/recall

Compute precision/recall on the labeled set (the SME-confirmed positives + negatives, treated as the partial gold). Track corrections-per-round. Stop when corrections-per-round drops below 3 or after 5 rounds.

**Files:**
- Create: `src/contract_intel_mvp/discovery/convergence.py`
- Create: `tests/test_discovery_convergence.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_convergence.py`:
```python
from pathlib import Path
import json
from contract_intel_mvp.discovery.convergence import (
    record_round, current_metrics, should_stop
)
from contract_intel_mvp.discovery.signature import init_signature, update_signature_from_label


def _seed_sig(tmp_root: Path):
    init_signature(tmp_root, interview={
        "target_class": "X", "target_description": "x",
        "must_contain_terms": [], "must_not_contain_terms": [],
    })


def test_record_round_persists(tmp_root: Path):
    _seed_sig(tmp_root)
    record_round(tmp_root, round_index=0, corrections=8, batch_size=20,
                 classifications_count=100)
    record_round(tmp_root, round_index=1, corrections=4, batch_size=20,
                 classifications_count=100)
    state = json.loads((tmp_root / "data" / "discovery" / "rounds.json").read_text())
    assert len(state["rounds"]) == 2
    assert state["rounds"][1]["corrections"] == 4


def test_should_stop_after_corrections_below_threshold(tmp_root: Path):
    _seed_sig(tmp_root)
    record_round(tmp_root, round_index=0, corrections=12, batch_size=20, classifications_count=100)
    record_round(tmp_root, round_index=1, corrections=5, batch_size=20, classifications_count=100)
    record_round(tmp_root, round_index=2, corrections=2, batch_size=20, classifications_count=100)
    assert should_stop(tmp_root, threshold=3, max_rounds=5) is True


def test_should_stop_after_max_rounds(tmp_root: Path):
    _seed_sig(tmp_root)
    for i in range(5):
        record_round(tmp_root, round_index=i, corrections=10, batch_size=20, classifications_count=100)
    assert should_stop(tmp_root, threshold=3, max_rounds=5) is True


def test_metrics_on_partial_gold(tmp_root: Path):
    _seed_sig(tmp_root)
    update_signature_from_label(tmp_root, doc_id="a", verdict="yes", evidence_excerpt="e")
    update_signature_from_label(tmp_root, doc_id="b", verdict="yes", evidence_excerpt="e")
    update_signature_from_label(tmp_root, doc_id="c", verdict="no", evidence_excerpt="e")
    classifications = [
        {"doc_id": "a", "verdict": "yes", "confidence": 0.9, "engine": "ollama"},
        {"doc_id": "b", "verdict": "no",  "confidence": 0.7, "engine": "ollama"},  # FN
        {"doc_id": "c", "verdict": "yes", "confidence": 0.6, "engine": "ollama"},  # FP
        {"doc_id": "d", "verdict": "yes", "confidence": 0.8, "engine": "ollama"},  # not in gold; ignore
    ]
    m = current_metrics(tmp_root, classifications=classifications)
    assert m["true_positives"] == 1
    assert m["false_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement**

`src/contract_intel_mvp/discovery/convergence.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.signature import load_signature


def _path(root: Path) -> Path:
    return root / "data" / "discovery" / "rounds.json"


def record_round(root: Path, *, round_index: int, corrections: int,
                 batch_size: int, classifications_count: int) -> None:
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(p.read_text()) if p.exists() else {"rounds": []}
    state["rounds"].append({
        "round_index": round_index,
        "corrections": corrections,
        "batch_size": batch_size,
        "classifications_count": classifications_count,
    })
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_stop(root: Path, *, threshold: int = 3, max_rounds: int = 5) -> bool:
    p = _path(root)
    if not p.exists():
        return False
    rounds = json.loads(p.read_text()).get("rounds", [])
    if len(rounds) >= max_rounds:
        return True
    if rounds and rounds[-1]["corrections"] < threshold:
        return True
    return False


def current_metrics(root: Path, *, classifications: list[dict[str, Any]]) -> dict[str, Any]:
    sig = load_signature(root)
    pos_gold = set(sig.confirmed_positive_doc_ids)
    neg_gold = set(sig.confirmed_negative_doc_ids)
    by_id = {c["doc_id"]: c for c in classifications}
    tp = sum(1 for d in pos_gold if by_id.get(d, {}).get("verdict") == "yes")
    fn = sum(1 for d in pos_gold if by_id.get(d, {}).get("verdict") == "no")
    fp = sum(1 for d in neg_gold if by_id.get(d, {}).get("verdict") == "yes")
    tn = sum(1 for d in neg_gold if by_id.get(d, {}).get("verdict") == "no")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"true_positives": tp, "true_negatives": tn,
            "false_positives": fp, "false_negatives": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "labeled_count": len(pos_gold) + len(neg_gold)}
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_discovery_convergence.py -v
git add src/contract_intel_mvp/discovery/convergence.py tests/test_discovery_convergence.py
git commit -m "feat(discovery): convergence detector and precision/recall on partial gold"
```

---

## Task 7: Discovery loop driver

One function that runs a single round: rank → classify (top-K candidates) → sample → save. Records round in `rounds.json`. Persists classifications to `data/discovery/classifications_round_{i}.json`.

**Files:**
- Create: `src/contract_intel_mvp/discovery/loop.py`
- Create: `tests/test_discovery_loop.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_loop.py`:
```python
import json
from pathlib import Path
from contract_intel_mvp.discovery.loop import run_round
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.embeddings import embed_corpus


def _fake_embed(text: str):
    if "PUBLISHING" in text or "Author grants Publisher" in text:
        return [1.0, 0.0, 0.0, 0.1]
    if "LICENSE" in text: return [0.0, 1.0, 0.0, 0.1]
    if "NON-DISCLOSURE" in text: return [0.0, 0.0, 1.0, 0.1]
    return [0.0, 0.0, 0.0, 1.0]


def test_run_round_writes_classifications_and_review_queue(tmp_root: Path, discovery_corpus: Path, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text")
    init_signature(tmp_root, interview={
        "target_class": "Publishing Agreement",
        "target_description": "PUBLISHING AGREEMENT Author Publisher royalties.",
        "must_contain_terms": ["author", "publisher"],
        "must_not_contain_terms": ["non-disclosure"],
    })
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **kw: {
        "verdict": "yes" if "PUBLISHING" in kw["prompt"] else "no",
        "confidence": 0.85, "evidence_excerpt": "x", "rationale": "x"
    })
    out = run_round(tmp_root, classifier_model="qwen3:4b",
                    top_k=15, batch_size=8, round_index=0, seed=1)
    assert out["round_index"] == 0
    assert out["classifications_count"] == 15
    cls_path = tmp_root / "data" / "discovery" / "classifications_round_0.json"
    assert cls_path.exists()
    rev_path = tmp_root / "data" / "discovery" / "review_queue_round_0.json"
    assert rev_path.exists()
    queue = json.loads(rev_path.read_text())
    assert len(queue["items"]) <= 8
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement**

`src/contract_intel_mvp/discovery/loop.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.ranker import rank_corpus
from contract_intel_mvp.discovery.classifier import classify_candidates
from contract_intel_mvp.discovery.sampler import sample_for_review
from contract_intel_mvp.discovery.convergence import current_metrics, record_round


def run_round(root: Path, *, classifier_model: str, top_k: int = 200,
              batch_size: int = 20, round_index: int, seed: int = 0) -> dict[str, Any]:
    ranked = rank_corpus(root, top_k=top_k)
    classifications = classify_candidates(root, candidates=ranked, model=classifier_model)
    queue = sample_for_review(root, classifications=classifications,
                              batch_size=batch_size, seed=seed)
    metrics = current_metrics(root, classifications=classifications)
    out_dir = root / "data" / "discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"classifications_round_{round_index}.json").write_text(
        json.dumps(classifications, indent=2), encoding="utf-8")
    (out_dir / f"review_queue_round_{round_index}.json").write_text(
        json.dumps({"round_index": round_index, "items": queue, "metrics": metrics},
                   indent=2), encoding="utf-8")
    return {"round_index": round_index,
            "classifications_count": len(classifications),
            "review_queue_size": len(queue),
            "metrics": metrics}


def submit_labels(root: Path, *, round_index: int,
                  labels: list[dict[str, Any]]) -> dict[str, Any]:
    """labels: [{doc_id, verdict: yes|no|borderline, evidence_excerpt}]."""
    from contract_intel_mvp.discovery.signature import update_signature_from_label
    queue_path = root / "data" / "discovery" / f"review_queue_round_{round_index}.json"
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {"items": []}
    by_id = {it["doc_id"]: it for it in queue.get("items", [])}
    corrections = 0
    for lbl in labels:
        doc_id = lbl["doc_id"]
        verdict = lbl["verdict"]
        if verdict == "borderline":
            continue
        item = by_id.get(doc_id, {})
        agent_verdict = item.get("verdict")
        if verdict != agent_verdict:
            corrections += 1
        update_signature_from_label(root, doc_id=doc_id, verdict=verdict,
                                     evidence_excerpt=lbl.get("evidence_excerpt", ""))
    cls_count = len(json.loads(
        (root / "data" / "discovery" / f"classifications_round_{round_index}.json").read_text()
    ))
    record_round(root, round_index=round_index, corrections=corrections,
                 batch_size=len(labels), classifications_count=cls_count)
    return {"round_index": round_index, "corrections": corrections,
            "labels_received": len(labels)}
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_discovery_loop.py -v
git add src/contract_intel_mvp/discovery/loop.py tests/test_discovery_loop.py
git commit -m "feat(discovery): single-round driver and label-submit handler"
```

---

## Task 8: Final output

When `should_stop` is true, produce `data/discovery/final.json` with the predicted positives, their confidence, and the precision/recall. Also write `data/discovery/borderline.json` for low-confidence positives the user might want to double-check.

**Files:**
- Modify: `src/contract_intel_mvp/discovery/loop.py` (append `finalize`)
- Create: `tests/test_discovery_finalize.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_finalize.py`:
```python
from pathlib import Path
import json
from contract_intel_mvp.discovery.loop import finalize
from contract_intel_mvp.discovery.signature import init_signature, update_signature_from_label


def test_finalize_emits_positives_and_borderline(tmp_root: Path):
    init_signature(tmp_root, interview={
        "target_class": "Publishing Agreement", "target_description": "x",
        "must_contain_terms": [], "must_not_contain_terms": [],
    })
    update_signature_from_label(tmp_root, doc_id="a", verdict="yes", evidence_excerpt="e")
    update_signature_from_label(tmp_root, doc_id="b", verdict="no", evidence_excerpt="e")
    classifications = [
        {"doc_id": "a", "verdict": "yes", "confidence": 0.95, "engine": "ollama"},
        {"doc_id": "x", "verdict": "yes", "confidence": 0.92, "engine": "ollama"},
        {"doc_id": "y", "verdict": "yes", "confidence": 0.55, "engine": "ollama"},  # borderline
        {"doc_id": "b", "verdict": "no",  "confidence": 0.90, "engine": "ollama"},
    ]
    (tmp_root / "data" / "discovery" / "classifications_round_2.json").write_text(
        json.dumps(classifications))
    out = finalize(tmp_root, round_index=2, borderline_threshold=0.7)
    assert out["positives_count"] == 3
    assert out["borderline_count"] == 1
    final = json.loads((tmp_root / "data" / "discovery" / "final.json").read_text())
    pos_ids = {p["doc_id"] for p in final["positives"]}
    assert pos_ids == {"a", "x", "y"}
    border = json.loads((tmp_root / "data" / "discovery" / "borderline.json").read_text())
    assert border["items"][0]["doc_id"] == "y"
```

- [ ] **Step 2: Append to `loop.py`**

```python
def finalize(root: Path, *, round_index: int,
             borderline_threshold: float = 0.7) -> dict[str, Any]:
    cls = json.loads((root / "data" / "discovery" / f"classifications_round_{round_index}.json").read_text())
    positives = [c for c in cls if c["verdict"] == "yes"]
    borderline = [c for c in positives if c["confidence"] < borderline_threshold]
    metrics = current_metrics(root, classifications=cls)
    final = {
        "target_class": __import__('contract_intel_mvp.discovery.signature',
                                    fromlist=['load_signature']).load_signature(root).target_class,
        "round_index": round_index,
        "positives_count": len(positives),
        "borderline_count": len(borderline),
        "metrics": metrics,
        "positives": [{"doc_id": c["doc_id"], "confidence": c["confidence"],
                       "evidence_excerpt": c.get("evidence_excerpt", "")} for c in positives],
    }
    out_dir = root / "data" / "discovery"
    (out_dir / "final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (out_dir / "borderline.json").write_text(
        json.dumps({"items": borderline, "threshold": borderline_threshold}, indent=2),
        encoding="utf-8")
    return {"positives_count": len(positives),
            "borderline_count": len(borderline),
            "final": str((out_dir / "final.json").relative_to(root))}
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_discovery_finalize.py -v
git add src/contract_intel_mvp/discovery/loop.py tests/test_discovery_finalize.py
git commit -m "feat(discovery): final output with positives and borderline review queue"
```

---

## Task 9: Reframe interview prompt for class-signature elicitation

The OpenAI interview agent currently asks for general goal/types. Add a discovery-mode prompt that asks one type at a time and produces `target_class`, `target_description`, `must_contain_terms`, `must_not_contain_terms`.

**Files:**
- Modify: `src/contract_intel_mvp/web.py` (extend `_call_openai_interview` with mode flag)
- Modify: `src/contract_intel_mvp/web.py` (add `/api/interview/discovery-chat` endpoint)
- Create: `tests/test_discovery_interview.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_interview.py`:
```python
import os, json
from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def test_discovery_chat_endpoint_routes_to_local_when_no_key(tmp_root: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_INTERVIEW", raising=False)
    app = build_app(root=tmp_root)
    client = TestClient(app)
    resp = client.post("/api/interview/discovery-chat", json={
        "signature": {"target_class": "", "target_description": "",
                      "must_contain_terms": [], "must_not_contain_terms": []},
        "message": "I'm looking for publishing agreements",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "signature" in body and "assistant" in body
    assert body["engine"] in {"local_discovery_fallback", "openai_api"}


def test_discovery_chat_initializes_signature_on_save(tmp_root: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_INTERVIEW", raising=False)
    app = build_app(root=tmp_root)
    client = TestClient(app)
    resp = client.post("/api/interview/discovery-chat", json={
        "signature": {"target_class": "Publishing Agreement",
                      "target_description": "Author grants Publisher rights; royalties.",
                      "must_contain_terms": ["author", "publisher", "royalties"],
                      "must_not_contain_terms": ["non-disclosure"]},
        "message": "Save this and start discovery",
        "save": True,
    })
    body = resp.json()
    assert body.get("saved") is True
    sig_path = tmp_root / "data" / "discovery" / "signature.json"
    assert sig_path.exists()
    sig = json.loads(sig_path.read_text())
    assert sig["target_class"] == "Publishing Agreement"
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement endpoint**

In `src/contract_intel_mvp/web.py`, inside `build_app`:
```python
from .discovery.signature import init_signature, load_signature

DISCOVERY_SYSTEM_PROMPT = (
    "You are a discovery interview agent. Help the user define ONE target contract "
    "class they want to find in a corpus. Ask focused questions about defining "
    "characteristics, must-have clauses, party patterns, and what should be excluded. "
    "Output strict JSON only."
)

@app.post("/api/interview/discovery-chat")
def discovery_chat(payload: dict):
    sig_in = payload.get("signature") or {}
    message = str(payload.get("message", "")).strip()
    save = bool(payload.get("save"))
    if not message and not save:
        return {"error": "message or save required"}
    if save and sig_in.get("target_class") and sig_in.get("target_description"):
        init_signature(root, interview=sig_in)
        return {"signature": sig_in, "assistant": "Saved. Ready to embed and start discovery.",
                "saved": True, "engine": "local_discovery_fallback"}
    # Try OpenAI
    if _openai_interview_enabled():
        prompt = {
            "task": "Continue a discovery interview to define ONE target contract class.",
            "current_signature": sig_in,
            "user_message": message,
            "schema": {
                "assistant": "string",
                "signature_updates": {
                    "target_class": "string",
                    "target_description": "string",
                    "must_contain_terms": ["string"],
                    "must_not_contain_terms": ["string"],
                },
                "ready_to_save": "boolean",
            },
        }
        body = {
            "model": _openai_model(),
            "messages": [
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{_openai_base_url()}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','').strip()}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                payload_resp = json.loads(r.read().decode("utf-8"))
            content = payload_resp["choices"][0]["message"]["content"]
            parsed = _extract_json_object(content)
            if isinstance(parsed, dict):
                updates = parsed.get("signature_updates") or {}
                merged = dict(sig_in)
                for k, v in updates.items():
                    if v: merged[k] = v
                return {"signature": merged,
                        "assistant": str(parsed.get("assistant") or ""),
                        "ready_to_save": bool(parsed.get("ready_to_save")),
                        "engine": "openai_api", "model": _openai_model()}
        except Exception:
            pass
    # Local fallback: echo + canned next question
    return {"signature": sig_in,
            "assistant": "Tell me one defining clause this contract type always has.",
            "engine": "local_discovery_fallback"}
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_discovery_interview.py -v
git add src/contract_intel_mvp/web.py tests/test_discovery_interview.py
git commit -m "feat(discovery): openai discovery-chat endpoint that elicits a class signature"
```

---

## Task 10: Discovery API endpoints

POST `/api/discovery/embed`, `/api/discovery/run-round`, `/api/discovery/submit-labels`, `/api/discovery/finalize`. GET `/api/discovery/state`.

**Files:**
- Modify: `src/contract_intel_mvp/web.py`
- Create: `tests/test_discovery_api.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_api.py`:
```python
import json
from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def test_state_endpoint_reflects_progress(tmp_root: Path, monkeypatch, discovery_corpus: Path):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t: [1.0, 0.0, 0.0, 0.0])
    from contract_intel_mvp.discovery.signature import init_signature
    init_signature(tmp_root, interview={"target_class": "T", "target_description": "x",
                                        "must_contain_terms": [], "must_not_contain_terms": []})
    from contract_intel_mvp.discovery.embeddings import embed_corpus
    embed_corpus(tmp_root, model="nomic-embed-text")
    app = build_app(root=tmp_root)
    client = TestClient(app)
    state = client.get("/api/discovery/state").json()
    assert state["embedded_count"] == 20
    assert state["target_class"] == "T"
    assert state["rounds"] == []
    assert state["finalized"] is False


def test_submit_labels_flow(tmp_root: Path, monkeypatch, discovery_corpus: Path):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t: [1.0 if "PUBLISHING" in t else 0.0,
                                                              0.0, 0.0, 0.1])
    from contract_intel_mvp.discovery.signature import init_signature
    init_signature(tmp_root, interview={"target_class": "Publishing", "target_description": "PUBLISHING",
                                         "must_contain_terms": [], "must_not_contain_terms": []})
    from contract_intel_mvp.discovery.embeddings import embed_corpus
    embed_corpus(tmp_root, model="nomic-embed-text")
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **kw: {
        "verdict": "yes" if "PUBLISHING" in kw["prompt"] else "no",
        "confidence": 0.85, "evidence_excerpt": "x", "rationale": "x"
    })
    app = build_app(root=tmp_root)
    client = TestClient(app)
    r = client.post("/api/discovery/run-round",
                    json={"classifier_model": "qwen3:4b", "top_k": 10,
                          "batch_size": 4, "round_index": 0}).json()
    assert r["classifications_count"] == 10
    queue = json.loads((tmp_root / "data" / "discovery" / "review_queue_round_0.json").read_text())
    first = queue["items"][0]
    sub = client.post("/api/discovery/submit-labels", json={
        "round_index": 0,
        "labels": [{"doc_id": first["doc_id"], "verdict": "yes",
                    "evidence_excerpt": "looks right"}],
    }).json()
    assert sub["labels_received"] == 1
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement endpoints**

Inside `build_app`:
```python
from .discovery.embeddings import embed_corpus
from .discovery.loop import run_round, submit_labels, finalize
from .discovery.convergence import should_stop, current_metrics

@app.get("/api/discovery/state")
def discovery_state():
    sig_path = root / "data" / "discovery" / "signature.json"
    emb_path = root / "data" / "discovery" / "embeddings.jsonl"
    rounds_path = root / "data" / "discovery" / "rounds.json"
    final_path = root / "data" / "discovery" / "final.json"
    return {
        "target_class": load_signature(root).target_class if sig_path.exists() else None,
        "embedded_count": sum(1 for _ in emb_path.read_text().splitlines() if _.strip()) if emb_path.exists() else 0,
        "rounds": json.loads(rounds_path.read_text()).get("rounds", []) if rounds_path.exists() else [],
        "finalized": final_path.exists(),
        "should_stop": should_stop(root),
    }

@app.post("/api/discovery/embed")
def discovery_embed(payload: dict):
    return embed_corpus(root, model=payload.get("model", "nomic-embed-text"))

@app.post("/api/discovery/run-round")
def discovery_run_round(payload: dict):
    return run_round(root,
                     classifier_model=payload.get("classifier_model", "qwen3:4b"),
                     top_k=int(payload.get("top_k", 200)),
                     batch_size=int(payload.get("batch_size", 20)),
                     round_index=int(payload.get("round_index", 0)),
                     seed=int(payload.get("seed", 0)))

@app.post("/api/discovery/submit-labels")
def discovery_submit_labels(payload: dict):
    return submit_labels(root,
                         round_index=int(payload.get("round_index", 0)),
                         labels=payload.get("labels", []))

@app.post("/api/discovery/finalize")
def discovery_finalize(payload: dict):
    return finalize(root, round_index=int(payload.get("round_index", 0)),
                    borderline_threshold=float(payload.get("borderline_threshold", 0.7)))
```

Also mirror these in the `Handler` class in the http.server section so the live UI hits them — wrap in a helper `_discovery_dispatch` or add per-path branches in `do_POST` and `do_GET` exactly mirroring the FastAPI routes.

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_discovery_api.py -v
git add src/contract_intel_mvp/web.py tests/test_discovery_api.py
git commit -m "feat(discovery): http endpoints for embed/run-round/submit-labels/finalize/state"
```

---

## Task 11: Discovery UI

Replace the current Setup/Agent tabs' middle content with a single Discovery tab. Keep the legacy tabs hidden behind a "show legacy panels" disclosure.

**Files:**
- Modify: `src/contract_intel_mvp/static/index.html`
- Modify: `src/contract_intel_mvp/static/app.js`

- [ ] **Step 1: Add a `Discovery` button to the nav**

In `index.html`:
```html
<button class="nav" data-view="discovery">Discovery</button>
```

- [ ] **Step 2: Add the Discovery view section**

```html
<section id="discovery" class="view">
  <section class="panel">
    <h3>1. Drop your contracts</h3>
    <p>Drop a folder, or click Choose folder. The system will embed each contract once for fast pre-screening.</p>
    <div id="discDropzone" style="border: 2px dashed #ccc; padding: 30px; text-align: center; background: #fafafa; border-radius: 8px;">
      <p>Drop a folder of contracts here</p>
      <label style="cursor: pointer; padding: 8px 16px; background: #4a7; color: white; border-radius: 4px;">
        Choose folder
        <input id="discFolderInput" type="file" multiple webkitdirectory directory style="display: none;" />
      </label>
    </div>
    <div id="discUploadStatus" style="font-family: monospace; padding: 8px;"></div>
    <p><button id="discEmbedBtn" type="button">Embed corpus</button>
       <span id="discEmbedStatus" style="margin-left: 12px; font-family: monospace;"></span></p>
  </section>

  <section class="panel">
    <h3>2. Define what you're looking for</h3>
    <p>Tell the agent what kind of contract you're looking for. It will ask follow-up questions.</p>
    <div id="discChat" style="max-height: 280px; overflow: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 8px;"></div>
    <div style="display: flex; gap: 8px;">
      <input id="discChatInput" placeholder="e.g. I'm looking for publishing agreements..." style="flex: 1;" />
      <button id="discChatSend" type="button">Send</button>
    </div>
    <div id="discSignaturePreview" style="margin-top: 12px; font-size: 12px; background: #fafafa; padding: 8px; border-radius: 4px; font-family: monospace;"></div>
    <p style="margin-top: 8px;"><button id="discSaveSig" type="button">Save signature & enable discovery</button></p>
  </section>

  <section class="panel">
    <h3>3. Run a round</h3>
    <p>Each round: rank all contracts, classify the top candidates with the small model, surface 20 borderline cases for your review.</p>
    <p>
      <label>Round <input id="discRoundIdx" type="number" value="0" style="width: 50px;"/></label>
      <label>Top-K <input id="discTopK" type="number" value="200" style="width: 80px;"/></label>
      <label>Batch <input id="discBatch" type="number" value="20" style="width: 50px;"/></label>
      <button id="discRunRoundBtn" type="button">Run round</button>
    </p>
    <pre id="discRoundResult" style="font-size: 12px; background: #fafafa; padding: 8px;"></pre>
  </section>

  <section class="panel">
    <h3>4. Review borderline cases</h3>
    <p>The agent picked these for your eyes. Yes / No / Skip.</p>
    <div id="discReviewQueue"></div>
    <p><button id="discSubmitLabels" type="button">Submit labels for this round</button></p>
  </section>

  <section class="panel">
    <h3>5. Final output</h3>
    <p>When the loop converges, the agent writes the predicted positives.</p>
    <p><button id="discFinalizeBtn" type="button">Finalize</button></p>
    <pre id="discFinalResult" style="font-size: 12px; background: #fafafa; padding: 8px;"></pre>
  </section>

  <section class="panel">
    <h3>State</h3>
    <pre id="discStateJson" style="font-size: 11px; background: #fafafa; padding: 8px;"></pre>
  </section>
</section>
```

- [ ] **Step 3: Append JS bindings**

In `app.js`, append a self-contained block (mirroring the existing per-tab IIFE pattern) that wires:
- folder input → POST `/api/upload` → POST `/api/discovery/embed` → show counts.
- chat send → POST `/api/interview/discovery-chat` → render reply + signature preview.
- save signature → POST `/api/interview/discovery-chat` with `save: true`.
- run round → POST `/api/discovery/run-round` → render result; fetch the review queue file from `/api/file?path=data/discovery/review_queue_round_<i>.json` and render labelable rows.
- submit labels → POST `/api/discovery/submit-labels`.
- finalize → POST `/api/discovery/finalize`.
- state polling → GET `/api/discovery/state` every 5s.

All dynamic data rendered with `textContent` / `createElement` only.

(See Task 14 for the full JS code; deferred to keep this task scoped to "wire the structure.")

- [ ] **Step 4: Smoke test the UI structure**

```bash
.venv/bin/contract-intel ui &
curl -s http://127.0.0.1:8765/ | grep -c 'data-view="discovery"'
```
Expected: `1`.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/static/index.html src/contract_intel_mvp/static/app.js
git commit -m "feat(discovery): UI tab structure (HTML)"
```

---

## Task 12: Discovery UI JS bindings

Full JS for the Discovery tab.

**Files:**
- Modify: `src/contract_intel_mvp/static/app.js`

- [ ] **Step 1: Append JS**

```javascript
// === Discovery tab ===
(function() {
  function $id(id) { return document.getElementById(id); }
  function setText(id, t) { const el = $id(id); if (el) el.textContent = t; }
  async function gj(url) { return (await fetch(url)).json(); }
  async function pj(url, body) {
    const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
                                body: JSON.stringify(body || {})});
    return r.json();
  }

  function readB64(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve({filename: file.webkitRelativePath || file.name,
                                content_b64: String(r.result).split(",")[1] || ""});
      r.onerror = () => reject(r.error);
      r.readAsDataURL(file);
    });
  }

  async function uploadAndEmbed(fileList) {
    const accepted = /\.(txt|md|html?|docx|pdf)$/i;
    const files = Array.from(fileList || []).filter(f => accepted.test(f.name));
    if (!files.length) { setText("discUploadStatus", "no accepted files"); return; }
    setText("discUploadStatus", `Reading ${files.length}...`);
    const payload = [];
    for (const f of files) try { payload.push(await readB64(f)); } catch {}
    const up = await pj("/api/upload", {files: payload});
    setText("discUploadStatus", `Uploaded ${up.received}, ingested ${up.ingested}.`);
  }

  async function embed() {
    setText("discEmbedStatus", "embedding...");
    const r = await pj("/api/discovery/embed", {model: "nomic-embed-text"});
    setText("discEmbedStatus", `Embedded ${r.embedded}, skipped ${r.skipped}, failed ${r.failed}.`);
    pollState();
  }

  let chatLog = [];
  let currentSig = {target_class: "", target_description: "", must_contain_terms: [], must_not_contain_terms: []};

  function renderChat() {
    const el = $id("discChat"); if (!el) return;
    while (el.firstChild) el.removeChild(el.firstChild);
    chatLog.forEach(m => {
      const div = document.createElement("div");
      div.style.margin = "4px 0";
      div.style.padding = "6px 10px";
      div.style.background = m.role === "user" ? "#eef" : "#f4faf4";
      div.style.borderRadius = "4px";
      div.textContent = (m.role === "user" ? "You: " : "Agent: ") + m.content;
      el.appendChild(div);
    });
    el.scrollTop = el.scrollHeight;
  }

  function renderSig() {
    setText("discSignaturePreview", JSON.stringify(currentSig, null, 2));
  }

  async function chatSend() {
    const input = $id("discChatInput");
    const msg = (input.value || "").trim();
    if (!msg) return;
    chatLog.push({role: "user", content: msg});
    renderChat();
    input.value = "";
    const r = await pj("/api/interview/discovery-chat", {signature: currentSig, message: msg});
    if (r.signature) { currentSig = r.signature; renderSig(); }
    chatLog.push({role: "agent", content: r.assistant || "(no reply)"});
    renderChat();
  }

  async function saveSig() {
    const r = await pj("/api/interview/discovery-chat", {signature: currentSig, message: "save", save: true});
    chatLog.push({role: "agent", content: r.assistant || "saved"});
    renderChat();
    pollState();
  }

  async function runRound() {
    const idx = parseInt($id("discRoundIdx").value || "0", 10);
    const topK = parseInt($id("discTopK").value || "200", 10);
    const batch = parseInt($id("discBatch").value || "20", 10);
    setText("discRoundResult", "running...");
    const r = await pj("/api/discovery/run-round", {round_index: idx, top_k: topK,
                                                      batch_size: batch, classifier_model: "qwen3:4b"});
    setText("discRoundResult", JSON.stringify(r, null, 2));
    await loadReviewQueue(idx);
    pollState();
  }

  let queueState = {round_index: 0, items: []};

  async function loadReviewQueue(idx) {
    const url = "/api/file?path=" + encodeURIComponent(`data/discovery/review_queue_round_${idx}.json`);
    try {
      const txt = await (await fetch(url)).text();
      queueState = JSON.parse(txt);
    } catch { queueState = {round_index: idx, items: []}; }
    const root = $id("discReviewQueue");
    while (root.firstChild) root.removeChild(root.firstChild);
    queueState.items.forEach((it, i) => {
      const card = document.createElement("div");
      card.style.border = "1px solid #ccc"; card.style.borderRadius = "4px";
      card.style.padding = "8px"; card.style.margin = "6px 0";
      const head = document.createElement("div");
      head.style.fontWeight = "bold";
      head.textContent = `${it.doc_id}  —  agent: ${it.verdict} (${(it.confidence||0).toFixed(2)})  —  reason: ${it.reason}`;
      card.appendChild(head);
      const ev = document.createElement("div");
      ev.style.fontSize = "12px"; ev.style.color = "#666"; ev.style.marginTop = "4px";
      ev.textContent = (it.evidence_excerpt || "").slice(0, 300);
      card.appendChild(ev);
      const btns = document.createElement("div"); btns.style.marginTop = "6px";
      ["yes","no","borderline"].forEach(v => {
        const b = document.createElement("button");
        b.textContent = v; b.style.marginRight = "6px";
        b.addEventListener("click", () => { it.userVerdict = v; b.style.background = "#cfc"; });
        btns.appendChild(b);
      });
      card.appendChild(btns);
      root.appendChild(card);
    });
  }

  async function submitLabels() {
    const labels = queueState.items.filter(it => it.userVerdict).map(it => ({
      doc_id: it.doc_id, verdict: it.userVerdict,
      evidence_excerpt: it.evidence_excerpt || "",
    }));
    if (!labels.length) { alert("no labels selected"); return; }
    const r = await pj("/api/discovery/submit-labels",
                       {round_index: queueState.round_index, labels});
    alert(`submitted ${r.labels_received} labels, corrections: ${r.corrections}`);
    pollState();
  }

  async function finalize() {
    const idx = parseInt($id("discRoundIdx").value || "0", 10);
    const r = await pj("/api/discovery/finalize", {round_index: idx, borderline_threshold: 0.7});
    setText("discFinalResult", JSON.stringify(r, null, 2));
    pollState();
  }

  async function pollState() {
    try { setText("discStateJson", JSON.stringify(await gj("/api/discovery/state"), null, 2)); }
    catch (e) { setText("discStateJson", "error: " + e); }
  }

  function bind() {
    const fi = $id("discFolderInput");
    if (fi) fi.addEventListener("change", e => uploadAndEmbed(e.target.files));
    const dz = $id("discDropzone");
    if (dz) {
      ["dragover","dragenter"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.style.background = "#eef9ee"; }));
      ["dragleave","drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.style.background = "#fafafa"; }));
      dz.addEventListener("drop", e => { e.preventDefault(); uploadAndEmbed(e.dataTransfer.files); });
    }
    const eb = $id("discEmbedBtn"); if (eb) eb.addEventListener("click", embed);
    const cs = $id("discChatSend"); if (cs) cs.addEventListener("click", chatSend);
    const ci = $id("discChatInput"); if (ci) ci.addEventListener("keydown", e => { if (e.key === "Enter") chatSend(); });
    const ss = $id("discSaveSig"); if (ss) ss.addEventListener("click", saveSig);
    const rr = $id("discRunRoundBtn"); if (rr) rr.addEventListener("click", runRound);
    const sl = $id("discSubmitLabels"); if (sl) sl.addEventListener("click", submitLabels);
    const fb = $id("discFinalizeBtn"); if (fb) fb.addEventListener("click", finalize);
    pollState();
    setInterval(pollState, 5000);
  }
  if (document.readyState !== "loading") bind();
  else document.addEventListener("DOMContentLoaded", bind);
})();
```

- [ ] **Step 2: Restart UI, manual smoke**

```bash
pkill -9 -f "contract-intel ui" 2>/dev/null; sleep 1
set -a && . .env.local && set +a
.venv/bin/contract-intel ui &
sleep 1
curl -s http://127.0.0.1:8765/ | grep -c 'discFolderInput'
```
Expected: `1`.

- [ ] **Step 3: Commit**

```bash
git add src/contract_intel_mvp/static/app.js
git commit -m "feat(discovery): UI bindings — upload, chat, run-round, label, finalize"
```

---

## Task 13: E2E discovery on synthetic corpus

End-to-end smoke on the synthetic 20-doc fixture (5 publishing, 5 license, 5 NDA, 5 unknown). Real Ollama. Target: find the 5 publishing agreements.

**Files:**
- Create: `scripts/e2e_discovery_smoke.sh`

- [ ] **Step 1: Write script**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
set -a; source .env.local; set +a

contract-intel reset --keep-raw >/dev/null || true
contract-intel init >/dev/null

# Stage a tiny synthetic corpus directly into documents.jsonl.
python3 - <<'PY'
import json
from pathlib import Path
docs = []
samples = [
    ("pub", "PUBLISHING AGREEMENT. Author grants Publisher exclusive rights to publish the Work in print and digital editions. Royalties paid quarterly at 12% of net receipts. Term 7 years."),
    ("lic", "SOFTWARE LICENSE AGREEMENT. Licensor grants Licensee a non-exclusive license to use the Software. Annual fee $50,000. Term 1 year, auto-renew."),
    ("nda", "MUTUAL NON-DISCLOSURE AGREEMENT. Each party agrees to keep Confidential Information secret for 5 years. No fees. No license granted."),
    ("unk", "EQUIPMENT LEASE. Lessor leases the Equipment to Lessee for monthly rent of $2,000. Term 36 months."),
]
for cls, text in samples:
    for i in range(5):
        docs.append({"doc_id": f"doc_{cls}_{i}", "title": f"{cls.upper()} {i}",
                     "text": f"{text} (instance {i})", "source": "fixture"})
p = Path("data/corpus/documents.jsonl")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("\n".join(json.dumps(d) for d in docs))
print("staged", len(docs), "docs")
PY

python3 - <<'PY'
import json
from pathlib import Path
from contract_intel_mvp.discovery.signature import init_signature
init_signature(Path.cwd(), interview={
    "target_class": "Publishing Agreement",
    "target_description": "Author grants Publisher exclusive rights to publish a Work; royalties stated.",
    "must_contain_terms": ["author", "publisher", "royalties"],
    "must_not_contain_terms": ["non-disclosure"],
})
print("signature initialized")
PY

python3 - <<'PY'
from pathlib import Path
from contract_intel_mvp.discovery.embeddings import embed_corpus
print(embed_corpus(Path.cwd(), model="nomic-embed-text"))
PY

python3 - <<'PY'
from pathlib import Path
from contract_intel_mvp.discovery.loop import run_round
print(run_round(Path.cwd(), classifier_model="qwen3:4b",
                top_k=20, batch_size=8, round_index=0, seed=1))
PY

python3 - <<'PY'
import json
from pathlib import Path
queue = json.loads((Path("data/discovery/review_queue_round_0.json")).read_text())
print("review queue (n=%d):" % len(queue["items"]))
for it in queue["items"]:
    print(f"  {it['doc_id']}  agent={it['verdict']} conf={it['confidence']:.2f} reason={it['reason']}")
PY

# Auto-label using the synthetic ground truth (doc_pub_* → yes, others → no)
python3 - <<'PY'
import json
from pathlib import Path
from contract_intel_mvp.discovery.loop import submit_labels
queue = json.loads((Path("data/discovery/review_queue_round_0.json")).read_text())
labels = []
for it in queue["items"]:
    truth = "yes" if it["doc_id"].startswith("doc_pub_") else "no"
    labels.append({"doc_id": it["doc_id"], "verdict": truth,
                   "evidence_excerpt": it.get("evidence_excerpt", "")})
print(submit_labels(Path.cwd(), round_index=0, labels=labels))
PY

# Round 2 with updated signature
python3 - <<'PY'
from pathlib import Path
from contract_intel_mvp.discovery.loop import run_round, finalize
print(run_round(Path.cwd(), classifier_model="qwen3:4b",
                top_k=20, batch_size=8, round_index=1, seed=2))
print("--- finalize ---")
print(finalize(Path.cwd(), round_index=1, borderline_threshold=0.7))
PY

echo "=== final ==="
python3 -m json.tool < data/discovery/final.json
```

- [ ] **Step 2: Make executable + run**

```bash
chmod +x scripts/e2e_discovery_smoke.sh
bash scripts/e2e_discovery_smoke.sh 2>&1 | tee /tmp/disc_smoke.log
```
Expected wall time: ~3–5 min on Aurora. Outputs `data/discovery/final.json` with `target_class: "Publishing Agreement"`, ideally 5 positives matching `doc_pub_*`.

- [ ] **Step 3: Verify**

```bash
python3 -c "
import json
f = json.load(open('data/discovery/final.json'))
ids = {p['doc_id'] for p in f['positives']}
expected = {f'doc_pub_{i}' for i in range(5)}
print('precision:', len(ids & expected) / len(ids) if ids else 0.0)
print('recall:   ', len(ids & expected) / len(expected))
print('positives:', sorted(ids))
"
```
Expected: precision and recall both ≥ 0.8 on the synthetic corpus. If precision is low, rerun (small model variance) or check that the ranker is surfacing pub docs first.

- [ ] **Step 4: Commit**

```bash
git add scripts/e2e_discovery_smoke.sh
git commit -m "test(discovery): synthetic e2e smoke script"
```

---

## Task 14: Demo dress rehearsal + version bump

**Files:**
- Modify: `docs/demo/presenter_script.md`
- Modify: `VERSION`

- [ ] **Step 1: Rewrite the presenter script**

`docs/demo/presenter_script.md`:
```markdown
# Code Games Agentic Edition — Discovery Demo (8 min)

## The customer's problem
A media company brought us 30,000 contracts and asked for the publishing agreements. No metadata, no labels, no easy search. The current process is humans skimming files for weeks.

## What we built
An agent-driven discovery loop that:
1. embeds the whole corpus once (free, local),
2. interviews the user to define ONE target class,
3. ranks all contracts by similarity,
4. classifies the top candidates with a small local model,
5. shows the user 20 borderline cases per round,
6. learns from yes/no corrections,
7. converges in ~3 rounds and outputs a labelled list with precision/recall.

## Live arc

1. (1m) Open `/`. Click Discovery tab. Drop a folder of 200 mixed contracts.
2. (1m) Watch the embedding count tick up. Click Embed corpus → "embedded 200".
3. (2m) Open the chat. "I'm looking for publishing agreements." Agent asks: parties? clauses? exclusions? Build the signature live. Click Save.
4. (1m) Click Run round → "200 classified, 20 in review queue, P=?, R=?".
5. (2m) Label the 20 cards: yes / no. Click Submit labels.
6. (1m) Click Run round (round 1). Numbers improve. Submit again. Round 2 → corrections drop. Finalize.
7. Headline: "From 200 contracts to N publishing agreements in 3 rounds, precision X, recall Y."

## What to call out, what to admit
- ✅ Local-only, no cloud LLM costs, no data leaves the box.
- ✅ Auditable: every decision in `data/discovery/`.
- ✅ Scales linearly with embeddings (cheap), classification only on top-K (cheap).
- ⚠️ Demoed on 200 docs. Production target: 30k. Embedding 30k takes ~30 min on a 4090; classification still only ~500 docs. Same loop.
- ⚠️ Single target class per run today. Multi-class is a follow-up.
```

- [ ] **Step 2: Bump VERSION**

```bash
echo "0.3.0" > VERSION
git add VERSION docs/demo/presenter_script.md
git commit -m "release: v0.3.0 — discovery pivot"
git tag -a v0.3.0 -m "discovery pivot — find one type in a haystack"
```

- [ ] **Step 3: Final test sweep**

```bash
pytest -v
```
Expected: all tests pass.

---

## Self-Review

**Spec coverage:**
- Embed once → Task 1 ✓
- Class signature from interview → Tasks 2, 9 ✓
- Pre-screen rank → Task 3 ✓
- Classify top-K → Task 4 ✓
- Active-learning sample → Task 5 ✓
- SME loop + signature update from corrections → Tasks 2, 7 ✓
- Convergence detector + precision/recall → Task 6 ✓
- Final labelled-list output + borderline drawer → Task 8 ✓
- API endpoints (run-round, submit-labels, finalize, state) → Task 10 ✓
- UI (drop, embed, chat, run, label, finalize) → Tasks 11, 12 ✓
- Reframed interview prompt for one-target-class → Task 9 ✓
- E2E smoke against real Ollama → Task 13 ✓
- Presenter script aligned to discovery framing → Task 14 ✓

**What this plan deliberately does NOT do (kept out of scope):**
- Multi-class discovery in one run.
- Embedding > 30k contracts in one run; nothing about chunking long docs into multiple embeddings.
- LoRA fine-tuning on confirmed examples.
- Removing the existing extraction/agent code — both stay in place; the discovery tab is added next to them. Cleanup is post-Code-Games.

**Type/name consistency:**
- Verdict values: `"yes" | "no" | "borderline"` everywhere.
- `engine` field: `"ollama" | "heuristic_fallback"` (matches existing convention).
- File paths under `data/discovery/`: `signature.json`, `embeddings.jsonl`, `classifications_round_<i>.json`, `review_queue_round_<i>.json`, `rounds.json`, `final.json`, `borderline.json`. Used consistently across Tasks 1–13.
- API paths: `/api/discovery/{embed,run-round,submit-labels,finalize,state}` and `/api/interview/discovery-chat`. Mirrored across FastAPI `build_app` and the http.server `Handler`.
- `ClassSignature` field names match between dataclass (Task 2) and update logic (Tasks 2, 9).
- `top_k`, `batch_size`, `round_index`, `borderline_threshold` parameter names consistent across loop driver, API, and UI.

**Risks:**
- Embedding throughput: 1 contract/second is realistic with `nomic-embed-text` over Ollama HTTP. 200 docs = ~3 min, 30k = ~8 hours (overnight). Document as a known caveat; chunking + parallelism is a follow-up.
- Heuristic classifier fallback could quietly degrade results. Tag every output with `engine`; the UI should warn if any final positive came from heuristic fallback.
- Synthetic e2e is small; precision/recall numbers there don't generalize. Plan to repeat on a CUAD subset (~200 contracts, mix of types) before the demo.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-04-discovery-pivot.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
