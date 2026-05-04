# Discovery + Clause Library (Cut-Down) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a discovery agent that finds one specific contract type in a folder of contracts, learns from SME corrections by harvesting verbatim clause quotes into a structured **clause library**, and uses that library as few-shot context for subsequent classification rounds. Final output: a labelled positives list, borderline drawer, and the clause library itself as a browseable artifact. Cut-down means: whole-doc embedding pre-screen only, library write-back without SME-level variation approval, library used in the LLM classify prompt rather than via chunk-cosine matching. The structured library schema is the *full-version* schema, so v0.4.x can layer chunk-level matching on top with zero rewrite.

**Architecture:** New `discovery/` module. (1) Embed each contract once with `nomic-embed-text`. (2) Interview agent elicits a `ClassSignature` with **clause types and seed variations**. (3) Whole-doc cosine + filename rule produces top-K candidates. (4) Small Ollama model classifies each candidate, returning `verdict + confidence + evidence_per_clause_type` (verbatim quotes per clause type). (5) After SME labels, the verbatim quotes from confirmed-yes docs are appended to `clause_library.json` under their clause types with full provenance. (6) Next round's classify prompt includes the library as few-shot examples. (7) Loop until corrections plateau. The library structure is `{clause_types: [{type, description, is_must_have, variations: [{text, source_doc_id, confirmed_by, added_at, embedding_id: null}]}]}` so v0.4.x can backfill `embedding_id` and add chunk-cosine matcher without migrating the file.

**Tech Stack:** Python 3.10+, pytest, http.server / FastAPI test client (existing `web.py`), Ollama HTTP (`nomic-embed-text` for embeddings, `qwen3:4b` for classification), OpenAI `gpt-4o-mini` for the interview agent (already wired), `numpy` for cosine, no SQLite. State persists as JSON under `data/discovery/`.

---

## Pre-flight (Task 0): numpy dep + corpus fixture

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add numpy**

Append to `pyproject.toml [project.optional-dependencies]`:
```toml
dev = ["pytest", "pytest-asyncio", "httpx", "fastapi", "numpy"]
```

```bash
. .venv/bin/activate && pip install -q numpy
```

- [ ] **Step 2: Add `discovery_corpus` fixture**

Append to `tests/conftest.py`:
```python
@pytest.fixture
def discovery_corpus(tmp_root):
    """20 fake contracts: 5 license, 5 distribution, 5 strategic alliance, 5 unknown.
    Designed for the License Agreement discovery scenario with realistic close-negatives."""
    import json
    samples = [
        ("lic", "TRADEMARK LICENSE AGREEMENT. Licensor hereby grants to Licensee a non-exclusive, royalty-bearing license to use the Marks in the Territory for the Term. Royalty: 8% of net receipts."),
        ("dis", "DISTRIBUTOR AGREEMENT. Company hereby appoints Distributor as the exclusive distributor of the Products in the Territory. Distributor is also granted a license to use the Marks in connection with marketing of the Products."),
        ("sa",  "STRATEGIC ALLIANCE AGREEMENT. The Parties shall jointly develop the Project. Each Party grants the other a non-exclusive license to its background IP solely for performance under this Agreement."),
        ("unk", "EQUIPMENT LEASE. Lessor leases the Equipment to Lessee for monthly rent of $2,000. Term 36 months. No license rights granted."),
    ]
    docs = []
    for cls, text in samples:
        for i in range(5):
            doc_id = f"doc_{cls}_{i}"
            docs.append({"doc_id": doc_id, "title": f"{cls.upper()} {i}",
                         "text": f"{text} (instance {i})", "source": "fixture"})
    path = tmp_root / "data" / "corpus" / "documents.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    return path
```

- [ ] **Step 3: Verify nothing broke**

```bash
pytest -q
```
Expected: 42 passed.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/conftest.py
git commit -m "chore: numpy dep + discovery_corpus fixture (license/dist/SA/unk)"
```

---

## Task 1: Whole-doc embeddings store

Embed every contract once with `nomic-embed-text`, persist to `data/discovery/embeddings.jsonl`. Idempotent.

**Files:**
- Create: `src/contract_intel_mvp/discovery/__init__.py`
- Create: `src/contract_intel_mvp/discovery/embeddings.py`
- Create: `tests/test_discovery_embeddings.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_embeddings.py`:
```python
from pathlib import Path
import json
from contract_intel_mvp.discovery.embeddings import (
    embed_corpus, load_embeddings, EmbeddingsStore
)


def test_embed_corpus_writes_jsonl(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", lambda text: [float(len(text) % 7), 0.1, 0.2, 0.3])
    out = embed_corpus(tmp_root, model="nomic-embed-text")
    assert out["embedded"] == 20
    rows = [json.loads(l) for l in (tmp_root / "data" / "discovery" / "embeddings.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    assert all(len(r["embedding"]) == 4 for r in rows)


def test_embed_corpus_skips_existing(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    calls = []
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t: calls.append(t) or [0.0]*4)
    embed_corpus(tmp_root, model="nomic-embed-text")
    n = len(calls)
    embed_corpus(tmp_root, model="nomic-embed-text")
    assert len(calls) == n


def test_load_embeddings_returns_store(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t: [1.0, 0.0, 0.0, 0.0])
    embed_corpus(tmp_root, model="nomic-embed-text")
    store = load_embeddings(tmp_root)
    assert isinstance(store, EmbeddingsStore)
    assert len(store.doc_ids) == 20
    assert store.matrix.shape == (20, 4)
```

- [ ] **Step 2: Implement**

`src/contract_intel_mvp/discovery/__init__.py`:
```python
"""Discovery: find contracts of one target class in a haystack via clause library + active learning."""
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
    embedded = skipped = failed = 0
    with out_path.open("a", encoding="utf-8") as f:
        for line in docs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            doc = json.loads(line); doc_id = doc["doc_id"]
            if doc_id in existing and existing[doc_id].get("model") == model:
                skipped += 1; continue
            vec = _call_ollama_embed((doc.get("text") or "")[:max_chars], model=model)
            if vec is None: failed += 1; continue
            f.write(json.dumps({"doc_id": doc_id, "model": model, "embedding": vec}) + "\n")
            embedded += 1
    return {"embedded": embedded, "skipped": skipped, "failed": failed,
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
```

- [ ] **Step 3: Test passes + commit**

```bash
pytest tests/test_discovery_embeddings.py -v
git add src/contract_intel_mvp/discovery/ tests/test_discovery_embeddings.py
git commit -m "feat(discovery): per-doc embeddings store via ollama nomic-embed-text"
```

---

## Task 2: Class signature with clause-type schema (full-version shape)

The signature has clause TYPES with seed variations, not just term lists. **This is the critical schema decision** — full-version compatible from day one.

**Files:**
- Create: `src/contract_intel_mvp/discovery/signature.py`
- Create: `tests/test_discovery_signature.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_signature.py`:
```python
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
    # Persisted
    loaded = load_signature(tmp_root)
    assert loaded.target_class == "License Agreement"
    assert loaded.clause_types[0].type == "license_grant"


def test_signature_has_confirmed_doc_lists(tmp_root):
    sig = init_signature(tmp_root, interview={
        "target_class": "X", "target_description": "x", "clause_types": [],
    })
    assert sig.confirmed_positive_doc_ids == []
    assert sig.confirmed_negative_doc_ids == []
```

- [ ] **Step 2: Implement**

`src/contract_intel_mvp/discovery/signature.py`:
```python
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ClauseType:
    type: str
    description: str = ""
    is_must_have: bool = True
    seed_variations: list[str] = field(default_factory=list)


@dataclass
class ClassSignature:
    target_class: str
    target_description: str
    clause_types: list[ClauseType] = field(default_factory=list)
    confirmed_positive_doc_ids: list[str] = field(default_factory=list)
    confirmed_negative_doc_ids: list[str] = field(default_factory=list)


def _path(root): return root / "data" / "discovery" / "signature.json"


def init_signature(root: Path, *, interview: dict[str, Any]) -> ClassSignature:
    clause_types = [ClauseType(**ct) for ct in (interview.get("clause_types") or [])]
    sig = ClassSignature(
        target_class=str(interview.get("target_class", "")).strip() or "Target Class",
        target_description=str(interview.get("target_description", "")).strip(),
        clause_types=clause_types,
    )
    save_signature(root, sig)
    return sig


def save_signature(root: Path, sig: ClassSignature) -> None:
    p = _path(root); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(sig), indent=2), encoding="utf-8")


def load_signature(root: Path) -> ClassSignature:
    p = _path(root)
    if not p.exists():
        raise ValueError(f"no signature at {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["clause_types"] = [ClauseType(**ct) for ct in raw.get("clause_types", [])]
    return ClassSignature(**raw)
```

- [ ] **Step 3: Test passes + commit**

```bash
pytest tests/test_discovery_signature.py -v
git add src/contract_intel_mvp/discovery/signature.py tests/test_discovery_signature.py
git commit -m "feat(discovery): clause-type-aware ClassSignature schema (full-version shape)"
```

---

## Task 3: Clause library (structured, full-version schema)

`clause_library.json` mirrors `signature.clause_types` but stores **all confirmed variations with provenance and a placeholder for embedding_id** (filled in v0.4.x).

**Files:**
- Create: `src/contract_intel_mvp/discovery/library.py`
- Create: `tests/test_discovery_library.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_library.py`:
```python
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
    assert new_v["embedding_id"] is None  # full-version placeholder


def test_append_variations_dedupes_exact_text(tmp_root):
    _seed_sig(tmp_root); init_library_from_signature(tmp_root)
    text = "Owner grants User a license"
    append_variations(tmp_root, clause_type="license_grant", variations=[
        {"text": text, "source_doc_id": "doc_a", "confirmed_by": "auto"},
        {"text": text, "source_doc_id": "doc_b", "confirmed_by": "auto"},
    ])
    lib = load_library(tmp_root)
    license_grant = next(ct for ct in lib["clause_types"] if ct["type"] == "license_grant")
    # 1 seed + 1 deduped new (second occurrence skipped)
    assert len(license_grant["variations"]) == 2


def test_render_library_text_for_few_shot_prompt(tmp_root):
    _seed_sig(tmp_root); init_library_from_signature(tmp_root)
    text = render_library_text(tmp_root, max_per_type=5)
    assert "license_grant" in text
    assert "Licensor hereby grants" in text
    assert "primary_distribution" in text
```

- [ ] **Step 2: Implement**

`src/contract_intel_mvp/discovery/library.py`:
```python
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.signature import load_signature


def _path(root): return root / "data" / "discovery" / "clause_library.json"


def init_library_from_signature(root: Path) -> dict[str, Any]:
    sig = load_signature(root)
    now = datetime.now(timezone.utc).isoformat()
    clause_types = []
    for ct in sig.clause_types:
        variations = [{
            "text": text, "source_doc_id": "seed",
            "confirmed_by": "interview_seed", "added_at": now,
            "embedding_id": None,
        } for text in ct.seed_variations]
        clause_types.append({
            "type": ct.type, "description": ct.description,
            "is_must_have": ct.is_must_have, "variations": variations,
        })
    lib = {"target_class": sig.target_class, "clause_types": clause_types}
    p = _path(root); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lib, indent=2), encoding="utf-8")
    return lib


def load_library(root: Path) -> dict[str, Any]:
    p = _path(root)
    if not p.exists():
        raise ValueError(f"no library at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_library(root: Path, lib: dict[str, Any]) -> None:
    _path(root).write_text(json.dumps(lib, indent=2), encoding="utf-8")


def append_variations(root: Path, *, clause_type: str,
                      variations: list[dict[str, Any]]) -> dict[str, Any]:
    lib = load_library(root)
    target = next((ct for ct in lib["clause_types"] if ct["type"] == clause_type), None)
    if target is None:
        return lib  # unknown clause type; ignore silently
    existing_texts = {v["text"].strip() for v in target["variations"]}
    now = datetime.now(timezone.utc).isoformat()
    for v in variations:
        text = (v.get("text") or "").strip()
        if not text or text in existing_texts:
            continue
        target["variations"].append({
            "text": text,
            "source_doc_id": v.get("source_doc_id", ""),
            "confirmed_by": v.get("confirmed_by", "auto_from_sme_yes"),
            "added_at": now,
            "embedding_id": None,
        })
        existing_texts.add(text)
    save_library(root, lib)
    return lib


def render_library_text(root: Path, *, max_per_type: int = 5) -> str:
    """Render the library as few-shot context for the classify prompt."""
    lib = load_library(root)
    chunks = [f"Library for target class: {lib['target_class']}\n"]
    for ct in lib["clause_types"]:
        marker = "MUST HAVE" if ct["is_must_have"] else "MUST NOT HAVE"
        chunks.append(f"\n[{marker}] {ct['type']} — {ct['description']}")
        for v in ct["variations"][:max_per_type]:
            chunks.append(f"  • \"{v['text']}\"")
    return "\n".join(chunks)
```

- [ ] **Step 3: Test passes + commit**

```bash
pytest tests/test_discovery_library.py -v
git add src/contract_intel_mvp/discovery/library.py tests/test_discovery_library.py
git commit -m "feat(discovery): clause library with structured schema and dedupe"
```

---

## Task 4: Pre-screen ranker (whole-doc cosine + filename rule)

Whole-doc cosine to a query embedded from `target_description + library` plus a cheap title/filename rule. Demote confirmed negatives, boost confirmed positives.

**Files:**
- Create: `src/contract_intel_mvp/discovery/ranker.py`
- Create: `tests/test_discovery_ranker.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_ranker.py`:
```python
from pathlib import Path
from contract_intel_mvp.discovery.ranker import rank_corpus
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature
from contract_intel_mvp.discovery.embeddings import embed_corpus


def _fake_embed(text):
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
    monkeypatch.setattr(e, "_call_ollama_embed", lambda t: [0.5, 0.5, 0.5, 0.5])
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
```

- [ ] **Step 2: Implement**

`src/contract_intel_mvp/discovery/ranker.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

from contract_intel_mvp.discovery.embeddings import load_embeddings, _call_ollama_embed
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
    # words like "license" individually
    for w in cls.split():
        if len(w) >= 5 and w in t: score += 0.05
    for a in anti_keywords:
        if a.lower() in t: score -= 0.30
    return score


def rank_corpus(root: Path, *, top_k: int | None = None,
                positive_boost: float = 0.20, negative_demote: float = 0.30) -> list[dict[str, Any]]:
    sig = load_signature(root)
    try:
        lib = load_library(root)
    except ValueError:
        lib = {"clause_types": []}
    store = load_embeddings(root)

    # Build the query text from description + library positive variations
    query_text = sig.target_description
    for ct in lib.get("clause_types", []):
        if ct.get("is_must_have"):
            for v in ct["variations"][:3]:
                query_text += "\n" + v["text"]
    qvec = _call_ollama_embed(query_text, model=store.model)
    if qvec is None:
        raise RuntimeError("could not embed signature query")
    q = np.array(qvec, dtype=np.float32)
    M = _normalize(store.matrix); qn = _normalize(q[None, :])[0]
    sims = (M @ qn).tolist()

    # Anti-keywords from must-not-have clause-type "type" names (split by underscore)
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
```

- [ ] **Step 3: Test passes + commit**

```bash
pytest tests/test_discovery_ranker.py -v
git add src/contract_intel_mvp/discovery/ranker.py tests/test_discovery_ranker.py
git commit -m "feat(discovery): pre-screen ranker (whole-doc cosine + filename rule + boost)"
```

---

## Task 5: Classifier with per-clause-type evidence extraction

The LLM doesn't just say yes/no — it extracts a verbatim quote per clause type. The library is rendered into the prompt as few-shot context.

**Files:**
- Create: `src/contract_intel_mvp/discovery/classifier.py`
- Modify: `src/contract_intel_mvp/prompts.py` (append `DISCOVERY_CLASSIFY_PROMPT`)
- Create: `tests/test_discovery_classifier.py`

- [ ] **Step 1: Append prompt**

In `src/contract_intel_mvp/prompts.py`:
```python
DISCOVERY_CLASSIFY_PROMPT = """You are deciding whether a single contract belongs to a target class.

{library_block}

Contract under review (first 6000 chars):
{doc_text}

Answer in JSON only. Schema:
{{
  "verdict": "yes" | "no",
  "confidence": <float 0..1>,
  "evidence_per_clause_type": {{
     "<clause_type_name>": "<verbatim substring from the contract that exemplifies that clause type, max 300 chars, or empty string if not present>"
  }},
  "rationale": "<one sentence>"
}}

Rules:
- evidence_per_clause_type MUST contain a key for every clause_type listed above.
- Each evidence value MUST be an exact substring of the contract or "" if not present.
- "yes" only if ALL must-have clause types are present and NO must-not-have clause type is the primary purpose.
"""
```

- [ ] **Step 2: Failing test**

`tests/test_discovery_classifier.py`:
```python
from pathlib import Path
from contract_intel_mvp.discovery.classifier import classify_candidates
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature


def _seed(tmp_root):
    init_signature(tmp_root, interview={
        "target_class": "License Agreement",
        "target_description": "primary IP grant",
        "clause_types": [
            {"type": "license_grant", "description": "right to use IP",
             "is_must_have": True, "seed_variations": ["Licensor grants Licensee a license"]},
            {"type": "primary_distribution", "description": "appoints distributor",
             "is_must_have": False, "seed_variations": ["appoints Distributor"]},
        ],
    })
    init_library_from_signature(tmp_root)


def test_classifier_extracts_per_clause_evidence(tmp_root, discovery_corpus, monkeypatch):
    _seed(tmp_root)
    import contract_intel_mvp.discovery.classifier as c
    def stub(*, model, prompt):
        is_lic = "TRADEMARK LICENSE" in prompt
        return {
            "verdict": "yes" if is_lic else "no",
            "confidence": 0.9,
            "evidence_per_clause_type": {
                "license_grant": "Licensor hereby grants" if is_lic else "",
                "primary_distribution": "" if is_lic else "appoints Distributor",
            },
            "rationale": "stub",
        }
    monkeypatch.setattr(c, "_call_ollama_json", stub)
    cands = [{"doc_id": "doc_lic_0", "score": 0.9},
             {"doc_id": "doc_dis_0", "score": 0.5}]
    out = classify_candidates(tmp_root, candidates=cands, model="qwen3:4b")
    by_id = {r["doc_id"]: r for r in out}
    assert by_id["doc_lic_0"]["verdict"] == "yes"
    assert "license_grant" in by_id["doc_lic_0"]["evidence_per_clause_type"]
    assert by_id["doc_lic_0"]["evidence_per_clause_type"]["license_grant"] == "Licensor hereby grants"
    assert by_id["doc_dis_0"]["verdict"] == "no"
    assert all(r["engine"] == "ollama" for r in out)


def test_classifier_falls_back_when_model_returns_none(tmp_root, discovery_corpus, monkeypatch):
    _seed(tmp_root)
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **_: None)
    out = classify_candidates(tmp_root, candidates=[{"doc_id": "doc_lic_0", "score": 0.9}],
                              model="qwen3:4b")
    assert out[0]["engine"] == "heuristic_fallback"
    assert out[0]["verdict"] in {"yes", "no"}
    assert "evidence_per_clause_type" in out[0]
```

- [ ] **Step 3: Implement**

`src/contract_intel_mvp/discovery/classifier.py`:
```python
from __future__ import annotations
from pathlib import Path
from typing import Any

from contract_intel_mvp.pipeline import _call_ollama_json, _load_documents
from contract_intel_mvp.prompts import DISCOVERY_CLASSIFY_PROMPT
from contract_intel_mvp.discovery.signature import load_signature
from contract_intel_mvp.discovery.library import render_library_text, load_library


def _heuristic_classify(doc_text: str, lib: dict) -> dict[str, Any]:
    text = doc_text.lower()
    have = {}; pos = neg = 0
    for ct in lib.get("clause_types", []):
        hit = any(v["text"].lower() in text for v in ct["variations"])
        have[ct["type"]] = doc_text[:160] if hit else ""
        if hit and ct["is_must_have"]: pos += 1
        if hit and not ct["is_must_have"]: neg += 1
    must = sum(1 for ct in lib.get("clause_types", []) if ct["is_must_have"]) or 1
    score = pos / must - 0.5 * neg
    return {"verdict": "yes" if score > 0.5 else "no",
            "confidence": max(0.0, min(1.0, abs(score - 0.5) + 0.4)),
            "evidence_per_clause_type": have,
            "rationale": "heuristic fallback"}


def classify_candidates(root: Path, *, candidates: list[dict[str, Any]],
                        model: str) -> list[dict[str, Any]]:
    lib = load_library(root)
    sig = load_signature(root)
    library_block = render_library_text(root, max_per_type=5)
    docs_by_id = {d.doc_id: d for d in _load_documents(root)}
    results = []
    for cand in candidates:
        doc = docs_by_id.get(cand["doc_id"])
        if doc is None: continue
        prompt = DISCOVERY_CLASSIFY_PROMPT.format(
            library_block=library_block,
            doc_text=(doc.text or "")[:6000],
        )
        parsed = _call_ollama_json(model=model, prompt=prompt)
        valid = (
            parsed and isinstance(parsed.get("verdict"), str)
            and parsed["verdict"] in {"yes", "no"}
            and isinstance(parsed.get("evidence_per_clause_type"), dict)
        )
        if valid:
            try:
                parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            except (TypeError, ValueError):
                parsed["confidence"] = 0.5
            engine = "ollama"
        else:
            parsed = _heuristic_classify(doc.text or "", lib)
            engine = "heuristic_fallback"
        parsed["doc_id"] = cand["doc_id"]
        parsed["screen_score"] = cand.get("score")
        parsed["engine"] = engine
        results.append(parsed)
    return results
```

- [ ] **Step 4: Test passes + commit**

```bash
pytest tests/test_discovery_classifier.py -v
git add src/contract_intel_mvp/discovery/classifier.py src/contract_intel_mvp/prompts.py tests/test_discovery_classifier.py
git commit -m "feat(discovery): classifier extracts evidence_per_clause_type with library few-shot"
```

---

## Task 6: Library write-back from confirmed positives

When the SME confirms a doc as `yes`, harvest its `evidence_per_clause_type` quotes and append them to `clause_library.json`. When the SME confirms `no`, harvest evidence into the `must_not_have` clause types only.

**Files:**
- Create: `src/contract_intel_mvp/discovery/harvest.py`
- Create: `tests/test_discovery_harvest.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_harvest.py`:
```python
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
```

- [ ] **Step 2: Implement**

`src/contract_intel_mvp/discovery/harvest.py`:
```python
from __future__ import annotations
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.signature import load_signature, save_signature
from contract_intel_mvp.discovery.library import append_variations, load_library


def harvest_from_label(root: Path, *, classification: dict[str, Any],
                       sme_verdict: str) -> dict[str, Any]:
    """Update the library and signature based on the SME's verdict for one doc."""
    if sme_verdict not in {"yes", "no", "borderline"}:
        raise ValueError("sme_verdict must be yes/no/borderline")
    if sme_verdict == "borderline":
        return {"updated_clause_types": [], "library_growth": 0}

    sig = load_signature(root)
    doc_id = classification["doc_id"]
    if sme_verdict == "yes":
        if doc_id not in sig.confirmed_positive_doc_ids:
            sig.confirmed_positive_doc_ids.append(doc_id)
    else:
        if doc_id not in sig.confirmed_negative_doc_ids:
            sig.confirmed_negative_doc_ids.append(doc_id)
    save_signature(root, sig)

    lib = load_library(root)
    must_have = {ct["type"] for ct in lib["clause_types"] if ct["is_must_have"]}
    must_not  = {ct["type"] for ct in lib["clause_types"] if not ct["is_must_have"]}
    evidence = classification.get("evidence_per_clause_type", {}) or {}
    growth = 0
    updated_types: list[str] = []

    if sme_verdict == "yes":
        # Harvest must-have evidence; ignore must-not-have (it was wrong agent extraction)
        for clause_type, text in evidence.items():
            if clause_type in must_have and (text or "").strip():
                before = len(next((ct["variations"] for ct in lib["clause_types"]
                                    if ct["type"] == clause_type), []))
                append_variations(root, clause_type=clause_type, variations=[
                    {"text": text, "source_doc_id": doc_id, "confirmed_by": "auto_from_sme_yes"}
                ])
                lib_after = load_library(root)
                after = len(next((ct["variations"] for ct in lib_after["clause_types"]
                                   if ct["type"] == clause_type), []))
                if after > before:
                    growth += after - before
                    updated_types.append(clause_type)
    else:
        # SME said no — harvest only must-not-have evidence (the disqualifying signals)
        for clause_type, text in evidence.items():
            if clause_type in must_not and (text or "").strip():
                before = len(next((ct["variations"] for ct in lib["clause_types"]
                                    if ct["type"] == clause_type), []))
                append_variations(root, clause_type=clause_type, variations=[
                    {"text": text, "source_doc_id": doc_id, "confirmed_by": "auto_from_sme_no"}
                ])
                lib_after = load_library(root)
                after = len(next((ct["variations"] for ct in lib_after["clause_types"]
                                   if ct["type"] == clause_type), []))
                if after > before:
                    growth += after - before
                    updated_types.append(clause_type)

    return {"updated_clause_types": updated_types, "library_growth": growth}
```

- [ ] **Step 3: Test passes + commit**

```bash
pytest tests/test_discovery_harvest.py -v
git add src/contract_intel_mvp/discovery/harvest.py tests/test_discovery_harvest.py
git commit -m "feat(discovery): harvest verbatim quotes from SME-confirmed labels into library"
```

---

## Task 7: Uncertainty sampler

20 docs per round: 5 highest-confidence-positive, 5 lowest-confidence-positive, 5 borderline-negative, 5 random borderline. Skip already-labeled.

**Files:**
- Create: `src/contract_intel_mvp/discovery/sampler.py`
- Create: `tests/test_discovery_sampler.py`

- [ ] **Step 1: Failing test + implementation are identical to Task 5 of `2026-05-04-discovery-pivot.md`** (uncertainty sampler logic doesn't change between cut-down and full). Copy the test file (`test_discovery_sampler.py`) and module (`sampler.py`) from that plan verbatim.

- [ ] **Step 2: Pass + commit**

```bash
pytest tests/test_discovery_sampler.py -v
git add src/contract_intel_mvp/discovery/sampler.py tests/test_discovery_sampler.py
git commit -m "feat(discovery): uncertainty sampler with reason codes"
```

---

## Task 8: Convergence + precision/recall

`record_round`, `should_stop`, `current_metrics`. Stop when corrections-per-round < threshold AND library growth in last round == 0, or after `max_rounds`.

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
```

- [ ] **Step 2: Implement**

`src/contract_intel_mvp/discovery/convergence.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.signature import load_signature


def _path(root): return root / "data" / "discovery" / "rounds.json"


def record_round(root: Path, *, round_index: int, corrections: int,
                 library_growth: int, batch_size: int) -> None:
    p = _path(root); p.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(p.read_text()) if p.exists() else {"rounds": []}
    state["rounds"].append({
        "round_index": round_index, "corrections": corrections,
        "library_growth": library_growth, "batch_size": batch_size,
    })
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_stop(root: Path, *, threshold: int = 3, max_rounds: int = 5) -> bool:
    p = _path(root)
    if not p.exists(): return False
    rounds = json.loads(p.read_text()).get("rounds", [])
    if len(rounds) >= max_rounds: return True
    if rounds and rounds[-1]["corrections"] < threshold and rounds[-1]["library_growth"] == 0:
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

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_discovery_convergence.py -v
git add src/contract_intel_mvp/discovery/convergence.py tests/test_discovery_convergence.py
git commit -m "feat(discovery): convergence (corrections + library_growth) and partial-gold metrics"
```

---

## Task 9: Loop driver — `run_round`, `submit_labels`, `finalize`

Single-file orchestrator stitching the pieces above.

**Files:**
- Create: `src/contract_intel_mvp/discovery/loop.py`
- Create: `tests/test_discovery_loop.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_loop.py`:
```python
import json
from pathlib import Path
from contract_intel_mvp.discovery.loop import run_round, submit_labels, finalize
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature
from contract_intel_mvp.discovery.embeddings import embed_corpus


def _fake_embed(text):
    if "TRADEMARK LICENSE" in text or "Licensor" in text:
        return [1.0, 0.0, 0.0, 0.1]
    if "DISTRIBUTOR" in text: return [0.0, 1.0, 0.0, 0.1]
    if "STRATEGIC" in text:   return [0.0, 0.0, 1.0, 0.1]
    return [0.0, 0.0, 0.0, 1.0]


def test_run_round_writes_artifacts(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text")
    init_signature(tmp_root, interview={
        "target_class": "License Agreement", "target_description": "TRADEMARK LICENSE",
        "clause_types": [
            {"type": "license_grant", "description": "x", "is_must_have": True, "seed_variations": []},
            {"type": "primary_distribution", "description": "x", "is_must_have": False, "seed_variations": []},
        ],
    })
    init_library_from_signature(tmp_root)
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **kw: {
        "verdict": "yes" if "TRADEMARK LICENSE" in kw["prompt"] else "no",
        "confidence": 0.85,
        "evidence_per_clause_type": {"license_grant": "Licensor hereby grants" if "TRADEMARK" in kw["prompt"] else "",
                                       "primary_distribution": "" if "TRADEMARK" in kw["prompt"] else "appoints"},
        "rationale": "x"
    })
    out = run_round(tmp_root, classifier_model="qwen3:4b",
                    top_k=15, batch_size=8, round_index=0, seed=1)
    assert out["round_index"] == 0
    assert out["classifications_count"] == 15
    assert (tmp_root / "data" / "discovery" / "review_queue_round_0.json").exists()


def test_submit_labels_grows_library(tmp_root, discovery_corpus, monkeypatch):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed", _fake_embed)
    embed_corpus(tmp_root, model="nomic-embed-text")
    init_signature(tmp_root, interview={
        "target_class": "License", "target_description": "TRADEMARK LICENSE",
        "clause_types": [
            {"type": "license_grant", "description": "x", "is_must_have": True, "seed_variations": []},
        ],
    })
    init_library_from_signature(tmp_root)
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **kw: {
        "verdict": "yes", "confidence": 0.9,
        "evidence_per_clause_type": {"license_grant": "Licensor hereby grants Licensee a license"},
        "rationale": "x"
    })
    run_round(tmp_root, classifier_model="qwen3:4b", top_k=5, batch_size=3, round_index=0, seed=1)
    queue = json.loads((tmp_root / "data" / "discovery" / "review_queue_round_0.json").read_text())
    first = queue["items"][0]
    res = submit_labels(tmp_root, round_index=0, labels=[
        {"doc_id": first["doc_id"], "verdict": "yes"}
    ])
    assert res["library_growth"] >= 1
    from contract_intel_mvp.discovery.library import load_library
    lib = load_library(tmp_root)
    lg = next(ct for ct in lib["clause_types"] if ct["type"] == "license_grant")
    assert any(v["confirmed_by"] == "auto_from_sme_yes" for v in lg["variations"])


def test_finalize_emits_positives_and_borderline(tmp_root):
    init_signature(tmp_root, interview={"target_class": "X", "target_description": "x",
                                        "clause_types": []})
    init_library_from_signature(tmp_root)
    classifications = [
        {"doc_id": "a", "verdict": "yes", "confidence": 0.95, "engine": "ollama",
         "evidence_per_clause_type": {}},
        {"doc_id": "b", "verdict": "yes", "confidence": 0.55, "engine": "ollama",
         "evidence_per_clause_type": {}},
        {"doc_id": "c", "verdict": "no",  "confidence": 0.9,  "engine": "ollama",
         "evidence_per_clause_type": {}},
    ]
    (tmp_root / "data" / "discovery" / "classifications_round_0.json").write_text(json.dumps(classifications))
    out = finalize(tmp_root, round_index=0, borderline_threshold=0.7)
    assert out["positives_count"] == 2
    assert out["borderline_count"] == 1  # b is borderline (conf 0.55 < 0.7)
    final = json.loads((tmp_root / "data" / "discovery" / "final.json").read_text())
    assert {p["doc_id"] for p in final["positives"]} == {"a", "b"}
```

- [ ] **Step 2: Implement**

`src/contract_intel_mvp/discovery/loop.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.ranker import rank_corpus
from contract_intel_mvp.discovery.classifier import classify_candidates
from contract_intel_mvp.discovery.sampler import sample_for_review
from contract_intel_mvp.discovery.harvest import harvest_from_label
from contract_intel_mvp.discovery.convergence import current_metrics, record_round
from contract_intel_mvp.discovery.library import load_library
from contract_intel_mvp.discovery.signature import load_signature


def run_round(root: Path, *, classifier_model: str, top_k: int = 200,
              batch_size: int = 20, round_index: int, seed: int = 0) -> dict[str, Any]:
    ranked = rank_corpus(root, top_k=top_k)
    classifications = classify_candidates(root, candidates=ranked, model=classifier_model)
    queue = sample_for_review(root, classifications=classifications,
                              batch_size=batch_size, seed=seed)
    metrics = current_metrics(root, classifications=classifications)
    out_dir = root / "data" / "discovery"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"classifications_round_{round_index}.json").write_text(
        json.dumps(classifications, indent=2), encoding="utf-8")
    (out_dir / f"review_queue_round_{round_index}.json").write_text(
        json.dumps({"round_index": round_index, "items": queue, "metrics": metrics}, indent=2),
        encoding="utf-8")
    return {"round_index": round_index,
            "classifications_count": len(classifications),
            "review_queue_size": len(queue),
            "metrics": metrics}


def submit_labels(root: Path, *, round_index: int,
                  labels: list[dict[str, Any]]) -> dict[str, Any]:
    cls_path = root / "data" / "discovery" / f"classifications_round_{round_index}.json"
    classifications = json.loads(cls_path.read_text()) if cls_path.exists() else []
    by_id = {c["doc_id"]: c for c in classifications}
    corrections = 0
    library_growth = 0
    for lbl in labels:
        doc_id = lbl["doc_id"]; sme = lbl["verdict"]
        if sme not in {"yes", "no", "borderline"}:
            continue
        cls = by_id.get(doc_id)
        if cls is None:
            continue
        if sme != cls.get("verdict") and sme != "borderline":
            corrections += 1
        result = harvest_from_label(root, classification=cls, sme_verdict=sme)
        library_growth += result.get("library_growth", 0)
    record_round(root, round_index=round_index, corrections=corrections,
                 library_growth=library_growth, batch_size=len(labels))
    return {"round_index": round_index, "labels_received": len(labels),
            "corrections": corrections, "library_growth": library_growth}


def finalize(root: Path, *, round_index: int,
             borderline_threshold: float = 0.7) -> dict[str, Any]:
    cls = json.loads((root / "data" / "discovery" / f"classifications_round_{round_index}.json").read_text())
    positives = [c for c in cls if c["verdict"] == "yes"]
    borderline = [c for c in positives if c.get("confidence", 0) < borderline_threshold]
    metrics = current_metrics(root, classifications=cls)
    sig = load_signature(root)
    out = {
        "target_class": sig.target_class,
        "round_index": round_index,
        "positives_count": len(positives),
        "borderline_count": len(borderline),
        "metrics": metrics,
        "positives": [{"doc_id": c["doc_id"], "confidence": c["confidence"],
                       "evidence_per_clause_type": c.get("evidence_per_clause_type", {})}
                      for c in positives],
    }
    out_dir = root / "data" / "discovery"
    (out_dir / "final.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (out_dir / "borderline.json").write_text(
        json.dumps({"items": borderline, "threshold": borderline_threshold}, indent=2),
        encoding="utf-8")
    return {"positives_count": len(positives),
            "borderline_count": len(borderline),
            "final_path": str((out_dir / "final.json").relative_to(root))}
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_discovery_loop.py -v
git add src/contract_intel_mvp/discovery/loop.py tests/test_discovery_loop.py
git commit -m "feat(discovery): loop driver — run_round, submit_labels (with library write-back), finalize"
```

---

## Task 10: Reframed interview prompt (clause-type elicitation) + opening monologue

The OpenAI interview chat now elicits clause TYPES with seed variations, and the agent's first message is a scripted onboarding monologue.

**Files:**
- Modify: `src/contract_intel_mvp/web.py`
- Create: `tests/test_discovery_interview.py`

- [ ] **Step 1: Failing test**

`tests/test_discovery_interview.py`:
```python
from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def test_discovery_chat_initial_returns_monologue(tmp_root, monkeypatch):
    monkeypatch.delenv("OPENAI_INTERVIEW", raising=False)
    app = build_app(root=tmp_root)
    client = TestClient(app)
    resp = client.post("/api/interview/discovery-chat", json={
        "signature": {"target_class": "", "target_description": "", "clause_types": []},
        "message": "",
        "initial": True,
    }).json()
    assert resp["engine"] == "scripted_opening"
    assert "discovery agent" in resp["assistant"].lower()
    assert "three rounds" in resp["assistant"].lower() or "three" in resp["assistant"].lower()


def test_discovery_chat_save_initializes_library(tmp_root, monkeypatch):
    monkeypatch.delenv("OPENAI_INTERVIEW", raising=False)
    app = build_app(root=tmp_root)
    client = TestClient(app)
    resp = client.post("/api/interview/discovery-chat", json={
        "signature": {
            "target_class": "License Agreement",
            "target_description": "Primary IP grant",
            "clause_types": [
                {"type": "license_grant", "description": "right to use", "is_must_have": True,
                 "seed_variations": ["Licensor grants Licensee a license"]},
                {"type": "primary_distribution", "description": "appoints distributor",
                 "is_must_have": False,
                 "seed_variations": ["appoints Distributor as the exclusive distributor"]},
            ],
        },
        "message": "save",
        "save": True,
    }).json()
    assert resp["saved"] is True
    assert (tmp_root / "data" / "discovery" / "signature.json").exists()
    assert (tmp_root / "data" / "discovery" / "clause_library.json").exists()
```

- [ ] **Step 2: Implement endpoint**

In `src/contract_intel_mvp/web.py`, inside `build_app(root)`:
```python
from .discovery.signature import init_signature, load_signature
from .discovery.library import init_library_from_signature

DISCOVERY_OPENING = (
    "Hi. I'm a discovery agent — give me a folder of contracts and I'll find the "
    "ones of a specific type you're looking for, even if you have thousands of "
    "them and no metadata.\n\n"
    "Here's how this works in three rounds, about 15 minutes total:\n\n"
    "(1) You tell me what to look for. I'll ask 4-5 questions to build a clear "
    "signature: the contract type, what clauses it always has, what parties or "
    "relationships it involves, and what would look similar but isn't actually it.\n\n"
    "(2) I do the heavy lifting. I embed your whole corpus once, rank every "
    "contract by similarity to your signature, and run a small local model on "
    "the top candidates to make a yes/no judgment with a confidence score.\n\n"
    "(3) I ask you to look at 20 borderline cases. I learn from your corrections "
    "and re-rank. After 2-3 rounds I converge — you get a final list of "
    "confirmed positives, plus a clause library showing every variation I've "
    "seen of each defining clause.\n\n"
    "To start: what type of contract are you looking for? Give me a one-line "
    "description in your own words."
)

DISCOVERY_SYSTEM_PROMPT = (
    "You are a discovery interview agent. Help the user define ONE target contract "
    "class. Build a structured signature with clause TYPES, each with a "
    "description, is_must_have flag, and 1-2 example phrasings. Ask focused "
    "follow-ups about defining clauses and what should be excluded. Output strict "
    "JSON only."
)

@app.post("/api/interview/discovery-chat")
def discovery_chat(payload: dict):
    sig_in = payload.get("signature") or {}
    message = str(payload.get("message", "")).strip()
    save = bool(payload.get("save"))
    initial = bool(payload.get("initial"))

    if initial:
        return {"signature": sig_in, "assistant": DISCOVERY_OPENING,
                "engine": "scripted_opening"}

    if save and sig_in.get("target_class") and sig_in.get("target_description"):
        init_signature(root, interview=sig_in)
        init_library_from_signature(root)
        return {"signature": sig_in, "saved": True,
                "assistant": "Signature saved. Library seeded from your examples. Embed the corpus and run round 0.",
                "engine": "local_save"}

    if _openai_interview_enabled():
        prompt = {
            "task": "Continue a discovery interview. Refine the structured signature.",
            "current_signature": sig_in,
            "user_message": message,
            "schema": {
                "assistant": "string",
                "signature_updates": {
                    "target_class": "string",
                    "target_description": "string",
                    "clause_types": [{
                        "type": "string", "description": "string",
                        "is_must_have": "boolean",
                        "seed_variations": ["string"],
                    }],
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
            "max_tokens": 800,
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
                resp = json.loads(r.read().decode("utf-8"))
            content = resp["choices"][0]["message"]["content"]
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

    return {"signature": sig_in,
            "assistant": "Tell me one specific clause type that this contract type always contains, and give me a one-sentence example of how it usually reads.",
            "engine": "local_discovery_fallback"}
```

Mirror the same endpoint behavior in the http.server `Handler` (added in `do_POST` for `/api/interview/discovery-chat`).

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_discovery_interview.py -v
git add src/contract_intel_mvp/web.py tests/test_discovery_interview.py
git commit -m "feat(discovery): scripted opening + clause-type-aware interview agent"
```

---

## Task 11: HTTP endpoints

POST `/api/discovery/embed`, `/run-round`, `/submit-labels`, `/finalize`. GET `/api/discovery/state`, `/api/discovery/library`.

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


def _seed_minimal(tmp_root, monkeypatch, discovery_corpus):
    import contract_intel_mvp.discovery.embeddings as e
    monkeypatch.setattr(e, "_call_ollama_embed",
                        lambda t: [1.0, 0.0, 0.0, 0.0] if "TRADEMARK" in t else [0.0]*3 + [1.0])
    from contract_intel_mvp.discovery.signature import init_signature
    from contract_intel_mvp.discovery.library import init_library_from_signature
    init_signature(tmp_root, interview={
        "target_class": "License Agreement",
        "target_description": "TRADEMARK LICENSE",
        "clause_types": [{"type": "license_grant", "description": "x",
                          "is_must_have": True, "seed_variations": []}],
    })
    init_library_from_signature(tmp_root)
    from contract_intel_mvp.discovery.embeddings import embed_corpus
    embed_corpus(tmp_root, model="nomic-embed-text")


def test_state_endpoint(tmp_root, discovery_corpus, monkeypatch):
    _seed_minimal(tmp_root, monkeypatch, discovery_corpus)
    client = TestClient(build_app(root=tmp_root))
    state = client.get("/api/discovery/state").json()
    assert state["embedded_count"] == 20
    assert state["target_class"] == "License Agreement"
    assert state["finalized"] is False
    assert state["library_size"] >= 1  # at least one clause type


def test_library_endpoint_returns_full_library(tmp_root, discovery_corpus, monkeypatch):
    _seed_minimal(tmp_root, monkeypatch, discovery_corpus)
    client = TestClient(build_app(root=tmp_root))
    lib = client.get("/api/discovery/library").json()
    assert lib["target_class"] == "License Agreement"
    assert any(ct["type"] == "license_grant" for ct in lib["clause_types"])


def test_submit_labels_grows_library_via_api(tmp_root, discovery_corpus, monkeypatch):
    _seed_minimal(tmp_root, monkeypatch, discovery_corpus)
    import contract_intel_mvp.discovery.classifier as c
    monkeypatch.setattr(c, "_call_ollama_json", lambda **kw: {
        "verdict": "yes", "confidence": 0.9,
        "evidence_per_clause_type": {"license_grant": "Licensor hereby grants Licensee a license to use the Marks"},
        "rationale": "x"
    })
    client = TestClient(build_app(root=tmp_root))
    rr = client.post("/api/discovery/run-round",
                     json={"classifier_model": "qwen3:4b", "top_k": 5,
                           "batch_size": 3, "round_index": 0}).json()
    queue = json.loads((tmp_root / "data" / "discovery" / "review_queue_round_0.json").read_text())
    first = queue["items"][0]
    sub = client.post("/api/discovery/submit-labels", json={
        "round_index": 0,
        "labels": [{"doc_id": first["doc_id"], "verdict": "yes"}],
    }).json()
    assert sub["library_growth"] >= 1
```

- [ ] **Step 2: Implement endpoints**

Inside `build_app`:
```python
from .discovery.embeddings import embed_corpus
from .discovery.loop import run_round, submit_labels, finalize
from .discovery.convergence import should_stop
from .discovery.library import load_library

@app.get("/api/discovery/state")
def discovery_state():
    sig_path = root / "data" / "discovery" / "signature.json"
    emb_path = root / "data" / "discovery" / "embeddings.jsonl"
    rounds_path = root / "data" / "discovery" / "rounds.json"
    final_path = root / "data" / "discovery" / "final.json"
    target_class = None
    library_size = 0
    if sig_path.exists():
        target_class = load_signature(root).target_class
        try:
            lib = load_library(root)
            library_size = sum(len(ct["variations"]) for ct in lib["clause_types"])
        except Exception:
            library_size = 0
    return {
        "target_class": target_class,
        "embedded_count": sum(1 for l in emb_path.read_text().splitlines() if l.strip())
                          if emb_path.exists() else 0,
        "rounds": json.loads(rounds_path.read_text()).get("rounds", [])
                  if rounds_path.exists() else [],
        "finalized": final_path.exists(),
        "should_stop": should_stop(root),
        "library_size": library_size,
    }

@app.get("/api/discovery/library")
def discovery_library():
    try:
        return load_library(root)
    except ValueError:
        return {"target_class": None, "clause_types": []}

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

Mirror these paths in the http.server `Handler` for the live UI.

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_discovery_api.py -v
git add src/contract_intel_mvp/web.py tests/test_discovery_api.py
git commit -m "feat(discovery): http endpoints embed/run-round/submit-labels/finalize/state/library"
```

---

## Task 12: Discovery UI structure

Single Discovery tab. Library viewer is a first-class panel — it's the **artifact** that grows over rounds.

**Files:**
- Modify: `src/contract_intel_mvp/static/index.html`

- [ ] **Step 1: Add nav button**

```html
<button class="nav" data-view="discovery">Discovery</button>
```

- [ ] **Step 2: Add the discovery view**

```html
<section id="discovery" class="view">
  <section class="panel">
    <h3>1. Drop your contracts</h3>
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
    <div id="discChat" style="max-height: 320px; overflow: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 8px; background: white;"></div>
    <div style="display: flex; gap: 8px;">
      <input id="discChatInput" placeholder="e.g. I'm looking for license agreements..." style="flex: 1;" />
      <button id="discChatSend" type="button">Send</button>
    </div>
    <details style="margin-top: 12px;"><summary>Current signature (advanced)</summary>
      <pre id="discSignaturePreview" style="font-size: 11px; background: #fafafa; padding: 8px;"></pre>
    </details>
    <p style="margin-top: 8px;"><button id="discSaveSig" type="button">Save signature & start discovery</button></p>
  </section>

  <section class="panel">
    <h3>3. Run a round</h3>
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
    <div id="discReviewQueue"></div>
    <p><button id="discSubmitLabels" type="button">Submit labels for this round</button></p>
  </section>

  <section class="panel">
    <h3>5. Clause library — the artifact</h3>
    <p>Every confirmed contract contributes verbatim quotes to this library, organized by clause type. This is what the agent has learned.</p>
    <div id="discLibrary" style="font-size: 12px;"></div>
  </section>

  <section class="panel">
    <h3>6. Final output</h3>
    <p><button id="discFinalizeBtn" type="button">Finalize</button></p>
    <pre id="discFinalResult" style="font-size: 12px; background: #fafafa; padding: 8px;"></pre>
  </section>

  <section class="panel">
    <h3>State</h3>
    <pre id="discStateJson" style="font-size: 11px; background: #fafafa; padding: 8px;"></pre>
  </section>
</section>
```

- [ ] **Step 3: Smoke + commit**

```bash
.venv/bin/contract-intel ui >/dev/null 2>&1 &
sleep 1
curl -s http://127.0.0.1:8765/ | grep -c 'data-view="discovery"'
pkill -9 -f "contract-intel ui"
git add src/contract_intel_mvp/static/index.html
git commit -m "feat(discovery): UI structure with library-as-artifact panel"
```

---

## Task 13: Discovery UI bindings (JS) — including library viewer

**Files:**
- Modify: `src/contract_intel_mvp/static/app.js`

- [ ] **Step 1: Append the JS module**

```javascript
// === Discovery tab (cut-down with clause library viewer) ===
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

  let chatLog = [];
  let currentSig = {target_class: "", target_description: "", clause_types: []};
  let queueState = {round_index: 0, items: []};
  let openingShown = false;

  function renderChat() {
    const el = $id("discChat"); if (!el) return;
    while (el.firstChild) el.removeChild(el.firstChild);
    chatLog.forEach(m => {
      const div = document.createElement("div");
      div.style.margin = "4px 0"; div.style.padding = "8px 12px"; div.style.borderRadius = "6px";
      div.style.background = m.role === "user" ? "#eef" : "#f4faf4";
      div.style.whiteSpace = "pre-wrap";
      div.textContent = (m.role === "user" ? "You: " : "Agent: ") + m.content;
      el.appendChild(div);
    });
    el.scrollTop = el.scrollHeight;
  }
  function renderSig() { setText("discSignaturePreview", JSON.stringify(currentSig, null, 2)); }

  async function showOpening() {
    if (openingShown) return;
    const r = await pj("/api/interview/discovery-chat",
                       {signature: currentSig, message: "", initial: true});
    chatLog.push({role: "agent", content: r.assistant || ""});
    renderChat(); openingShown = true;
  }

  async function chatSend() {
    const input = $id("discChatInput"); const msg = (input.value || "").trim();
    if (!msg) return;
    chatLog.push({role: "user", content: msg}); renderChat(); input.value = "";
    const r = await pj("/api/interview/discovery-chat",
                       {signature: currentSig, message: msg});
    if (r.signature) { currentSig = r.signature; renderSig(); }
    chatLog.push({role: "agent", content: r.assistant || "(no reply)"});
    renderChat();
  }

  async function saveSig() {
    const r = await pj("/api/interview/discovery-chat",
                       {signature: currentSig, message: "save", save: true});
    chatLog.push({role: "agent", content: r.assistant || "saved"}); renderChat();
    pollState(); pollLibrary();
  }

  async function uploadAndIngest(fileList) {
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
    setText("discEmbedStatus", "embedding (slow)...");
    const r = await pj("/api/discovery/embed", {model: "nomic-embed-text"});
    setText("discEmbedStatus", `Embedded ${r.embedded}, skipped ${r.skipped}, failed ${r.failed}.`);
    pollState();
  }

  async function runRound() {
    const idx = parseInt($id("discRoundIdx").value || "0", 10);
    const topK = parseInt($id("discTopK").value || "200", 10);
    const batch = parseInt($id("discBatch").value || "20", 10);
    setText("discRoundResult", "running... (this calls the LLM once per top-K candidate)");
    const r = await pj("/api/discovery/run-round",
                       {round_index: idx, top_k: topK, batch_size: batch,
                        classifier_model: "qwen3:4b"});
    setText("discRoundResult", JSON.stringify(r, null, 2));
    await loadReviewQueue(idx);
    pollState(); pollLibrary();
  }

  async function loadReviewQueue(idx) {
    const url = "/api/file?path=" + encodeURIComponent(`data/discovery/review_queue_round_${idx}.json`);
    try {
      const txt = await (await fetch(url)).text();
      queueState = JSON.parse(txt);
    } catch { queueState = {round_index: idx, items: []}; }
    const root = $id("discReviewQueue");
    while (root.firstChild) root.removeChild(root.firstChild);
    queueState.items.forEach(it => {
      const card = document.createElement("div");
      card.style.border = "1px solid #ccc"; card.style.borderRadius = "4px";
      card.style.padding = "10px"; card.style.margin = "8px 0"; card.style.background = "white";
      const head = document.createElement("div"); head.style.fontWeight = "bold";
      head.textContent = `${it.doc_id}  —  agent: ${it.verdict} (${(it.confidence||0).toFixed(2)})  —  reason: ${it.reason}`;
      card.appendChild(head);
      // Clause-by-clause evidence
      const ev = it.evidence_per_clause_type || {};
      Object.keys(ev).forEach(k => {
        const row = document.createElement("div");
        row.style.fontSize = "11px"; row.style.color = "#555"; row.style.marginTop = "4px";
        row.style.paddingLeft = "12px"; row.style.borderLeft = "2px solid #ddd";
        row.textContent = `[${k}]  ${(ev[k] || "(none)").slice(0, 240)}`;
        card.appendChild(row);
      });
      const btns = document.createElement("div"); btns.style.marginTop = "8px";
      ["yes","no","borderline"].forEach(v => {
        const b = document.createElement("button");
        b.textContent = v; b.style.marginRight = "6px";
        b.addEventListener("click", () => {
          it.userVerdict = v;
          [...btns.children].forEach(x => x.style.background = "");
          b.style.background = v === "yes" ? "#cfc" : v === "no" ? "#fcc" : "#ffc";
        });
        btns.appendChild(b);
      });
      card.appendChild(btns);
      root.appendChild(card);
    });
  }

  async function submitLabels() {
    const labels = queueState.items.filter(it => it.userVerdict).map(it => ({
      doc_id: it.doc_id, verdict: it.userVerdict,
    }));
    if (!labels.length) { alert("no labels selected"); return; }
    const r = await pj("/api/discovery/submit-labels",
                       {round_index: queueState.round_index, labels});
    alert(`Submitted ${r.labels_received}. Corrections: ${r.corrections}. Library grew by ${r.library_growth} variations.`);
    pollState(); pollLibrary();
  }

  async function finalize() {
    const idx = parseInt($id("discRoundIdx").value || "0", 10);
    const r = await pj("/api/discovery/finalize",
                       {round_index: idx, borderline_threshold: 0.7});
    setText("discFinalResult", JSON.stringify(r, null, 2));
    pollState();
  }

  async function pollState() {
    try { setText("discStateJson", JSON.stringify(await gj("/api/discovery/state"), null, 2)); }
    catch (e) { setText("discStateJson", "error: " + e); }
  }

  async function pollLibrary() {
    const root = $id("discLibrary"); if (!root) return;
    while (root.firstChild) root.removeChild(root.firstChild);
    let lib;
    try { lib = await gj("/api/discovery/library"); }
    catch { return; }
    if (!lib.clause_types || !lib.clause_types.length) {
      const p = document.createElement("p"); p.textContent = "(library not yet seeded)";
      root.appendChild(p); return;
    }
    const h = document.createElement("h4"); h.textContent = `Target class: ${lib.target_class}`;
    root.appendChild(h);
    lib.clause_types.forEach(ct => {
      const div = document.createElement("div");
      div.style.border = "1px solid #ddd"; div.style.borderRadius = "4px";
      div.style.padding = "10px"; div.style.margin = "8px 0";
      div.style.background = ct.is_must_have ? "#f4faf4" : "#faf4f4";
      const head = document.createElement("div"); head.style.fontWeight = "bold";
      const flag = ct.is_must_have ? "[MUST HAVE]" : "[MUST NOT HAVE]";
      head.textContent = `${flag} ${ct.type}  (${ct.variations.length} variations)`;
      div.appendChild(head);
      const desc = document.createElement("div"); desc.style.fontSize = "11px"; desc.style.color = "#666";
      desc.textContent = ct.description; div.appendChild(desc);
      ct.variations.forEach(v => {
        const item = document.createElement("div");
        item.style.fontSize = "11px"; item.style.marginTop = "6px"; item.style.paddingLeft = "12px";
        item.style.borderLeft = "2px solid #ccc";
        item.textContent = `• "${v.text}"   — from ${v.source_doc_id} (${v.confirmed_by})`;
        div.appendChild(item);
      });
      root.appendChild(div);
    });
  }

  function bind() {
    const fi = $id("discFolderInput");
    if (fi) fi.addEventListener("change", e => uploadAndIngest(e.target.files));
    const dz = $id("discDropzone");
    if (dz) {
      ["dragover","dragenter"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.style.background = "#eef9ee"; }));
      ["dragleave","drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.style.background = "#fafafa"; }));
      dz.addEventListener("drop", e => { e.preventDefault(); uploadAndIngest(e.dataTransfer.files); });
    }
    const eb = $id("discEmbedBtn"); if (eb) eb.addEventListener("click", embed);
    const cs = $id("discChatSend"); if (cs) cs.addEventListener("click", chatSend);
    const ci = $id("discChatInput"); if (ci) ci.addEventListener("keydown", e => { if (e.key === "Enter") chatSend(); });
    const ss = $id("discSaveSig"); if (ss) ss.addEventListener("click", saveSig);
    const rr = $id("discRunRoundBtn"); if (rr) rr.addEventListener("click", runRound);
    const sl = $id("discSubmitLabels"); if (sl) sl.addEventListener("click", submitLabels);
    const fb = $id("discFinalizeBtn"); if (fb) fb.addEventListener("click", finalize);
    showOpening(); pollState(); pollLibrary();
    setInterval(pollState, 5000); setInterval(pollLibrary, 8000);
  }
  if (document.readyState !== "loading") bind();
  else document.addEventListener("DOMContentLoaded", bind);
})();
```

- [ ] **Step 2: Smoke + commit**

```bash
pkill -9 -f "contract-intel ui" 2>/dev/null; sleep 1
set -a; . .env.local; set +a
.venv/bin/contract-intel ui >/dev/null 2>&1 &
sleep 1
curl -s http://127.0.0.1:8765/ | grep -c 'discFolderInput'   # → 1
git add src/contract_intel_mvp/static/app.js
git commit -m "feat(discovery): UI bindings — chat, embed, round, label, library viewer"
```

---

## Task 14: E2E smoke on curated CUAD demo corpus

Stage 50 CUAD contracts (10 licenses + 10 distribution + 8 strategic alliance + 5 service-with-IP + 17 unrelated) and run the full discovery loop with target = "License Agreement". Auto-label using filename ground truth so the smoke is reproducible.

**Files:**
- Create: `scripts/e2e_discovery_cuad.sh`

- [ ] **Step 1: Write script**

`scripts/e2e_discovery_cuad.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
set -a; . .env.local; set +a

contract-intel reset --keep-raw >/dev/null || true
contract-intel init >/dev/null

CUAD=/home/gui/archive/lexi-ai/Knowledge/Dataset/CUAD_v1/full_contract_txt
DEMO=data/raw_contracts/demo_licenses
mkdir -p "$DEMO"
rm -f "$DEMO"/*

# 10 licenses
for f in \
  "ArconicRolledProductsCorp_20191217_10-12B_EX-2.7_11923804_EX-2.7_Trademark License Agreement.txt" \
  "AlliedEsportsEntertainmentInc_20190815_8-K_EX-10.19_11788293_EX-10.19_Content License Agreement.txt" \
  "Array BioPharma Inc. - LICENSE, DEVELOPMENT AND COMMERCIALIZATION AGREEMENT.txt" \
  "ArtaraTherapeuticsInc_20200110_8-K_EX-10.5_11943350_EX-10.5_License Agreement.txt" \
  "CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.txt" \
  "ChinaRealEstateInformationCorp_20090929_F-1_EX-10.32_4771615_EX-10.32_Content License Agreement.txt" \
  "CytodynInc_20200109_10-Q_EX-10.5_11941634_EX-10.5_License Agreement.txt" \
  "DataCallTechnologies_20060918_SB-2A_EX-10.9_944510_EX-10.9_Content License Agreement.txt" \
  "EhaveInc_20190515_20-F_EX-4.44_11678816_EX-4.44_License Agreement_ Reseller Agreement.txt" \
  "CORIOINC_07_20_2000-EX-10.5-LICENSE AND HOSTING AGREEMENT.txt"; do
  cp "$CUAD/$f" "$DEMO/"
done
ls "$CUAD" | grep -iE "distributor|reseller" | head -10 | while read f; do cp "$CUAD/$f" "$DEMO/"; done
ls "$CUAD" | grep -iE "strategic alliance|collaboration" | head -8 | while read f; do cp "$CUAD/$f" "$DEMO/"; done
ls "$CUAD" | grep -iE "outsourcing|maintenance|consulting" | head -5 | while read f; do cp "$CUAD/$f" "$DEMO/"; done
ls "$CUAD" | grep -iE "joint venture|agency|franchise|supply|endorsement|sponsorship" | shuf -n 17 | while read f; do cp "$CUAD/$f" "$DEMO/"; done

echo "staged $(ls "$DEMO" | wc -l) files"

contract-intel ingest --input "$DEMO" >/dev/null

python3 - <<'PY'
from pathlib import Path
from contract_intel_mvp.discovery.signature import init_signature
from contract_intel_mvp.discovery.library import init_library_from_signature
init_signature(Path.cwd(), interview={
    "target_class": "License Agreement",
    "target_description": "Primary purpose is granting one party the right to use IP, trademarks, or content owned by the other, in exchange for fees or royalties.",
    "clause_types": [
        {"type": "license_grant", "description": "Grantor gives Grantee right to use IP",
         "is_must_have": True,
         "seed_variations": ["Licensor hereby grants to Licensee a non-exclusive license"]},
        {"type": "scope_definition", "description": "Field of use, territory, exclusivity, term",
         "is_must_have": True,
         "seed_variations": ["the license shall be exclusive in the Territory"]},
        {"type": "royalty_or_fee", "description": "Payment in exchange for the license",
         "is_must_have": True,
         "seed_variations": ["royalty of \\d+% of net receipts"]},
        {"type": "primary_distribution_appointment",
         "description": "Appoints distributor as primary purpose",
         "is_must_have": False,
         "seed_variations": ["Company hereby appoints Distributor as the exclusive distributor"]},
        {"type": "joint_venture_formation",
         "description": "Establishes a joint venture as primary purpose",
         "is_must_have": False,
         "seed_variations": ["the Parties hereby establish a joint venture"]},
    ],
})
init_library_from_signature(Path.cwd())
print("signature + library seeded")
PY

python3 - <<'PY'
from pathlib import Path
from contract_intel_mvp.discovery.embeddings import embed_corpus
print(embed_corpus(Path.cwd(), model="nomic-embed-text"))
PY

# Helper: auto-label using filename ground truth (filename contains "License" → yes; else no)
auto_label () {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
from contract_intel_mvp.discovery.loop import submit_labels
idx = int(sys.argv[1])
queue = json.loads(Path(f"data/discovery/review_queue_round_{idx}.json").read_text())
labels = []
for it in queue["items"]:
    # Use ingested doc title as proxy for filename ground truth
    docs = [json.loads(l) for l in Path("data/corpus/documents.jsonl").read_text().splitlines() if l.strip()]
    title = next((d.get("title", "") for d in docs if d["doc_id"] == it["doc_id"]), "")
    is_license_filename = ("license" in title.lower() and
                           "distributor" not in title.lower() and
                           "reseller" not in title.lower() and
                           "alliance" not in title.lower())
    labels.append({"doc_id": it["doc_id"], "verdict": "yes" if is_license_filename else "no"})
print(submit_labels(Path.cwd(), round_index=idx, labels=labels))
PY
}

# Round 0
python3 - <<'PY'
from pathlib import Path
from contract_intel_mvp.discovery.loop import run_round
print(run_round(Path.cwd(), classifier_model="qwen3:4b",
                top_k=30, batch_size=12, round_index=0, seed=1))
PY
auto_label 0

# Round 1 (library has grown from confirmed positives)
python3 - <<'PY'
from pathlib import Path
from contract_intel_mvp.discovery.loop import run_round
print(run_round(Path.cwd(), classifier_model="qwen3:4b",
                top_k=30, batch_size=12, round_index=1, seed=2))
PY
auto_label 1

# Finalize
python3 - <<'PY'
from pathlib import Path
from contract_intel_mvp.discovery.loop import finalize
print(finalize(Path.cwd(), round_index=1, borderline_threshold=0.7))
PY

echo "=== final.json ==="
python3 -m json.tool < data/discovery/final.json | head -40
echo "=== library size ==="
python3 -c "
import json
lib = json.load(open('data/discovery/clause_library.json'))
for ct in lib['clause_types']:
    print(f\"  [{('must_have' if ct['is_must_have'] else 'must_not_have')}] {ct['type']}: {len(ct['variations'])} variations\")
"
```

- [ ] **Step 2: Make executable, run, verify**

```bash
chmod +x scripts/e2e_discovery_cuad.sh
bash scripts/e2e_discovery_cuad.sh 2>&1 | tee /tmp/disc_cuad.log
```

Expected: `final.json` lists ~8-10 docs, mostly the 10 license filenames. Library shows growth across must-have and must-not-have clause types.

```bash
python3 -c "
import json
final = json.load(open('data/discovery/final.json'))
ids = {p['doc_id'] for p in final['positives']}
docs = [json.loads(l) for l in open('data/corpus/documents.jsonl').read_text().splitlines() if l.strip()]
license_ids = {d['doc_id'] for d in docs if 'license' in d['title'].lower()
                                          and 'distributor' not in d['title'].lower()
                                          and 'reseller' not in d['title'].lower()
                                          and 'alliance' not in d['title'].lower()}
tp = ids & license_ids
print(f'Predicted positives: {len(ids)}')
print(f'Filename-truth licenses: {len(license_ids)}')
print(f'TP: {len(tp)}')
print(f'Precision: {len(tp)/len(ids) if ids else 0:.2f}')
print(f'Recall:    {len(tp)/len(license_ids) if license_ids else 0:.2f}')
"
```

Expected: precision ≥ 0.7, recall ≥ 0.7 on a 50-doc curated corpus. If lower, inspect the borderline cases — most often the agent has flagged a `License and Hosting Agreement` (mixed type) as borderline correctly.

- [ ] **Step 3: Commit**

```bash
git add scripts/e2e_discovery_cuad.sh
git commit -m "test(discovery): CUAD e2e smoke for License Agreement target"
```

---

## Task 15: Dress rehearsal + version bump v0.4.0

**Files:**
- Modify: `docs/demo/presenter_script.md`
- Modify: `VERSION`

- [ ] **Step 1: Rewrite presenter script**

`docs/demo/presenter_script.md`:
```markdown
# Code Games Agentic Edition — Discovery Demo (8 min)

## The customer's problem
A media company brought us 30,000 contracts and asked for the publishing agreements among them. No metadata, no labels. Manual skimming would take weeks. We did it once by interviewing the user about what publishing agreements look like, running analysis, and showing "right or wrong" for each candidate. We turned that into an agent.

## Live arc

1. (1m) Open Discovery tab. Agent's opening message explains the three-round process.
2. (1m) Drop a folder of 50 mixed contracts. Click Embed corpus → "embedded 50".
3. (2m) Chat with the agent: "I'm looking for license agreements." Agent asks about license-grant clause, scope, royalty, exclusions like distribution / strategic alliance. Save the signature → library is seeded with the user's example phrasings.
4. (1m) Click Run round → top-30 candidates classified, 12 surfaced for review.
5. (2m) Label cards: yes on real licenses, no on distribution agreements that mention licensing language, no on strategic alliances with embedded IP grants. Submit labels. Each "yes" grows the library with verbatim quotes from the confirmed contract; each "no" on a close-negative grows the must-not-have side.
6. (1m) Run round 1. Library has grown from 5 seed variations to ~15 confirmed variations. Re-classification is sharper. Submit labels. Finalize.
7. Headline: "From 50 mixed contracts to N confirmed licenses, with a clause library that anyone can audit. Library has X variations of license-grant, Y variations of primary-distribution-appointment that we now reject. Same loop runs on 30,000 contracts as an overnight job."

## What to admit
- Demo n=50. Production target n=30,000 needs chunk-level embedding (v0.4.x roadmap, ~1 week).
- One target class per run today. Multi-class is a follow-up.
- Library entries auto-appended without per-variation SME approval. Variation clustering + approval is v0.4.x.

## The artifact judges scroll through
The clause library file. Every entry has provenance: which doc it came from, which round, who confirmed. That's the audit trail customers ask for.
```

- [ ] **Step 2: Bump VERSION**

```bash
echo "0.4.0" > VERSION
git add VERSION docs/demo/presenter_script.md
git commit -m "release: v0.4.0 — discovery + clause library (cut-down)"
git tag -a v0.4.0 -m "discovery + clause library cut-down for codegames"
```

- [ ] **Step 3: Final test sweep**

```bash
pytest -v
```
Expected: all tests pass.

---

## Self-Review

**Spec coverage:**
- Embed corpus → Task 1 ✓
- Class signature with clause-types schema → Tasks 2, 10 ✓
- Clause library with structured schema (full-version compatible) → Task 3 ✓
- Pre-screen ranker (whole-doc cosine + filename rule) → Task 4 ✓
- Classifier extracts evidence per clause type → Task 5 ✓
- Library write-back from confirmed positives AND negatives → Task 6 ✓
- Uncertainty sampler → Task 7 ✓
- Convergence (corrections + library_growth) + precision/recall → Task 8 ✓
- Loop driver + finalize → Task 9 ✓
- Scripted opening monologue + clause-type interview → Task 10 ✓
- HTTP endpoints → Task 11 ✓
- UI structure with library-as-artifact panel → Task 12 ✓
- UI bindings including library viewer → Task 13 ✓
- E2E smoke on curated CUAD → Task 14 ✓
- Dress rehearsal + v0.4.0 tag → Task 15 ✓

**What this plan deliberately does NOT do (cut-down scope):**
- Chunk-level embeddings — v0.4.1 (1 day).
- Variation embedder + cache — v0.4.2 (0.5 day).
- Cosine-based library matcher with calibrated thresholds — v0.4.3 (1 day).
- Variation clustering + SME approval cards — v0.4.4 (1.5 days).
- Switch from "LLM-classifies-everything" to "matcher-first, LLM-tiebreaker" — v0.5.0 (0.5 day).

The full-version schema is *already in place* — these are additive layers, no rewrite of cut-down code.

**Type/name consistency:**
- `verdict`: `"yes" | "no" | "borderline"` everywhere.
- `engine`: `"ollama" | "heuristic_fallback"` everywhere (matches existing convention).
- `confirmed_by`: `"interview_seed" | "auto_from_sme_yes" | "auto_from_sme_no"` (and v0.4.x will add `"sme_explicit"`).
- File paths under `data/discovery/`: `signature.json`, `clause_library.json`, `embeddings.jsonl`, `classifications_round_<i>.json`, `review_queue_round_<i>.json`, `rounds.json`, `final.json`, `borderline.json`.
- API paths: `/api/discovery/{embed,run-round,submit-labels,finalize,state,library}` and `/api/interview/discovery-chat`.
- `ClassSignature.clause_types` (signature) ↔ `clause_library.json#clause_types` (library) share the same shape; only the variations diverge in storage location.

**Risks:**
- **Cold-start recall**: round 0 has only seed variations. If user's seed phrasings are too narrow, recall@30 might miss real positives. Mitigation: filename rule in the ranker + `top_k` set to 30 for n=50 demo / 500 for n=30k.
- **Heuristic fallback contamination**: every classification is tagged with `engine`. Final output should warn if any positive came from heuristic_fallback. (Worth adding to Task 9's `finalize` if testing reveals issues.)
- **Library size at scale**: at 30k contracts the library could grow to 100+ variations per type. v0.4.x's clustering + cosine matcher addresses this.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-04-discovery-cutdown.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between, fast iteration.

**2. Inline Execution** — batch with checkpoints in this session.

**Which approach?**
