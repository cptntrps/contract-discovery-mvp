# Code Games Agentic-Edition MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing single-shot Ollama-extraction MVP into a defensible Code Games agentic-edition entry: a planner-driven agent that extracts contracts with a large primary model and a shadow small model in parallel, verifies its own evidence spans against source text, escalates uncertain documents to a human, then reruns the small model with reviewed context — all benchmarked on a held-out CUAD split with engine integrity guards and a live counterfactual toggle.

**Architecture:** Add an `agent/` module with a planner loop, tool registry, evidence verifier, shadow-model wrapper, triage scorer, and JSONL decision log on top of the existing `pipeline.py` functions (which become tools). Add a `benchmark/` module that scores three model conditions — large cold, small cold (from the shadow log), small + reviewed-context — on a held-out split and refuses to publish when any extraction used the heuristic fallback. State stays in JSON files under `data/`; no SQLite migration in MVP scope. UI gains a Decisions tab streaming the live agent log and a Benchmark tab with the counterfactual toggle.

**Tech Stack:** Python 3.10+, pytest, FastAPI (existing `web.py`), Ollama HTTP API at `127.0.0.1:11434` (`qwen2.5:14b` primary, `qwen3:4b` shadow), `requests`, `asyncio` for parallel model calls, vanilla JS for UI rendering using safe DOM APIs (no innerHTML on dynamic content).

---

## Pre-flight (Task 0): Repo, deps, test harness

The current project has no git repo and no test directory. Bootstrap before any work.

**Files:**
- Create: `/home/gui/code-games-contract-intelligence-mvp/.gitignore`
- Create: `/home/gui/code-games-contract-intelligence-mvp/VERSION`
- Create: `/home/gui/code-games-contract-intelligence-mvp/tests/__init__.py`
- Create: `/home/gui/code-games-contract-intelligence-mvp/tests/conftest.py`
- Modify: `/home/gui/code-games-contract-intelligence-mvp/pyproject.toml`

- [ ] **Step 1: Initialize git, write `.gitignore`, `VERSION`, baseline commit**

```bash
cd /home/gui/code-games-contract-intelligence-mvp
git init
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.pytest_cache/
data/runs/
data/reviews/
data/memory/
data/training/
data/corpus/
data/raw_contracts/
adapters/
*.db
*.db-wal
*.db-shm
```

`VERSION`:
```
0.1.0
```

```bash
git add .gitignore VERSION README.md ROADMAP.md pyproject.toml src config sample_corpus scripts
git commit -m "chore: project scaffolding"
```

- [ ] **Step 2: Create venv, install package + dev deps**

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install pytest pytest-asyncio httpx fastapi uvicorn requests
```

- [ ] **Step 3: Add test config to `pyproject.toml`**

Append to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx"]
```

- [ ] **Step 4: Add `tests/conftest.py` with a `tmp_root` fixture**

```python
import json
from pathlib import Path
import pytest


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Project root layout under a temp dir. Mirrors what `init` produces."""
    for sub in ("data/corpus", "data/runs", "data/reviews",
                "data/memory", "data/training", "data/raw_contracts"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def docs_jsonl(tmp_root: Path) -> Path:
    """Five fake documents with predictable structure."""
    docs = [
        {"doc_id": f"doc_{i:03}", "title": f"Doc {i}",
         "text": f"This License Agreement covers product {i}. "
                 f"Termination clause: 30 days notice. "
                 f"Governing law: Delaware.",
         "source": "fixture"}
        for i in range(5)
    ]
    path = tmp_root / "data" / "corpus" / "documents.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    return path
```

- [ ] **Step 5: Run pytest to confirm harness works (no tests yet)**

Run: `pytest -v`
Expected: `no tests ran` exit 5 (or 0 with no-tests warning) — harness loads cleanly.

- [ ] **Step 6: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "chore: add pytest harness with tmp_root fixture"
```

---

## Task 1: Held-out split

Add a deterministic stratified split that writes `data/corpus/splits.json` so every downstream stage knows which docs are reviewable and which are held out.

**Files:**
- Create: `src/contract_intel_mvp/splits.py`
- Create: `tests/test_splits.py`
- Modify: `src/contract_intel_mvp/cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_splits.py`:
```python
import json
from pathlib import Path
from contract_intel_mvp.splits import make_splits, load_splits


def test_make_splits_partitions_all_docs(tmp_root: Path, docs_jsonl: Path):
    out = make_splits(tmp_root, review_frac=0.6, seed=42)
    assert set(out["review_set"]) | set(out["holdout_set"]) == \
           {f"doc_{i:03}" for i in range(5)}
    assert set(out["review_set"]) & set(out["holdout_set"]) == set()
    assert out["split_seed"] == 42


def test_make_splits_is_deterministic(tmp_root: Path, docs_jsonl: Path):
    a = make_splits(tmp_root, review_frac=0.6, seed=42)
    b = make_splits(tmp_root, review_frac=0.6, seed=42)
    assert a["review_set"] == b["review_set"]


def test_load_splits_round_trip(tmp_root: Path, docs_jsonl: Path):
    written = make_splits(tmp_root, review_frac=0.6, seed=42)
    loaded = load_splits(tmp_root)
    assert loaded == written


def test_make_splits_refuses_empty_corpus(tmp_root: Path):
    import pytest
    with pytest.raises(ValueError, match="no documents"):
        make_splits(tmp_root, review_frac=0.6, seed=42)
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: contract_intel_mvp.splits`.

- [ ] **Step 3: Implement `splits.py`**

```python
"""Held-out / review-set partitioning."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any


def make_splits(root: Path, *, review_frac: float = 0.57, seed: int = 42) -> dict[str, Any]:
    docs_path = root / "data" / "corpus" / "documents.jsonl"
    if not docs_path.exists():
        raise ValueError("no documents.jsonl - run ingest first")
    doc_ids = [json.loads(line)["doc_id"]
               for line in docs_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not doc_ids:
        raise ValueError("no documents in corpus")
    rng = random.Random(seed)
    shuffled = sorted(doc_ids)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * review_frac))
    payload = {
        "review_set": sorted(shuffled[:cut]),
        "holdout_set": sorted(shuffled[cut:]),
        "split_seed": seed,
        "split_strategy": "random",
        "review_frac": review_frac,
    }
    out = root / "data" / "corpus" / "splits.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_splits(root: Path) -> dict[str, Any]:
    return json.loads((root / "data" / "corpus" / "splits.json").read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_splits.py -v`
Expected: 4 passed.

- [ ] **Step 5: Wire into CLI**

In `src/contract_intel_mvp/cli.py`, add the subparser registration alongside the others (see existing pattern at `cli.py:33-84`):
```python
split = sub.add_parser("split", help="Partition the corpus into review_set and holdout_set")
split.add_argument("--review-frac", type=float, default=0.57)
split.add_argument("--seed", type=int, default=42)
```

And in the dispatch block (after the existing handlers):
```python
elif args.cmd == "split":
    from contract_intel_mvp.splits import make_splits
    out = make_splits(Path.cwd(), review_frac=args.review_frac, seed=args.seed)
    print(f"split: {len(out['review_set'])} review, {len(out['holdout_set'])} holdout, seed={out['split_seed']}")
```

- [ ] **Step 6: Smoke test the CLI**

```bash
cd /home/gui/code-games-contract-intelligence-mvp
. .venv/bin/activate
contract-intel init
contract-intel ingest --input sample_corpus
contract-intel split --review-frac 0.6 --seed 42
cat data/corpus/splits.json
```
Expected: JSON with `review_set` and `holdout_set` covering all sample docs.

- [ ] **Step 7: Commit**

```bash
git add src/contract_intel_mvp/splits.py src/contract_intel_mvp/cli.py tests/test_splits.py
git commit -m "feat: held-out corpus split with deterministic seed"
```

---

## Task 2: Engine integrity gate

Refuse to publish a benchmark if any extraction row used `heuristic_fallback`. Provide an explicit override flag for diagnostic-only runs.

**Files:**
- Create: `src/contract_intel_mvp/engine_gate.py`
- Create: `tests/test_engine_gate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_engine_gate.py`:
```python
import pytest
from contract_intel_mvp.engine_gate import check_engine_integrity, EngineContamination


def test_all_ollama_passes():
    rows = [{"doc_id": "a", "engine": "ollama"},
            {"doc_id": "b", "engine": "ollama"}]
    report = check_engine_integrity(rows)
    assert report["ok"] is True
    assert report["fallback_count"] == 0


def test_any_fallback_fails():
    rows = [{"doc_id": "a", "engine": "ollama"},
            {"doc_id": "b", "engine": "heuristic_fallback"}]
    with pytest.raises(EngineContamination) as exc:
        check_engine_integrity(rows)
    assert "1 of 2" in str(exc.value)


def test_allow_fallback_returns_tagged_report():
    rows = [{"doc_id": "a", "engine": "heuristic_fallback"}]
    report = check_engine_integrity(rows, allow_fallback=True)
    assert report["ok"] is False
    assert report["fallback_count"] == 1
    assert report["fallback_doc_ids"] == ["a"]


def test_empty_rows_is_explicit_failure():
    with pytest.raises(EngineContamination, match="no rows"):
        check_engine_integrity([])
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_engine_gate.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `engine_gate.py`**

```python
"""Engine-integrity guard: refuse benchmarks that mix model with heuristic_fallback rows."""
from __future__ import annotations
from typing import Any


class EngineContamination(Exception):
    pass


def check_engine_integrity(rows: list[dict[str, Any]],
                           *, allow_fallback: bool = False) -> dict[str, Any]:
    if not rows:
        raise EngineContamination("no rows to check")
    fallback_ids = [r["doc_id"] for r in rows if r.get("engine") == "heuristic_fallback"]
    n_total = len(rows)
    n_fb = len(fallback_ids)
    report = {
        "ok": n_fb == 0,
        "total": n_total,
        "fallback_count": n_fb,
        "fallback_doc_ids": fallback_ids,
    }
    if n_fb and not allow_fallback:
        raise EngineContamination(
            f"engine contamination: {n_fb} of {n_total} rows used heuristic_fallback "
            f"(doc_ids: {fallback_ids[:5]}...). Pass allow_fallback=True to override."
        )
    return report
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_engine_gate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/engine_gate.py tests/test_engine_gate.py
git commit -m "feat: engine integrity gate refuses heuristic_fallback contamination"
```

---

## Task 3: Decision log writer

JSONL append-only audit trail for every agent action. Foundation for the streaming UI tab.

**Files:**
- Create: `src/contract_intel_mvp/agent/__init__.py`
- Create: `src/contract_intel_mvp/agent/decisions.py`
- Create: `tests/test_decisions.py`

- [ ] **Step 1: Write the failing test**

`tests/test_decisions.py`:
```python
import json
from pathlib import Path
from contract_intel_mvp.agent.decisions import DecisionLog


def test_log_writes_jsonl(tmp_root: Path):
    log = DecisionLog(tmp_root, run_id="run-1")
    log.append(action="extract_doc", args={"doc_id": "doc_001"},
               result={"ok": True}, rationale="first doc in queue")
    path = tmp_root / "data" / "runs" / "agent_decisions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["action"] == "extract_doc"
    assert row["run_id"] == "run-1"
    assert row["args"] == {"doc_id": "doc_001"}
    assert row["rationale"] == "first doc in queue"
    assert "ts" in row and "decision_id" in row


def test_log_appends_multiple(tmp_root: Path):
    log = DecisionLog(tmp_root, run_id="run-2")
    for i in range(3):
        log.append(action="noop", args={"i": i}, result={}, rationale="")
    path = tmp_root / "data" / "runs" / "agent_decisions.jsonl"
    assert len(path.read_text().splitlines()) == 3


def test_log_iter_filters_by_run(tmp_root: Path):
    DecisionLog(tmp_root, run_id="r1").append(action="a", args={}, result={}, rationale="")
    DecisionLog(tmp_root, run_id="r2").append(action="b", args={}, result={}, rationale="")
    rows = list(DecisionLog.iter(tmp_root, run_id="r1"))
    assert len(rows) == 1
    assert rows[0]["action"] == "a"
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_decisions.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `decisions.py`**

`src/contract_intel_mvp/agent/__init__.py`:
```python
"""Agent loop: planner, tools, verifier, shadow, triage, decision log."""
```

`src/contract_intel_mvp/agent/decisions.py`:
```python
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class DecisionLog:
    def __init__(self, root: Path, *, run_id: str):
        self.root = root
        self.run_id = run_id
        self.path = root / "data" / "runs" / "agent_decisions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, action: str, args: dict[str, Any],
               result: dict[str, Any], rationale: str,
               model_call_id: str | None = None) -> dict[str, Any]:
        row = {
            "decision_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "args": args,
            "result": result,
            "rationale": rationale,
            "model_call_id": model_call_id,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    @staticmethod
    def iter(root: Path, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
        path = root / "data" / "runs" / "agent_decisions.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if run_id is None or row.get("run_id") == run_id:
                yield row
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_decisions.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/agent/ tests/test_decisions.py
git commit -m "feat: agent decision log writer (jsonl)"
```

---

## Task 4: Evidence verifier

Substring-check every `evidence_snippet` against the source document text. Re-prompt up to 2 times on misses; flag for human review on persistent miss.

**Files:**
- Create: `src/contract_intel_mvp/agent/verifier.py`
- Create: `tests/test_verifier.py`
- Modify: `src/contract_intel_mvp/prompts.py`

- [ ] **Step 1: Write the failing test**

`tests/test_verifier.py`:
```python
from contract_intel_mvp.agent.verifier import (
    verify_spans, EvidenceReport, _normalize_for_match
)


def test_all_spans_verified():
    text = "This Agreement may be terminated upon 30 days written notice."
    clauses = [
        {"family": "termination", "evidence_snippet": "terminated upon 30 days written notice"}
    ]
    report = verify_spans(clauses, text)
    assert report.verified == 1
    assert report.missing == 0
    assert report.is_clean is True


def test_some_spans_missing():
    text = "Termination requires 30 days notice."
    clauses = [
        {"family": "termination", "evidence_snippet": "30 days notice"},
        {"family": "indemnity",   "evidence_snippet": "Acme indemnifies Buyer for losses"},
    ]
    report = verify_spans(clauses, text)
    assert report.verified == 1
    assert report.missing == 1
    assert report.missing_families == ["indemnity"]
    assert report.is_clean is False


def test_normalization_handles_whitespace_and_case():
    text = "Governing\nLaw:  Delaware."
    clauses = [{"family": "law", "evidence_snippet": "governing law: delaware"}]
    report = verify_spans(clauses, text)
    assert report.is_clean is True


def test_empty_clauses_is_clean():
    report = verify_spans([], "any text")
    assert report.is_clean is True
    assert report.verified == 0
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_verifier.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `verifier.py`**

```python
"""Evidence-span verification against source text."""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class EvidenceReport:
    verified: int
    missing: int
    missing_families: list[str]
    verified_families: list[str]

    @property
    def is_clean(self) -> bool:
        return self.missing == 0


_WS = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


def verify_spans(clauses: list[dict], source_text: str) -> EvidenceReport:
    haystack = _normalize_for_match(source_text)
    verified, missing = [], []
    for c in clauses:
        snippet = (c.get("evidence_snippet") or "").strip()
        if not snippet:
            missing.append(c.get("family", "unknown"))
            continue
        needle = _normalize_for_match(snippet)
        if needle and needle in haystack:
            verified.append(c.get("family", "unknown"))
        else:
            missing.append(c.get("family", "unknown"))
    return EvidenceReport(
        verified=len(verified),
        missing=len(missing),
        missing_families=missing,
        verified_families=verified,
    )
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_verifier.py -v`
Expected: 4 passed.

- [ ] **Step 5: Add a re-grounding prompt template**

In `src/contract_intel_mvp/prompts.py`, append:
```python
REGROUND_PROMPT = """You previously extracted clauses from this contract, but the following clause families had evidence quotes that do NOT appear verbatim in the source: {missing_families}.

Re-extract ONLY those clause families. Each evidence_snippet MUST be an exact substring of the source. If you cannot find a verbatim quote, return null for that family.

Source text:
{source_text}

Return JSON: {{"key_clauses": [{{"family": "<family>", "evidence_snippet": "<exact substring or null>"}}]}}
"""
```

- [ ] **Step 6: Add the retry-loop function**

In `src/contract_intel_mvp/agent/verifier.py`, append:
```python
from contract_intel_mvp.prompts import REGROUND_PROMPT
from contract_intel_mvp.pipeline import _call_ollama_json


def extract_with_verification(*, source_text: str, model: str,
                              initial_extraction: dict, max_retries: int = 2
                              ) -> tuple[dict, list[EvidenceReport]]:
    """Run verification on an initial extraction; re-prompt on misses up to max_retries.

    Returns the final extraction (with verified spans only) and the list of reports per attempt.
    """
    extraction = dict(initial_extraction)
    reports: list[EvidenceReport] = []
    for attempt in range(max_retries + 1):
        report = verify_spans(extraction.get("key_clauses", []), source_text)
        reports.append(report)
        if report.is_clean:
            break
        prompt = REGROUND_PROMPT.format(
            missing_families=report.missing_families,
            source_text=source_text[:8000],
        )
        regrounded = _call_ollama_json(model=model, prompt=prompt) or {}
        new_clauses = regrounded.get("key_clauses", [])
        existing = {c["family"]: c for c in extraction.get("key_clauses", [])
                    if c.get("family") in report.verified_families}
        for c in new_clauses:
            existing[c.get("family")] = c
        extraction["key_clauses"] = list(existing.values())
    final_report = verify_spans(extraction.get("key_clauses", []), source_text)
    extraction["key_clauses"] = [
        c for c in extraction.get("key_clauses", [])
        if c.get("family") in final_report.verified_families
    ]
    extraction["evidence_verification"] = {
        "attempts": len(reports),
        "final_verified": final_report.verified,
        "final_missing": final_report.missing,
        "rejected_families": final_report.missing_families,
    }
    return extraction, reports
```

- [ ] **Step 7: Add an integration test that mocks `_call_ollama_json`**

Add to `tests/test_verifier.py`:
```python
def test_extract_with_verification_retries_then_drops(monkeypatch):
    from contract_intel_mvp.agent import verifier as v
    text = "This Agreement may be terminated upon 30 days written notice."
    initial = {"key_clauses": [
        {"family": "termination", "evidence_snippet": "30 days written notice"},
        {"family": "indemnity",   "evidence_snippet": "fabricated quote not in source"},
    ]}
    monkeypatch.setattr(v, "_call_ollama_json", lambda **_: {
        "key_clauses": [{"family": "indemnity",
                         "evidence_snippet": "still fabricated"}]
    })
    final, reports = v.extract_with_verification(
        source_text=text, model="qwen3:4b", initial_extraction=initial, max_retries=2)
    families = [c["family"] for c in final["key_clauses"]]
    assert "termination" in families
    assert "indemnity" not in families
    assert final["evidence_verification"]["rejected_families"] == ["indemnity"]
```

- [ ] **Step 8: Run all verifier tests**

Run: `pytest tests/test_verifier.py -v`
Expected: 5 passed.

- [ ] **Step 9: Commit**

```bash
git add src/contract_intel_mvp/agent/verifier.py src/contract_intel_mvp/prompts.py tests/test_verifier.py
git commit -m "feat: evidence-span verifier with re-grounding retry loop"
```

---

## Task 5: Shadow-model parallel extraction

Run primary (`qwen2.5:14b`) and shadow (`qwen3:4b`) on the same prompt; only the primary feeds the review packet, both are persisted for the three-way benchmark.

**Files:**
- Create: `src/contract_intel_mvp/agent/shadow.py`
- Create: `tests/test_shadow.py`

- [ ] **Step 1: Write the failing test**

`tests/test_shadow.py`:
```python
import asyncio
from contract_intel_mvp.agent import shadow


def test_pair_extraction_calls_both_models(monkeypatch):
    calls = []

    async def fake_call(*, model, prompt):
        calls.append(model)
        return {"contract_type": f"type-from-{model}", "key_clauses": [], "coversheet": {}}

    monkeypatch.setattr(shadow, "_async_call_ollama_json", fake_call)
    result = asyncio.run(shadow.run_shadow_pair(
        prompt="extract this", primary_model="qwen2.5:14b", shadow_model="qwen3:4b"))

    assert sorted(calls) == ["qwen2.5:14b", "qwen3:4b"]
    assert result.primary["contract_type"] == "type-from-qwen2.5:14b"
    assert result.shadow["contract_type"] == "type-from-qwen3:4b"
    assert result.primary_engine == "ollama"
    assert result.shadow_engine == "ollama"


def test_pair_records_fallback_when_shadow_returns_none(monkeypatch):
    async def fake_call(*, model, prompt):
        if "qwen3" in model:
            return None
        return {"contract_type": "License", "key_clauses": [], "coversheet": {}}

    monkeypatch.setattr(shadow, "_async_call_ollama_json", fake_call)
    result = asyncio.run(shadow.run_shadow_pair(
        prompt="x", primary_model="qwen2.5:14b", shadow_model="qwen3:4b"))
    assert result.primary_engine == "ollama"
    assert result.shadow_engine == "heuristic_fallback"
    assert result.shadow is not None
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_shadow.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `shadow.py`**

```python
"""Parallel primary + shadow model extraction."""
from __future__ import annotations
import asyncio
import json as _json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ShadowPair:
    primary: dict[str, Any]
    shadow: dict[str, Any]
    primary_engine: str
    shadow_engine: str


async def _async_call_ollama_json(*, model: str, prompt: str,
                                  base_url: str = "http://127.0.0.1:11434"
                                  ) -> dict[str, Any] | None:
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{base_url}/api/generate", json=payload)
            r.raise_for_status()
            return _json.loads(r.json()["response"])
    except Exception:
        return None


def _heuristic_extraction_stub() -> dict[str, Any]:
    return {"contract_type": "unknown", "coversheet": {}, "key_clauses": [],
            "rationale": "heuristic fallback (shadow path)"}


async def run_shadow_pair(*, prompt: str, primary_model: str, shadow_model: str
                          ) -> ShadowPair:
    primary_task = _async_call_ollama_json(model=primary_model, prompt=prompt)
    shadow_task = _async_call_ollama_json(model=shadow_model, prompt=prompt)
    primary_raw, shadow_raw = await asyncio.gather(primary_task, shadow_task)
    return ShadowPair(
        primary=primary_raw or _heuristic_extraction_stub(),
        shadow=shadow_raw or _heuristic_extraction_stub(),
        primary_engine="ollama" if primary_raw else "heuristic_fallback",
        shadow_engine="ollama" if shadow_raw else "heuristic_fallback",
    )
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_shadow.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/agent/shadow.py tests/test_shadow.py
git commit -m "feat: parallel primary+shadow Ollama extraction"
```

---

## Task 6: Triage scorer

Decide which documents go to the human review queue based on uncertainty signals.

**Files:**
- Create: `src/contract_intel_mvp/agent/triage.py`
- Create: `tests/test_triage.py`

- [ ] **Step 1: Write the failing test**

`tests/test_triage.py`:
```python
from contract_intel_mvp.agent.triage import score_document, build_review_queue


def test_clean_extraction_low_uncertainty():
    extraction = {
        "doc_id": "doc_001",
        "contract_type": "License",
        "type_alternatives": [{"type": "Service", "score": 0.05}],
        "evidence_verification": {"attempts": 1, "final_missing": 0, "rejected_families": []},
        "key_clauses": [{"family": "termination"}, {"family": "law"}],
    }
    interview = {"key_clause_families": ["termination", "law"]}
    score = score_document(extraction, interview)
    assert score["uncertainty"] < 0.3
    assert score["reasons"] == []


def test_unverifiable_spans_raise_uncertainty():
    extraction = {
        "doc_id": "doc_002",
        "contract_type": "License",
        "type_alternatives": [],
        "evidence_verification": {"attempts": 3, "final_missing": 2,
                                  "rejected_families": ["indemnity", "ip"]},
        "key_clauses": [],
    }
    score = score_document(extraction, {"key_clause_families": []})
    assert "unverifiable_spans" in score["reasons"]
    assert score["uncertainty"] >= 0.4


def test_missing_expected_clauses_flagged():
    extraction = {
        "doc_id": "doc_003",
        "contract_type": "License",
        "type_alternatives": [],
        "evidence_verification": {"attempts": 1, "final_missing": 0, "rejected_families": []},
        "key_clauses": [{"family": "termination"}],
    }
    interview = {"key_clause_families": ["termination", "indemnity", "ip"]}
    score = score_document(extraction, interview)
    assert "missing_expected_clauses" in score["reasons"]


def test_close_type_alternatives_flagged():
    extraction = {
        "doc_id": "doc_004",
        "contract_type": "License",
        "type_alternatives": [{"type": "Service", "score": 0.85}],
        "evidence_verification": {"attempts": 1, "final_missing": 0, "rejected_families": []},
        "key_clauses": [],
    }
    score = score_document(extraction, {"key_clause_families": []})
    assert "close_type_alternative" in score["reasons"]


def test_build_review_queue_sorts_descending(tmp_root):
    extractions = [
        {"doc_id": "a", "contract_type": "X", "type_alternatives": [],
         "evidence_verification": {"attempts": 1, "final_missing": 0, "rejected_families": []},
         "key_clauses": [{"family": "t"}]},
        {"doc_id": "b", "contract_type": "Y", "type_alternatives": [],
         "evidence_verification": {"attempts": 3, "final_missing": 5,
                                   "rejected_families": ["x", "y", "z", "w", "v"]},
         "key_clauses": []},
    ]
    queue = build_review_queue(extractions, {"key_clause_families": ["t"]}, threshold=0.3)
    assert queue[0]["doc_id"] == "b"
    assert all(item["uncertainty"] >= 0.3 for item in queue)
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_triage.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `triage.py`**

```python
"""Uncertainty scoring → review queue."""
from __future__ import annotations
from typing import Any


def score_document(extraction: dict[str, Any], interview: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    score = 0.0

    ver = extraction.get("evidence_verification", {})
    n_missing = ver.get("final_missing", 0)
    if n_missing > 0:
        reasons.append("unverifiable_spans")
        score += min(0.5, 0.15 * n_missing)

    alts = extraction.get("type_alternatives", [])
    if alts and (alts[0].get("score") or 0) >= 0.4:
        reasons.append("close_type_alternative")
        score += 0.25

    expected = set(interview.get("key_clause_families", []) or [])
    found = {c.get("family") for c in extraction.get("key_clauses", [])}
    missing = expected - found
    if missing:
        reasons.append("missing_expected_clauses")
        score += min(0.3, 0.1 * len(missing))

    if ver.get("attempts", 1) > 1:
        reasons.append("required_retries")
        score += 0.1 * (ver["attempts"] - 1)

    return {
        "doc_id": extraction["doc_id"],
        "uncertainty": round(min(1.0, score), 3),
        "reasons": reasons,
    }


def build_review_queue(extractions: list[dict[str, Any]],
                       interview: dict[str, Any],
                       *, threshold: float = 0.3) -> list[dict[str, Any]]:
    scored = [score_document(e, interview) for e in extractions]
    flagged = [s for s in scored if s["uncertainty"] >= threshold]
    flagged.sort(key=lambda s: s["uncertainty"], reverse=True)
    return flagged
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_triage.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/agent/triage.py tests/test_triage.py
git commit -m "feat: triage uncertainty scorer with reason codes"
```

---

## Task 7: Tool registry

Tools the planner can call. Wraps existing pipeline functions plus the new agent-only ones.

**Files:**
- Create: `src/contract_intel_mvp/agent/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tools.py`:
```python
import pytest
from contract_intel_mvp.agent.tools import ToolRegistry, ToolError


def test_registry_dispatch_calls_registered_tool():
    reg = ToolRegistry()

    @reg.register("ping")
    def _ping(*, msg: str) -> dict:
        return {"echo": msg}

    out = reg.call("ping", {"msg": "hi"})
    assert out == {"echo": "hi"}


def test_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="unknown tool"):
        reg.call("nope", {})


def test_arg_validation_failure_is_tool_error():
    reg = ToolRegistry()

    @reg.register("strict")
    def _strict(*, n: int) -> dict:
        return {"n": n}

    with pytest.raises(ToolError):
        reg.call("strict", {"wrong": 1})


def test_list_tools_returns_signatures():
    reg = ToolRegistry()
    @reg.register("foo", description="foo doc")
    def _foo(*, a: str) -> dict: return {}
    listing = reg.list_tools()
    assert listing[0]["name"] == "foo"
    assert listing[0]["description"] == "foo doc"
    assert "a" in listing[0]["parameters"]
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `tools.py`**

```python
"""Tool registry for the planner loop."""
from __future__ import annotations
import inspect
from typing import Any, Callable


class ToolError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, *, description: str = ""):
        def deco(fn: Callable[..., dict]):
            sig = inspect.signature(fn)
            params = {p.name: str(p.annotation) for p in sig.parameters.values()
                      if p.kind == p.KEYWORD_ONLY}
            self._tools[name] = {"fn": fn, "description": description,
                                 "parameters": params}
            return fn
        return deco

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self._tools.get(name)
        if not spec:
            raise ToolError(f"unknown tool: {name}")
        try:
            return spec["fn"](**args)
        except TypeError as e:
            raise ToolError(f"bad args for {name}: {e}") from e

    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": n, "description": s["description"], "parameters": s["parameters"]}
                for n, s in self._tools.items()]
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_tools.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/agent/tools.py tests/test_tools.py
git commit -m "feat: tool registry with signature introspection"
```

---

## Task 8: Planner loop

Inspect state → decide next action → execute → log → repeat. Uses a deterministic state machine (no planner LLM needed for MVP) so the demo runs reliably.

**Files:**
- Create: `src/contract_intel_mvp/agent/planner.py`
- Create: `tests/test_planner.py`

- [ ] **Step 1: Write the failing test**

`tests/test_planner.py`:
```python
from contract_intel_mvp.agent.planner import deterministic_next_action


def test_first_action_is_ingest_when_no_corpus():
    state = {"docs_extracted": 0, "docs_total": 0, "review_pending": 0,
             "holdout_remaining": 0, "ingested": False}
    assert deterministic_next_action(state)["action"] == "ingest_corpus"


def test_after_ingest_extract_review_set():
    state = {"docs_extracted": 0, "docs_total": 40, "review_pending": 0,
             "holdout_remaining": 30, "ingested": True, "phase": "review"}
    assert deterministic_next_action(state)["action"] == "extract_review_batch"


def test_after_extraction_runs_triage():
    state = {"docs_extracted": 40, "docs_total": 40, "review_pending": 0,
             "holdout_remaining": 30, "ingested": True, "phase": "review",
             "triage_done": False}
    assert deterministic_next_action(state)["action"] == "triage"


def test_after_triage_awaits_human():
    state = {"docs_extracted": 40, "docs_total": 40, "review_pending": 12,
             "holdout_remaining": 30, "ingested": True, "phase": "review",
             "triage_done": True, "review_completed": False}
    assert deterministic_next_action(state)["action"] == "await_human"


def test_after_review_extracts_holdout():
    state = {"docs_extracted": 40, "docs_total": 40, "review_pending": 0,
             "holdout_remaining": 30, "ingested": True, "phase": "holdout",
             "triage_done": True, "review_completed": True, "cold_done": False}
    assert deterministic_next_action(state)["action"] == "extract_holdout_batch"


def test_after_holdout_runs_cold_small():
    state = {"docs_extracted": 30, "docs_total": 30, "review_pending": 0,
             "holdout_remaining": 0, "ingested": True, "phase": "holdout",
             "triage_done": True, "review_completed": True,
             "cold_done": False, "benchmarked": False}
    assert deterministic_next_action(state)["action"] == "extract_holdout_cold_small"


def test_terminates_when_done():
    state = {"docs_extracted": 70, "docs_total": 70, "review_pending": 0,
             "holdout_remaining": 0, "ingested": True, "phase": "holdout",
             "triage_done": True, "review_completed": True,
             "cold_done": True, "benchmarked": True}
    assert deterministic_next_action(state)["action"] == "stop"
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `planner.py`**

```python
"""Agent planner: state -> next action."""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Any

from contract_intel_mvp.agent.decisions import DecisionLog


def deterministic_next_action(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("ingested"):
        return {"action": "ingest_corpus", "args": {},
                "rationale": "corpus not ingested yet"}
    phase = state.get("phase", "review")
    if phase == "review":
        if state["docs_extracted"] < state["docs_total"]:
            return {"action": "extract_review_batch", "args": {},
                    "rationale": f"{state['docs_extracted']}/{state['docs_total']} review docs extracted"}
        if not state.get("triage_done"):
            return {"action": "triage", "args": {},
                    "rationale": "extraction complete; triage pending"}
        if not state.get("review_completed"):
            return {"action": "await_human",
                    "args": {"queue_size": state["review_pending"]},
                    "rationale": f"{state['review_pending']} docs awaiting human review"}
        return {"action": "advance_phase", "args": {"to": "holdout"},
                "rationale": "review phase complete"}
    if phase == "holdout":
        if state.get("holdout_remaining", 0) > 0:
            return {"action": "extract_holdout_batch", "args": {},
                    "rationale": f"{state['holdout_remaining']} holdout docs remaining"}
        if not state.get("cold_done"):
            return {"action": "extract_holdout_cold_small", "args": {},
                    "rationale": "compute small_cold column for benchmark"}
        if not state.get("benchmarked"):
            return {"action": "benchmark_three_way", "args": {},
                    "rationale": "all extractions complete; benchmark pending"}
        return {"action": "stop", "args": {}, "rationale": "all phases complete"}
    return {"action": "stop", "args": {}, "rationale": f"unknown phase {phase}"}


def run_agent(*, root: Path, registry, primary_model: str, shadow_model: str,
              max_steps: int = 50) -> str:
    run_id = f"run-{uuid.uuid4()}"
    log = DecisionLog(root, run_id=run_id)
    state = inspect_state(root)
    for _ in range(max_steps):
        decision = deterministic_next_action(state)
        log.append(action=decision["action"], args=decision["args"],
                   result={}, rationale=decision["rationale"])
        if decision["action"] in ("stop", "await_human"):
            return run_id
        result = registry.call(decision["action"], {
            **decision["args"],
            "root": root,
            "primary_model": primary_model,
            "shadow_model": shadow_model,
        })
        log.append(action=decision["action"] + ":result", args={},
                   result=result, rationale="tool result")
        state = inspect_state(root)
    return run_id


def inspect_state(root: Path) -> dict[str, Any]:
    base = root / "data"
    splits_path = base / "corpus" / "splits.json"
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else \
             {"review_set": [], "holdout_set": []}
    docs_path = base / "corpus" / "documents.jsonl"
    ingested = docs_path.exists() and docs_path.stat().st_size > 0
    baseline_path = base / "runs" / "baseline_results.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else []
    second_path = base / "runs" / "second_run_results.json"
    second = json.loads(second_path.read_text()) if second_path.exists() else []
    triage_path = base / "runs" / "triage_queue.json"
    triage_done = triage_path.exists()
    review_path = base / "reviews" / "review_packet.reviewed.json"
    review_completed = review_path.exists()
    cold_path = base / "runs" / "shadow_holdout_cold_results.json"
    cold_done = cold_path.exists()
    bench_path = base / "runs" / "benchmark.json"
    benchmarked = bench_path.exists()
    review_set = set(splits.get("review_set", []))
    holdout_set = set(splits.get("holdout_set", []))
    review_extracted = sum(1 for r in baseline if r.get("doc_id") in review_set)
    holdout_extracted = sum(1 for r in second if r.get("doc_id") in holdout_set)
    phase = "holdout" if review_completed else "review"
    return {
        "ingested": ingested,
        "phase": phase,
        "docs_extracted": review_extracted if phase == "review" else holdout_extracted,
        "docs_total": len(review_set) if phase == "review" else len(holdout_set),
        "holdout_remaining": len(holdout_set) - holdout_extracted,
        "review_pending": _pending_count(triage_path) if not review_completed else 0,
        "triage_done": triage_done,
        "review_completed": review_completed,
        "cold_done": cold_done,
        "benchmarked": benchmarked,
    }


def _pending_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(json.loads(path.read_text()).get("queue", []))
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_planner.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/agent/planner.py tests/test_planner.py
git commit -m "feat: deterministic agent planner (state machine)"
```

---

## Task 9: Wire pipeline functions as tools, add `agent run` CLI

Glue: register `extract_review_batch`, `extract_holdout_batch`, `extract_holdout_cold_small`, `triage`, `benchmark_three_way` as tools that call the existing/new functions, then expose `contract-intel agent run`.

**Files:**
- Create: `src/contract_intel_mvp/agent/wiring.py`
- Modify: `src/contract_intel_mvp/cli.py`
- Modify: `src/contract_intel_mvp/pipeline.py` (add `extract_split` and `extract_holdout_cold` helpers)
- Modify: `src/contract_intel_mvp/prompts.py` (promote inline prompt to `EXTRACTION_PROMPT`)

- [ ] **Step 1: Lift the existing extraction prompt into `prompts.py`**

Inspect `src/contract_intel_mvp/pipeline.py:run_extraction` (around line 451) and locate the inline prompt string. Copy it verbatim into `prompts.py` as a module-level constant `EXTRACTION_PROMPT` with placeholders `{title}`, `{text}`, `{interview}`, `{taxonomy}`. Replace the inline construction in `run_extraction` with `EXTRACTION_PROMPT.format(...)` so both the legacy CLI path and the new agent path use the same template.

- [ ] **Step 2: Add `extract_split` to `pipeline.py`**

Append to `src/contract_intel_mvp/pipeline.py`:
```python
def extract_split(root: Path, *, split: str, primary_model: str, shadow_model: str
                  ) -> dict[str, Any]:
    """Extract one corpus split (review or holdout) with primary+shadow + verifier."""
    import asyncio
    from contract_intel_mvp.agent.shadow import run_shadow_pair
    from contract_intel_mvp.agent.verifier import extract_with_verification
    from contract_intel_mvp.splits import load_splits

    splits = load_splits(root)
    target_ids = set(splits[f"{split}_set"])
    docs = [d for d in _load_documents(root) if d.doc_id in target_ids]
    interview = _load_json(root / "data" / "memory" / "interview.json", {})
    taxonomy = _load_json(root / "data" / "memory" / "taxonomy.json", _empty_taxonomy())
    use_memory = split == "holdout"

    primary_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []

    for doc in docs:
        prompt = _build_extraction_prompt(doc, interview, taxonomy if use_memory else {})
        pair = asyncio.run(run_shadow_pair(
            prompt=prompt, primary_model=primary_model, shadow_model=shadow_model))
        primary_verified, _ = extract_with_verification(
            source_text=doc.text, model=primary_model, initial_extraction=pair.primary)
        primary_verified["doc_id"] = doc.doc_id
        primary_verified["engine"] = pair.primary_engine
        primary_verified["role"] = "primary"
        primary_rows.append(primary_verified)
        shadow_row = dict(pair.shadow)
        shadow_row["doc_id"] = doc.doc_id
        shadow_row["engine"] = pair.shadow_engine
        shadow_row["role"] = "shadow"
        shadow_rows.append(shadow_row)

    runs = root / "data" / "runs"
    if split == "review":
        (runs / "baseline_results.json").write_text(
            json.dumps(primary_rows, indent=2), encoding="utf-8")
        (runs / "shadow_review_results.json").write_text(
            json.dumps(shadow_rows, indent=2), encoding="utf-8")
    else:
        (runs / "second_run_primary_holdout.json").write_text(
            json.dumps(primary_rows, indent=2), encoding="utf-8")
        (runs / "second_run_results.json").write_text(
            json.dumps(shadow_rows, indent=2), encoding="utf-8")
    return {"split": split, "n": len(docs)}


def extract_holdout_cold(root: Path, *, shadow_model: str) -> dict:
    """Run the small model on the holdout WITHOUT reviewed taxonomy. Cold baseline column."""
    from contract_intel_mvp.splits import load_splits
    splits = load_splits(root)
    holdout_ids = set(splits["holdout_set"])
    interview = _load_json(root / "data" / "memory" / "interview.json", {})
    rows: list[dict] = []
    for doc in _load_documents(root):
        if doc.doc_id not in holdout_ids:
            continue
        prompt = _build_extraction_prompt(doc, interview, {})
        result = _call_ollama_json(model=shadow_model, prompt=prompt)
        engine = "ollama" if result else "heuristic_fallback"
        if not result:
            result = _heuristic_extract(doc, interview, {})
        result["doc_id"] = doc.doc_id
        result["engine"] = engine
        result["role"] = "shadow_cold"
        rows.append(result)
    (root / "data" / "runs" / "shadow_holdout_cold_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    return {"n": len(rows)}


def _build_extraction_prompt(doc, interview: dict, taxonomy: dict) -> str:
    from contract_intel_mvp.prompts import EXTRACTION_PROMPT
    return EXTRACTION_PROMPT.format(
        title=doc.title,
        text=doc.text[:8000],
        interview=json.dumps(interview, indent=2),
        taxonomy=json.dumps(taxonomy, indent=2),
    )
```

- [ ] **Step 3: Implement `agent/wiring.py`**

```python
"""Wire pipeline functions into the tool registry."""
from __future__ import annotations
import json
from pathlib import Path

from contract_intel_mvp.agent.tools import ToolRegistry
from contract_intel_mvp.agent.triage import build_review_queue


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register("ingest_corpus", description="(no-op; ingest is operator-driven)")
    def _ingest(*, root: Path, **_) -> dict:
        return {"note": "ingest is run separately; agent assumes corpus exists"}

    @reg.register("extract_review_batch", description="primary+shadow on review_set")
    def _extract_review(*, root: Path, primary_model: str, shadow_model: str, **_) -> dict:
        from contract_intel_mvp.pipeline import extract_split
        return extract_split(root, split="review",
                             primary_model=primary_model, shadow_model=shadow_model)

    @reg.register("extract_holdout_batch", description="primary+shadow on holdout_set with reviewed context")
    def _extract_holdout(*, root: Path, primary_model: str, shadow_model: str, **_) -> dict:
        from contract_intel_mvp.pipeline import extract_split
        return extract_split(root, split="holdout",
                             primary_model=primary_model, shadow_model=shadow_model)

    @reg.register("extract_holdout_cold_small", description="cold small-model run on holdout")
    def _cold(*, root: Path, shadow_model: str, **_) -> dict:
        from contract_intel_mvp.pipeline import extract_holdout_cold
        return extract_holdout_cold(root, shadow_model=shadow_model)

    @reg.register("triage", description="score review_set extractions; build review queue")
    def _triage(*, root: Path, **_) -> dict:
        baseline = json.loads((root / "data" / "runs" / "baseline_results.json").read_text())
        interview = json.loads((root / "data" / "memory" / "interview.json").read_text())
        queue = build_review_queue(baseline, interview, threshold=0.3)
        out = {"queue": queue, "total_extracted": len(baseline), "flagged": len(queue)}
        (root / "data" / "runs" / "triage_queue.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        return {"flagged": len(queue), "total": len(baseline)}

    @reg.register("benchmark_three_way", description="large/small_cold/small_reviewed")
    def _bench(*, root: Path, primary_model: str, shadow_model: str, **_) -> dict:
        from contract_intel_mvp.benchmark.three_way import run_three_way
        return run_three_way(root, large=primary_model, small=shadow_model)

    @reg.register("advance_phase", description="phase transition marker")
    def _advance(*, root: Path, to: str, **_) -> dict:
        return {"phase": to}

    return reg
```

- [ ] **Step 4: Add `agent run` and `agent resume` to CLI**

In `src/contract_intel_mvp/cli.py`, add:
```python
agent_grp = sub.add_parser("agent", help="Run the agentic loop")
agent_sub = agent_grp.add_subparsers(dest="agent_cmd", required=True)
ar = agent_sub.add_parser("run", help="Start a new agent run")
ar.add_argument("--primary-model", default="qwen2.5:14b")
ar.add_argument("--shadow-model", default="qwen3:4b")
ar2 = agent_sub.add_parser("resume", help="Resume after human review")
ar2.add_argument("--primary-model", default="qwen2.5:14b")
ar2.add_argument("--shadow-model", default="qwen3:4b")
```

And dispatch:
```python
elif args.cmd == "agent":
    from contract_intel_mvp.agent.planner import run_agent
    from contract_intel_mvp.agent.wiring import build_registry
    registry = build_registry()
    run_id = run_agent(root=Path.cwd(), registry=registry,
                       primary_model=args.primary_model,
                       shadow_model=args.shadow_model)
    print(f"agent {args.agent_cmd}: run_id={run_id}")
```

- [ ] **Step 5: Smoke test (Ollama up, sample_corpus is small)**

```bash
. .venv/bin/activate
contract-intel reset
contract-intel init
contract-intel interview --config config/interview.example.json
contract-intel ingest --input sample_corpus
contract-intel split --review-frac 0.6 --seed 42
contract-intel agent run --primary-model qwen2.5:14b --shadow-model qwen3:4b
ls data/runs/
head -5 data/runs/agent_decisions.jsonl
head -20 data/runs/triage_queue.json
```
Expected: `baseline_results.json`, `shadow_review_results.json`, `triage_queue.json`, `agent_decisions.jsonl` all present. Engine column reflects whether Ollama responded.

- [ ] **Step 6: Commit**

```bash
git add src/contract_intel_mvp/agent/wiring.py src/contract_intel_mvp/cli.py src/contract_intel_mvp/pipeline.py src/contract_intel_mvp/prompts.py
git commit -m "feat: agent CLI with planner, tools, and shadow extraction wiring"
```

---

## Task 10: Three-way benchmark

Score `large` (from `second_run_primary_holdout.json`), `small_cold` (from `shadow_holdout_cold_results.json`), `small_reviewed` (from `second_run_results.json`). All three on the **holdout split only**. Engine-gated.

**Files:**
- Create: `src/contract_intel_mvp/benchmark/__init__.py`
- Create: `src/contract_intel_mvp/benchmark/three_way.py`
- Create: `tests/test_three_way.py`

- [ ] **Step 1: Write the failing test**

`tests/test_three_way.py`:
```python
import json
from pathlib import Path
from contract_intel_mvp.benchmark.three_way import run_three_way


def _seed_runs(root: Path):
    (root / "data" / "corpus" / "splits.json").write_text(json.dumps({
        "review_set": ["doc_001"],
        "holdout_set": ["doc_002", "doc_003"],
    }))
    (root / "data" / "runs" / "second_run_primary_holdout.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "License",
         "key_clauses": [{"family": "termination"}], "coversheet": {}},
        {"doc_id": "doc_003", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [{"family": "law"}], "coversheet": {}},
    ]))
    (root / "data" / "runs" / "shadow_holdout_cold_results.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [], "coversheet": {}},
        {"doc_id": "doc_003", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [{"family": "law"}], "coversheet": {}},
    ]))
    (root / "data" / "runs" / "second_run_results.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "License",
         "key_clauses": [{"family": "termination"}], "coversheet": {}},
        {"doc_id": "doc_003", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [{"family": "law"}], "coversheet": {}},
    ]))
    (root / "data" / "reviews" / "holdout_gold.json").write_text(json.dumps([
        {"doc_id": "doc_002", "accepted_contract_type": "License",
         "accepted_key_clauses": [{"family": "termination"}], "accepted_coversheet": {}},
        {"doc_id": "doc_003", "accepted_contract_type": "Service",
         "accepted_key_clauses": [{"family": "law"}], "accepted_coversheet": {}},
    ]))


def test_three_way_engine_gate_passes(tmp_root: Path):
    _seed_runs(tmp_root)
    out = run_three_way(tmp_root, large="qwen2.5:14b", small="qwen3:4b")
    assert out["engine_integrity"] == "ok"
    assert out["n_docs"] == 2
    m = out["metrics"]["contract_type_accuracy"]
    assert m["large"] == 1.0
    assert m["small_cold"] == 0.5
    assert m["small_reviewed"] == 1.0


def test_three_way_engine_gate_fails_on_fallback(tmp_root: Path):
    _seed_runs(tmp_root)
    runs = json.loads((tmp_root / "data" / "runs" / "second_run_results.json").read_text())
    runs[0]["engine"] = "heuristic_fallback"
    (tmp_root / "data" / "runs" / "second_run_results.json").write_text(json.dumps(runs))
    out = run_three_way(tmp_root, large="qwen2.5:14b", small="qwen3:4b",
                        allow_fallback=False)
    assert out["engine_integrity"] == "contaminated"
    assert "metrics" not in out
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_three_way.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `three_way.py`**

`src/contract_intel_mvp/benchmark/__init__.py`:
```python
"""Three-way benchmark and counterfactual."""
```

`src/contract_intel_mvp/benchmark/three_way.py`:
```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.engine_gate import check_engine_integrity, EngineContamination


def _load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["doc_id"]: r for r in rows}


def _accuracy(rows, key_pred, key_gold) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r[key_pred] == r[key_gold]) / len(rows)


def _clause_f1(pred: list[dict], gold: list[dict]) -> tuple[float, float, float]:
    p = {c.get("family") for c in pred if c.get("family")}
    g = {c.get("family") for c in gold if c.get("family")}
    if not p and not g:
        return 1.0, 1.0, 1.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def run_three_way(root: Path, *, large: str, small: str,
                  allow_fallback: bool = False) -> dict[str, Any]:
    splits = _load(root / "data" / "corpus" / "splits.json", {"holdout_set": []})
    holdout_ids = set(splits["holdout_set"])
    large_rows  = [r for r in _load(root / "data" / "runs" / "second_run_primary_holdout.json", []) if r["doc_id"] in holdout_ids]
    cold_rows   = [r for r in _load(root / "data" / "runs" / "shadow_holdout_cold_results.json", []) if r["doc_id"] in holdout_ids]
    revd_rows   = [r for r in _load(root / "data" / "runs" / "second_run_results.json", []) if r["doc_id"] in holdout_ids]
    gold        = _by_id(_load(root / "data" / "reviews" / "holdout_gold.json", []))

    try:
        for label, rows in [("large", large_rows), ("small_cold", cold_rows),
                            ("small_reviewed", revd_rows)]:
            check_engine_integrity(rows, allow_fallback=allow_fallback)
        integrity = "ok"
    except EngineContamination as e:
        out = {"engine_integrity": "contaminated", "error": str(e),
               "n_docs": len(holdout_ids)}
        (root / "data" / "runs" / "benchmark.json").write_text(json.dumps(out, indent=2))
        return out

    def _aligned(rows):
        return [{
            "doc_id": r["doc_id"],
            "pred_type": r["contract_type"],
            "gold_type": gold[r["doc_id"]]["accepted_contract_type"],
            "pred_clauses": r.get("key_clauses", []),
            "gold_clauses": gold[r["doc_id"]].get("accepted_key_clauses", []),
        } for r in rows if r["doc_id"] in gold]

    metrics: dict[str, dict[str, float]] = {}
    for label, rows in [("large", large_rows), ("small_cold", cold_rows),
                        ("small_reviewed", revd_rows)]:
        a = _aligned(rows)
        type_acc = _accuracy(a, "pred_type", "gold_type")
        f1s = [_clause_f1(r["pred_clauses"], r["gold_clauses"])[2] for r in a]
        metrics.setdefault("contract_type_accuracy", {})[label] = type_acc
        metrics.setdefault("clause_family_f1", {})[label] = sum(f1s) / len(f1s) if f1s else 0.0

    out = {
        "engine_integrity": integrity,
        "n_docs": len(holdout_ids),
        "models": {"large": large, "small_cold": small, "small_reviewed": small},
        "metrics": metrics,
    }
    (root / "data" / "runs" / "benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_three_way.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/benchmark/ tests/test_three_way.py
git commit -m "feat: three-way benchmark (large vs small_cold vs small_reviewed)"
```

---

## Task 11: Counterfactual toggle

Recompute the benchmark with verifier disabled or reviewed-context disabled to prove each component's contribution.

**Files:**
- Create: `src/contract_intel_mvp/benchmark/counterfactual.py`
- Create: `tests/test_counterfactual.py`

- [ ] **Step 1: Write the failing test**

`tests/test_counterfactual.py`:
```python
import json
from pathlib import Path
from contract_intel_mvp.benchmark.counterfactual import (
    recompute_without_verification, recompute_without_reviewed_context
)


def test_recompute_without_verification_drops_unverified_clauses(tmp_root: Path):
    (tmp_root / "data" / "runs" / "second_run_results.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "License",
         "key_clauses": [
             {"family": "termination", "evidence_snippet": "verified"},
         ],
         "coversheet": {},
         "evidence_verification": {"final_missing": 1, "rejected_families": ["ip"]}},
    ]))
    (tmp_root / "data" / "corpus" / "splits.json").write_text(
        json.dumps({"review_set": [], "holdout_set": ["doc_002"]}))
    (tmp_root / "data" / "reviews" / "holdout_gold.json").write_text(json.dumps([
        {"doc_id": "doc_002", "accepted_contract_type": "License",
         "accepted_key_clauses": [{"family": "termination"}, {"family": "ip"}],
         "accepted_coversheet": {}},
    ]))
    out = recompute_without_verification(tmp_root, model="qwen3:4b")
    assert "f1_with_verifier_on" in out
    assert "f1_with_verifier_off" in out
    assert out["f1_with_verifier_off"] != out["f1_with_verifier_on"]


def test_recompute_without_reviewed_context_uses_cold(tmp_root: Path):
    (tmp_root / "data" / "runs" / "shadow_holdout_cold_results.json").write_text(json.dumps([
        {"doc_id": "doc_002", "engine": "ollama", "contract_type": "Service",
         "key_clauses": [], "coversheet": {}},
    ]))
    (tmp_root / "data" / "corpus" / "splits.json").write_text(
        json.dumps({"review_set": [], "holdout_set": ["doc_002"]}))
    (tmp_root / "data" / "reviews" / "holdout_gold.json").write_text(json.dumps([
        {"doc_id": "doc_002", "accepted_contract_type": "License",
         "accepted_key_clauses": [], "accepted_coversheet": {}},
    ]))
    out = recompute_without_reviewed_context(tmp_root)
    assert out["contract_type_accuracy"] == 0.0
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_counterfactual.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `counterfactual.py`**

```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from contract_intel_mvp.benchmark.three_way import _load, _accuracy, _clause_f1, _by_id


def recompute_without_verification(root: Path, *, model: str) -> dict[str, Any]:
    """Score the small_reviewed column with rejected clauses RE-INCLUDED to show verifier value."""
    splits = _load(root / "data" / "corpus" / "splits.json", {"holdout_set": []})
    holdout_ids = set(splits["holdout_set"])
    revd = [r for r in _load(root / "data" / "runs" / "second_run_results.json", []) if r["doc_id"] in holdout_ids]
    gold = _by_id(_load(root / "data" / "reviews" / "holdout_gold.json", []))

    def _f1_over(rows, with_verifier: bool) -> float:
        f1s = []
        for r in rows:
            if r["doc_id"] not in gold:
                continue
            preds = r.get("key_clauses", [])
            if not with_verifier:
                rejected = r.get("evidence_verification", {}).get("rejected_families", [])
                preds = preds + [{"family": fam} for fam in rejected]
            _, _, f1 = _clause_f1(preds, gold[r["doc_id"]].get("accepted_key_clauses", []))
            f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else 0.0

    on = _f1_over(revd, True)
    off = _f1_over(revd, False)
    out = {
        "f1_with_verifier_on":  on,
        "f1_with_verifier_off": off,
        "delta": on - off,
        "n_docs": len(revd),
        "model": model,
    }
    (root / "data" / "runs" / "counterfactual_verifier.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out


def recompute_without_reviewed_context(root: Path) -> dict[str, Any]:
    splits = _load(root / "data" / "corpus" / "splits.json", {"holdout_set": []})
    holdout_ids = set(splits["holdout_set"])
    cold = [r for r in _load(root / "data" / "runs" / "shadow_holdout_cold_results.json", []) if r["doc_id"] in holdout_ids]
    gold = _by_id(_load(root / "data" / "reviews" / "holdout_gold.json", []))
    aligned = [{"pred_type": r["contract_type"],
                "gold_type": gold[r["doc_id"]]["accepted_contract_type"]}
               for r in cold if r["doc_id"] in gold]
    f1s = []
    for r in cold:
        if r["doc_id"] not in gold:
            continue
        _, _, f1 = _clause_f1(r.get("key_clauses", []),
                              gold[r["doc_id"]].get("accepted_key_clauses", []))
        f1s.append(f1)
    out = {
        "contract_type_accuracy": _accuracy(aligned, "pred_type", "gold_type"),
        "clause_family_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "n_docs": len(aligned),
    }
    (root / "data" / "runs" / "counterfactual_context.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `pytest tests/test_counterfactual.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/contract_intel_mvp/benchmark/counterfactual.py tests/test_counterfactual.py
git commit -m "feat: counterfactual toggles for verifier and reviewed-context"
```

---

## Task 12: Decision-log streaming UI tab

Endpoint returns `agent_decisions.jsonl` rows; new "Decisions" tab tails them with `setInterval`. Uses safe DOM APIs (`textContent` only — no innerHTML on dynamic content).

**Files:**
- Modify: `src/contract_intel_mvp/web.py`
- Create: `tests/test_web_decisions.py`

- [ ] **Step 1: Write the failing test**

`tests/test_web_decisions.py`:
```python
from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def test_decisions_endpoint_returns_jsonl_rows(tmp_root: Path):
    (tmp_root / "data" / "runs" / "agent_decisions.jsonl").write_text(
        '{"decision_id":"1","run_id":"r1","action":"a","args":{},"result":{},"rationale":"x","ts":"2026-05-04T00:00:00Z","model_call_id":null}\n'
        '{"decision_id":"2","run_id":"r1","action":"b","args":{},"result":{},"rationale":"y","ts":"2026-05-04T00:00:01Z","model_call_id":null}\n'
    )
    app = build_app(root=tmp_root)
    client = TestClient(app)
    resp = client.get("/api/decisions?run_id=r1")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 2
    assert rows[0]["action"] == "a"


def test_decisions_filters_by_run(tmp_root: Path):
    (tmp_root / "data" / "runs" / "agent_decisions.jsonl").write_text(
        '{"decision_id":"1","run_id":"r1","action":"a","args":{},"result":{},"rationale":"","ts":"t","model_call_id":null}\n'
        '{"decision_id":"2","run_id":"r2","action":"b","args":{},"result":{},"rationale":"","ts":"t","model_call_id":null}\n'
    )
    app = build_app(root=tmp_root)
    client = TestClient(app)
    rows = client.get("/api/decisions?run_id=r1").json()["rows"]
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_web_decisions.py -v`
Expected: FAIL — `build_app` factory missing or endpoint missing.

- [ ] **Step 3: Refactor `web.py` to expose `build_app(root)`**

If the existing module has a module-level `app` only, wrap setup in a factory:
```python
def build_app(root: Path) -> FastAPI:
    app = FastAPI()
    # ... existing route registrations, all referencing `root` instead of a global ROOT
    return app

# preserve module-level `app` for backwards compat with `uvicorn web:app`
app = build_app(Path.cwd())
```

- [ ] **Step 4: Add the decisions endpoint**

Inside `build_app`:
```python
from contract_intel_mvp.agent.decisions import DecisionLog

@app.get("/api/decisions")
def get_decisions(run_id: str | None = None):
    rows = list(DecisionLog.iter(root, run_id=run_id))
    return {"rows": rows}
```

- [ ] **Step 5: Add the Decisions tab UI (textContent-only rendering)**

In the existing HTML template, add a Decisions tab:
```html
<div id="decisions-tab" class="tab">
  <h2>Agent Decisions</h2>
  <input id="decisions-run-id" placeholder="run_id (or leave blank)" />
  <button id="decisions-load">Load</button>
  <pre id="decisions-log" style="height: 60vh; overflow: auto; font-size: 11px;"></pre>
</div>
<script>
async function loadDecisions() {
  const runIdEl = document.getElementById('decisions-run-id');
  const runId = runIdEl ? runIdEl.value : '';
  const url = runId ? '/api/decisions?run_id=' + encodeURIComponent(runId) : '/api/decisions';
  const res = await fetch(url).then(r => r.json());
  const lines = res.rows.map(function(r) {
    var args = JSON.stringify(r.args || {}).slice(0, 60);
    return r.ts + '  ' + (r.action + '').padEnd(28) + ' ' + args + '  ' + (r.rationale || '');
  });
  document.getElementById('decisions-log').textContent = lines.join('\n');
}
document.getElementById('decisions-load').addEventListener('click', loadDecisions);
setInterval(loadDecisions, 2000);
</script>
```

`textContent` is XSS-safe. No `innerHTML` on user-derived data anywhere in this tab.

- [ ] **Step 6: Run tests, confirm pass**

Run: `pytest tests/test_web_decisions.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/contract_intel_mvp/web.py tests/test_web_decisions.py
git commit -m "feat: web Decisions tab tailing agent_decisions.jsonl"
```

---

## Task 13: Three-way Benchmark UI + counterfactual toggle

UI tab that shows the three-column scoreboard and two toggle buttons that POST to recompute the counterfactual numbers. Renders the table with `createElement` + `textContent` only (no innerHTML on dynamic data).

**Files:**
- Modify: `src/contract_intel_mvp/web.py`
- Create: `tests/test_web_benchmark.py`

- [ ] **Step 1: Write the failing test**

`tests/test_web_benchmark.py`:
```python
import json
from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def test_benchmark_endpoint_returns_three_way(tmp_root: Path):
    (tmp_root / "data" / "runs" / "benchmark.json").write_text(json.dumps({
        "engine_integrity": "ok",
        "n_docs": 30,
        "models": {"large": "qwen2.5:14b", "small_cold": "qwen3:4b", "small_reviewed": "qwen3:4b"},
        "metrics": {
            "contract_type_accuracy": {"large": 0.97, "small_cold": 0.83, "small_reviewed": 0.93},
            "clause_family_f1":        {"large": 0.69, "small_cold": 0.41, "small_reviewed": 0.62},
        },
    }))
    app = build_app(root=tmp_root)
    client = TestClient(app)
    out = client.get("/api/benchmark/three-way").json()
    assert out["engine_integrity"] == "ok"
    assert out["metrics"]["clause_family_f1"]["small_reviewed"] == 0.62


def test_counterfactual_endpoint_recomputes(tmp_root: Path, monkeypatch):
    import contract_intel_mvp.web as w
    monkeypatch.setattr(w, "recompute_without_verification",
                        lambda root, model: {"f1_with_verifier_on": 0.62,
                                              "f1_with_verifier_off": 0.51, "delta": 0.11})
    monkeypatch.setattr(w, "recompute_without_reviewed_context",
                        lambda root: {"clause_family_f1": 0.41})
    app = build_app(root=tmp_root)
    client = TestClient(app)
    a = client.post("/api/benchmark/counterfactual",
                    json={"toggle": "verifier_off", "model": "qwen3:4b"}).json()
    assert a["f1_with_verifier_off"] == 0.51
    b = client.post("/api/benchmark/counterfactual",
                    json={"toggle": "context_off"}).json()
    assert b["clause_family_f1"] == 0.41
```

- [ ] **Step 2: Run test, confirm failure**

Run: `pytest tests/test_web_benchmark.py -v`
Expected: FAIL.

- [ ] **Step 3: Add endpoints to `web.py`**

Inside `build_app`:
```python
import json as _json
from contract_intel_mvp.benchmark.counterfactual import (
    recompute_without_verification, recompute_without_reviewed_context
)

@app.get("/api/benchmark/three-way")
def benchmark_three_way():
    p = root / "data" / "runs" / "benchmark.json"
    if not p.exists():
        return {"engine_integrity": "missing"}
    return _json.loads(p.read_text())

@app.post("/api/benchmark/counterfactual")
def counterfactual(payload: dict):
    toggle = payload.get("toggle")
    if toggle == "verifier_off":
        return recompute_without_verification(root, model=payload.get("model", "qwen3:4b"))
    if toggle == "context_off":
        return recompute_without_reviewed_context(root)
    return {"error": "unknown toggle: " + str(toggle)}
```

Module-level imports for monkeypatch reachability:
```python
from contract_intel_mvp.benchmark.counterfactual import (
    recompute_without_verification, recompute_without_reviewed_context
)
```
(These need to be at module scope so the test can monkeypatch them.)

- [ ] **Step 4: Add the Benchmark tab UI (textContent-only rendering)**

```html
<div id="benchmark-tab" class="tab">
  <h2>Three-Way Benchmark</h2>
  <div id="bench-banner"></div>
  <table id="bench-table"></table>

  <h3>Counterfactuals</h3>
  <button id="cf-verifier">Verifier OFF</button>
  <button id="cf-context">Reviewed-context OFF</button>
  <pre id="cf-result"></pre>
</div>
<script>
async function loadBench() {
  const b = await fetch('/api/benchmark/three-way').then(r => r.json());
  document.getElementById('bench-banner').textContent =
    'Engine integrity: ' + b.engine_integrity + ' | n=' + (b.n_docs || 0);
  const table = document.getElementById('bench-table');
  while (table.firstChild) table.removeChild(table.firstChild);
  if (!b.metrics) return;
  const cols = ['large', 'small_cold', 'small_reviewed'];
  const metrics = ['contract_type_accuracy', 'clause_family_f1'];
  const head = document.createElement('tr');
  const th0 = document.createElement('th');
  th0.textContent = 'metric'; head.appendChild(th0);
  cols.forEach(function(c) {
    const th = document.createElement('th');
    th.textContent = c; head.appendChild(th);
  });
  table.appendChild(head);
  metrics.forEach(function(m) {
    const tr = document.createElement('tr');
    const td0 = document.createElement('td');
    td0.textContent = m; tr.appendChild(td0);
    cols.forEach(function(c) {
      const td = document.createElement('td');
      const v = (b.metrics[m] && b.metrics[m][c]) || 0;
      td.textContent = v.toFixed(2);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
}
async function cf(toggle) {
  const out = await fetch('/api/benchmark/counterfactual', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({toggle: toggle, model: 'qwen3:4b'})
  }).then(r => r.json());
  document.getElementById('cf-result').textContent = JSON.stringify(out, null, 2);
}
document.getElementById('cf-verifier').addEventListener('click', function(){ cf('verifier_off'); });
document.getElementById('cf-context').addEventListener('click', function(){ cf('context_off'); });
loadBench();
</script>
```

All dynamic data rendered via `textContent` or `createElement` — no innerHTML on user-derived content.

- [ ] **Step 5: Run tests, confirm pass**

Run: `pytest tests/test_web_benchmark.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/contract_intel_mvp/web.py tests/test_web_benchmark.py
git commit -m "feat: three-way benchmark UI with counterfactual toggles"
```

---

## Task 14: End-to-end smoke on CUAD n=70

Drive the entire agent loop on CUAD to confirm the pipeline produces a green benchmark with real Ollama models.

**Files:**
- Create: `scripts/e2e_codegames_demo.sh`

- [ ] **Step 1: Write the script**

`scripts/e2e_codegames_demo.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

PRIMARY="${PRIMARY:-qwen2.5:14b}"
SHADOW="${SHADOW:-qwen3:4b}"

contract-intel reset
contract-intel init
contract-intel interview --config config/interview.example.json
contract-intel cuad-sample --limit 70 --contains license
contract-intel ingest --input data/raw_contracts/cuad_samples
contract-intel split --review-frac 0.57 --seed 42

echo "=== Phase 1: agent run (review set) ==="
contract-intel agent run --primary-model "$PRIMARY" --shadow-model "$SHADOW"

echo "=== Phase 2: apply CUAD gold to review queue ==="
contract-intel cuad-apply-gold
contract-intel apply-review --review data/reviews/review_packet.reviewed.json

echo "=== Phase 3: agent resume (holdout) ==="
contract-intel agent resume --primary-model "$PRIMARY" --shadow-model "$SHADOW"

echo "=== Benchmark ==="
python3 -m json.tool < data/runs/benchmark.json
echo
echo "Artifacts:"
ls -la data/runs/
```

- [ ] **Step 2: Make executable, prerequisite check**

```bash
chmod +x scripts/e2e_codegames_demo.sh
curl -s http://127.0.0.1:11434/api/tags | python3 -c "
import sys,json; m={x['name'] for x in json.load(sys.stdin)['models']}
need={'qwen2.5:14b','qwen3:4b'}; missing=need-m
print('missing:', missing) if missing else print('models present')
"
```
If any model is missing: `ollama pull qwen2.5:14b` and/or `ollama pull qwen3:4b`.

Note: `cuad-sample` requires a local CUAD archive. If unavailable, substitute `edgar-sample --limit 70` and write a hand-curated `holdout_gold.json` from EDGAR exhibits, OR use the `sample_corpus/` synthetic data as a final fallback (with the understanding that a 5-doc smoke is not a real benchmark).

- [ ] **Step 3: Run the script**

```bash
bash scripts/e2e_codegames_demo.sh 2>&1 | tee /tmp/codegames_e2e.log
```
Expected wall time: ~6–12 min depending on Ollama throughput.

- [ ] **Step 4: Verify benchmark output**

Inspect `data/runs/benchmark.json`:
- `engine_integrity == "ok"`.
- `n_docs == 30` (or your holdout count).
- Three columns present in `metrics`.
- All `engine` fields in `data/runs/*.json` are `"ollama"`.

If `engine_integrity == "contaminated"`: investigate which Ollama call failed (timeout? wrong model name? OOM?). Rerun.

- [ ] **Step 5: Verify counterfactuals via UI**

```bash
contract-intel ui &
sleep 2
# Open http://127.0.0.1:8765/ manually
```
On the Benchmark tab, click both counterfactual buttons. Confirm:
- Verifier-OFF F1 is *lower* than Verifier-ON F1.
- Context-OFF F1 equals the `small_cold` column from the three-way table.

- [ ] **Step 6: Commit**

```bash
git add scripts/e2e_codegames_demo.sh
git commit -m "test: end-to-end CUAD smoke for codegames demo"
```

---

## Task 15: Demo dress rehearsal + version bump

Final pass to lock the artifacts the presentation depends on.

- [ ] **Step 1: Run the e2e script three times**, record each run's `benchmark.json` headline numbers. Variance > 0.05 F1 across runs means n is too small or the prompt is unstable — investigate before pitching.

- [ ] **Step 2: Capture screenshots** of:
  - The Run tab decision log mid-run.
  - The Review Queue with flagged docs and the reasons column.
  - The Benchmark tab with the three-column scoreboard.
  - Each counterfactual toggle's recomputed result.

Save to `docs/demo/screenshots/`.

- [ ] **Step 3: Write a one-page presenter script**

`docs/demo/presenter_script.md`:
```markdown
# Code Games Agentic Edition — Presenter Script (8 min)

1. (1m) Setup tab — show interview, splits.json, models picked.
2. (2m) Click Start Run — narrate the streaming decision log: planner → extract → verify → retry → triage. Highlight verifier rejecting fabricated spans.
3. (2m) Open Review Queue — flagged docs with reasons. Apply CUAD gold, click Resume.
4. (1m) Watch the holdout pass — engine banner stays green.
5. (1m) Benchmark tab — read the three numbers. "75% of large quality at much lower latency."
6. (1m) Counterfactual: toggle Verifier OFF → F1 drops. Toggle Context OFF → F1 drops to small_cold. "Each component earns its number."
```

- [ ] **Step 4: Bump VERSION and tag**

```bash
echo "0.2.0" > VERSION
git add VERSION docs/demo/
git commit -m "release: v0.2.0 — codegames agentic-edition MVP"
git tag -a v0.2.0 -m "codegames agentic-edition MVP"
```

- [ ] **Step 5: Final test run**

```bash
pytest -v
```
Expected: all tests pass. Any failure blocks the demo.

---

## Self-Review

**Spec coverage:**
- Held-out split → Task 1 ✓
- Engine gate → Task 2, integrated in Task 10 ✓
- Real agent loop with planner + tools + decision log → Tasks 3, 7, 8, 9 ✓
- Evidence verification with retry → Task 4 ✓
- Shadow-model parallel call → Task 5 ✓
- Triage scorer + review queue → Task 6 ✓
- Three-way benchmark → Task 10 ✓
- Counterfactual toggle (live, on stage) → Tasks 11, 13 ✓
- Streaming decision log UI → Task 12 ✓
- End-to-end CUAD smoke + dress rehearsal → Tasks 14, 15 ✓
- Pre-flight (git init, VERSION, tests harness) → Task 0 ✓

**Cuts that match the MVP scope decision:**
- No SQLite migration (state stays in JSON).
- No retrieval/chunking (full doc text in prompt).
- No LoRA fine-tuning (small_reviewed = small + taxonomy injection, not weight tuning).
- No agent-proposed taxonomy updates (human-only review).
- No cross-corpus generalization run (CUAD held-out only).

**Type/name consistency check:**
- `engine` field values: `"ollama" | "heuristic_fallback"` — used consistently across `pipeline.py`, `engine_gate.py`, `shadow.py`, three-way benchmark.
- `role` field on extraction rows: `"primary" | "shadow" | "shadow_cold"` — added in Tasks 9 and 10.
- File names: `baseline_results.json` (review-set primary), `shadow_review_results.json` (review-set shadow), `second_run_results.json` (holdout small reviewed), `second_run_primary_holdout.json` (holdout large), `shadow_holdout_cold_results.json` (holdout small cold), `holdout_gold.json` (gold for holdout). Used consistently in Tasks 9, 10, 11, 13.
- `extract_with_verification` signature: `(*, source_text, model, initial_extraction, max_retries)` returning `(extraction, reports)` — matched in Tasks 4 and 9.
- Tool names match across `wiring.py` and `planner.deterministic_next_action`: `extract_review_batch`, `extract_holdout_batch`, `extract_holdout_cold_small`, `triage`, `benchmark_three_way`, `advance_phase`, `ingest_corpus`.
- DOM rendering: every UI tab in Tasks 12 and 13 uses `textContent` or `createElement` for dynamic data — no `innerHTML` on user-derived content.

**Open assumption that needs verification on first execution of Task 9 Step 1:** `pipeline.py` may not currently expose `EXTRACTION_PROMPT` as a constant; the existing extraction prompt is built inline. Step 1 calls for promoting that inline prompt to `prompts.py:EXTRACTION_PROMPT` so both legacy CLI and the new agent path use the same template. Inspect `run_extraction` (`pipeline.py:451`) to find the current prompt string and lift it verbatim with placeholders `{title}`, `{text}`, `{interview}`, `{taxonomy}`.

**Risk register:**
- Ollama serializes requests internally — "parallel" primary+shadow is async-concurrent at the Python layer but wall-clock saving is only the smaller of the two latencies. Acceptable for MVP.
- `qwen2.5:14b` requires ~10 GB VRAM. Aurora's 4090 (24 GB) handles it; document the requirement for anyone else running this.
- Heuristic fallback still exists in `pipeline.py` for the legacy CLI path; the agent path treats it as contamination via `engine_gate`. Intentional — kept for backward-compat smoke tests.
- CUAD archive availability is the riskiest external dependency. Task 14 Step 2 documents three fallback paths.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-04-codegames-mvp.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
