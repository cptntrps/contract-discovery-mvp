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
