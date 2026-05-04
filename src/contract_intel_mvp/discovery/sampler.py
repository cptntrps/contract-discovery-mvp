from __future__ import annotations
import random
from pathlib import Path
from typing import Any

from contract_intel_mvp.discovery.signature import load_signature


def _strip_labeled(rows, labeled):
    return [r for r in rows if r["doc_id"] not in labeled]


def sample_for_review(root: Path, *, classifications: list[dict[str, Any]],
                      batch_size: int = 20, seed: int = 0) -> list[dict[str, Any]]:
    sig = load_signature(root)
    labeled = set(sig.confirmed_positive_doc_ids) | set(sig.confirmed_negative_doc_ids)
    rows = _strip_labeled(classifications, labeled)

    yes_rows = sorted([r for r in rows if r["verdict"] == "yes"],
                      key=lambda r: r["confidence"], reverse=True)
    no_rows = sorted([r for r in rows if r["verdict"] == "no"],
                     key=lambda r: r["confidence"])  # ascending: borderline negatives first

    quota = max(1, batch_size // 4)
    picks: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take(bucket, reason, n):
        added = 0
        for r in bucket:
            if len(picks) >= batch_size:
                return
            if r["doc_id"] in seen:
                continue
            entry = dict(r); entry["reason"] = reason
            picks.append(entry); seen.add(r["doc_id"])
            added += 1
            if added >= n:
                return

    take(yes_rows, "high_confidence_positive", quota)
    take(yes_rows[::-1], "low_confidence_positive", quota)
    take(no_rows, "borderline_negative", quota)

    rng = random.Random(seed)
    remaining = [r for r in rows if r["doc_id"] not in seen]
    rng.shuffle(remaining)
    take(remaining, "borderline_random", batch_size - len(picks))

    return picks[:batch_size]
