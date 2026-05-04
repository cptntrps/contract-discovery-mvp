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
