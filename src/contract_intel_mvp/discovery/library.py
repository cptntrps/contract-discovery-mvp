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
