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
