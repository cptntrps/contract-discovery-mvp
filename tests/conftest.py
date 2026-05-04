import json
from pathlib import Path
import pytest


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    for sub in ("data/corpus", "data/runs", "data/reviews",
                "data/memory", "data/training", "data/raw_contracts"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def docs_jsonl(tmp_root: Path) -> Path:
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
