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
