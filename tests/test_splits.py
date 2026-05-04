import pytest
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
    with pytest.raises(ValueError, match="no documents"):
        make_splits(tmp_root, review_frac=0.6, seed=42)
