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
        baseline_path = root / "data" / "runs" / "baseline_results.json"
        interview_path = root / "data" / "memory" / "interview.json"
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else []
        interview = json.loads(interview_path.read_text()) if interview_path.exists() else {}
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
