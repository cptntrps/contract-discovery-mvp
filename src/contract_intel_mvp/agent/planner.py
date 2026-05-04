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
