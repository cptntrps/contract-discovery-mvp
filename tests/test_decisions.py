import json
from pathlib import Path
from contract_intel_mvp.agent.decisions import DecisionLog


def test_log_writes_jsonl(tmp_root: Path):
    log = DecisionLog(tmp_root, run_id="run-1")
    log.append(action="extract_doc", args={"doc_id": "doc_001"},
               result={"ok": True}, rationale="first doc in queue")
    path = tmp_root / "data" / "runs" / "agent_decisions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["action"] == "extract_doc"
    assert row["run_id"] == "run-1"
    assert row["args"] == {"doc_id": "doc_001"}
    assert row["rationale"] == "first doc in queue"
    assert "ts" in row and "decision_id" in row


def test_log_appends_multiple(tmp_root: Path):
    log = DecisionLog(tmp_root, run_id="run-2")
    for i in range(3):
        log.append(action="noop", args={"i": i}, result={}, rationale="")
    path = tmp_root / "data" / "runs" / "agent_decisions.jsonl"
    assert len(path.read_text().splitlines()) == 3


def test_log_iter_filters_by_run(tmp_root: Path):
    DecisionLog(tmp_root, run_id="r1").append(action="a", args={}, result={}, rationale="")
    DecisionLog(tmp_root, run_id="r2").append(action="b", args={}, result={}, rationale="")
    rows = list(DecisionLog.iter(tmp_root, run_id="r1"))
    assert len(rows) == 1
    assert rows[0]["action"] == "a"
