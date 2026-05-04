from pathlib import Path
from fastapi.testclient import TestClient
from contract_intel_mvp.web import build_app


def test_decisions_endpoint_returns_jsonl_rows(tmp_root: Path):
    (tmp_root / "data" / "runs" / "agent_decisions.jsonl").write_text(
        '{"decision_id":"1","run_id":"r1","action":"a","args":{},"result":{},"rationale":"x","ts":"2026-05-04T00:00:00Z","model_call_id":null}\n'
        '{"decision_id":"2","run_id":"r1","action":"b","args":{},"result":{},"rationale":"y","ts":"2026-05-04T00:00:01Z","model_call_id":null}\n'
    )
    app = build_app(root=tmp_root)
    client = TestClient(app)
    resp = client.get("/api/decisions?run_id=r1")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 2
    assert rows[0]["action"] == "a"


def test_decisions_filters_by_run(tmp_root: Path):
    (tmp_root / "data" / "runs" / "agent_decisions.jsonl").write_text(
        '{"decision_id":"1","run_id":"r1","action":"a","args":{},"result":{},"rationale":"","ts":"t","model_call_id":null}\n'
        '{"decision_id":"2","run_id":"r2","action":"b","args":{},"result":{},"rationale":"","ts":"t","model_call_id":null}\n'
    )
    app = build_app(root=tmp_root)
    client = TestClient(app)
    rows = client.get("/api/decisions?run_id=r1").json()["rows"]
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"
