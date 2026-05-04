from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class DecisionLog:
    def __init__(self, root: Path, *, run_id: str):
        self.root = root
        self.run_id = run_id
        self.path = root / "data" / "runs" / "agent_decisions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, *, action: str, args: dict[str, Any],
               result: dict[str, Any], rationale: str,
               model_call_id: str | None = None) -> dict[str, Any]:
        row = {
            "decision_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "args": args,
            "result": result,
            "rationale": rationale,
            "model_call_id": model_call_id,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    @staticmethod
    def iter(root: Path, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
        path = root / "data" / "runs" / "agent_decisions.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if run_id is None or row.get("run_id") == run_id:
                yield row
