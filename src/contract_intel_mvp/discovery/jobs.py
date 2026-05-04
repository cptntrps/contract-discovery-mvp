"""Tiny disk-backed job runner so long discovery rounds can run async with progress polling."""
from __future__ import annotations
import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _jobs_dir(root: Path) -> Path:
    p = root / "data" / "discovery" / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _job_path(root: Path, job_id: str) -> Path:
    return _jobs_dir(root) / f"{job_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(root: Path, job_id: str, **fields: Any) -> None:
    p = _job_path(root, job_id)
    state = {}
    if p.exists():
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    state.update(fields)
    state["updated_at"] = _now()
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_status(root: Path, job_id: str) -> dict[str, Any] | None:
    p = _job_path(root, job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def start_job(root: Path, *, kind: str, target: Callable[..., Any],
              kwargs: dict[str, Any] | None = None) -> str:
    """Spawn target(progress_cb=..., **kwargs) in a daemon thread. Returns job_id."""
    kwargs = dict(kwargs or {})
    job_id = str(uuid.uuid4())
    write_status(root, job_id, job_id=job_id, kind=kind, status="running",
                 started_at=_now(), progress=0, total=0)

    def _progress_cb(done: int, total: int, note: str = "") -> None:
        write_status(root, job_id, progress=done, total=total, note=note)

    def _runner():
        try:
            result = target(progress_cb=_progress_cb, **kwargs)
            write_status(root, job_id, status="done", finished_at=_now(),
                         result=result)
        except Exception as e:
            write_status(root, job_id, status="error", finished_at=_now(),
                         error=str(e), traceback=traceback.format_exc()[-2000:])

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return job_id
