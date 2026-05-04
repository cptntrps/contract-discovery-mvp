"""Tiny local web UI for the Contract Intelligence MVP."""

from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

from .pipeline import (
    _empty_taxonomy,
    _load_documents,
    _load_json,
    accept_pending_review,
    apply_cuad_gold_review,
    apply_review,
    generate_benchmark,
    generate_demo_report,
    generate_review_packet,
    run_agent_analysis,
    run_cuad_demo,
    run_extraction,
    save_interview_payload,
)
from .agent.decisions import DecisionLog
from .benchmark.counterfactual import (
    recompute_without_verification, recompute_without_reviewed_context,
)


STATIC_DIR = Path(__file__).parent / "static"


DISCOVERY_OPENING_TEXT = (
    "Hi — I'm here to help you find a specific kind of contract in a big folder. "
    "I'll ask you a few questions, build up a description of what you're looking "
    "for, then go through your contracts and pull out the ones that match.\n\n"
    "Here's how this goes:\n"
    "  • A few short questions from me about what you're looking for — type, "
    "scope, key clauses, and what would disqualify a match.\n"
    "  • I read through your contracts and surface around 20 borderline cases for "
    "you to confirm yes or no.\n"
    "  • I learn from your corrections and look again. After 2-3 rounds I'm "
    "confident, and you get a clean list of matches plus the language I've seen "
    "across the contracts you confirmed.\n\n"
    "Ready when you are — what type of contracts are you looking for?"
)


DISCOVERY_SYSTEM_PROMPT = (
    "You are a discovery interview agent. Walk the user through a structured "
    "conversation, ONE step at a time, asking the user to confirm before moving "
    "to the next step. Output strict JSON only on every turn.\n\n"
    "USER-FACING LANGUAGE RULES (very important):\n"
    "  - Talk like a friendly assistant, not an engineer.\n"
    "  - NEVER say the words: signature, clause_type, is_must_have, ready_to_save, "
    "schema, JSON, target_class, target_description. These are INTERNAL field names.\n"
    "  - Use plain English instead: 'what we're looking for', 'key clauses', "
    "'things that should be in it', 'things that would disqualify it', 'this kind "
    "of contract'.\n"
    "  - Keep questions short. One sentence per question when possible.\n\n"
    "IMPORTANT — the user has already been greeted with an opening message that "
    "ends with 'What type of contracts are you looking for?'. The user's FIRST "
    "message in this conversation IS the answer to that question. Do not re-ask "
    "Step 1 on the first turn — go straight to Step 2.\n\n"
    "STEP ORDER (do not skip, do not combine, do not finalize early):\n"
    "  Step 1. (Already asked by the opening message.) On the user's first reply, "
    "treat it as the contract type. Save it to target_class and acknowledge "
    "briefly: 'Got it — <Target>.'\n"
    "  Step 2. Ask: 'What business units and geography should I focus on?' Wait "
    "for the user's answer. Stash the scope info into target_description.\n"
    "  Step 3. Propose the definition in plain language: 'OK — <Target> contracts "
    "are typically <one-sentence definition in plain English>. Does that sound "
    "right?' Do NOT add any clause_types yet. Set ready_to_save=false. Wait for "
    "yes / no / refine.\n"
    "  Step 4. After the user confirms the definition, propose 3-5 key clauses you "
    "would expect to find. Be helpful — many users won't know the legal-clause "
    "names, so YOU propose them based on what's typical for this contract type. "
    "Phrase it like: 'I'd typically expect these clauses in a <Target>: 1) "
    "<plain-name> — <short reason>; 2) ...; 3) ...; Do these look right? Want to "
    "add or remove any? If you have one or two real example contracts on hand, "
    "you can also paste a snippet — that helps me tune the language.' For each, "
    "internally add a clause_type with is_must_have=true, a one-line description, "
    "and one example phrasing. Wait for the user. If the user says 'I don't "
    "know', reassure them: 'No problem — I'll propose what's typical and you can "
    "tell me if anything looks off.'\n"
    "  Step 5. After the user confirms the expected clauses, ask: 'What contracts "
    "look similar but ARE NOT what you want? Anything that would disqualify a "
    "match — for example, distribution agreements, joint ventures, agency "
    "appointments? Tell me, or say \"none\".' Internally add each user-named "
    "competing type as a clause_type with is_must_have=false.\n"
    "  Step 6. Recap in plain language: 'OK — so we're looking for <Target> "
    "contracts that include <plain summary of must-haves>, and we'll skip any "
    "whose main purpose is <plain summary of must-not-haves>. Does that match "
    "what you want?' Set ready_to_save=false until the user explicitly confirms.\n"
    "  Step 7. When the user confirms the recap with 'yes', set ready_to_save=true "
    "and reply: 'Got it. Click Save signature on the right, then go to Step 3 to "
    "start finding matches.' (You may say 'Save signature' here because that's the "
    "literal label on the button the user clicks.)\n\n"
    "INTERNAL FIELD SEMANTICS (never expose to user):\n"
    "  is_must_have=true: a defining clause that MUST appear for a doc to belong "
    "to the class (e.g. license-grant, scope, royalty for a License Agreement).\n"
    "  is_must_have=false: a clause whose PRESENCE AS THE PRIMARY PURPOSE "
    "DISQUALIFIES the doc — i.e. the primary-purpose clause of a competing "
    "contract type. NOT 'clauses to avoid'. For a License Agreement, "
    "must-not-have items would be 'primary distribution appointment', 'joint "
    "venture formation', 'agency appointment', 'employment terms'.\n"
    "  Never put confidentiality, governing law, notices, dispute resolution, "
    "or indemnification into is_must_have=false — those are neutral and should "
    "be omitted from clause_types entirely.\n\n"
    "On every turn: do exactly ONE step, ask for the user's confirmation before "
    "advancing. The 'assistant' field is what the user sees — keep it warm, "
    "concise, and free of internal field names. Output strict JSON only."
)


def build_app(root: Path):
    """Return a FastAPI app exposing the agent-edition endpoints. Used by tests and as
    a programmatic API. The CLI's `run_server` keeps using the http.server-based UI
    below, which mirrors the same endpoints so the demo can run without uvicorn."""
    from fastapi import FastAPI
    app = FastAPI()
    root = root.resolve()

    @app.get("/api/decisions")
    def get_decisions(run_id: str | None = None):
        rows = list(DecisionLog.iter(root, run_id=run_id))
        return {"rows": rows}

    @app.get("/api/benchmark/three-way")
    def benchmark_three_way():
        p = root / "data" / "runs" / "benchmark.json"
        if not p.exists():
            return {"engine_integrity": "missing"}
        return json.loads(p.read_text())

    @app.post("/api/benchmark/counterfactual")
    def counterfactual(payload: dict):
        toggle = payload.get("toggle")
        if toggle == "verifier_off":
            return recompute_without_verification(root, model=payload.get("model", "qwen3:4b"))
        if toggle == "context_off":
            return recompute_without_reviewed_context(root)
        return {"error": "unknown toggle: " + str(toggle)}

    from .discovery.signature import init_signature, load_signature
    from .discovery.library import init_library_from_signature

    DISCOVERY_OPENING = DISCOVERY_OPENING_TEXT

    # Use the module-level DISCOVERY_SYSTEM_PROMPT defined above.

    @app.post("/api/interview/discovery-chat")
    def discovery_chat(payload: dict):
        sig_in = payload.get("signature") or {}
        message = str(payload.get("message", "")).strip()
        save = bool(payload.get("save"))
        initial = bool(payload.get("initial"))

        if initial:
            return {"signature": sig_in, "assistant": DISCOVERY_OPENING,
                    "engine": "scripted_opening"}

        if save and sig_in.get("target_class") and sig_in.get("target_description"):
            init_signature(root, interview=sig_in)
            init_library_from_signature(root)
            return {"signature": sig_in, "saved": True,
                    "assistant": "Signature saved. Library seeded from your examples. Embed the corpus and run round 0.",
                    "engine": "local_save"}

        if _openai_interview_enabled():
            prompt = {
                "task": "Continue a discovery interview. Refine the structured signature.",
                "current_signature": sig_in,
                "user_message": message,
                "schema": {
                    "assistant": "string",
                    "signature_updates": {
                        "target_class": "string",
                        "target_description": "string",
                        "clause_types": [{
                            "type": "string", "description": "string",
                            "is_must_have": "boolean",
                            "seed_variations": ["string"],
                        }],
                    },
                    "ready_to_save": "boolean",
                },
            }
            body = {
                "model": _openai_model(),
                "messages": [
                    {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt, indent=2)},
                ],
                "max_completion_tokens": 1500,
                "response_format": {"type": "json_object"},
            }
            request = urllib.request.Request(
                f"{_openai_base_url()}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','').strip()}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                content = resp["choices"][0]["message"]["content"]
                parsed = _extract_json_object(content)
                if isinstance(parsed, dict):
                    updates = parsed.get("signature_updates") or {}
                    merged = dict(sig_in)
                    for k, v in updates.items():
                        if v: merged[k] = v
                    return {"signature": merged,
                            "assistant": str(parsed.get("assistant") or ""),
                            "ready_to_save": bool(parsed.get("ready_to_save")),
                            "engine": "openai_api", "model": _openai_model()}
            except Exception:
                pass

        return {"signature": sig_in,
                "assistant": "Tell me one specific clause type that this contract type always contains, and give me a one-sentence example of how it usually reads.",
                "engine": "local_discovery_fallback"}

    from .discovery.embeddings import embed_corpus
    from .discovery.loop import run_round, submit_labels, finalize
    from .discovery.convergence import should_stop
    from .discovery.library import load_library

    @app.get("/api/discovery/state")
    def discovery_state():
        sig_path = root / "data" / "discovery" / "signature.json"
        emb_path = root / "data" / "discovery" / "embeddings.jsonl"
        rounds_path = root / "data" / "discovery" / "rounds.json"
        final_path = root / "data" / "discovery" / "final.json"
        target_class = None
        library_size = 0
        if sig_path.exists():
            target_class = load_signature(root).target_class
            try:
                lib = load_library(root)
                library_size = len(lib["clause_types"]) + sum(
                    len(ct["variations"]) for ct in lib["clause_types"])
            except Exception:
                library_size = 0
        return {
            "target_class": target_class,
            "embedded_count": sum(1 for l in emb_path.read_text().splitlines() if l.strip())
                              if emb_path.exists() else 0,
            "rounds": json.loads(rounds_path.read_text()).get("rounds", [])
                      if rounds_path.exists() else [],
            "finalized": final_path.exists(),
            "should_stop": should_stop(root),
            "library_size": library_size,
        }

    @app.get("/api/discovery/library")
    def discovery_library():
        try:
            return load_library(root)
        except ValueError:
            return {"target_class": None, "clause_types": []}

    @app.post("/api/discovery/embed")
    def discovery_embed(payload: dict):
        return embed_corpus(root, model=payload.get("model", "nomic-embed-text"))

    @app.post("/api/discovery/run-round")
    def discovery_run_round(payload: dict):
        return run_round(root,
                         classifier_model=payload.get("classifier_model", "qwen3:4b"),
                         top_k=int(payload.get("top_k", 200)),
                         batch_size=int(payload.get("batch_size", 20)),
                         round_index=int(payload.get("round_index", 0)),
                         seed=int(payload.get("seed", 0)))

    @app.post("/api/discovery/submit-labels")
    def discovery_submit_labels(payload: dict):
        return submit_labels(root,
                             round_index=int(payload.get("round_index", 0)),
                             labels=payload.get("labels", []))

    @app.post("/api/discovery/finalize")
    def discovery_finalize(payload: dict):
        return finalize(root, round_index=int(payload.get("round_index", 0)),
                        borderline_threshold=float(payload.get("borderline_threshold", 0.7)))

    return app


def run_server(root: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    root = root.resolve()

    from .discovery.signature import init_signature, load_signature
    from .discovery.library import init_library_from_signature, load_library
    from .discovery.embeddings import embed_corpus
    from .discovery.loop import run_round, submit_labels, finalize
    from .discovery.convergence import should_stop

    def _discovery_state() -> dict[str, object]:
        sig_path = root / "data" / "discovery" / "signature.json"
        emb_path = root / "data" / "discovery" / "embeddings.jsonl"
        rounds_path = root / "data" / "discovery" / "rounds.json"
        final_path = root / "data" / "discovery" / "final.json"
        target_class = None
        library_size = 0
        if sig_path.exists():
            target_class = load_signature(root).target_class
            try:
                lib = load_library(root)
                library_size = len(lib["clause_types"]) + sum(
                    len(ct["variations"]) for ct in lib["clause_types"])
            except Exception:
                library_size = 0
        return {
            "target_class": target_class,
            "embedded_count": sum(1 for l in emb_path.read_text().splitlines() if l.strip())
                              if emb_path.exists() else 0,
            "rounds": json.loads(rounds_path.read_text()).get("rounds", [])
                      if rounds_path.exists() else [],
            "finalized": final_path.exists(),
            "should_stop": should_stop(root),
            "library_size": library_size,
        }

    def _discovery_library_payload() -> dict[str, object]:
        try:
            return load_library(root)
        except ValueError:
            return {"target_class": None, "clause_types": []}

    def _discovery_chat_handler(payload: dict) -> dict[str, object]:
        sig_in = payload.get("signature") or {}
        message = str(payload.get("message", "")).strip()
        save = bool(payload.get("save"))
        initial = bool(payload.get("initial"))

        DISCOVERY_OPENING = DISCOVERY_OPENING_TEXT
        # Use the module-level DISCOVERY_SYSTEM_PROMPT defined at top of file.

        if initial:
            return {"signature": sig_in, "assistant": DISCOVERY_OPENING,
                    "engine": "scripted_opening"}

        if save and sig_in.get("target_class") and sig_in.get("target_description"):
            init_signature(root, interview=sig_in)
            init_library_from_signature(root)
            return {"signature": sig_in, "saved": True,
                    "assistant": "Signature saved. Library seeded from your examples. Embed the corpus and run round 0.",
                    "engine": "local_save"}

        if _openai_interview_enabled():
            prompt = {
                "task": "Continue a discovery interview. Refine the structured signature.",
                "current_signature": sig_in,
                "user_message": message,
                "schema": {
                    "assistant": "string",
                    "signature_updates": {
                        "target_class": "string",
                        "target_description": "string",
                        "clause_types": [{
                            "type": "string", "description": "string",
                            "is_must_have": "boolean",
                            "seed_variations": ["string"],
                        }],
                    },
                    "ready_to_save": "boolean",
                },
            }
            body = {
                "model": _openai_model(),
                "messages": [
                    {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt, indent=2)},
                ],
                "max_completion_tokens": 1500,
                "response_format": {"type": "json_object"},
            }
            request = urllib.request.Request(
                f"{_openai_base_url()}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','').strip()}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                content = resp["choices"][0]["message"]["content"]
                parsed = _extract_json_object(content)
                if isinstance(parsed, dict):
                    updates = parsed.get("signature_updates") or {}
                    merged = dict(sig_in)
                    for k, v in updates.items():
                        if v: merged[k] = v
                    return {"signature": merged,
                            "assistant": str(parsed.get("assistant") or ""),
                            "ready_to_save": bool(parsed.get("ready_to_save")),
                            "engine": "openai_api", "model": _openai_model()}
            except Exception:
                pass

        return {"signature": sig_in,
                "assistant": "Tell me one specific clause type that this contract type always contains, and give me a one-sentence example of how it usually reads.",
                "engine": "local_discovery_fallback"}

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/file":
                rel = parse_qs(parsed.query).get("path", [""])[0]
                path = _safe_artifact_path(root, rel)
                if not path or not path.exists():
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", _artifact_content_type(path))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
            elif parsed.path in {"/", "/app.js", "/styles.css", "/api/state"}:
                self.send_response(200)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            else:
                self.send_error(404)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif parsed.path == "/app.js":
                self._send_file(STATIC_DIR / "app.js", "text/javascript; charset=utf-8")
            elif parsed.path == "/styles.css":
                self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            elif parsed.path == "/api/state":
                self._send_json(_ui_state(root))
            elif parsed.path == "/api/file":
                self._send_artifact(root, parsed.query)
            elif parsed.path == "/agent":
                self._send_file(STATIC_DIR / "agent.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/decisions":
                run_id = parse_qs(parsed.query).get("run_id", [None])[0]
                rows = list(DecisionLog.iter(root, run_id=run_id))
                self._send_json({"rows": rows})
            elif parsed.path == "/api/benchmark/three-way":
                p = root / "data" / "runs" / "benchmark.json"
                if not p.exists():
                    self._send_json({"engine_integrity": "missing"})
                else:
                    self._send_json(json.loads(p.read_text()))
            elif parsed.path == "/api/discovery/state":
                self._send_json(_discovery_state())
            elif parsed.path == "/api/discovery/library":
                self._send_json(_discovery_library_payload())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/review/save":
                    payload = self._read_json()
                    self._send_json(_save_review(root, payload))
                elif parsed.path == "/api/interview/chat":
                    payload = self._read_json()
                    self._send_json(_chat_interview(root, payload))
                elif parsed.path == "/api/interview/save":
                    payload = self._read_json()
                    self._send_json(_save_interview(root, payload))
                elif parsed.path == "/api/actions/review-packet":
                    self._send_json({"review_packet": str(generate_review_packet(root))})
                elif parsed.path == "/api/actions/cuad-gold":
                    self._send_json(apply_cuad_gold_review(root))
                elif parsed.path == "/api/actions/demo-accept-review":
                    self._send_json(accept_pending_review(root, note="Accepted from UI demo action."))
                elif parsed.path == "/api/actions/apply-review":
                    self._send_json(apply_review(root, root / "data" / "reviews" / "review_packet.reviewed.json"))
                elif parsed.path == "/api/actions/baseline":
                    self._send_json(run_extraction(root, model="qwen3:4b", mode="baseline"))
                elif parsed.path == "/api/actions/demo-cuad":
                    self._send_json(run_cuad_demo(root, model="qwen3:4b", limit=4, contains="license"))
                elif parsed.path == "/api/actions/agent-analyze":
                    self._send_json(run_agent_analysis(root, model="qwen3:4b", run="baseline"))
                elif parsed.path == "/api/actions/second-run":
                    self._send_json(run_extraction(root, model="qwen3:4b", mode="second"))
                elif parsed.path == "/api/actions/benchmark":
                    self._send_json(generate_benchmark(root))
                elif parsed.path == "/api/actions/demo-report":
                    self._send_json(generate_demo_report(root))
                elif parsed.path == "/api/benchmark/counterfactual":
                    payload = self._read_json()
                    toggle = payload.get("toggle")
                    if toggle == "verifier_off":
                        self._send_json(recompute_without_verification(root, model=payload.get("model", "qwen3:4b")))
                    elif toggle == "context_off":
                        self._send_json(recompute_without_reviewed_context(root))
                    else:
                        self._send_json({"error": "unknown toggle: " + str(toggle)}, status=400)
                elif parsed.path == "/api/upload":
                    import base64, time, re
                    payload = self._read_json()
                    files = payload.get("files", [])
                    if not files:
                        self._send_json({"error": "no files"}, status=400); return
                    upload_dir = root / "data" / "raw_contracts" / f"upload_{int(time.time())}"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    written = 0
                    for f in files:
                        name = f.get("filename", "")
                        b64 = f.get("content_b64", "")
                        if not name or not b64:
                            continue
                        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:200]
                        try:
                            (upload_dir / safe).write_bytes(base64.b64decode(b64))
                            written += 1
                        except Exception:
                            continue
                    from .pipeline import ingest_folder
                    ingested = ingest_folder(upload_dir, root)
                    self._send_json({"received": written, "ingested": ingested,
                                     "upload_dir": str(upload_dir.relative_to(root))})
                elif parsed.path == "/api/split":
                    payload = self._read_json()
                    from .splits import make_splits
                    out = make_splits(root,
                                      review_frac=float(payload.get("review_frac", 0.6)),
                                      seed=int(payload.get("seed", 42)))
                    self._send_json(out)
                elif parsed.path == "/api/agent/run" or parsed.path == "/api/agent/resume":
                    payload = self._read_json()
                    from .agent.planner import run_agent
                    from .agent.wiring import build_registry
                    import threading
                    primary = str(payload.get("primary_model", "qwen2.5:14b"))
                    shadow = str(payload.get("shadow_model", "qwen3:4b"))
                    registry = build_registry()
                    state = {"run_id": None}
                    def _go():
                        state["run_id"] = run_agent(root=root, registry=registry,
                                                    primary_model=primary, shadow_model=shadow)
                    t = threading.Thread(target=_go, daemon=True); t.start()
                    self._send_json({"started": True, "primary_model": primary, "shadow_model": shadow,
                                     "note": "agent running in background; tail /api/decisions for progress"})
                elif parsed.path == "/api/cuad-apply-holdout-gold":
                    from .pipeline import cuad_apply_holdout_gold
                    self._send_json(cuad_apply_holdout_gold(root))
                elif parsed.path == "/api/interview/discovery-chat":
                    payload = self._read_json()
                    self._send_json(_discovery_chat_handler(payload))
                elif parsed.path == "/api/discovery/embed":
                    payload = self._read_json()
                    self._send_json(embed_corpus(root, model=payload.get("model", "nomic-embed-text")))
                elif parsed.path == "/api/discovery/run-round":
                    payload = self._read_json()
                    self._send_json(run_round(
                        root,
                        classifier_model=payload.get("classifier_model", "qwen3:4b"),
                        top_k=int(payload.get("top_k", 200)),
                        batch_size=int(payload.get("batch_size", 20)),
                        round_index=int(payload.get("round_index", 0)),
                        seed=int(payload.get("seed", 0)),
                    ))
                elif parsed.path == "/api/discovery/submit-labels":
                    payload = self._read_json()
                    self._send_json(submit_labels(
                        root,
                        round_index=int(payload.get("round_index", 0)),
                        labels=payload.get("labels", []),
                    ))
                elif parsed.path == "/api/discovery/finalize":
                    payload = self._read_json()
                    self._send_json(finalize(
                        root,
                        round_index=int(payload.get("round_index", 0)),
                        borderline_threshold=float(payload.get("borderline_threshold", 0.7)),
                    ))
                else:
                    self.send_error(404)
            except Exception as exc:  # Keep UI errors visible instead of crashing server.
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[contract-intel-ui] {self.address_string()} - {fmt % args}")

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: object, *, status: int = 200) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_artifact(self, root: Path, query: str) -> None:
            rel = parse_qs(query).get("path", [""])[0]
            path = _safe_artifact_path(root, rel)
            if not path or not path.exists():
                self.send_error(404)
                return
            content_type = _artifact_content_type(path)
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Contract Intelligence UI: http://{host}:{port}")
    print(f"Project root: {root}")
    server.serve_forever()


def _ui_state(root: Path) -> dict[str, object]:
    docs = _load_documents(root)
    baseline = _load_json(root / "data" / "runs" / "baseline_results.json", [])
    second = _load_json(root / "data" / "runs" / "second_run_results.json", [])
    benchmark = _load_json(root / "data" / "runs" / "benchmark.json", {})
    ingest_manifest = _load_json(root / "data" / "corpus" / "ingest_manifest.json", {})
    agent_analysis = _load_json(root / "data" / "runs" / "agent_analysis.json", [])
    review_pending = _load_json(root / "data" / "reviews" / "review_packet.pending.json", {"items": []})
    review_done = _load_json(root / "data" / "reviews" / "review_packet.reviewed.json", {"items": []})
    taxonomy = _load_json(root / "data" / "memory" / "taxonomy.json", _empty_taxonomy())
    interview = _load_json(root / "data" / "memory" / "interview.json", {})
    report = _load_json(root / "data" / "runs" / "demo_report.json", {})
    report_markdown = _read_text(root / "data" / "runs" / "demo_report.md")
    roadmap_markdown = _read_text(root / "ROADMAP.md")
    run_manifest = _load_json(root / "data" / "runs" / "demo_run_manifest.json", {})
    gold = _load_json(root / "data" / "gold" / "cuad_review_labels.json", {"items": []})
    training_pairs = _read_jsonl(root / "data" / "training" / "training_pairs.jsonl")

    baseline_by_doc = {item.get("doc_id"): item for item in baseline}
    second_by_doc = {item.get("doc_id"): item for item in second}
    analysis_by_doc = {item.get("doc_id"): item for item in agent_analysis}
    gold_by_doc = {item.get("doc_id"): item for item in gold.get("items", [])}
    reviewed_by_doc = {item.get("doc_id"): item for item in review_done.get("items", [])}
    benchmark_by_doc = {item.get("doc_id"): item for item in benchmark.get("rows", [])}

    rows = []
    provenance = []
    for doc in docs:
        baseline_item = baseline_by_doc.get(doc.doc_id, {})
        second_item = second_by_doc.get(doc.doc_id, {})
        gold_item = gold_by_doc.get(doc.doc_id, {})
        review_item = reviewed_by_doc.get(doc.doc_id, {})
        analysis_item = analysis_by_doc.get(doc.doc_id, {})
        benchmark_item = benchmark_by_doc.get(doc.doc_id, {})
        rows.append({
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source_path": doc.source_path,
            "baseline_type": baseline_item.get("contract_type"),
            "baseline_confidence": baseline_item.get("confidence"),
            "baseline_engine": baseline_item.get("engine"),
            "second_type": second_item.get("contract_type"),
            "second_confidence": second_item.get("confidence"),
            "second_engine": second_item.get("engine"),
            "gold_type": gold_item.get("contract_type") or review_item.get("accepted_contract_type"),
            "review_status": review_item.get("status"),
            "coversheet": review_item.get("accepted_coversheet") or baseline_item.get("coversheet", {}),
            "key_clauses": review_item.get("accepted_key_clauses") or baseline_item.get("key_clauses", []),
            "agent_analysis": analysis_item,
            "text_preview": doc.text[:1200],
        })
        provenance.append({
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source_path": doc.source_path,
            "baseline": {
                "contract_type": baseline_item.get("contract_type"),
                "engine": baseline_item.get("engine"),
                "model": baseline_item.get("model"),
                "confidence": baseline_item.get("confidence"),
                "evidence_count": len(baseline_item.get("evidence", [])),
            },
            "agent": {
                "priority": analysis_item.get("review_priority"),
                "engine": analysis_item.get("engine"),
                "model": analysis_item.get("model"),
                "challenge_summary": analysis_item.get("challenge_summary"),
            },
            "review": {
                "status": review_item.get("status"),
                "authority": review_item.get("reviewer_authority"),
                "accepted_contract_type": review_item.get("accepted_contract_type") or gold_item.get("contract_type"),
                "evidence_sufficient": review_item.get("evidence_sufficient"),
            },
            "second_run": {
                "contract_type": second_item.get("contract_type"),
                "engine": second_item.get("engine"),
                "model": second_item.get("model"),
                "confidence": second_item.get("confidence"),
            },
            "benchmark": {
                "baseline_correct": benchmark_item.get("baseline_correct"),
                "second_run_correct": benchmark_item.get("second_run_correct"),
            },
        })

    return {
        "summary": {
            "documents": len(docs),
            "reviewed": len([item for item in review_done.get("items", []) if item.get("status") in {"accepted", "edited", "rejected"}]),
            "training_pairs": len(training_pairs),
            "baseline_accuracy": benchmark.get("baseline_contract_type_accuracy"),
            "second_accuracy": benchmark.get("second_run_contract_type_accuracy"),
            "baseline_engines": sorted({item.get("engine", "unknown") for item in baseline}),
            "second_engines": sorted({item.get("engine", "unknown") for item in second}),
            "agent_high_priority": len([item for item in agent_analysis if item.get("review_priority") == "high"]),
            "agent_medium_priority": len([item for item in agent_analysis if item.get("review_priority") == "medium"]),
            **{k: v for k, v in report.items() if k not in {"report_json", "report_markdown"}},
        },
        "documents": rows,
        "ingest_manifest": ingest_manifest,
        "benchmark": benchmark,
        "pending_review": review_pending,
        "reviewed": review_done,
        "agent_analysis": agent_analysis,
        "taxonomy": taxonomy,
        "interview": interview,
        "livingos_api": _livingos_api_status(),
        "training_pairs": training_pairs[:50],
        "report_markdown": report_markdown,
        "roadmap_markdown": roadmap_markdown,
        "output_files": _output_files(root),
        "run_manifest": run_manifest,
        "provenance": provenance,
    }


def _save_review(root: Path, payload: dict[str, object]) -> dict[str, object]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Expected JSON body with an items list.")
    pending = _load_json(root / "data" / "reviews" / "review_packet.pending.json", {"instructions": "", "items": []})
    pending_by_doc = {item.get("doc_id"): item for item in pending.get("items", [])}
    reviewed_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        merged = dict(pending_by_doc.get(doc_id, {}))
        merged.update(item)
        merged.setdefault("accepted_coversheet", merged.get("coversheet", {}))
        merged.setdefault("accepted_key_clauses", merged.get("key_clauses", []))
        merged.setdefault("reviewer_authority", "business_sme_confirmed")
        merged.setdefault("evidence_sufficient", True)
        reviewed_items.append(merged)

    packet = {
        "instructions": pending.get("instructions", "Reviewed in local UI."),
        "items": reviewed_items,
    }
    out = root / "data" / "reviews" / "review_packet.reviewed.json"
    out.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return {"reviewed_packet": str(out), "items": len(reviewed_items)}


def _save_interview(root: Path, payload: dict[str, object]) -> dict[str, object]:
    interview = payload.get("interview")
    if not isinstance(interview, dict):
        raise ValueError("Expected JSON body with an interview object.")
    normalized = _normalize_interview(interview)
    if not normalized["goal"]:
        raise ValueError("Interview goal is required.")
    if not normalized["expected_contract_types"]:
        raise ValueError("At least one expected contract type is required.")
    if not normalized["key_clause_families"]:
        raise ValueError("At least one key clause family is required.")
    result = save_interview_payload(root, normalized)
    result["interview"] = normalized
    return result


def _chat_interview(root: Path, payload: dict[str, object]) -> dict[str, object]:
    interview = payload.get("interview", {})
    if not isinstance(interview, dict):
        interview = {}
    message = str(payload.get("message", "")).strip()
    field = str(payload.get("field", "")).strip()
    if not message:
        raise ValueError("Interview chat message is required.")

    normalized = _normalize_interview(interview)
    if _interview_complete(normalized) and not _looks_like_memory_update(message):
        # Try OpenAI first so casual follow-ups get a real conversational reply.
        api_result = _call_openai_interview(
            interview=normalized, field=field or "review_priorities",
            message=message, next_field="", ready=True,
        )
        if api_result:
            normalized = _merge_chat_updates(normalized, api_result.get("updates"))
            saved = save_interview_payload(root, normalized)
            return {
                "interview": normalized,
                "field": "",
                "next_field": "",
                "ready": True,
                "saved": saved,
                "assistant": str(api_result.get("assistant") or _post_completion_reply(normalized, message)),
                "engine": "openai_api",
                "model": _openai_model(),
                "codex_error": None,
            }
        response = _post_completion_reply(normalized, message)
        saved = save_interview_payload(root, normalized)
        return {
            "interview": normalized,
            "field": "",
            "next_field": "",
            "ready": True,
            "saved": saved,
            "assistant": response,
            "engine": "local_interview_fallback",
            "model": None,
            "codex_error": None,
        }

    if _interview_complete(normalized):
        field = _infer_update_field(message) or field
    field = field if field in _INTERVIEW_FIELDS else _next_interview_field(normalized)
    if not field:
        field = "review_priorities"
    normalized[field] = _parse_interview_answer(field, message, normalized)
    next_field = _next_interview_field(normalized)
    ready = _interview_ready(normalized)
    response = _interview_reply(normalized, next_field)
    engine = "local_interview_fallback"
    model = None
    codex_error = None
    api_result = _call_openai_interview(
        interview=normalized, field=field, message=message,
        next_field=next_field, ready=ready,
    )
    if api_result:
        normalized = _merge_chat_updates(normalized, api_result.get("updates"))
        next_field = str(api_result.get("next_field") or _next_interview_field(normalized))
        ready = _interview_ready(normalized)
        response = str(api_result.get("assistant") or _interview_reply(normalized, next_field))
        engine = "openai_api"
        model = _openai_model()
    else:
        if _openai_interview_enabled():
            codex_error = "OpenAI API unavailable or returned invalid JSON; falling through."
        codex_result = _call_livingos_codex_interview(
            interview=normalized, field=field, message=message,
            next_field=next_field, ready=ready,
        )
        if codex_result:
            normalized = _merge_chat_updates(normalized, codex_result.get("updates"))
            next_field = str(codex_result.get("next_field") or _next_interview_field(normalized))
            ready = _interview_ready(normalized)
            response = str(codex_result.get("assistant") or _interview_reply(normalized, next_field))
            engine = "livingos_api_codex"
            model = _livingos_api_model()
        elif _livingos_api_configured():
            codex_error = (codex_error or "") + " LivingOS API Codex also unavailable; local interview fallback used."
    saved = None
    if ready:
        saved = save_interview_payload(root, normalized)
    return {
        "interview": normalized,
        "field": field,
        "next_field": next_field,
        "ready": ready,
        "saved": saved,
        "assistant": response,
        "engine": engine,
        "model": model,
        "codex_error": codex_error,
    }


def _normalize_interview(interview: dict[object, object]) -> dict[str, object]:
    normalized = {
        "goal": str(interview.get("goal", "")).strip(),
        "business_unit": str(interview.get("business_unit", "")).strip(),
        "region": str(interview.get("region", "")).strip(),
        "expected_contract_types": _string_list(interview.get("expected_contract_types")),
        "contract_type_aliases": _alias_map(interview.get("contract_type_aliases")),
        "key_clause_families": _string_list(interview.get("key_clause_families")),
        "not_expected": _string_list(interview.get("not_expected")),
        "review_priorities": _string_list(interview.get("review_priorities")),
    }
    return normalized


_INTERVIEW_FIELDS = {
    "goal": "What business decision should this contract corpus support?",
    "business_unit": "Which business unit or operating group owns this review?",
    "region": "Which region or governing jurisdiction should bias the taxonomy?",
    "expected_contract_types": "Which contract types should the model expect first?",
    "key_clause_families": "Which clause families matter most for review?",
    "not_expected": "What document types should be treated as out of scope?",
    "review_priorities": "Which gaps or risks should create high-priority review queues?",
}


def _next_interview_field(interview: dict[str, object]) -> str:
    for field in _INTERVIEW_FIELDS:
        value = interview.get(field)
        if isinstance(value, list):
            if not value:
                return field
        elif not str(value or "").strip():
            return field
    return ""


def _interview_ready(interview: dict[str, object]) -> bool:
    return bool(
        interview.get("goal")
        and interview.get("business_unit")
        and interview.get("region")
        and interview.get("expected_contract_types")
        and interview.get("key_clause_families")
    )


def _interview_complete(interview: dict[str, object]) -> bool:
    return _interview_ready(interview) and not _next_interview_field(interview)


def _looks_like_memory_update(message: str) -> bool:
    lowered = message.strip().lower()
    update_words = {
        "add",
        "change",
        "update",
        "remove",
        "replace",
        "set",
        "include",
        "exclude",
        "also",
        "instead",
    }
    field_words = {
        "goal",
        "business unit",
        "region",
        "contract type",
        "clause",
        "alias",
        "priority",
        "priorities",
        "out of scope",
        "not expected",
        "exclude",
    }
    return any(word in lowered for word in update_words) and any(word in lowered for word in field_words)


def _infer_update_field(message: str) -> str:
    lowered = message.lower()
    if "contract type" in lowered or "agreement type" in lowered:
        return "expected_contract_types"
    if "clause" in lowered or "provision" in lowered:
        return "key_clause_families"
    if "out of scope" in lowered or "not expected" in lowered or "exclude" in lowered:
        return "not_expected"
    if "priority" in lowered or "priorities" in lowered or "review queue" in lowered:
        return "review_priorities"
    if "business unit" in lowered:
        return "business_unit"
    if "region" in lowered or "jurisdiction" in lowered:
        return "region"
    if "goal" in lowered:
        return "goal"
    return ""


def _post_completion_reply(interview: dict[str, object], message: str) -> str:
    lowered = message.strip().lower()
    type_count = len(interview.get("expected_contract_types", []))
    clause_count = len(interview.get("key_clause_families", []))
    if lowered in {"hi", "hello", "hey", "yo", "hi there", "hello there"}:
        return "Hi. The setup interview is complete, and I can still help you adjust it or run the workflow."
    if "what can you do" in lowered or "help" in lowered:
        return (
            "I can update the interview memory, add expected contract types or clause families, explain the current "
            "38-contract corpus, point you to review queues, or help rerun the baseline, reviewed-context run, and benchmark."
        )
    if "status" in lowered or "ready" in lowered or "complete" in lowered:
        return f"Interview memory is complete with {type_count} expected contract types and {clause_count} clause families. The next useful step is reviewing corpus intelligence or rerunning the benchmark after changes."
    return (
        "Interview memory is already complete. Tell me what to change, for example: "
        "'add Data Processing Addendum as an expected contract type' or 'add indemnity as a key clause family.'"
    )


def _parse_interview_answer(field: str, message: str, interview: dict[str, object]) -> object:
    if field in {"expected_contract_types", "key_clause_families", "not_expected", "review_priorities"}:
        existing = interview.get(field)
        values = _string_list(existing) + _string_list(_clean_update_message(field, message))
        return list(dict.fromkeys(values))
    return message


def _clean_update_message(field: str, message: str) -> str:
    cleaned = re.sub(r"^\s*(add|include|set|change|update|also add|please add)\s+", "", message, flags=re.I)
    suffixes = {
        "expected_contract_types": r"\s+(as|to|for)\s+(an?\s+)?(expected\s+)?(contract|agreement)\s+type(s)?\.?$",
        "key_clause_families": r"\s+(as|to|for)\s+(a\s+)?(key\s+)?clause(\s+family)?\.?$",
        "not_expected": r"\s+(as|to|for)\s+(out\s+of\s+scope|not\s+expected)\.?$",
        "review_priorities": r"\s+(as|to|for)\s+(a\s+)?(review\s+)?priorit(y|ies|ized\s+queue)\.?$",
    }
    pattern = suffixes.get(field)
    if pattern:
        cleaned = re.sub(pattern, "", cleaned, flags=re.I)
    return cleaned.strip()


def _interview_reply(interview: dict[str, object], next_field: str) -> str:
    if next_field:
        return _INTERVIEW_FIELDS[next_field]
    type_count = len(interview.get("expected_contract_types", []))
    clause_count = len(interview.get("key_clause_families", []))
    return f"Interview memory is ready. I saved {type_count} expected contract types and {clause_count} clause families into taxonomy memory."


def _merge_chat_updates(interview: dict[str, object], updates: object) -> dict[str, object]:
    if not isinstance(updates, dict):
        return interview
    merged = dict(interview)
    for field in [*_INTERVIEW_FIELDS.keys(), "contract_type_aliases"]:
        if field not in updates:
            continue
        if field == "contract_type_aliases":
            aliases = _alias_map(updates.get(field))
            if aliases:
                current = dict(merged.get("contract_type_aliases") or {})
                current.update(aliases)
                merged[field] = current
        elif field in {"expected_contract_types", "key_clause_families", "not_expected", "review_priorities"}:
            values = _string_list(updates.get(field))
            if values:
                merged[field] = list(dict.fromkeys(_string_list(merged.get(field)) + values))
        else:
            value = str(updates.get(field) or "").strip()
            if value:
                merged[field] = value
    return _normalize_interview(merged)


def _livingos_api_model() -> str:
    return os.getenv("LIVINGOS_API_MODEL", "bare-sonnet-codex").strip() or "bare-sonnet-codex"


def _livingos_api_base_url() -> str:
    return os.getenv("LIVINGOS_API_BASE_URL", "http://127.0.0.1:8510/v1").rstrip("/")


def _livingos_api_configured() -> bool:
    return bool(os.getenv("LIVINGOS_API_KEY", "").strip())


def _livingos_api_enabled() -> bool:
    value = os.getenv("LIVINGOS_API_INTERVIEW", "").strip().lower()
    return _livingos_api_configured() and value in {"1", "true", "yes", "on"}


def _livingos_api_status() -> dict[str, object]:
    return {
        "configured": _livingos_api_configured(),
        "interview_enabled": _livingos_api_enabled(),
        "base_url": _livingos_api_base_url(),
        "model": _livingos_api_model() if _livingos_api_configured() else None,
        "openai_configured": _openai_configured(),
        "openai_interview_enabled": _openai_interview_enabled(),
        "openai_model": _openai_model() if _openai_configured() else None,
    }


def _openai_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _openai_interview_enabled() -> bool:
    value = os.getenv("OPENAI_INTERVIEW", "").strip().lower()
    return _openai_configured() and value in {"1", "true", "yes", "on"}


def _openai_model() -> str:
    return os.getenv("OPENAI_API_MODEL", "gpt-4o-mini").strip()


def _openai_base_url() -> str:
    return os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _call_openai_interview(
    *,
    interview: dict[str, object],
    field: str,
    message: str,
    next_field: str,
    ready: bool,
) -> dict[str, object] | None:
    if not _openai_interview_enabled():
        return None
    prompt = {
        "task": "Continue a business-level contract corpus setup interview.",
        "rules": [
            "Return JSON only.",
            "Do not ask the user to validate raw entities.",
            "Ask exactly one concise next question unless ready is true and no next_field remains.",
            "Preserve existing memory. Only include updates that are supported by the user's latest answer.",
        ],
        "schema": {
            "assistant": "string",
            "updates": {
                "goal": "string",
                "business_unit": "string",
                "region": "string",
                "expected_contract_types": ["string"],
                "contract_type_aliases": {"Canonical Type": ["Alias"]},
                "key_clause_families": ["string"],
                "not_expected": ["string"],
                "review_priorities": ["string"],
            },
            "next_field": "string",
            "ready": "boolean",
        },
        "current_interview_memory": interview,
        "latest_answer": {"field": field, "message": message},
        "local_next_field": next_field,
        "local_ready": ready,
        "questions": _INTERVIEW_FIELDS,
    }
    body = {
        "model": _openai_model(),
        "messages": [
            {
                "role": "system",
                "content": "You are an interview agent for a contract intelligence MVP. Output strict JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt, indent=2)},
        ],
        "max_completion_tokens": 1500,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{_openai_base_url()}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '').strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        if isinstance(parsed, dict):
            return parsed
        return {
            "assistant": content.strip(),
            "updates": {},
            "next_field": next_field,
            "ready": ready,
        }
    except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError):
        return None


def _call_livingos_codex_interview(
    *,
    interview: dict[str, object],
    field: str,
    message: str,
    next_field: str,
    ready: bool,
) -> dict[str, object] | None:
    if not _livingos_api_enabled():
        return None
    prompt = {
        "task": "Continue a business-level contract corpus setup interview.",
        "rules": [
            "Return JSON only.",
            "Do not ask the user to validate raw entities.",
            "Ask exactly one concise next question unless ready is true and no next_field remains.",
            "Preserve existing memory. Only include updates that are supported by the user's latest answer.",
        ],
        "schema": {
            "assistant": "string",
            "updates": {
                "goal": "string",
                "business_unit": "string",
                "region": "string",
                "expected_contract_types": ["string"],
                "contract_type_aliases": {"Canonical Type": ["Alias"]},
                "key_clause_families": ["string"],
                "not_expected": ["string"],
                "review_priorities": ["string"],
            },
            "next_field": "string",
            "ready": "boolean",
        },
        "current_interview_memory": interview,
        "latest_answer": {"field": field, "message": message},
        "local_next_field": next_field,
        "local_ready": ready,
        "questions": _INTERVIEW_FIELDS,
    }
    body = {
        "model": _livingos_api_model(),
        "messages": [
            {
                "role": "system",
                "content": "You are an interview agent for a contract intelligence MVP. Output strict JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt, indent=2)},
        ],
        "max_tokens": 420,
    }
    request = urllib.request.Request(
        f"{_livingos_api_base_url()}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.getenv('LIVINGOS_API_KEY', '').strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        if isinstance(parsed, dict):
            return parsed
        # Codex occasionally answers conversationally despite the JSON instruction.
        # The local parser has already updated structured memory, so keep the
        # Codex-authored next question and avoid dropping the turn to fallback.
        return {
            "assistant": content.strip(),
            "updates": {},
            "next_field": next_field,
            "ready": ready,
        }
    except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError):
        return None


def _extract_json_object(text: str) -> object:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    return None


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
    return []


def _alias_map(value: object) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    aliases: dict[str, list[str]] = {}
    for key, raw_aliases in value.items():
        label = str(key).strip()
        if label:
            aliases[label] = _string_list(raw_aliases)
    return aliases


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _output_files(root: Path) -> list[dict[str, object]]:
    files = [
        ("Interview Memory", "data/memory/interview.json"),
        ("Corpus Documents", "data/corpus/documents.jsonl"),
        ("Ingest Manifest", "data/corpus/ingest_manifest.json"),
        ("Baseline Results", "data/runs/baseline_results.json"),
        ("Agent Analysis", "data/runs/agent_analysis.json"),
        ("Pending Review Packet", "data/reviews/review_packet.pending.json"),
        ("Reviewed Packet", "data/reviews/review_packet.reviewed.json"),
        ("Taxonomy Memory", "data/memory/taxonomy.json"),
        ("Reviewed Examples JSONL", "data/training/training_pairs.jsonl"),
        ("Second Run Results", "data/runs/second_run_results.json"),
        ("Benchmark", "data/runs/benchmark.json"),
        ("Demo Report", "data/runs/demo_report.md"),
        ("Demo Run Manifest", "data/runs/demo_run_manifest.json"),
        ("Roadmap", "ROADMAP.md"),
    ]
    output = []
    for label, rel in files:
        path = root / rel
        output.append({
            "label": label,
            "path": rel,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        })
    return output


def _safe_artifact_path(root: Path, rel: str) -> Path | None:
    allowed = {item["path"] for item in _output_files(root)}
    if rel not in allowed:
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _artifact_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".jsonl":
        return "application/x-ndjson; charset=utf-8"
    if suffix == ".md":
        return "text/markdown; charset=utf-8"
    if suffix == ".txt":
        return "text/plain; charset=utf-8"
    return "application/octet-stream"
