"""CLI for the Contract Corpus Intelligence MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    apply_review,
    accept_pending_review,
    apply_cuad_gold_review,
    fetch_edgar_samples,
    generate_benchmark,
    generate_demo_report,
    generate_review_packet,
    ingest_folder,
    init_project,
    prepare_cuad_sample,
    reset_project,
    run_agent_analysis,
    run_cuad_demo,
    save_interview,
    run_extraction,
)
from .web import run_server


def main() -> None:
    parser = argparse.ArgumentParser(prog="contract-intel")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create local data folders and seed interview config")

    reset = sub.add_parser("reset", help="Clear generated data and reinitialize the project")
    reset.add_argument("--keep-raw", action="store_true", help="Keep data/raw_contracts while clearing derived outputs")

    interview = sub.add_parser("interview", help="Save or refresh the run interview config")
    interview.add_argument("--config", default="config/interview.example.json", help="Interview JSON to copy into data/memory/interview.json")

    ingest = sub.add_parser("ingest", help="Ingest a folder of contract files")
    ingest.add_argument("--input", required=True, help="Folder containing contracts")

    edgar = sub.add_parser("edgar-sample", help="Download a small public EDGAR exhibit pack")
    edgar.add_argument("--limit", type=int, default=10, help="Maximum exhibits to download")
    edgar.add_argument("--user-agent", default=None, help="SEC user agent string")

    cuad = sub.add_parser("cuad-sample", help="Prepare a sample from the public CUAD contract corpus")
    cuad.add_argument("--source", default=None, help="Path to CUAD_v1 folder")
    cuad.add_argument("--limit", type=int, default=12, help="Maximum CUAD contracts to stage")
    cuad.add_argument("--contains", default=None, help="Optional filename/type substring filter")

    demo_cuad = sub.add_parser("demo-cuad", help="Run the full CUAD reviewed-context demo path")
    demo_cuad.add_argument("--model", default="qwen3:4b")
    demo_cuad.add_argument("--limit", type=int, default=4)
    demo_cuad.add_argument("--contains", default="license")
    demo_cuad.add_argument("--source", default=None, help="Path to CUAD_v1 folder")

    baseline = sub.add_parser("baseline", help="Run baseline extraction")
    baseline.add_argument("--model", default="qwen3:4b")

    agent = sub.add_parser("agent-analyze", help="Run an agentic critique pass over extraction output")
    agent.add_argument("--model", default="qwen3:4b")
    agent.add_argument("--run", choices=["baseline", "second"], default="baseline")

    review = sub.add_parser("review-packet", help="Generate SME review packet")
    review.add_argument("--limit", type=int, default=50)

    apply = sub.add_parser("apply-review", help="Apply reviewed labels and export training pairs")
    apply.add_argument("--review", required=True)

    demo_accept = sub.add_parser("demo-accept-review", help="Demo-only: accept pending review packet as reviewed labels")
    demo_accept.add_argument("--note", default="Demo dry-run accepted pending model outputs.")

    sub.add_parser("cuad-apply-gold", help="Apply CUAD expert labels to the pending review packet")

    second = sub.add_parser("second-run", help="Run extraction with reviewed taxonomy context")
    second.add_argument("--model", default="qwen3:4b")

    sub.add_parser("benchmark", help="Compare baseline and reviewed-context runs")

    sub.add_parser("demo-report", help="Write a Markdown/JSON demo report")

    ui = sub.add_parser("ui", help="Run the local web UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    if args.command == "init":
        paths = init_project(Path.cwd())
        print(json.dumps({"created": [str(p) for p in paths]}, indent=2))
    elif args.command == "reset":
        result = reset_project(Path.cwd(), keep_raw=args.keep_raw)
        print(json.dumps(result, indent=2))
    elif args.command == "interview":
        result = save_interview(Path.cwd(), Path(args.config))
        print(json.dumps(result, indent=2))
    elif args.command == "ingest":
        count = ingest_folder(Path(args.input), Path.cwd())
        print(json.dumps({"documents_ingested": count}, indent=2))
    elif args.command == "edgar-sample":
        result = fetch_edgar_samples(Path.cwd(), limit=args.limit, user_agent=args.user_agent)
        print(json.dumps(result, indent=2))
    elif args.command == "cuad-sample":
        result = prepare_cuad_sample(Path.cwd(), source_dir=Path(args.source) if args.source else None, limit=args.limit, contains=args.contains)
        print(json.dumps(result, indent=2))
    elif args.command == "demo-cuad":
        result = run_cuad_demo(Path.cwd(), model=args.model, limit=args.limit, contains=args.contains, source_dir=Path(args.source) if args.source else None)
        print(json.dumps(result, indent=2))
    elif args.command == "baseline":
        result = run_extraction(Path.cwd(), model=args.model, mode="baseline")
        print(json.dumps(result, indent=2))
    elif args.command == "agent-analyze":
        result = run_agent_analysis(Path.cwd(), model=args.model, run=args.run)
        print(json.dumps(result, indent=2))
    elif args.command == "review-packet":
        packet_path = generate_review_packet(Path.cwd(), limit=args.limit)
        print(json.dumps({"review_packet": str(packet_path)}, indent=2))
    elif args.command == "apply-review":
        result = apply_review(Path.cwd(), Path(args.review))
        print(json.dumps(result, indent=2))
    elif args.command == "demo-accept-review":
        result = accept_pending_review(Path.cwd(), note=args.note)
        print(json.dumps(result, indent=2))
    elif args.command == "cuad-apply-gold":
        result = apply_cuad_gold_review(Path.cwd())
        print(json.dumps(result, indent=2))
    elif args.command == "second-run":
        result = run_extraction(Path.cwd(), model=args.model, mode="second")
        print(json.dumps(result, indent=2))
    elif args.command == "benchmark":
        result = generate_benchmark(Path.cwd())
        print(json.dumps(result, indent=2))
    elif args.command == "demo-report":
        result = generate_demo_report(Path.cwd())
        print(json.dumps(result, indent=2))
    elif args.command == "ui":
        run_server(Path.cwd(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
