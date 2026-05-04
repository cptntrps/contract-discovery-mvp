# Contract Corpus Intelligence MVP

Standalone Code Games prototype for an agentic contract corpus discovery loop.

The MVP uses a small EDGAR-style contract corpus, runs a local Ollama model, asks for low-friction business review, captures corrections as training pairs, and reruns the same model with learned context to measure improvement.

## MVP Flow

1. Interview the user for goal, region, business unit, expected contract types, aliases, key clauses, and exclusions.
2. Ingest a folder of contracts into normalized text chunks.
3. Run a baseline Ollama extraction with no learned context.
4. Run an agentic critique pass that challenges the first extraction and suggests review questions.
5. Generate a review packet focused on business objects, not raw entities.
6. Capture human corrections as labels.
7. Update taxonomy, clause-family seeds, aliases, examples, playbook, and rejected patterns.
8. Export training pairs from reviewed labels.
9. Rerun the same Ollama model with reviewed context.
10. Compare baseline vs reviewed-context output.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

contract-intel init
contract-intel interview --config config/interview.example.json
contract-intel ingest --input sample_corpus
contract-intel baseline --model qwen3:4b
contract-intel agent-analyze --model qwen3:4b
contract-intel review-packet
cp data/reviews/review_packet.pending.json data/reviews/review_packet.reviewed.json
# edit review_packet.reviewed.json: set status/accepted values
contract-intel apply-review --review data/reviews/review_packet.reviewed.json
contract-intel second-run --model qwen3:4b
contract-intel benchmark
contract-intel demo-report
```

Use `contract-intel reset` to clear generated `data/` outputs before a clean demo run.
Run `contract-intel ui` to open the local dashboard at `http://127.0.0.1:8765`.

If Ollama is not running, the CLI falls back to a deterministic heuristic so the end-to-end flow can still be tested.

## Executable User Stories

The Interview tab is the agent setup surface. It is chat-first: the agent asks for the business goal, business unit, region, expected contract types, aliases, key clause families, exclusions, and review priorities. The chat writes structured memory to `data/memory/interview.json`, then merges the expected types and clause families into `data/memory/taxonomy.json`.

By default, the chat can run locally with deterministic interview logic. To connect the interview to LivingOS API Codex, create `.env.local` from `.env.example`, set `LIVINGOS_API_KEY`, and start the UI with the env file loaded:

```bash
set -a
. ./.env.local
set +a
PYTHONPATH=src python3 -m contract_intel_mvp.cli ui --host 127.0.0.1 --port 8788
```

When `LIVINGOS_API_INTERVIEW=1`, `/api/interview/chat` calls `LIVINGOS_API_MODEL` through `LIVINGOS_API_BASE_URL` and falls back locally if the API is unavailable or returns invalid JSON.

Start the UI on the same port used by the smoke harness:

```bash
PYTHONPATH=src python3 -m contract_intel_mvp.cli ui --host 127.0.0.1 --port 8788
```

In another terminal, run the executable user-story check:

```bash
python3 scripts/user_story_smoke.py --base-url http://127.0.0.1:8788
```

The smoke check exercises:

- US-1: capture interview goals and scope through `/api/interview/chat`
- US-2: seed expected contract types for extraction
- US-3: seed key clause families for evidence review
- US-4: confirm interview context updates taxonomy memory
- US-5: confirm the pipeline can proceed after interview setup without losing corpus state

## One-Command CUAD Demo

For the most repeatable demo path, use the public CUAD corpus sample and expert labels:

```bash
contract-intel demo-cuad --model qwen3:4b --limit 4 --contains license
contract-intel ui
```

This command resets generated `data/`, loads the interview config, stages CUAD contracts, ingests them, runs baseline extraction, runs agent analysis, generates the review packet, applies CUAD expert labels, exports reviewed examples, reruns the same model with reviewed context, benchmarks, and writes the demo report.

## Real EDGAR Contracts

Download a public SEC EDGAR exhibit pack, then ingest it:

```bash
contract-intel edgar-sample --limit 10
contract-intel ingest --input data/raw_contracts/edgar_samples
contract-intel baseline --model qwen3:4b
contract-intel agent-analyze --model qwen3:4b
contract-intel review-packet
```

For a real review loop, edit `data/reviews/review_packet.pending.json` into `data/reviews/review_packet.reviewed.json` with SME corrections, then apply it. For a non-authoritative dry run only:

```bash
contract-intel demo-accept-review
contract-intel apply-review --review data/reviews/review_packet.reviewed.json
contract-intel second-run --model qwen3:4b
contract-intel benchmark
contract-intel demo-report
contract-intel ui
```

You can also put downloaded EDGAR exhibits in `data/raw_contracts/` or pass another folder to `ingest`.

Supported inputs:

- `.txt`, `.md`
- `.html`, `.htm`
- `.docx`
- `.pdf` if `pdftotext` is installed

The included `sample_corpus/` files are tiny synthetic EDGAR-style documents for local smoke tests only.

## CUAD Contract Corpus

The recommended realistic demo corpus is CUAD v1, the Contract Understanding Atticus Dataset used in the prior LivingOS/TRI validation work. CUAD contains 510 public commercial contracts with expert clause annotations. If the local CUAD copy is present, stage a sample and use CUAD annotations as expert-reviewed labels:

```bash
contract-intel reset
contract-intel cuad-sample --limit 12
contract-intel ingest --input data/raw_contracts/cuad_samples
contract-intel baseline --model qwen3:4b
contract-intel agent-analyze --model qwen3:4b
contract-intel review-packet
contract-intel cuad-apply-gold
contract-intel apply-review --review data/reviews/review_packet.reviewed.json
contract-intel second-run --model qwen3:4b
contract-intel benchmark
contract-intel demo-report
```

Use `--contains license`, `--contains distributor`, or another substring to stage a narrower corpus. `cuad-apply-gold` is not model fine-tuning and not a raw entity review step; it maps CUAD expert annotations into the same business-level review packet fields: contract type, coversheet fields, key clause families, evidence spans, and playbook/taxonomy memory.

## Output

- `data/corpus/documents.jsonl` - normalized document text
- `data/runs/baseline_results.json` - first-pass model outputs
- `data/runs/agent_analysis.json` - agentic critique, review questions, and memory suggestions
- `data/reviews/review_packet.pending.json` - SME review packet
- `data/memory/taxonomy.json` - learned taxonomy/playbook context
- `data/training/training_pairs.jsonl` - exported reviewed examples in input/output JSONL form
- `data/runs/second_run_results.json` - reviewed-context rerun
- `data/runs/benchmark.json` - before/after comparison
- `data/runs/demo_report.md` - short demo summary for stakeholders

## Local UI

```bash
contract-intel ui
```

The UI is a dependency-light local web workbench over the generated `data/` files. The primary navigation now follows the BRD workflow: Setup, Corpus, Hypotheses, Review, Intelligence, and Benchmark. Diagnostic surfaces such as pipeline readiness, source documents, provenance, run report, BRD coverage, roadmap, memory, and exported examples still exist in the app code/output for inspection, but they are no longer the main product journey.

## Roadmap

See `ROADMAP.md` for the path from this POC to a thousand-document contract intelligence workbench. The key product principle is that cards are acceptable for demo detail views, but large-corpus operations need dense tables, filters, saved queues, virtualization, bulk actions, and detail drawers.

## Implementation Note

What is real in this MVP: the CLI ingests local contract files, calls Ollama at `127.0.0.1:11434` when reachable, creates an agentic critique pass over first-pass outputs, creates business-level review packets, applies reviewed labels, updates taxonomy/playbook/example/rejected-pattern memory, exports JSONL training pairs, reruns the same model with reviewed context, and benchmarks baseline vs reviewed-context contract type accuracy.

What is fallback: if Ollama is unavailable or returns invalid JSON, extraction and agent analysis use deterministic heuristics so the full demo path remains runnable. Output rows include an `engine` field showing `ollama` or `heuristic_fallback`.

What future fine-tuning would add: the exported `data/training/training_pairs.jsonl` contains reviewed input/output examples that can become supervised examples for later local-model tuning, but this MVP does not fine-tune weights. The improvement claim here is reviewed-context learning through taxonomy, playbook, examples, and rejected-pattern memory.
