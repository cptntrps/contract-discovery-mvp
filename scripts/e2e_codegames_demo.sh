#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

PRIMARY="${PRIMARY:-qwen2.5:14b}"
SHADOW="${SHADOW:-qwen3:4b}"
LIMIT="${LIMIT:-20}"
REVIEW_FRAC="${REVIEW_FRAC:-0.6}"
SEED="${SEED:-42}"

contract-intel reset --keep-raw >/dev/null || true
contract-intel init >/dev/null
contract-intel interview --config config/interview.example.json >/dev/null
contract-intel cuad-sample --limit "$LIMIT" --contains license >/dev/null
contract-intel ingest --input data/raw_contracts/cuad_samples >/dev/null
contract-intel split --review-frac "$REVIEW_FRAC" --seed "$SEED"

echo
echo "=== Phase 1: agent run (review set extraction + triage) ==="
contract-intel agent run --primary-model "$PRIMARY" --shadow-model "$SHADOW"

echo
echo "=== Phase 2: apply CUAD gold to review packet ==="
contract-intel review-packet >/dev/null
contract-intel cuad-apply-gold
contract-intel apply-review --review data/reviews/review_packet.reviewed.json

echo
echo "=== Phase 3: project CUAD gold onto holdout for benchmark ==="
contract-intel cuad-apply-holdout-gold

echo
echo "=== Phase 4: agent resume (holdout extraction + cold + benchmark) ==="
contract-intel agent resume --primary-model "$PRIMARY" --shadow-model "$SHADOW"

echo
echo "=== Benchmark ==="
python3 -m json.tool < data/runs/benchmark.json
echo
echo "Artifacts:"
ls -la data/runs/
