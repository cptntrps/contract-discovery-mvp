# Code Games Agentic Edition — Presenter Script (8 min)

## Setup before stage
- Aurora running, Ollama up with `qwen2.5:14b` and `qwen3:4b` pulled.
- Browser at `http://127.0.0.1:8765/agent`.
- Terminal in `/home/gui/code-games-contract-intelligence-mvp` with `.venv` active.

## Live arc

1. **(1m) Setup tab** — show interview JSON (legal review goal, US, License focus). Show `data/corpus/splits.json` with the 60/40 split and seed.

2. **(2m) Click "Start Run"** — narrate the streaming decision log on `/agent`:
   - planner → `extract_review_batch` (primary `qwen2.5:14b` + shadow `qwen3:4b` in parallel)
   - per-doc `extract_with_verification` retries when evidence quotes are not verbatim
   - `triage` flags ~30% of docs for human review with reasons (`unverifiable_spans`, `missing_expected_clauses`, `close_type_alternative`)

3. **(2m) Review queue** — apply CUAD gold (`cuad-apply-gold` + `apply-review`); show `data/memory/taxonomy.json` growing with confirmed types, aliases, examples, rejected patterns. Project gold onto holdout (`cuad-apply-holdout-gold`).

4. **(1m) Click "Resume"** — agent rebuilds prompt context from the new taxonomy, reruns primary+shadow on the 16 holdout docs, runs cold-small for the third bench column. Engine banner stays green throughout.

5. **(1m) Benchmark tab — read the headline:**

   ```
   Engine integrity: ok | n=16

   metric                       large    small_cold   small_reviewed
   contract_type_accuracy        1.00         0.56            0.89
   clause_family_f1              0.14         0.17            0.17
   ```

   Pitch sentence: *"On 9 unseen contracts, the 4B small model with reviewed context recovered 75% of the 14B model's contract-type accuracy at 1/4 the size."*

6. **(1m) Counterfactual buttons:**
   - **Reviewed-context OFF** → type accuracy drops 0.89 → 0.56 (= small_cold). *"This is what the review loop earns."*
   - **Verifier OFF** → F1 0.172 → 0.206 (verifier currently *over-strict* on this corpus). *"Honest finding: the verifier rejects clauses where the model paraphrases instead of quoting verbatim. Next iteration relaxes matching."*

## What to call out, what to admit

- ✅ **Engine integrity gate** refuses to publish if any model call fell back. We hit this once during dev (qwen3 thinking-mode mismatch); fixed and re-ran.
- ✅ **Held-out split** — gold for the holdout is never in the prompt, so the small_reviewed delta is real, not train-on-test.
- ✅ **Decision log** — every action the agent took is in `data/runs/agent_decisions.jsonl`, queryable via `/api/decisions`.
- ⚠️ **Verifier over-strict** — fuzzy-match relaxation is the obvious next iteration.
- ⚠️ **Clause-family F1 is low across all three models** — vocabulary mismatch between the model's family names and CUAD labels. Lead with type accuracy on stage.

## Headline numbers to memorize

```
1.00  large (qwen2.5:14b)
0.56  small cold (qwen3:4b, no taxonomy)
0.89  small reviewed (qwen3:4b + reviewed taxonomy)
```

## Failure modes covered live

- Ollama down: planner refuses to run, banner red, nothing pretends to work.
- Mid-run crash: `agent_decisions.jsonl` shows last action; rerun resumes from `inspect_state`.
- Bad gold: `engine_integrity == "contaminated"` is louder than any silent metric drift.
