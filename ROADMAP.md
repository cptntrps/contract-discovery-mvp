# Contract Intelligence Roadmap

This MVP is a proof of concept. It proves the learning loop with a small public contract sample:

1. ingest public contracts;
2. run baseline local Ollama extraction;
3. challenge the result with an agentic review pass;
4. validate business-level outputs;
5. update taxonomy/playbook/examples;
6. rerun the same model with reviewed context;
7. benchmark the before/after result.

The ideal product must handle thousands to tens of thousands of contracts. That changes the product shape. Card-heavy views are acceptable for a four-document demo, but they are not sustainable for production-scale review, triage, monitoring, or corpus management.

## Design Principle

At scale, the primary UI should be dense, sortable, filterable, and queue-oriented.

Use cards only for:

- a single selected contract;
- a modal/detail drawer;
- small repeated evidence snippets;
- presentation/demo surfaces.

Use tables, split panes, virtualization, facets, saved filters, and bulk actions for:

- corpus lists;
- review queues;
- benchmark rows;
- clause libraries;
- taxonomy terms;
- exception handling;
- processing status.

## Phase 0: POC Hardening

Goal: make the current standalone demo reliable, honest, and easy to show.

- Add editable coversheet fields and clause evidence in the review UI.
- Add step-by-step progress for `Run CUAD Demo`.
- Add a tiny packaged public sample so the demo runs without the local CUAD archive.
- Add smoke/unit tests for title extraction, CUAD mapping, reviewed example export, benchmark metrics, and ingestion manifest.
- Add screenshots and a short presenter script.

## Phase 1: Scale-Oriented UI

Goal: replace demo-card surfaces with operational views that work for thousands of contracts.

- Add a dense corpus table with pagination or virtualization.
- Add filters for contract type, review status, confidence, source, engine, clause family, missing evidence, and benchmark result.
- Add saved views such as `Needs Review`, `High-Risk Misses`, `Missing Coversheet Fields`, and `Clause Exceptions`.
- Add row selection and bulk actions for assigning review, accepting low-risk outputs, exporting subsets, and rerunning selected contracts.
- Move coversheet, provenance, agent critique, and clause evidence into a right-side detail drawer for the selected row.

## Phase 2: Review Workbench

Goal: make human review productive and authority-aware.

- Edit contract type, coversheet fields, clause families, evidence spans, and reviewer notes in one workbench.
- Add confidence/evidence sufficiency controls.
- Track reviewer authority: contributor, business SME, senior SME, legal-reviewed.
- Prevent legal-reviewed claims from non-legal labels.
- Add queue routing by required authority, uncertainty, contract type, and risk.
- Preserve review decisions with provenance and scope.

## Phase 3: Corpus Processing

Goal: support large messy folders.

- Add PDF OCR/layout extraction.
- Track OCR status, page count, extraction quality, and unreadable pages.
- Add duplicate and near-duplicate detection.
- Add template clustering.
- Chunk documents with page/offset evidence spans.
- Add resumable processing and per-document run status.
- Add retry/error queues.

## Phase 4: Graph and Memory

Goal: move from JSON demo files to a durable contract intelligence graph.

- Add database-backed documents, chunks, evidence spans, hypotheses, coversheets, reviews, training examples, clause families, taxonomy terms, aliases, and relationships.
- Add scoped taxonomy: business unit, region, country, product line, customer, and validity dates.
- Add memory-assisted interview: retrieve known types, aliases, clauses, rejected assumptions, and prior benchmark weaknesses before processing a new corpus.
- Add relationship types: parent/child, equivalent, near-equivalent, deprecated, local alias, and regional variant.

## Phase 5: Evaluation and Model Improvement

Goal: make improvement measurable and promotable.

- Extend benchmark metrics for coversheet fields, clause family precision/recall/F1, evidence span quality, high-risk false positives, and reviewer correction rate.
- Add benchmark datasets and train/dev/test splits.
- Compare multiple local models and prompt/playbook configurations.
- Promote a prompt/playbook/model only when it beats baseline on reviewed metrics.
- Keep fine-tuning as an explicit optional workflow, separate from reviewed-context learning.

## Phase 6: Productionization

Goal: make it safe and operable for real customers.

- Add authentication and role-based permissions.
- Add audit logs.
- Add customer/workspace separation.
- Add background job orchestration.
- Add observability for processing throughput, failure rate, model latency, and review throughput.
- Add export APIs for BI, legal ops, and downstream systems.
- Add deployment packaging.

## Non-Goals for the Current POC

- No fine-tuning claim.
- No legal advice claim.
- No production permission model.
- No raw entity validation UI.
- No PDF OCR for this pass.
- No LivingOS dependency in the standalone MVP.

