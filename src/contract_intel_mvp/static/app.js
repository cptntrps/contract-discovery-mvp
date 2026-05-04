let state = null;
let selectedQueueDocId = null;
let selectedWorkbenchKey = null;
let queuePage = 1;
let workbenchPage = 1;
let interviewDraft = null;
let interviewMessages = [];
let interviewActiveField = null;

const interviewFieldQuestions = {
  goal: "What business decision should this contract corpus support?",
  business_unit: "Which business unit or operating group owns this review?",
  region: "Which region or governing jurisdiction should bias the taxonomy?",
  expected_contract_types: "Which contract types should the model expect first?",
  key_clause_families: "Which clause families matter most for review?",
  not_expected: "What document types should be treated as out of scope?",
  review_priorities: "Which gaps or risks should create high-priority review queues?"
};

const interviewFieldLabels = {
  engine: "Engine",
  goal: "Goal",
  business_unit: "Business Unit",
  region: "Region",
  expected_contract_types: "Expected Types",
  contract_type_aliases: "Aliases",
  key_clause_families: "Clause Families",
  not_expected: "Out Of Scope",
  review_priorities: "Review Priorities"
};

const titles = {
  overview: ["Benchmark", "Baseline vs reviewed-context performance and demo claim boundary."],
  interview: ["Setup", "Start with the SME interview and scope the contract intelligence run."],
  pipeline: ["Pipeline", "End-to-end run readiness, output artifacts, and rerun commands."],
  intelligence: ["Intelligence", "Taxonomy, clause families, corpus mix, provenance, and graph-shaped findings."],
  queue: ["Hypotheses", "Candidate contract classifications, alternatives, evidence gaps, and review routing."],
  workbench: ["Review", "Business-level validation of coversheet fields, clause families, evidence, and authority."],
  corpus: ["Corpus", "Corpus intake, accepted inputs, skipped files, and OCR/text boundary."],
  documents: ["Documents", "Contract-level coversheets, labels, and evidence."],
  coversheets: ["Coversheets", "Reviewed business coversheets with authority and evidence."],
  review: ["Review", "Business-level review packet and accepted clause evidence."],
  agent: ["Agent", "Local critique pass that challenges extraction before human review."],
  clauses: ["Clause Library", "Reviewed clause families, variants, answers, authority, and evidence."],
  provenance: ["Provenance", "Document lineage from source file through model runs, review, and benchmark."],
  demo: ["Demo Script", "A concise click path and claim boundary for presenting the MVP."],
  report: ["Run Report", "Stakeholder report and generated output files for the current run."],
  memory: ["Memory", "Taxonomy, playbook, and reviewed examples learned from labels."],
  training: ["Reviewed Examples", "JSONL correction examples exported from reviewed labels."],
  brd: ["BRD Coverage", "What this MVP satisfies, partially covers, and leaves for the roadmap."],
  roadmap: ["Roadmap", "Path from POC to thousand-document contract intelligence workbench."]
};

const workflowSteps = [
  ["step-corpus",   "1", "Corpus",    "Drop your contracts and embed"],
  ["step-define",   "2", "Define",    "Tell the agent what to look for"],
  ["step-classify", "3", "Classify",  "Rank and run a round"],
  ["step-review",   "4", "Review",    "Confirm borderline cases"],
  ["step-library",  "5", "Library",   "Clause language the agent learned"],
  ["step-results",  "6", "Results",   "Final list with confidence"]
];

const brdCoverage = [
  ["FR-1", "Interview-led setup", "partial", "Config-backed interview seeds the local taxonomy; memory/graph-assisted questioning is future work."],
  ["FR-2", "Corpus graph creation", "partial", "Creates normalized document records from public contracts; no Postgres graph, chunks, OCR quality, or duplicate/template graph yet."],
  ["FR-3", "Evidence-backed hypotheses", "mvp", "Baseline extraction returns contract type, coversheet fields, key clauses, confidence, and evidence spans."],
  ["FR-4", "Human authority model", "partial", "Review labels preserve authority strings such as CUAD expert annotation and SME; no formal permission model or legal-reviewed guardrail."],
  ["FR-5", "Automatic coversheets", "mvp", "Generates and reviews coversheet fields with evidence; richer cover-page artifact is still a polish item."],
  ["FR-6", "Clause library", "partial", "Reviewed clause families and examples update memory; no dedicated clause-variant promotion workflow yet."],
  ["FR-7", "HITL learning loop", "mvp", "Reviewed labels update memory and export reviewed correction examples."],
  ["FR-8", "Ollama benchmark loop", "mvp", "Benchmarks baseline vs reviewed-context rerun with the same Ollama model."],
  ["FR-9", "Taxonomy and ontology creation", "partial", "Tracks contract types, aliases, examples, playbook, and rejected patterns; scoped BU/region ontology remains future work."],
  ["FR-10", "Memory and graph-assisted interview", "future", "No LivingOS memory/graph dependency in the standalone MVP; this is intentionally deferred."],
  ["NFR-1", "Provenance", "partial", "Outputs keep doc id, source path, engine/model, evidence, and authority; not every field has normalized evidence-span IDs."],
  ["NFR-2", "Legal safety", "mvp", "UI and README state no fine-tuning and no legal truth claim; formal legal-reviewed role controls are future work."],
  ["NFR-3", "Demo reliability", "mvp", "Bounded CUAD sample, deterministic expert-label path, fallback behavior, benchmark, and one-command demo runner."]
];

const demoSteps = [
  ["1", "Open Overview", "Show the bounded CUAD sample, baseline accuracy, reviewed-context accuracy, and engine badges."],
  ["2", "Open Documents", "Show that the corpus contains actual public contracts and that titles are now contract-like, not boilerplate such as CONFIDENTIAL."],
  ["3", "Open Coversheets", "Show reviewed business fields, authority, evidence, and clause families for one contract."],
  ["4", "Open Agent", "Show how the critique pass challenges broad labels and turns uncertainty into reviewer questions."],
  ["5", "Open Review", "Show the human validation surface: contract type, coversheet fields, clauses, and evidence, not raw entity validation."],
  ["6", "Open Examples", "Show reviewed correction examples: contract plus baseline draft becomes a reviewed business answer."],
  ["7", "Open BRD", "Show what is MVP, partial, and future so the demo does not overclaim."]
];

const demoBoundaries = [
  ["Real", "Public CUAD contracts, local Ollama calls when reachable, reviewed-context rerun, business-level review packets, reviewed examples, benchmark output."],
  ["Fallback", "If Ollama is unavailable or returns invalid JSON, deterministic fallback keeps the path runnable and marks the engine."],
  ["Not Claimed", "No model fine-tuning, no legal advice, no production permissions model, no PDF OCR/layout pipeline, no LivingOS graph dependency in this standalone MVP."],
  ["Demo Claim", "Expert review becomes reusable taxonomy, playbook, examples, and benchmark context; the same model improves when rerun with that reviewed context."]
];

const roadmapRows = [
  ["0", "POC Hardening", "Next", "Editable review fields, run progress, packaged sample, smoke tests, screenshots."],
  ["1", "Scale-Oriented UI", "Next", "Dense tables, filters, saved views, virtualized lists, bulk actions, detail drawers."],
  ["2", "Review Workbench", "Next", "Coversheet/clause evidence editing, authority controls, review queues, routing."],
  ["3", "Corpus Processing", "Later", "PDF OCR/layout, duplicate detection, template clusters, resumable runs, error queues."],
  ["4", "Graph and Memory", "Later", "Database-backed evidence graph, scoped taxonomy, aliases, relationships, memory-assisted interview."],
  ["5", "Evaluation and Model Improvement", "Later", "Benchmarks, splits, model/prompt comparisons, promotion gates, optional fine-tuning lane."],
  ["6", "Productionization", "Later", "Auth, permissions, audit logs, job orchestration, observability, deployment packaging."]
];

document.querySelectorAll(".nav").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

document.getElementById("refreshBtn").addEventListener("click", load);
document.getElementById("saveReviewBtn").addEventListener("click", saveReviewEdits);
document.getElementById("saveInterviewBtn").addEventListener("click", saveInterview);
document.getElementById("interviewSendBtn").addEventListener("click", sendInterviewMessage);
document.getElementById("interviewChatInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendInterviewMessage();
});
document.querySelectorAll(".action").forEach((button) => {
  button.addEventListener("click", () => runAction(button.dataset.action));
});
document.querySelectorAll("[data-queue-preset]").forEach((button) => {
  button.addEventListener("click", () => applyQueuePreset(button.dataset.queuePreset));
});
document.querySelectorAll("[data-workbench-preset]").forEach((button) => {
  button.addEventListener("click", () => applyWorkbenchPreset(button.dataset.workbenchPreset));
});
document.getElementById("queueSearch").addEventListener("input", renderQueue);
document.getElementById("queueStatus").addEventListener("change", renderQueue);
document.getElementById("queueType").addEventListener("change", renderQueue);
document.getElementById("queueSort").addEventListener("change", renderQueue);
document.getElementById("queueDir").addEventListener("change", renderQueue);
document.getElementById("queueExport").addEventListener("click", exportQueueCsv);
document.getElementById("queuePageSize").addEventListener("change", () => {
  queuePage = 1;
  renderQueue();
});
document.getElementById("queuePrev").addEventListener("click", () => {
  queuePage = Math.max(1, queuePage - 1);
  renderQueue();
});
document.getElementById("queueNext").addEventListener("click", () => {
  queuePage += 1;
  renderQueue();
});
document.getElementById("workbenchSearch").addEventListener("input", renderWorkbench);
document.getElementById("workbenchMode").addEventListener("change", () => {
  selectedWorkbenchKey = null;
  renderWorkbench();
});
document.getElementById("workbenchStatus").addEventListener("change", renderWorkbench);
document.getElementById("workbenchType").addEventListener("change", renderWorkbench);
document.getElementById("workbenchSort").addEventListener("change", renderWorkbench);
document.getElementById("workbenchDir").addEventListener("change", renderWorkbench);
document.getElementById("workbenchExport").addEventListener("click", exportWorkbenchCsv);
document.getElementById("workbenchPageSize").addEventListener("change", () => {
  workbenchPage = 1;
  renderWorkbench();
});
document.getElementById("workbenchPrev").addEventListener("click", () => {
  workbenchPage = Math.max(1, workbenchPage - 1);
  renderWorkbench();
});
document.getElementById("workbenchNext").addEventListener("click", () => {
  workbenchPage += 1;
  renderWorkbench();
});

load();

async function load() {
  const response = await fetch("/api/state", { cache: "no-store" });
  state = await response.json();
  render();
}

function setView(name) {
  document.querySelectorAll(".nav").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  document.querySelectorAll(".workflowStep").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === name));
  document.getElementById("viewTitle").textContent = titles[name][0];
  document.getElementById("viewSub").textContent = titles[name][1];
}

function render() {
  renderWorkflowProgress();
  renderOverview();
  renderInterview();
  renderPipeline();
  renderIntelligence();
  renderQueueFilters();
  renderQueue();
  renderWorkbenchFilters();
  renderWorkbench();
  renderCorpus();
  renderDocuments();
  renderCoversheets();
  renderReview();
  renderAgent();
  renderClauses();
  renderProvenance();
  renderDemo();
  renderReport();
  renderMemory();
  renderTraining();
  renderBRD();
  renderRoadmap();
}

function renderWorkflowProgress() {
  html("workflowSteps", workflowSteps.map(([anchor, number, label, detail]) => `
    <button type="button" class="workflowStep" data-anchor="${escape(anchor)}">
      <span>${escape(number)}</span>
      <strong>${escape(label)}</strong>
      <small>${escape(detail)}</small>
    </button>
  `).join(""));
  document.querySelectorAll(".workflowStep").forEach((button) => {
    button.addEventListener("click", () => {
      const target = document.getElementById(button.dataset.anchor);
      if (target) target.scrollIntoView({behavior: "smooth", block: "start"});
    });
  });
}

function renderOverview() {
  const s = state.summary;
  text("metricDocs", s.documents);
  text("metricReviewed", s.reviewed);
  text("metricPairs", s.training_pairs);
  text("metricAgentFlags", Number(s.agent_high_priority || 0) + Number(s.agent_medium_priority || 0));
  text("metricBaseline", pct(s.baseline_accuracy));
  text("metricSecond", pct(s.second_accuracy));
  text("metricCoversheetDelta", metricDelta(s.baseline_coversheet_field_accuracy, s.second_run_coversheet_field_accuracy));
  text("metricClauseDelta", metricDelta(s.baseline_clause_family_f1, s.second_run_clause_family_f1));

  const rows = state.benchmark.rows || [];
  html("benchmarkRows", rows.map((row) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(row.doc_id)}</span>
        <span class="badge ${row.second_run_correct ? "good" : "bad"}">${row.second_run_correct ? "Improved" : "Miss"}</span>
      </div>
      <div class="labels">
        <span class="badge bad">Baseline: ${escape(row.baseline_contract_type)}</span>
        <span class="badge accent">Reviewed: ${escape(row.gold_contract_type)}</span>
        <span class="badge good">Second: ${escape(row.second_run_contract_type)}</span>
        <span class="badge">Cover: ${pct(row.baseline_coversheet_field_accuracy)} → ${pct(row.second_run_coversheet_field_accuracy)}</span>
        <span class="badge">Clause F1: ${pct(row.baseline_clause_family_f1)} → ${pct(row.second_run_clause_family_f1)}</span>
      </div>
    </div>
  `).join(""));

  const engines = [
    ...(s.baseline_engines || []).map((value) => `Baseline: ${value}`),
    ...(s.second_engines || []).map((value) => `Second: ${value}`)
  ];
  html("engineBadges", engines.map((value) => `<span class="badge accent">${escape(value)}</span>`).join(""));
}

function renderInterview() {
  ensureInterviewDraft();
  renderInterviewChat();
  renderInterviewMemory(interviewDraft);
  renderInterviewQuestions(interviewDraft);
  renderInterviewStories(interviewDraft);
}

function ensureInterviewDraft() {
  if (!interviewDraft) {
    interviewDraft = structuredCloneSafe(state.interview || {});
    interviewActiveField = nextInterviewField(interviewDraft);
  }
  if (!interviewMessages.length) {
    interviewMessages.push({
      role: "agent",
      text: interviewActiveField ? interviewFieldQuestions[interviewActiveField] : "Interview memory is ready. You can add aliases, exclusions, or review priorities."
    });
  }
}

function renderInterviewChat() {
  html("interviewChat", interviewMessages.map((message) => `
    <div class="chatMessage ${message.role === "user" ? "fromUser" : "fromAgent"}">
      <span>${message.role === "user" ? "You" : "Agent"}</span>
      <p>${escape(message.text)}</p>
    </div>
  `).join(""));
  const chat = document.getElementById("interviewChat");
  if (chat) chat.scrollTop = chat.scrollHeight;
}

function renderInterviewMemory(interview) {
  const api = state.livingos_api || {};
  const rows = [
    ["engine", api.interview_enabled ? `LivingOS API Codex (${api.model})` : "Local fallback"],
    ["goal", interview.goal],
    ["business_unit", interview.business_unit],
    ["region", interview.region],
    ["expected_contract_types", listText(interview.expected_contract_types)],
    ["key_clause_families", listText(interview.key_clause_families)],
    ["not_expected", listText(interview.not_expected)],
    ["review_priorities", listText(interview.review_priorities)]
  ];
  html("interviewMemorySummary", rows.map(([field, value]) => `
    <div>
      <span>${escape(interviewFieldLabels[field])}</span>
      <strong>${escape(value || "-")}</strong>
    </div>
  `).join(""));
}

function renderInterviewQuestions(interview) {
  const questions = interviewQuestions(interview);
  const ready = interviewReady(interview);
  const badge = document.getElementById("interviewReadiness");
  badge.textContent = ready ? "Ready" : "Needs Input";
  badge.className = `badge ${ready ? "good" : "warn"}`;
  html("interviewQuestions", questions.map((question) => `
    <div class="row">
      <div class="rowTitle"><span>${escape(question)}</span></div>
    </div>
  `).join(""));
}

function renderInterviewStories(interview) {
  const stories = [
    ["US-1", "As a business reviewer, I can save interview goals and scope.", Boolean(interview.goal && interview.business_unit && interview.region), "POST /api/interview/save"],
    ["US-2", "As a reviewer, I can seed expected contract types for extraction.", Boolean((interview.expected_contract_types || []).length), "GET /api/state interview.expected_contract_types"],
    ["US-3", "As a reviewer, I can seed clause families for evidence review.", Boolean((interview.key_clause_families || []).length), "GET /api/state interview.key_clause_families"],
    ["US-4", "As the system, saved interview context updates taxonomy memory.", taxonomyContainsInterview(interview), "GET /api/state taxonomy"],
    ["US-5", "As an operator, the pipeline can proceed after interview setup.", interviewReady(interview), "Pipeline Interview / Init ready"],
    ["US-6", "As a reviewer, I can keep chatting after setup without corrupting memory.", interviewReady(interview), "POST /api/interview/chat post-completion help"]
  ];
  html("interviewStories", `
    <div class="storyHeader">
      <span>ID</span>
      <span>User Story</span>
      <span>Status</span>
      <span>Executable Check</span>
    </div>
    ${stories.map(([id, story, passed, check]) => `
      <div class="storyRow">
        <strong>${escape(id)}</strong>
        <span>${escape(story)}</span>
        <span class="badge ${passed ? "good" : "warn"}">${passed ? "Pass" : "Needs Input"}</span>
        <code>${escape(check)}</code>
      </div>
    `).join("")}
  `);
}

function interviewQuestions(interview) {
  const questions = [];
  if (!interview.goal) questions.push("What business decision should this contract corpus support?");
  if (!interview.business_unit) questions.push("Which business unit or operating group owns this review?");
  if (!interview.region) questions.push("Which region or governing jurisdiction should bias the taxonomy?");
  if (!(interview.expected_contract_types || []).length) questions.push("Which contract types should the model expect first?");
  if (!(interview.key_clause_families || []).length) questions.push("Which clause families matter most for review?");
  if (!(interview.not_expected || []).length) questions.push("What document types should be treated as out of scope?");
  if (!questions.length) {
    questions.push("Should any contract type aliases be added from the current corpus?");
    questions.push("Which missing fields should create high-priority review queues?");
    questions.push("Which clause families should be benchmarked as must-find evidence?");
  }
  return questions;
}

function interviewReady(interview) {
  return Boolean(interview.goal && interview.business_unit && interview.region && (interview.expected_contract_types || []).length && (interview.key_clause_families || []).length);
}

function taxonomyContainsInterview(interview) {
  const taxonomy = state.taxonomy || {};
  const types = new Set(taxonomy.contract_types || []);
  const clauses = new Set(taxonomy.clause_families || []);
  const expectedTypes = interview.expected_contract_types || [];
  const expectedClauses = interview.key_clause_families || [];
  return expectedTypes.length > 0 &&
    expectedClauses.length > 0 &&
    expectedTypes.every((item) => types.has(item)) &&
    expectedClauses.every((item) => clauses.has(item));
}

async function saveInterview() {
  ensureInterviewDraft();
  const result = await post("/api/interview/save", { interview: interviewDraft });
  if (result.error) {
    setStatus(result.error);
    interviewMessages.push({ role: "agent", text: result.error });
    renderInterviewChat();
    return;
  }
  interviewDraft = result.interview;
  setStatus(`Saved interview with ${result.expected_contract_types} contract types and ${result.key_clause_families} clause families.`);
  await load();
}

async function sendInterviewMessage() {
  ensureInterviewDraft();
  const input = document.getElementById("interviewChatInput");
  const message = input.value.trim();
  if (!message) return;
  const field = interviewActiveField || nextInterviewField(interviewDraft) || "review_priorities";
  input.value = "";
  interviewMessages.push({ role: "user", text: message });
  renderInterviewChat();
  const result = await post("/api/interview/chat", { interview: interviewDraft, field, message });
  if (result.error) {
    interviewMessages.push({ role: "agent", text: result.error });
    renderInterviewChat();
    setStatus(result.error);
    return;
  }
  interviewDraft = result.interview || interviewDraft;
  interviewActiveField = result.next_field || nextInterviewField(interviewDraft);
  const engine = result.engine === "livingos_api_codex" ? "Codex" : "Local";
  interviewMessages.push({ role: "agent", text: `${result.assistant || "Saved."} (${engine})` });
  if (result.saved) setStatus(`Saved interview with ${result.saved.expected_contract_types} contract types and ${result.saved.key_clause_families} clause families.`);
  if (result.codex_error) setStatus(result.codex_error);
  renderInterview();
  if (result.saved) await load();
}

function nextInterviewField(interview) {
  for (const field of Object.keys(interviewFieldQuestions)) {
    const value = interview[field];
    if (Array.isArray(value)) {
      if (!value.length) return field;
    } else if (!String(value || "").trim()) {
      return field;
    }
  }
  return "";
}

function structuredCloneSafe(value) {
  return JSON.parse(JSON.stringify(value || {}));
}

function splitLines(value) {
  return value.split(/\n|,/).map((item) => item.trim()).filter(Boolean);
}

function parseAliasText(value) {
  const aliases = {};
  for (const line of value.split(/\n/)) {
    const [label, raw] = line.split(/:(.*)/s);
    if (label && raw) aliases[label.trim()] = splitLines(raw);
  }
  return aliases;
}

function listText(value) {
  return (value || []).join("\n");
}

function aliasText(value) {
  return Object.entries(value || {}).map(([label, aliases]) => `${label}: ${(aliases || []).join(", ")}`).join("\n");
}

function renderPipeline() {
  const rows = pipelineRows();
  html("pipelineSummary", `
    <span>${rows.filter((row) => row.done).length} complete</span>
    <span>${rows.length} stages</span>
    <span>${state.summary.documents || 0} docs</span>
    <span>${state.summary.training_pairs || 0} reviewed examples</span>
  `);
  html("pipelineRows", `
    <div class="pipelineHeader">
      <span>Step</span>
      <span>Stage</span>
      <span>Status</span>
      <span>Artifact</span>
      <span>Command</span>
    </div>
    ${rows.map((row) => `
      <div class="pipelineRow">
        <span>${escape(row.step)}</span>
        <strong>${escape(row.stage)}</strong>
        <span class="badge ${row.done ? "good" : "warn"}">${row.done ? "Ready" : "Missing"}</span>
        <span title="${escape(row.artifact)}">${escape(row.artifact)}</span>
        <code>${escape(row.command)}</code>
      </div>
    `).join("")}
  `);
}

function pipelineRows() {
  const output = outputFileMap();
  const hasCuad = Boolean((state.ingest_manifest?.input_dir || "").includes("cuad_samples") || Number(state.summary.cuad_contracts_prepared || 0));
  return [
    {
      step: "1",
      stage: "Interview / Init",
      done: true,
      artifact: "config/interview.example.json",
      command: "contract-intel init && contract-intel interview --config config/interview.example.json"
    },
    {
      step: "2",
      stage: "Stage Public Corpus",
      done: hasCuad || Number(state.summary.edgar_files_downloaded || 0) > 0,
      artifact: "data/raw_contracts/cuad_samples",
      command: "contract-intel cuad-sample --limit 4 --contains license"
    },
    {
      step: "3",
      stage: "Ingest Contracts",
      done: Number(state.summary.documents || 0) > 0 || Boolean(output.get("data/corpus/documents.jsonl")?.exists),
      artifact: "data/corpus/documents.jsonl",
      command: "contract-intel ingest --input data/raw_contracts/cuad_samples"
    },
    {
      step: "4",
      stage: "Baseline Extraction",
      done: Number(state.summary.baseline_documents || 0) > 0 || Boolean(output.get("data/runs/baseline_results.json")?.exists),
      artifact: "data/runs/baseline_results.json",
      command: "contract-intel baseline --model qwen3:4b"
    },
    {
      step: "5",
      stage: "Agent Analysis",
      done: Boolean(output.get("data/runs/agent_analysis.json")?.exists),
      artifact: "data/runs/agent_analysis.json",
      command: "contract-intel agent-analyze --model qwen3:4b"
    },
    {
      step: "6",
      stage: "Review Packet",
      done: Boolean(output.get("data/reviews/review_packet.pending.json")?.exists),
      artifact: "data/reviews/review_packet.pending.json",
      command: "contract-intel review-packet"
    },
    {
      step: "7",
      stage: "Apply Reviewed Labels",
      done: Number(state.summary.reviewed || 0) > 0 || Boolean(output.get("data/memory/taxonomy.json")?.exists),
      artifact: "data/memory/taxonomy.json",
      command: "contract-intel cuad-apply-gold && contract-intel apply-review --review data/reviews/review_packet.reviewed.json"
    },
    {
      step: "8",
      stage: "Export Reviewed Examples",
      done: Number(state.summary.training_pairs || 0) > 0 || Boolean(output.get("data/training/training_pairs.jsonl")?.exists),
      artifact: "data/training/training_pairs.jsonl",
      command: "contract-intel apply-review --review data/reviews/review_packet.reviewed.json"
    },
    {
      step: "9",
      stage: "Second Run",
      done: Number(state.summary.second_run_documents || 0) > 0 || Boolean(output.get("data/runs/second_run_results.json")?.exists),
      artifact: "data/runs/second_run_results.json",
      command: "contract-intel second-run --model qwen3:4b"
    },
    {
      step: "10",
      stage: "Benchmark",
      done: Boolean(output.get("data/runs/benchmark.json")?.exists),
      artifact: "data/runs/benchmark.json",
      command: "contract-intel benchmark"
    },
    {
      step: "11",
      stage: "Demo Report",
      done: Boolean(output.get("data/runs/demo_report.md")?.exists),
      artifact: "data/runs/demo_report.md",
      command: "contract-intel demo-report"
    }
  ];
}

function outputFileMap() {
  return new Map((state.output_files || []).map((file) => [file.path, file]));
}

function renderIntelligence() {
  const docs = state.documents || [];
  const contractMix = contractMixRows(docs);
  const clausePrevalence = clausePrevalenceRows(docs);
  const missingFields = missingFieldRows(docs);
  const agentFlags = docs.filter((doc) => ["high", "medium"].includes(doc.agent_analysis?.review_priority)).length;
  const misses = (state.benchmark.rows || []).filter((row) => row.second_run_correct === false).length;
  html("intelligenceSummary", `
    <span>${docs.length} contracts</span>
    <span>${contractMix.length} contract types</span>
    <span>${clausePrevalence.length} clause families</span>
    <span>${agentFlags} agent flags</span>
    <span>${misses} benchmark misses</span>
  `);
  renderQuestionRows(docs, contractMix, clausePrevalence, missingFields, agentFlags, misses);
  renderContractMix(contractMix, docs.length);
  renderClausePrevalence(clausePrevalence, docs.length);
  renderMissingFields(missingFields, docs.length);
}

function renderQuestionRows(docs, contractMix, clausePrevalence, missingFields, agentFlags, misses) {
  const topType = contractMix[0];
  const topClause = clausePrevalence[0];
  const topMissing = missingFields[0];
  const benchmark = state.summary || {};
  const rows = [
    ["What contract types are in this corpus?", topType ? `${topType.label} is most common: ${topType.count}/${docs.length}` : "No contracts loaded."],
    ["Which clause family appears most often?", topClause ? `${topClause.family}: ${topClause.doc_count}/${docs.length} contracts` : "No reviewed clauses loaded."],
    ["Which coversheet field needs the most cleanup?", topMissing ? `${topMissing.label}: missing in ${topMissing.missing}/${docs.length} contracts` : "No coversheet fields loaded."],
    ["Which contracts should be reviewed first?", `${agentFlags} agent-flagged contracts and ${misses} benchmark misses.`],
    ["Did reviewed context improve contract type extraction?", `${pct(benchmark.baseline_accuracy)} baseline -> ${pct(benchmark.second_accuracy)} second run.`],
    ["What can this corpus support next?", "Risk hotspot review, clause prevalence reporting, taxonomy expansion, and benchmark trend analysis."]
  ];
  html("questionRows", `
    <div class="intelligenceHeader questionMatrix">
      <span>Question</span>
      <span>Current Answer</span>
    </div>
    ${rows.map(([question, answer]) => `
      <div class="intelligenceRow questionMatrix">
        <strong>${escape(question)}</strong>
        <span>${escape(answer)}</span>
      </div>
    `).join("")}
  `);
}

function renderContractMix(rows, totalDocs) {
  html("contractMixRows", `
    <div class="intelligenceHeader">
      <span>Contract Type</span>
      <span>Count</span>
      <span>Share</span>
      <span>Examples</span>
    </div>
    ${rows.map((row) => `
      <div class="intelligenceRow">
        <strong>${escape(row.label)}</strong>
        <span>${row.count}</span>
        <span>${pctFraction(row.count, totalDocs)}</span>
        <span title="${escape(row.examples.join(" | "))}">${escape(row.examples.join(" | ") || "-")}</span>
      </div>
    `).join("")}
  `);
}

function renderClausePrevalence(rows, totalDocs) {
  html("clausePrevalenceRows", `
    <div class="intelligenceHeader clauseIntelMatrix">
      <span>Clause Family</span>
      <span>Contracts</span>
      <span>Prevalence</span>
      <span>Positive Evidence</span>
      <span>Categories</span>
    </div>
    ${rows.map((row) => `
      <div class="intelligenceRow clauseIntelMatrix">
        <strong>${escape(row.family)}</strong>
        <span>${row.doc_count}</span>
        <span>${pctFraction(row.doc_count, totalDocs)}</span>
        <span>${row.positive_count}</span>
        <span title="${escape(row.categories.join(", "))}">${escape(row.categories.join(", ") || "-")}</span>
      </div>
    `).join("")}
  `);
}

function renderMissingFields(rows, totalDocs) {
  html("missingFieldRows", `
    <div class="intelligenceHeader missingIntelMatrix">
      <span>Field</span>
      <span>Missing</span>
      <span>Missing Share</span>
      <span>Needs Review Examples</span>
    </div>
    ${rows.map((row) => `
      <div class="intelligenceRow missingIntelMatrix">
        <strong>${escape(row.label)}</strong>
        <span>${row.missing}</span>
        <span>${pctFraction(row.missing, totalDocs)}</span>
        <span title="${escape(row.examples.join(" | "))}">${escape(row.examples.join(" | ") || "-")}</span>
      </div>
    `).join("")}
  `);
}

function contractMixRows(docs) {
  const groups = new Map();
  for (const doc of docs) {
    const label = doc.gold_type || doc.second_type || doc.baseline_type || "Unknown";
    if (!groups.has(label)) groups.set(label, { label, count: 0, examples: [] });
    const group = groups.get(label);
    group.count += 1;
    if (group.examples.length < 3) group.examples.push(doc.title);
  }
  return [...groups.values()].sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
}

function clausePrevalenceRows(docs) {
  const groups = new Map();
  for (const doc of docs) {
    const seenFamilies = new Set();
    for (const clause of doc.key_clauses || []) {
      const family = clause.family || clause.cuad_category || "other";
      if (!groups.has(family)) groups.set(family, { family, doc_ids: new Set(), positive_count: 0, categories: new Set() });
      const group = groups.get(family);
      if (!seenFamilies.has(family)) {
        group.doc_ids.add(doc.doc_id);
        seenFamilies.add(family);
      }
      if ((clause.answer && clause.answer !== "No") || clause.evidence) group.positive_count += 1;
      if (clause.cuad_category) group.categories.add(clause.cuad_category);
    }
  }
  return [...groups.values()]
    .map((group) => ({
      family: group.family,
      doc_count: group.doc_ids.size,
      positive_count: group.positive_count,
      categories: [...group.categories].slice(0, 5)
    }))
    .sort((left, right) => right.doc_count - left.doc_count || right.positive_count - left.positive_count || left.family.localeCompare(right.family));
}

function missingFieldRows(docs) {
  const fields = [
    ["parties", "Parties"],
    ["agreement_date", "Agreement Date"],
    ["effective_date", "Effective Date"],
    ["expiration_date", "Expiration Date"],
    ["territory", "Territory"],
    ["governing_law", "Governing Law"]
  ];
  return fields.map(([field, label]) => {
    const missingDocs = docs.filter((doc) => !fieldValue(doc.coversheet?.[field]));
    return {
      field,
      label,
      missing: missingDocs.length,
      examples: missingDocs.slice(0, 3).map((doc) => doc.title)
    };
  }).sort((left, right) => right.missing - left.missing || left.label.localeCompare(right.label));
}

function renderQueueFilters() {
  const typeSelect = document.getElementById("queueType");
  const current = typeSelect.value;
  const types = [...new Set((state.documents || []).map((doc) => doc.gold_type || doc.second_type || doc.baseline_type).filter(Boolean))].sort();
  typeSelect.innerHTML = `<option value="">All types</option>` + types.map((type) => `<option value="${escape(type)}">${escape(type)}</option>`).join("");
  typeSelect.value = types.includes(current) ? current : "";
}

function renderQueue() {
  if (!state) return;
  const controls = currentQueueControls();
  queuePage = resetPageWhenFiltersChange("queue", [controls.query, controls.status, controls.type, controls.sortKey, controls.sortDir], queuePage);
  const rows = queueFilteredRows(controls);
  const page = paginateRows(rows, queuePage, Number(document.getElementById("queuePageSize").value || 25));
  queuePage = page.page;
  renderPager("queue", page);
  const visibleIds = new Set(page.rows.map((row) => row.doc_id));
  if (!selectedQueueDocId || !visibleIds.has(selectedQueueDocId)) {
    selectedQueueDocId = page.rows[0]?.doc_id || null;
  }

  html("queueSummary", `
    <span>${rows.length} visible</span>
    <span>${state.documents.length} total</span>
    <span>${rows.filter((row) => ["high", "medium"].includes(row.agent_priority)).length} agent flags</span>
    <span>${rows.filter((row) => !row.second_correct).length} misses</span>
  `);

  html("queueTable", `
    <div class="queueHeader">
      <span>Doc</span>
      <span>Reviewed Type</span>
      <span>Baseline</span>
      <span>Second</span>
      <span>Agent</span>
      <span>Cover</span>
      <span>Clause F1</span>
      <span>Source</span>
    </div>
    ${page.rows.map((row) => `
      <button class="queueRow ${row.doc_id === selectedQueueDocId ? "selected" : ""}" type="button" data-doc-id="${escape(row.doc_id)}">
        <span title="${escape(row.title)}">${escape(row.title)}</span>
        <strong>${escape(row.type || "-")}</strong>
        <span>${escape(row.baseline || "-")}</span>
        <span class="${row.second_correct ? "queueGood" : "queueWarn"}">${escape(row.second || "-")}</span>
        <span>${escape(row.agent_priority || "-")}</span>
        <span>${pct(row.cover_after)}</span>
        <span>${pct(row.clause_after)}</span>
        <span title="${escape(row.source_path)}">${escape(shortPath(row.source_path))}</span>
      </button>
    `).join("")}
  `);
  document.querySelectorAll(".queueRow").forEach((node) => {
    node.addEventListener("click", () => {
      selectedQueueDocId = node.dataset.docId;
      renderQueue();
    });
  });
  renderQueueDetail(page.rows.find((row) => row.doc_id === selectedQueueDocId));
}

function currentQueueControls() {
  return {
    query: document.getElementById("queueSearch").value.trim().toLowerCase(),
    status: document.getElementById("queueStatus").value,
    type: document.getElementById("queueType").value,
    sortKey: document.getElementById("queueSort").value,
    sortDir: document.getElementById("queueDir").value
  };
}

function queueFilteredRows(controls) {
  return sortRows(queueRows().filter((row) => {
    const haystack = [row.title, row.source_path, row.type, row.baseline, row.second, row.engine, row.agent].join(" ").toLowerCase();
    if (controls.query && !haystack.includes(controls.query)) return false;
    if (controls.type && row.type !== controls.type) return false;
    if (controls.status === "reviewed" && row.review_status !== "reviewed") return false;
    if (controls.status === "needs_review" && row.review_status === "reviewed" && row.second_correct) return false;
    if (controls.status === "agent_flag" && !["high", "medium"].includes(row.agent_priority)) return false;
    if (controls.status === "miss" && row.second_correct) return false;
    return true;
  }), controls.sortKey, controls.sortDir);
}

function exportQueueCsv() {
  const rows = queueFilteredRows(currentQueueControls()).map((row) => ({
    doc_id: row.doc_id,
    title: row.title,
    reviewed_type: row.type,
    baseline_type: row.baseline,
    second_run_type: row.second,
    engine: row.engine,
    review_status: row.review_status,
    agent_priority: row.agent_priority,
    coversheet_score: pct(row.cover_after),
    clause_f1: pct(row.clause_after),
    source_path: row.source_path
  }));
  downloadCsv(`contract_queue_${timestampSlug()}.csv`, rows);
  setStatus(`Exported ${rows.length} queue rows.`);
}

function applyQueuePreset(name) {
  selectedQueueDocId = null;
  queuePage = 1;
  setControl("queueSearch", "");
  setControl("queueType", "");
  setControl("queueDir", "asc");
  if (name === "agent_flag") {
    setControl("queueStatus", "agent_flag");
    setControl("queueSort", "agent_priority");
    setControl("queueDir", "desc");
  } else if (name === "miss") {
    setControl("queueStatus", "miss");
    setControl("queueSort", "title");
  } else if (name === "needs_review") {
    setControl("queueStatus", "needs_review");
    setControl("queueSort", "title");
  } else if (name === "lowest_clause") {
    setControl("queueStatus", "");
    setControl("queueSort", "clause_after");
  } else {
    setControl("queueStatus", "");
    setControl("queueSort", "title");
  }
  renderQueue();
}

function queueRows() {
  const benchmarkByDoc = new Map((state.benchmark.rows || []).map((row) => [row.doc_id, row]));
  return (state.documents || []).map((doc) => {
    const bench = benchmarkByDoc.get(doc.doc_id) || {};
    return {
      doc_id: doc.doc_id,
      title: doc.title,
      source_path: doc.source_path,
      type: doc.gold_type || doc.second_type || doc.baseline_type,
      baseline: doc.baseline_type,
      second: doc.second_type,
      engine: doc.second_engine || doc.baseline_engine,
      review_status: doc.review_status ? "reviewed" : "needs_review",
      agent: doc.agent_analysis?.challenge_summary || "",
      agent_priority: doc.agent_analysis?.review_priority || "",
      second_correct: bench.second_run_correct !== false,
      cover_after: bench.second_run_coversheet_field_accuracy,
      clause_after: bench.second_run_clause_family_f1
    };
  });
}

function renderQueueDetail(row) {
  if (!row) {
    html("queueDetail", `<p class="muted">No document matches the current queue filters.</p>`);
    return;
  }
  const doc = findDocument(row.doc_id);
  const bench = findBenchmark(row.doc_id);
  const coversheet = doc?.coversheet || {};
  const fields = [
    ["Parties", fieldValue(coversheet.parties)],
    ["Agreement Date", fieldValue(coversheet.agreement_date)],
    ["Effective Date", fieldValue(coversheet.effective_date)],
    ["Expiration", fieldValue(coversheet.expiration_date)],
    ["Territory", fieldValue(coversheet.territory)],
    ["Governing Law", fieldValue(coversheet.governing_law)]
  ];
  const clauses = (doc?.key_clauses || [])
    .filter((clause) => clause.evidence || clause.answer)
    .slice(0, 5);
  html("queueDetail", `
    <div class="queueDetailHeader">
      <h3>${escape(row.title)}</h3>
      <span class="badge ${row.review_status === "reviewed" ? "good" : "warn"}">${escape(row.review_status)}</span>
    </div>
    <p class="muted">${escape(row.source_path)}</p>
    <div class="labels">
      <span class="badge accent">${escape(row.engine || "engine unknown")}</span>
      <span class="badge ${priorityClass(row.agent_priority)}">Agent: ${escape(row.agent_priority || "none")}</span>
      <span class="badge ${row.second_correct ? "good" : "warn"}">${row.second_correct ? "Benchmark pass" : "Benchmark miss"}</span>
    </div>
    <div class="detailMetrics">
      <div><span>Reviewed</span><strong>${escape(row.type || "-")}</strong></div>
      <div><span>Baseline</span><strong>${escape(row.baseline || "-")}</strong></div>
      <div><span>Second Run</span><strong>${escape(row.second || "-")}</strong></div>
      <div><span>Cover</span><strong>${pct(row.cover_after)}</strong></div>
      <div><span>Clause F1</span><strong>${pct(row.clause_after)}</strong></div>
      <div><span>Before F1</span><strong>${pct(bench?.baseline_clause_family_f1)}</strong></div>
    </div>
    ${doc?.agent_analysis?.challenge_summary ? `
      <div class="detailSection">
        <strong>Agent Challenge</strong>
        <p>${escape(doc.agent_analysis.challenge_summary)}</p>
      </div>
    ` : ""}
    <div class="detailSection">
      <strong>Coversheet</strong>
      <div class="detailFields">
        ${fields.map(([label, value]) => `
          <div>
            <span>${escape(label)}</span>
            <strong>${escape(value || "-")}</strong>
          </div>
        `).join("")}
      </div>
    </div>
    <div class="detailSection">
      <strong>Top Clauses</strong>
      ${clauses.map((clause) => `
        <div class="detailClause">
          <span>${escape(clause.family || clause.cuad_category || "clause")}</span>
          <p>${escape(clause.evidence || clause.answer || "")}</p>
        </div>
      `).join("") || `<p class="muted">No reviewed clause evidence yet.</p>`}
    </div>
  `);
}

function findDocument(docId) {
  return (state.documents || []).find((doc) => doc.doc_id === docId) || null;
}

function findBenchmark(docId) {
  return (state.benchmark.rows || []).find((row) => row.doc_id === docId) || null;
}

function renderWorkbenchFilters() {
  const typeSelect = document.getElementById("workbenchType");
  const current = typeSelect.value;
  const types = [...new Set((state.documents || []).map((doc) => doc.gold_type || doc.second_type || doc.baseline_type).filter(Boolean))].sort();
  typeSelect.innerHTML = `<option value="">All types</option>` + types.map((type) => `<option value="${escape(type)}">${escape(type)}</option>`).join("");
  typeSelect.value = types.includes(current) ? current : "";
}

function renderWorkbench() {
  if (!state) return;
  const controls = currentWorkbenchControls();
  workbenchPage = resetPageWhenFiltersChange("workbench", [controls.mode, controls.query, controls.status, controls.type, controls.sortKey, controls.sortDir], workbenchPage);
  const { allRows, rows } = workbenchFilteredRows(controls);
  const page = paginateRows(rows, workbenchPage, Number(document.getElementById("workbenchPageSize").value || 25));
  workbenchPage = page.page;
  renderPager("workbench", page);
  const visibleKeys = new Set(page.rows.map((row) => row.key));
  if (!selectedWorkbenchKey || !visibleKeys.has(selectedWorkbenchKey)) {
    selectedWorkbenchKey = page.rows[0]?.key || null;
  }
  html("workbenchSummary", `
    <span>${rows.length} visible</span>
    <span>${allRows.length} total</span>
    <span>${rows.filter((row) => row.evidence).length} with evidence</span>
    <span>${rows.filter((row) => !row.evidence).length} needs evidence</span>
  `);
  html("workbenchTable", controls.mode === "coversheet" ? renderCoversheetMatrix(page.rows) : renderClauseMatrix(page.rows));
  document.querySelectorAll(".workbenchRow").forEach((node) => {
    node.addEventListener("click", () => {
      selectedWorkbenchKey = node.dataset.key;
      renderWorkbench();
    });
  });
  renderWorkbenchDetail(page.rows.find((row) => row.key === selectedWorkbenchKey));
}

function currentWorkbenchControls() {
  return {
    mode: document.getElementById("workbenchMode").value,
    query: document.getElementById("workbenchSearch").value.trim().toLowerCase(),
    status: document.getElementById("workbenchStatus").value,
    type: document.getElementById("workbenchType").value,
    sortKey: document.getElementById("workbenchSort").value,
    sortDir: document.getElementById("workbenchDir").value
  };
}

function workbenchFilteredRows(controls) {
  const allRows = workbenchRows(controls.mode);
  const rows = sortRows(allRows.filter((row) => {
    const haystack = [row.title, row.contract_type, row.label, row.value, row.evidence, row.authority, row.answer, row.category].join(" ").toLowerCase();
    if (controls.query && !haystack.includes(controls.query)) return false;
    if (controls.type && row.contract_type !== controls.type) return false;
    if (controls.status === "has_evidence" && !row.evidence) return false;
    if (controls.status === "needs_evidence" && row.evidence) return false;
    if (controls.status === "agent_flag" && !["high", "medium"].includes(row.agent_priority)) return false;
    if (controls.status === "miss" && row.second_correct) return false;
    return true;
  }), controls.sortKey, controls.sortDir);
  return { allRows, rows };
}

function exportWorkbenchCsv() {
  const { rows } = workbenchFilteredRows(currentWorkbenchControls());
  const csvRows = rows.map((row) => ({
    mode: row.mode,
    doc_id: row.doc_id,
    title: row.title,
    contract_type: row.contract_type,
    label: row.label,
    category: row.category || "",
    value: row.value || row.answer || "",
    evidence: row.evidence,
    authority: row.authority,
    agent_priority: row.agent_priority,
    benchmark_status: row.second_correct ? "pass" : "miss",
    source_path: row.source_path
  }));
  downloadCsv(`review_workbench_${timestampSlug()}.csv`, csvRows);
  setStatus(`Exported ${csvRows.length} workbench rows.`);
}

function applyWorkbenchPreset(name) {
  selectedWorkbenchKey = null;
  workbenchPage = 1;
  setControl("workbenchSearch", "");
  setControl("workbenchType", "");
  setControl("workbenchDir", "asc");
  if (name === "needs_evidence") {
    setControl("workbenchMode", "coversheet");
    setControl("workbenchStatus", "needs_evidence");
    setControl("workbenchSort", "evidence_status");
  } else if (name === "agent_flag") {
    setControl("workbenchMode", "coversheet");
    setControl("workbenchStatus", "agent_flag");
    setControl("workbenchSort", "contract_type");
  } else if (name === "miss") {
    setControl("workbenchMode", "coversheet");
    setControl("workbenchStatus", "miss");
    setControl("workbenchSort", "title");
  } else if (name === "clauses") {
    setControl("workbenchMode", "clauses");
    setControl("workbenchStatus", "");
    setControl("workbenchSort", "label");
  } else {
    setControl("workbenchMode", "coversheet");
    setControl("workbenchStatus", "");
    setControl("workbenchSort", "title");
  }
  renderWorkbench();
}

function workbenchRows(mode) {
  const benchmarkByDoc = new Map((state.benchmark.rows || []).map((row) => [row.doc_id, row]));
  if (mode === "clauses") {
    return (state.documents || []).flatMap((doc) => {
      const bench = benchmarkByDoc.get(doc.doc_id) || {};
      return (doc.key_clauses || []).map((clause, index) => ({
        key: `${doc.doc_id}:clause:${index}`,
        mode,
        doc_id: doc.doc_id,
        title: doc.title,
        source_path: doc.source_path,
        contract_type: doc.gold_type || doc.second_type || doc.baseline_type,
        label: clause.family || clause.cuad_category || "clause",
        category: clause.cuad_category || "",
        answer: clause.answer || "",
        value: clause.answer || "",
        evidence: clause.evidence || "",
        authority: clause.authority || clause.cuad_category || "",
        confidence: clause.confidence,
        agent_priority: doc.agent_analysis?.review_priority || "",
        second_correct: bench.second_run_correct !== false
      }));
    });
  }
  const fieldDefs = [
    ["parties", "Parties"],
    ["agreement_date", "Agreement Date"],
    ["effective_date", "Effective Date"],
    ["expiration_date", "Expiration Date"],
    ["territory", "Territory"],
    ["governing_law", "Governing Law"]
  ];
  return (state.documents || []).flatMap((doc) => {
    const bench = benchmarkByDoc.get(doc.doc_id) || {};
    const coversheet = doc.coversheet || {};
    return fieldDefs.map(([field, label]) => {
      const raw = coversheet[field];
      return {
        key: `${doc.doc_id}:field:${field}`,
        mode,
        doc_id: doc.doc_id,
        title: doc.title,
        source_path: doc.source_path,
        contract_type: doc.gold_type || doc.second_type || doc.baseline_type,
        label,
        field,
        value: fieldValue(raw),
        evidence: raw && typeof raw === "object" && !Array.isArray(raw) ? raw.evidence || "" : "",
        authority: raw && typeof raw === "object" && !Array.isArray(raw) ? raw.authority || "" : "",
        agent_priority: doc.agent_analysis?.review_priority || "",
        second_correct: bench.second_run_correct !== false
      };
    });
  });
}

function renderCoversheetMatrix(rows) {
  return `
    <div class="workbenchHeader">
      <span>Document</span>
      <span>Type</span>
      <span>Field</span>
      <span>Reviewed Value</span>
      <span>Evidence</span>
      <span>Authority</span>
    </div>
    ${rows.map((row) => `
      <button class="workbenchRow ${row.key === selectedWorkbenchKey ? "selected" : ""}" type="button" data-key="${escape(row.key)}">
        <span title="${escape(row.title)}">${escape(row.title)}</span>
        <strong>${escape(row.contract_type || "-")}</strong>
        <span>${escape(row.label)}</span>
        <span title="${escape(row.value)}">${escape(row.value || "-")}</span>
        <span class="${row.evidence ? "queueGood" : "queueWarn"}" title="${escape(row.evidence)}">${escape(row.evidence || "Needs evidence")}</span>
        <span>${escape(row.authority || "-")}</span>
      </button>
    `).join("")}
  `;
}

function renderClauseMatrix(rows) {
  return `
    <div class="workbenchHeader clauseMatrix">
      <span>Document</span>
      <span>Type</span>
      <span>Family</span>
      <span>CUAD Category</span>
      <span>Answer</span>
      <span>Evidence</span>
      <span>Authority</span>
    </div>
    ${rows.map((row) => `
      <button class="workbenchRow clauseMatrix ${row.key === selectedWorkbenchKey ? "selected" : ""}" type="button" data-key="${escape(row.key)}">
        <span title="${escape(row.title)}">${escape(row.title)}</span>
        <strong>${escape(row.contract_type || "-")}</strong>
        <span>${escape(row.label)}</span>
        <span title="${escape(row.category)}">${escape(row.category || "-")}</span>
        <span>${escape(row.answer || "-")}</span>
        <span class="${row.evidence ? "queueGood" : "queueWarn"}" title="${escape(row.evidence)}">${escape(row.evidence || "Needs evidence")}</span>
        <span>${escape(row.authority || "-")}</span>
      </button>
    `).join("")}
  `;
}

function renderWorkbenchDetail(row) {
  if (!row) {
    html("workbenchDetail", `<p class="muted">No review item matches the current filters.</p>`);
    return;
  }
  const doc = findDocument(row.doc_id);
  html("workbenchDetail", `
    <div class="queueDetailHeader">
      <h3>${escape(row.label)}</h3>
      <span class="badge ${row.evidence ? "good" : "warn"}">${row.evidence ? "Evidence" : "Needs evidence"}</span>
    </div>
    <p class="muted">${escape(row.title)}</p>
    <div class="labels">
      <span class="badge accent">${escape(row.contract_type || "-")}</span>
      <span class="badge ${priorityClass(row.agent_priority)}">Agent: ${escape(row.agent_priority || "none")}</span>
      <span class="badge ${row.second_correct ? "good" : "warn"}">${row.second_correct ? "Benchmark pass" : "Benchmark miss"}</span>
    </div>
    <div class="detailSection">
      <strong>Reviewed Value</strong>
      <p>${escape(row.value || row.answer || "-")}</p>
    </div>
    ${row.category ? `
      <div class="detailSection">
        <strong>Category</strong>
        <p>${escape(row.category)}</p>
      </div>
    ` : ""}
    <div class="detailSection">
      <strong>Evidence</strong>
      <p>${escape(row.evidence || "No evidence span recorded for this reviewed item.")}</p>
    </div>
    <div class="detailSection">
      <strong>Authority</strong>
      <p>${escape(row.authority || "No authority recorded.")}</p>
    </div>
    ${doc?.agent_analysis?.challenge_summary ? `
      <div class="detailSection">
        <strong>Agent Challenge</strong>
        <p>${escape(doc.agent_analysis.challenge_summary)}</p>
      </div>
    ` : ""}
    <div class="detailSection">
      <strong>Source</strong>
      <p>${escape(row.source_path)}</p>
    </div>
  `);
}

function sortRows(rows, key, direction) {
  const multiplier = direction === "desc" ? -1 : 1;
  return [...rows].sort((left, right) => compareValues(sortValue(left, key), sortValue(right, key)) * multiplier);
}

function sortValue(row, key) {
  if (key === "agent_priority") {
    return { high: 3, medium: 2, low: 1 }[row.agent_priority] || 0;
  }
  if (key === "evidence_status") return row.evidence ? 1 : 0;
  if (key === "cover_after" || key === "clause_after") return Number(row[key] ?? -1);
  return row[key] ?? "";
}

function compareValues(left, right) {
  if (typeof left === "number" && typeof right === "number") return left - right;
  return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: "base" });
}

const pageFilterCache = {};

function resetPageWhenFiltersChange(scope, values, currentPage) {
  const signature = values.join("|");
  if (pageFilterCache[scope] !== signature) {
    pageFilterCache[scope] = signature;
    return 1;
  }
  return currentPage;
}

function paginateRows(rows, requestedPage, pageSize) {
  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const page = Math.min(Math.max(1, requestedPage), totalPages);
  const start = (page - 1) * pageSize;
  return {
    rows: rows.slice(start, start + pageSize),
    page,
    pageSize,
    totalRows: rows.length,
    totalPages,
    start: rows.length ? start + 1 : 0,
    end: Math.min(start + pageSize, rows.length)
  };
}

function renderPager(scope, page) {
  text(`${scope}PageInfo`, `${page.start}-${page.end} of ${page.totalRows} · page ${page.page}/${page.totalPages}`);
  const prev = document.getElementById(`${scope}Prev`);
  const next = document.getElementById(`${scope}Next`);
  if (prev) prev.disabled = page.page <= 1;
  if (next) next.disabled = page.page >= page.totalPages;
}

function downloadCsv(filename, rows) {
  const csv = toCsv(rows);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function toCsv(rows) {
  if (!rows.length) return "";
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  return [
    columns.map(csvCell).join(","),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(","))
  ].join("\n");
}

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return `"${text.replace(/"/g, '""')}"`;
}

function timestampSlug() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function setControl(id, value) {
  const node = document.getElementById(id);
  if (node) node.value = value;
}

function renderCorpus() {
  const manifest = state.ingest_manifest || {};
  const extensions = Object.entries(manifest.extensions || {});
  html("ingestSummary", `
    <div class="row">
      <div class="docGrid">
        <div class="field"><span>Scanned</span><strong>${escape(manifest.scanned_files ?? 0)}</strong></div>
        <div class="field"><span>Ingested</span><strong>${escape(manifest.ingested_files ?? state.summary.documents ?? 0)}</strong></div>
        <div class="field"><span>Skipped</span><strong>${escape(manifest.skipped_files ?? 0)}</strong></div>
        <div class="field"><span>Input Dir</span><strong>${escape(manifest.input_dir || "-")}</strong></div>
      </div>
      <div class="labels">
        ${(manifest.supported_inputs || []).map((item) => `<span class="badge accent">${escape(item)}</span>`).join("")}
      </div>
      <p class="muted">${escape(manifest.pdf_ocr_boundary || "PDF OCR is out of scope for this MVP.")}</p>
    </div>
    <div class="row">
      <div class="rowTitle">
        <span>Extensions</span>
      </div>
      <div class="labels">
        ${extensions.map(([ext, count]) => `<span class="badge">${escape(ext)}: ${escape(count)}</span>`).join("")}
      </div>
    </div>
  `);
  html("ingestFiles", (manifest.files || []).map((file) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(file.title || file.path)}</span>
        <span class="badge ${file.status === "ingested" ? "good" : "warn"}">${escape(file.status || "unknown")}</span>
      </div>
      <p class="muted">${escape(file.path)}</p>
      <div class="labels">
        <span class="badge">${escape(file.extension || "-")}</span>
        ${file.characters ? `<span class="badge accent">${escape(file.characters)} chars</span>` : ""}
        ${file.reason ? `<span class="badge warn">${escape(file.reason)}</span>` : ""}
      </div>
    </div>
  `).join(""));
}

function renderDocuments() {
  html("documentList", state.documents.map((doc) => `
    <article class="docCard">
      <div class="docTitle">
        <span>${escape(doc.title)}</span>
        <span class="badge ${doc.second_type === doc.gold_type ? "good" : "warn"}">${escape(doc.review_status || "reviewed")}</span>
      </div>
      <div class="muted">${escape(doc.source_path)}</div>
      <div class="docGrid">
        <div class="field"><span>Baseline</span><strong>${escape(doc.baseline_type)} (${escape(doc.baseline_engine)})</strong></div>
        <div class="field"><span>Reviewed Label</span><strong>${escape(doc.gold_type)}</strong></div>
        <div class="field"><span>Second Run</span><strong>${escape(doc.second_type)} (${escape(doc.second_engine)})</strong></div>
        <div class="field"><span>Agent Priority</span><strong>${escape(doc.agent_analysis?.review_priority || "-")}</strong></div>
      </div>
      <details>
        <summary>Preview</summary>
        <pre>${escape(doc.text_preview)}</pre>
      </details>
    </article>
  `).join(""));
}

function renderCoversheets() {
  html("coversheetList", state.documents.map((doc) => {
    const fields = [
      ["Contract Type", doc.gold_type || doc.second_type || doc.baseline_type, "Reviewed label"],
      ["Parties", fieldValue(doc.coversheet?.parties), fieldEvidence(doc.coversheet?.parties)],
      ["Agreement Date", fieldValue(doc.coversheet?.agreement_date), fieldEvidence(doc.coversheet?.agreement_date)],
      ["Effective Date", fieldValue(doc.coversheet?.effective_date), fieldEvidence(doc.coversheet?.effective_date)],
      ["Expiration Date", fieldValue(doc.coversheet?.expiration_date), fieldEvidence(doc.coversheet?.expiration_date)],
      ["Territory", fieldValue(doc.coversheet?.territory), fieldEvidence(doc.coversheet?.territory)],
      ["Governing Law", fieldValue(doc.coversheet?.governing_law), fieldEvidence(doc.coversheet?.governing_law)]
    ];
    const clauses = (doc.key_clauses || []).filter((clause) => clause.evidence || clause.answer).slice(0, 8);
    return `
      <article class="coverSheet">
        <div class="coverHeader">
          <div>
            <h3>${escape(doc.title)}</h3>
            <p class="muted">${escape(doc.source_path)}</p>
          </div>
          <span class="badge ${doc.second_type === doc.gold_type ? "good" : "warn"}">${escape(doc.review_status || "reviewed")}</span>
        </div>
        <div class="coverMeta">
          ${fields.map(([label, value, evidence]) => `
            <div class="coverField">
              <span>${escape(label)}</span>
              <strong>${escape(value || "-")}</strong>
              ${evidence ? `<p>${escape(evidence)}</p>` : ""}
            </div>
          `).join("")}
        </div>
        ${doc.agent_analysis?.challenge_summary ? `
          <div class="callout">
            <strong>Agent Challenge</strong>
            <p>${escape(doc.agent_analysis.challenge_summary)}</p>
          </div>
        ` : ""}
        <div class="clauseGrid">
          ${clauses.map((clause) => `
            <div class="clauseCard">
              <div class="rowTitle">
                <span>${escape(clause.family || clause.cuad_category || "clause")}</span>
                <span class="badge">${escape(clause.authority || clause.cuad_category || "evidence")}</span>
              </div>
              <p class="muted">${escape(clause.answer ? `Answer: ${clause.answer}` : "")}</p>
              <p>${escape(clause.evidence || "")}</p>
            </div>
          `).join("") || `<p class="muted">No reviewed clause evidence yet.</p>`}
        </div>
      </article>
    `;
  }).join(""));
}

function renderReview() {
  const reviewed = (state.reviewed.items && state.reviewed.items.length ? state.reviewed.items : state.pending_review.items) || [];
  html("reviewItems", `<p id="reviewStatus" class="statusLine"></p>` + reviewed.map((item) => `
    <div class="row reviewEdit" data-doc-id="${escape(item.doc_id)}">
      <div class="rowTitle">
        <span>${escape(item.title)}</span>
        <span class="badge accent">${escape(item.reviewer_authority || "business_sme_confirmed")}</span>
      </div>
      <div class="editGrid">
        <label>Status
          <select data-field="status">
            ${option("pending", item.status)}
            ${option("accepted", item.status)}
            ${option("edited", item.status)}
            ${option("rejected", item.status)}
          </select>
        </label>
        <label>Accepted Contract Type
          <input data-field="accepted_contract_type" value="${escape(item.accepted_contract_type || item.proposed_contract_type || "")}">
        </label>
      </div>
      <label class="reviewNotes">Review Notes
        <textarea data-field="review_notes">${escape(item.review_notes || "")}</textarea>
      </label>
      <div class="labels">
        <span class="badge warn">Proposed: ${escape(item.proposed_contract_type)}</span>
        ${(item.alternative_contract_types || []).slice(0, 2).map((alt) => `<span class="badge accent">Alt: ${escape(alt.label)}</span>`).join("")}
      </div>
      ${item.agent_analysis?.challenge_summary ? `<p class="muted">${escape(item.agent_analysis.challenge_summary)}</p>` : ""}
    </div>
  `).join(""));

  const clauses = reviewed.flatMap((item) => (item.accepted_key_clauses || []).slice(0, 8).map((clause) => ({ title: item.title, ...clause })));
  html("clauseItems", clauses.slice(0, 40).map((clause) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(clause.family || clause.cuad_category)}</span>
        <span class="badge">${escape(clause.cuad_category || "model")}</span>
      </div>
      <p class="muted">${escape(clause.title)}</p>
      <p>${escape(clause.evidence || clause.answer || "")}</p>
    </div>
  `).join(""));
}

function renderAgent() {
  const findings = state.agent_analysis || [];
  html("agentFindings", findings.map((item) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(item.title)}</span>
        <span class="badge ${priorityClass(item.review_priority)}">${escape(item.review_priority || "unscored")}</span>
      </div>
      <p>${escape(item.challenge_summary || "")}</p>
      ${renderNamedList("Alternatives", item.alternative_contract_types, (alt) => `${alt.label}: ${alt.reason}${alt.evidence ? ` | ${alt.evidence}` : ""}`)}
      ${renderNamedList("Missing Elements", item.missing_expected_elements, (gap) => `${gap.element}: ${gap.why_it_matters}`)}
      ${renderNamedList("Evidence Gaps", item.evidence_gaps, (gap) => gap)}
      ${renderNamedList("Reviewer Questions", item.reviewer_questions, (question) => question)}
      <div class="labels">
        <span class="badge accent">${escape(item.engine || "unknown")}</span>
        <span class="badge">${escape(item.run || "baseline")}</span>
      </div>
    </div>
  `).join(""));
}

function renderClauses() {
  const groups = clauseGroups();
  html("clauseLibrary", groups.map((group) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(group.family)}</span>
        <span class="badge accent">${group.items.length} findings</span>
      </div>
      <div class="labels">
        ${group.categories.slice(0, 8).map((category) => `<span class="badge">${escape(category)}</span>`).join("")}
      </div>
      <div class="clauseGrid">
        ${group.items.slice(0, 4).map((item) => `
          <div class="clauseCard">
            <div class="rowTitle">
              <span>${escape(item.category || item.family)}</span>
              <span class="badge ${item.answer === "No" ? "warn" : "good"}">${escape(item.answer || "evidence")}</span>
            </div>
            <p class="muted">${escape(item.title)}</p>
            <p>${escape(item.evidence || "No positive evidence span in CUAD annotation.")}</p>
          </div>
        `).join("")}
      </div>
    </div>
  `).join(""));
}

function renderProvenance() {
  html("provenanceRows", (state.provenance || []).map((item) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(item.title)}</span>
        <span class="badge ${item.benchmark?.second_run_correct ? "good" : "warn"}">${item.benchmark?.second_run_correct ? "Benchmark pass" : "Needs review"}</span>
      </div>
      <p class="muted">${escape(item.source_path)}</p>
      <div class="lineage">
        ${lineageStep("Source", item.doc_id, "Document ingested into normalized corpus.")}
        ${lineageStep("Baseline", `${item.baseline?.contract_type || "-"} · ${item.baseline?.engine || "-"}`, `Model: ${item.baseline?.model || "-"} · Evidence spans: ${item.baseline?.evidence_count ?? 0}`)}
        ${lineageStep("Agent", `${item.agent?.priority || "-"} · ${item.agent?.engine || "-"}`, item.agent?.challenge_summary || "No challenge recorded.")}
        ${lineageStep("Review", `${item.review?.accepted_contract_type || "-"} · ${item.review?.authority || "-"}`, `Status: ${item.review?.status || "-"} · Evidence sufficient: ${item.review?.evidence_sufficient ?? "-"}`)}
        ${lineageStep("Second Run", `${item.second_run?.contract_type || "-"} · ${item.second_run?.engine || "-"}`, `Model: ${item.second_run?.model || "-"} · Correct: ${item.benchmark?.second_run_correct ?? "-"}`)}
      </div>
    </div>
  `).join(""));
}

function lineageStep(label, value, detail) {
  return `
    <div class="lineageStep">
      <span>${escape(label)}</span>
      <strong>${escape(value)}</strong>
      <p>${escape(detail)}</p>
    </div>
  `;
}

function clauseGroups() {
  const byFamily = new Map();
  for (const doc of state.documents || []) {
    for (const clause of doc.key_clauses || []) {
      const family = clause.family || clause.cuad_category || "other";
      if (!byFamily.has(family)) byFamily.set(family, []);
      byFamily.get(family).push({
        family,
        category: clause.cuad_category,
        answer: clause.answer,
        evidence: clause.evidence,
        authority: clause.authority,
        title: doc.title
      });
    }
  }
  return [...byFamily.entries()]
    .map(([family, items]) => ({
      family,
      items,
      categories: [...new Set(items.map((item) => item.category).filter(Boolean))]
    }))
    .sort((a, b) => b.items.length - a.items.length || a.family.localeCompare(b.family));
}

function renderNamedList(title, values, formatter) {
  if (!values || !values.length) return "";
  return `
    <div class="miniList">
      <strong>${escape(title)}</strong>
      <ul>
        ${values.slice(0, 6).map((value) => `<li>${escape(formatter(value))}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderDemo() {
  html("demoScript", demoSteps.map(([number, title, body]) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(number)}. ${escape(title)}</span>
      </div>
      <p class="muted">${escape(body)}</p>
    </div>
  `).join(""));
  html("demoBoundaries", demoBoundaries.map(([title, body]) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(title)}</span>
      </div>
      <p class="muted">${escape(body)}</p>
    </div>
  `).join(""));
}

function renderReport() {
  html("reportMarkdown", markdownLite(state.report_markdown || ""));
  html("outputFiles", (state.output_files || []).map((file) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(file.label)}</span>
        <span class="badge ${file.exists ? "good" : "bad"}">${file.exists ? "Created" : "Missing"}</span>
      </div>
      <p class="muted">${escape(file.path)}${file.exists ? ` · ${escape(formatBytes(file.bytes))}` : ""}</p>
      ${file.exists ? `
        <div class="fileActions">
          <a href="/api/file?path=${encodeURIComponent(file.path)}" target="_blank" rel="noreferrer">Open</a>
          <a href="/api/file?path=${encodeURIComponent(file.path)}" download>Download</a>
        </div>
      ` : ""}
    </div>
  `).join(""));
}

async function saveReviewEdits() {
  const items = [...document.querySelectorAll(".reviewEdit")].map((node) => ({
    doc_id: node.dataset.docId,
    title: state.documents.find((doc) => doc.doc_id === node.dataset.docId)?.title || "",
    proposed_contract_type: findReviewItem(node.dataset.docId)?.proposed_contract_type,
    accepted_contract_type: node.querySelector('[data-field="accepted_contract_type"]').value.trim(),
    status: node.querySelector('[data-field="status"]').value,
    contract_type_correct: node.querySelector('[data-field="accepted_contract_type"]').value.trim() === findReviewItem(node.dataset.docId)?.proposed_contract_type,
    evidence_sufficient: true,
    reviewer_authority: "business_sme_confirmed",
    review_notes: node.querySelector('[data-field="review_notes"]').value.trim(),
    accepted_coversheet: findReviewItem(node.dataset.docId)?.accepted_coversheet || findReviewItem(node.dataset.docId)?.coversheet || {},
    accepted_key_clauses: findReviewItem(node.dataset.docId)?.accepted_key_clauses || findReviewItem(node.dataset.docId)?.key_clauses || []
  }));
  const result = await post("/api/review/save", { items });
  setStatus(`Saved ${result.items} reviewed items.`);
  await load();
}

async function runAction(action) {
  setStatus(`Running ${action}...`);
  const result = await post(`/api/actions/${action}`, {});
  if (result.error) {
    setStatus(result.error);
    return;
  }
  setStatus(`Completed ${action}.`);
  await load();
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  return response.json();
}

function setStatus(message) {
  const node = document.getElementById("reviewStatus");
  if (node) node.textContent = message;
  const global = document.getElementById("globalStatus");
  if (global) global.textContent = message;
}

function findReviewItem(docId) {
  return [...(state.reviewed.items || []), ...(state.pending_review.items || [])].find((item) => item.doc_id === docId) || {};
}

function option(value, selected) {
  return `<option value="${value}" ${value === selected ? "selected" : ""}>${value}</option>`;
}

function priorityClass(value) {
  if (value === "high") return "bad";
  if (value === "medium") return "warn";
  if (value === "low") return "good";
  return "";
}

function renderMemory() {
  const taxonomy = state.taxonomy || {};
  html("taxonomyTypes", (taxonomy.contract_types || []).map((item) => `<span class="chip">${escape(item)}</span>`).join(""));
  html("examples", (taxonomy.reviewed_examples || []).map((item) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(item.contract_type)}</span>
        <span class="badge">${escape(item.source_engine)}</span>
      </div>
      <p class="muted">${escape(item.title)}</p>
    </div>
  `).join(""));
}

function renderTraining() {
  html("trainingPairs", state.training_pairs.map((pair) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(pair.output?.reviewed_answer?.contract_type || pair.output?.contract_type)}</span>
        <span class="badge accent">${escape(pair.example_type || pair.label_source)}</span>
      </div>
      <p class="muted">${escape(pair.input?.contract?.title || pair.input?.title)}</p>
      <div class="labels">
        <span class="badge warn">Baseline: ${escape(pair.input?.baseline_model_output?.contract_type || pair.input?.model_output?.contract_type || "-")}</span>
        <span class="badge good">Reviewed: ${escape(pair.output?.reviewed_answer?.contract_type || pair.output?.contract_type || "-")}</span>
      </div>
      <details>
        <summary>JSON</summary>
        <pre>${escape(JSON.stringify(pair, null, 2))}</pre>
      </details>
    </div>
  `).join(""));
}

function renderBRD() {
  html("brdCoverage", brdCoverage.map(([id, name, status, note]) => `
    <div class="row">
      <div class="rowTitle">
        <span>${escape(id)} · ${escape(name)}</span>
        <span class="badge ${coverageClass(status)}">${escape(statusLabel(status))}</span>
      </div>
      <p class="muted">${escape(note)}</p>
    </div>
  `).join(""));
}

function renderRoadmap() {
  html("roadmapRows", `
    <div class="denseHeader">
      <span>Phase</span>
      <span>Name</span>
      <span>Horizon</span>
      <span>Scope</span>
    </div>
    ${roadmapRows.map(([phase, name, horizon, scope]) => `
      <div class="denseRow">
        <span>${escape(phase)}</span>
        <strong>${escape(name)}</strong>
        <span class="badge ${horizon === "Next" ? "accent" : ""}">${escape(horizon)}</span>
        <span>${escape(scope)}</span>
      </div>
    `).join("")}
  `);
  html("roadmapMarkdown", markdownLite(state.roadmap_markdown || ""));
}

function coverageClass(value) {
  if (value === "mvp") return "good";
  if (value === "partial") return "warn";
  if (value === "future") return "bad";
  return "";
}

function statusLabel(value) {
  if (value === "mvp") return "MVP";
  if (value === "partial") return "Partial";
  if (value === "future") return "Future";
  return value;
}

function fieldValue(value) {
  if (value === null || value === undefined || value === "") return "";
  if (Array.isArray(value)) return value.join("; ");
  if (typeof value === "object") return value.accepted_value || value.value || value.answer || "";
  return value;
}

function fieldEvidence(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "";
  const authority = value.authority ? `Authority: ${value.authority}` : "";
  const evidence = value.evidence ? `Evidence: ${value.evidence}` : "";
  return [authority, evidence].filter(Boolean).join(" | ");
}

function markdownLite(markdown) {
  if (!markdown) return "";
  const lines = markdown.split(/\r?\n/);
  return lines.map((line) => {
    if (line.startsWith("## ")) return `<h4>${escape(line.slice(3))}</h4>`;
    if (line.startsWith("# ")) return `<h3>${escape(line.slice(2))}</h3>`;
    if (line.startsWith("- ")) return `<p class="reportBullet">${escape(line.slice(2))}</p>`;
    if (!line.trim()) return "";
    return `<p>${escape(line)}</p>`;
  }).join("");
}

function formatBytes(value) {
  const number = Number(value || 0);
  if (number < 1024) return `${number} B`;
  if (number < 1024 * 1024) return `${Math.round(number / 1024)} KB`;
  return `${(number / 1024 / 1024).toFixed(1)} MB`;
}

function shortPath(path) {
  if (!path) return "";
  const parts = String(path).split("/");
  return parts.slice(-2).join("/");
}

function text(id, value) {
  document.getElementById(id).textContent = value ?? "-";
}

function html(id, value) {
  document.getElementById(id).innerHTML = value || `<p class="muted">No data yet.</p>`;
}

function pct(value) {
  return value === null || value === undefined ? "-" : `${Math.round(Number(value) * 100)}%`;
}

function metricDelta(before, after) {
  if (before === null || before === undefined || after === null || after === undefined) return "-";
  const delta = Number(after) - Number(before);
  const sign = delta > 0 ? "+" : "";
  return `${sign}${Math.round(delta * 100)} pts`;
}

function escape(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[char]));
}

// === Agent edition: upload, split, agent run, decision log, three-way benchmark, counterfactuals ===
(function() {
  function setText(id, txt) { const el = document.getElementById(id); if (el) el.textContent = txt; }
  function readFileAsB64(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => {
        const b64 = String(r.result).split(",")[1] || "";
        resolve({ filename: file.webkitRelativePath || file.name, content_b64: b64 });
      };
      r.onerror = () => reject(r.error);
      r.readAsDataURL(file);
    });
  }
  async function postJson(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    return r.json();
  }
  async function getJson(url) { const r = await fetch(url); return r.json(); }

  async function uploadFiles(fileList) {
    const accepted = /\.(txt|md|html?|docx|pdf)$/i;
    const files = Array.from(fileList || []).filter(f => accepted.test(f.name));
    if (files.length === 0) { setText("uploadStatus", "No accepted file types in selection."); return; }
    setText("uploadStatus", `Reading ${files.length} files...`);
    const payload = [];
    for (let i = 0; i < files.length; i++) {
      try { payload.push(await readFileAsB64(files[i])); }
      catch (e) { /* skip unreadable */ }
      if (i % 5 === 0) setText("uploadStatus", `Reading ${i + 1}/${files.length}...`);
    }
    setText("uploadStatus", `Uploading ${payload.length} files...`);
    try {
      const out = await postJson("/api/upload", { files: payload });
      const results = document.getElementById("uploadResults");
      if (results) {
        results.textContent = "";
        const div = document.createElement("div");
        div.style.padding = "8px";
        div.style.background = "#e6ffe6";
        div.style.borderRadius = "4px";
        div.textContent = `Ingested ${out.ingested} of ${out.received} files. Uploaded to ${out.upload_dir}.`;
        results.appendChild(div);
      }
      setText("uploadStatus", "");
    } catch (e) { setText("uploadStatus", "Upload error: " + e); }
  }

  function bindUpload() {
    const folderInput = document.getElementById("uploadFolderInput");
    const filesInput = document.getElementById("uploadFilesInput");
    const dropzone = document.getElementById("uploadDropzone");
    if (folderInput) folderInput.addEventListener("change", e => uploadFiles(e.target.files));
    if (filesInput) filesInput.addEventListener("change", e => uploadFiles(e.target.files));
    if (dropzone) {
      ["dragover", "dragenter"].forEach(ev => dropzone.addEventListener(ev, e => {
        e.preventDefault(); dropzone.style.background = "#eef9ee";
      }));
      ["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, e => {
        e.preventDefault(); dropzone.style.background = "#fafafa";
      }));
      dropzone.addEventListener("drop", async e => {
        e.preventDefault();
        const items = e.dataTransfer.items;
        const collected = [];
        async function walk(entry, prefix) {
          if (!entry) return;
          if (entry.isFile) {
            await new Promise(res => entry.file(f => { f.relPath = prefix + f.name; collected.push(f); res(); }));
          } else if (entry.isDirectory) {
            const reader = entry.createReader();
            await new Promise(res => reader.readEntries(async ents => {
              for (const ent of ents) await walk(ent, prefix + entry.name + "/");
              res();
            }));
          }
        }
        if (items && items[0] && items[0].webkitGetAsEntry) {
          for (const it of items) await walk(it.webkitGetAsEntry(), "");
          await uploadFiles(collected);
        } else {
          await uploadFiles(e.dataTransfer.files);
        }
      });
    }
  }

  function bindSplit() {
    const btn = document.getElementById("splitBtn");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const frac = parseFloat(document.getElementById("splitFrac").value || "0.6");
      const seed = parseInt(document.getElementById("splitSeed").value || "42", 10);
      setText("splitStatus", "Splitting...");
      try {
        const out = await postJson("/api/split", { review_frac: frac, seed });
        setText("splitStatus", `Split: ${out.review_set.length} review, ${out.holdout_set.length} holdout, seed=${out.split_seed}`);
      } catch (e) { setText("splitStatus", "Split error: " + e); }
    });
  }

  function bindAgent() {
    const startBtn = document.getElementById("agentStartBtn");
    const resumeBtn = document.getElementById("agentResumeBtn");
    async function fire(url) {
      const primary = document.getElementById("agentPrimary").value || "qwen2.5:14b";
      const shadow = document.getElementById("agentShadow").value || "qwen3:4b";
      setText("agentStatus", "Starting agent...");
      try {
        const out = await postJson(url, { primary_model: primary, shadow_model: shadow });
        setText("agentStatus", `Agent ${out.started ? "started" : "?"} (primary=${out.primary_model}, shadow=${out.shadow_model}). Decision log streaming below.`);
      } catch (e) { setText("agentStatus", "Agent error: " + e); }
    }
    if (startBtn) startBtn.addEventListener("click", () => fire("/api/agent/run"));
    if (resumeBtn) resumeBtn.addEventListener("click", () => fire("/api/agent/resume"));
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindUpload();
    bindSplit();
    bindAgent();
  });
  if (document.readyState !== "loading") {
    bindUpload(); bindSplit(); bindAgent();
  }
})();

// === Agent tab: decisions, three-way bench, counterfactuals (consolidated UI) ===
(function() {
  function setText(id, txt) { const el = document.getElementById(id); if (el) el.textContent = txt; }
  async function getJson(url) { const r = await fetch(url); return r.json(); }
  async function postJson(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    return r.json();
  }

  let autoTimer = null;

  async function loadDecisions() {
    const input = document.getElementById("agentDecisionsRunId");
    const runId = input ? input.value : "";
    const url = runId ? "/api/decisions?run_id=" + encodeURIComponent(runId) : "/api/decisions";
    try {
      const res = await getJson(url);
      const lines = (res.rows || []).map(r => {
        const args = JSON.stringify(r.args || {}).slice(0, 60);
        return r.ts + "  " + (r.action + "").padEnd(28) + " " + args + "  " + (r.rationale || "");
      });
      setText("agentDecisionsLog", lines.length ? lines.join("\n") : "(no decisions yet)");
    } catch (e) { setText("agentDecisionsLog", "error: " + e); }
  }

  async function loadThreeWay() {
    const banner = document.getElementById("threeWayBanner");
    const table = document.getElementById("threeWayTable");
    if (!banner || !table) return;
    while (table.firstChild) table.removeChild(table.firstChild);
    try {
      const b = await getJson("/api/benchmark/three-way");
      banner.style.background = b.engine_integrity === "ok" ? "#e6ffe6" : "#ffe6e6";
      banner.textContent = "Engine integrity: " + b.engine_integrity + "  |  n=" + (b.n_docs || 0);
      if (!b.metrics) return;
      const cols = ["large", "small_cold", "small_reviewed"];
      const metrics = ["contract_type_accuracy", "clause_family_f1"];
      const head = document.createElement("tr");
      const th0 = document.createElement("th"); th0.textContent = "metric";
      th0.style.border = "1px solid #ccc"; th0.style.padding = "6px 12px"; th0.style.background = "#f4f4f4";
      head.appendChild(th0);
      cols.forEach(c => {
        const th = document.createElement("th"); th.textContent = c;
        th.style.border = "1px solid #ccc"; th.style.padding = "6px 12px"; th.style.background = "#f4f4f4";
        head.appendChild(th);
      });
      table.appendChild(head);
      metrics.forEach(m => {
        const tr = document.createElement("tr");
        const td0 = document.createElement("td"); td0.textContent = m;
        td0.style.border = "1px solid #ccc"; td0.style.padding = "6px 12px";
        tr.appendChild(td0);
        cols.forEach(c => {
          const td = document.createElement("td");
          const v = (b.metrics[m] && b.metrics[m][c]) || 0;
          td.textContent = (typeof v === "number") ? v.toFixed(2) : String(v);
          td.style.border = "1px solid #ccc"; td.style.padding = "6px 12px"; td.style.fontFamily = "monospace";
          tr.appendChild(td);
        });
        table.appendChild(tr);
      });
    } catch (e) {
      banner.style.background = "#ffe6e6";
      banner.textContent = "error: " + e;
    }
  }

  async function counterfactual(toggle) {
    const target = document.getElementById("cfResult");
    if (!target) return;
    target.textContent = "Recomputing...";
    try {
      const out = await postJson("/api/benchmark/counterfactual", { toggle, model: "qwen3:4b" });
      target.textContent = JSON.stringify(out, null, 2);
    } catch (e) { target.textContent = "error: " + e; }
  }

  function bindAgentTab() {
    const loadBtn = document.getElementById("agentDecisionsLoadBtn");
    const autoBtn = document.getElementById("agentDecisionsAutoBtn");
    const refreshBtn = document.getElementById("threeWayRefreshBtn");
    const cfVerifier = document.getElementById("cfVerifierBtn");
    const cfContext = document.getElementById("cfContextBtn");
    if (loadBtn) loadBtn.addEventListener("click", loadDecisions);
    if (refreshBtn) refreshBtn.addEventListener("click", loadThreeWay);
    if (cfVerifier) cfVerifier.addEventListener("click", () => counterfactual("verifier_off"));
    if (cfContext) cfContext.addEventListener("click", () => counterfactual("context_off"));
    if (autoBtn) {
      autoTimer = setInterval(loadDecisions, 2000);
      autoBtn.addEventListener("click", function() {
        if (autoTimer) { clearInterval(autoTimer); autoTimer = null; this.textContent = "Auto-refresh: OFF"; }
        else { autoTimer = setInterval(loadDecisions, 2000); this.textContent = "Auto-refresh: ON"; }
      });
    }
    loadDecisions();
    loadThreeWay();
  }

  if (document.readyState !== "loading") bindAgentTab();
  else document.addEventListener("DOMContentLoaded", bindAgentTab);
})();

// === Discovery tab (cut-down with clause library viewer) ===
(function() {
  function $id(id) { return document.getElementById(id); }
  function setText(id, t) { const el = $id(id); if (el) el.textContent = t; }
  async function gj(url) { return (await fetch(url)).json(); }
  async function pj(url, body) {
    const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
                                body: JSON.stringify(body || {})});
    return r.json();
  }
  function readB64(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve({filename: file.webkitRelativePath || file.name,
                                content_b64: String(r.result).split(",")[1] || ""});
      r.onerror = () => reject(r.error);
      r.readAsDataURL(file);
    });
  }

  let chatLog = [];
  let currentSig = {target_class: "", target_description: "", clause_types: []};
  let queueState = {round_index: 0, items: []};
  let openingShown = false;

  function renderChat() {
    const el = $id("discChat"); if (!el) return;
    while (el.firstChild) el.removeChild(el.firstChild);
    chatLog.forEach(m => {
      const wrap = document.createElement("div");
      wrap.className = "chatMessage " + (m.role === "user" ? "fromUser" : "fromAgent");
      const label = document.createElement("span");
      label.textContent = m.role === "user" ? "You" : "Agent";
      wrap.appendChild(label);
      const p = document.createElement("p");
      p.style.whiteSpace = "pre-wrap";
      p.textContent = m.content;
      wrap.appendChild(p);
      el.appendChild(wrap);
    });
    el.scrollTop = el.scrollHeight;
  }

  function syncSigFromDOM() {
    // Pull all editable values back into currentSig before sending or rendering.
    const tcEl = $id("discEditTargetClass");
    const tdEl = $id("discEditTargetDescription");
    if (tcEl) currentSig.target_class = tcEl.value;
    if (tdEl) currentSig.target_description = tdEl.value;
    const ctsRoot = $id("discClauseTypes");
    if (ctsRoot) {
      const cards = ctsRoot.querySelectorAll(".clauseCard");
      const out = [];
      cards.forEach(card => {
        const idx = parseInt(card.getAttribute("data-idx"), 10);
        const orig = (currentSig.clause_types || [])[idx] || {};
        const nameEl = card.querySelector(".clauseName");
        const descEl = card.querySelector(".clauseDesc");
        const examplesEl = card.querySelector(".clauseExamples");
        const mustEl = card.querySelector(".clauseMust");
        out.push({
          type: nameEl ? nameEl.value : orig.type,
          description: descEl ? descEl.value : (orig.description || ""),
          is_must_have: mustEl ? mustEl.value === "must" : !!orig.is_must_have,
          seed_variations: examplesEl
            ? examplesEl.value.split("\n").map(s => s.trim()).filter(Boolean)
            : (orig.seed_variations || []),
        });
      });
      currentSig.clause_types = out;
    }
  }

  function renderSig() {
    const tcEl = $id("discEditTargetClass");
    const tdEl = $id("discEditTargetDescription");
    if (tcEl && document.activeElement !== tcEl) tcEl.value = currentSig.target_class || "";
    if (tdEl && document.activeElement !== tdEl) tdEl.value = currentSig.target_description || "";

    const types = $id("discClauseTypes");
    const readiness = $id("discReadiness");
    if (types) {
      // Skip re-render if the user is currently editing one of the inputs inside a card.
      const focused = document.activeElement;
      const focusedInsideTypes = focused && types.contains(focused);
      if (!focusedInsideTypes) {
        while (types.firstChild) types.removeChild(types.firstChild);
        const cts = currentSig.clause_types || [];
        if (!cts.length) {
          const empty = document.createElement("p");
          empty.className = "muted"; empty.style.padding = "8px 12px";
          empty.textContent = "No clauses yet — the agent will propose some, and you can edit them here.";
          types.appendChild(empty);
        } else {
          cts.forEach((ct, i) => types.appendChild(buildClauseCard(ct, i)));
        }
      }
    }
    if (readiness) {
      const ready = (currentSig.target_class || "").trim() &&
                    (currentSig.target_description || "").trim() &&
                    (currentSig.clause_types || []).length > 0;
      readiness.textContent = ready ? "Ready" : "Draft";
      readiness.className = "badge " + (ready ? "ok" : "warn");
    }
  }

  function buildClauseCard(ct, idx) {
    const card = document.createElement("div");
    card.className = "clauseCard";
    card.setAttribute("data-idx", String(idx));
    card.style.padding = "10px 12px";
    card.style.borderBottom = "1px solid #eee";
    card.style.fontSize = "12px";
    card.style.background = ct.is_must_have ? "#f4faf4" : "#faf4f4";

    // Top row: must/not toggle, name input, delete button
    const top = document.createElement("div");
    top.style.display = "flex"; top.style.alignItems = "center"; top.style.gap = "6px";

    const mustSel = document.createElement("select");
    mustSel.className = "clauseMust";
    mustSel.style.fontSize = "10px"; mustSel.style.padding = "2px 4px";
    [["must", "MUST"], ["not", "NOT"]].forEach(([v, label]) => {
      const o = document.createElement("option"); o.value = v; o.textContent = label;
      if ((v === "must") === !!ct.is_must_have) o.selected = true;
      mustSel.appendChild(o);
    });
    mustSel.addEventListener("change", () => {
      // re-color
      card.style.background = mustSel.value === "must" ? "#f4faf4" : "#faf4f4";
      syncSigFromDOM();
    });
    top.appendChild(mustSel);

    const nameInput = document.createElement("input");
    nameInput.className = "clauseName";
    nameInput.value = ct.type || "";
    nameInput.style.flex = "1"; nameInput.style.fontSize = "12px";
    nameInput.style.fontWeight = "650"; nameInput.style.padding = "3px 6px";
    nameInput.addEventListener("change", syncSigFromDOM);
    top.appendChild(nameInput);

    const delBtn = document.createElement("button");
    delBtn.type = "button"; delBtn.textContent = "×";
    delBtn.title = "Remove this clause";
    delBtn.style.fontSize = "14px"; delBtn.style.padding = "0 8px";
    delBtn.style.cursor = "pointer"; delBtn.style.color = "#a31";
    delBtn.addEventListener("click", () => {
      syncSigFromDOM();
      currentSig.clause_types.splice(idx, 1);
      renderSig();
    });
    top.appendChild(delBtn);
    card.appendChild(top);

    // Description
    const descLbl = document.createElement("div");
    descLbl.style.fontSize = "10px"; descLbl.style.color = "#888"; descLbl.style.marginTop = "6px";
    descLbl.textContent = "Description";
    card.appendChild(descLbl);
    const descInput = document.createElement("textarea");
    descInput.className = "clauseDesc"; descInput.rows = 2;
    descInput.value = ct.description || "";
    descInput.style.width = "100%"; descInput.style.fontSize = "11px";
    descInput.style.boxSizing = "border-box"; descInput.style.padding = "4px 6px";
    descInput.style.fontFamily = "inherit";
    descInput.addEventListener("change", syncSigFromDOM);
    card.appendChild(descInput);

    // Examples (one per line)
    const exLbl = document.createElement("div");
    exLbl.style.fontSize = "10px"; exLbl.style.color = "#888"; exLbl.style.marginTop = "6px";
    exLbl.textContent = "Example phrasings (one per line)";
    card.appendChild(exLbl);
    const exInput = document.createElement("textarea");
    exInput.className = "clauseExamples"; exInput.rows = 2;
    exInput.value = (ct.seed_variations || []).join("\n");
    exInput.style.width = "100%"; exInput.style.fontSize = "11px";
    exInput.style.boxSizing = "border-box"; exInput.style.padding = "4px 6px";
    exInput.style.fontFamily = "inherit"; exInput.style.fontStyle = "italic";
    exInput.addEventListener("change", syncSigFromDOM);
    card.appendChild(exInput);

    return card;
  }

  function addBlankClause() {
    syncSigFromDOM();
    if (!Array.isArray(currentSig.clause_types)) currentSig.clause_types = [];
    currentSig.clause_types.push({
      type: "new_clause", description: "", is_must_have: true, seed_variations: [],
    });
    renderSig();
  }

  async function showOpening() {
    if (openingShown) return;
    const r = await pj("/api/interview/discovery-chat",
                       {signature: currentSig, message: "", initial: true});
    chatLog.push({role: "agent", content: r.assistant || ""});
    renderChat(); openingShown = true;
  }

  async function chatSend() {
    const input = $id("discChatInput"); const msg = (input.value || "").trim();
    if (!msg) return;
    syncSigFromDOM();   // capture user edits before sending
    chatLog.push({role: "user", content: msg}); renderChat(); input.value = "";
    const r = await pj("/api/interview/discovery-chat",
                       {signature: currentSig, message: msg});
    if (r.signature) { currentSig = r.signature; renderSig(); }
    chatLog.push({role: "agent", content: r.assistant || "(no reply)"});
    renderChat();
  }

  async function saveSig() {
    syncSigFromDOM();   // capture user edits before saving
    const r = await pj("/api/interview/discovery-chat",
                       {signature: currentSig, message: "save", save: true});
    chatLog.push({role: "agent", content: r.assistant || "saved"}); renderChat();
    pollState(); pollLibrary();
  }

  async function uploadAndIngest(fileList) {
    const accepted = /\.(txt|md|html?|docx|pdf)$/i;
    const files = Array.from(fileList || []).filter(f => accepted.test(f.name));
    if (!files.length) { setText("discUploadStatus", "no accepted files"); return; }
    setText("discUploadStatus", `Reading ${files.length}...`);
    const payload = [];
    for (const f of files) try { payload.push(await readB64(f)); } catch (e) { /* skip */ }
    const up = await pj("/api/upload", {files: payload});
    setText("discUploadStatus", `Uploaded ${up.received}, ingested ${up.ingested}.`);
  }
  async function embed() {
    setText("discEmbedStatus", "embedding (slow)...");
    const r = await pj("/api/discovery/embed", {model: "nomic-embed-text"});
    setText("discEmbedStatus", `Embedded ${r.embedded}, skipped ${r.skipped}, failed ${r.failed}.`);
    pollState();
  }

  async function runRound() {
    const idx = parseInt($id("discRoundIdx").value || "0", 10);
    const topK = parseInt($id("discTopK").value || "300", 10);
    const batch = parseInt($id("discBatch").value || "30", 10);
    const btn = $id("discRunRoundBtn");
    const progress = $id("discRoundProgress");
    if (btn) { btn.disabled = true; btn.textContent = "Starting…"; }
    if (progress) progress.textContent = `Queued. Reading up to ${topK} contracts. Each takes ~3 seconds.`;
    setText("discRoundResult", "");
    try {
      const modelEl = $id("discClassifierModel");
      const classifier = (modelEl && modelEl.value) || "gpt-4o-mini";
      const start = await pj("/api/discovery/run-round",
                             {round_index: idx, top_k: topK, batch_size: batch,
                              classifier_model: classifier, async: true});
      if (!start.job_id) {
        if (progress) progress.textContent = "Error starting job: " + JSON.stringify(start);
        return;
      }
      // Poll job status until done.
      let done = false;
      while (!done) {
        await new Promise(r => setTimeout(r, 2000));
        const s = await gj("/api/discovery/job/" + encodeURIComponent(start.job_id));
        if (!s || s.error) {
          if (progress) progress.textContent = "Job error: " + (s && s.error || "unknown");
          break;
        }
        const p = s.progress || 0, t = s.total || topK;
        if (btn) btn.textContent = `Reading… ${p}/${t}`;
        if (progress) progress.textContent =
          `Reading contracts… ${p}/${t} done` + (s.note ? ` (${s.note.slice(0, 80)})` : "");
        if (s.status === "done") {
          done = true;
          const r = s.result || {};
          if (progress) progress.textContent =
            `Done — looked at ${r.classifications_count || t} contracts; ${r.review_queue_size || batch} need your review in step 4.`;
          setText("discRoundResult", JSON.stringify(r, null, 2));
          await loadReviewQueue(idx);
        } else if (s.status === "error") {
          done = true;
          if (progress) progress.textContent = "Error: " + (s.error || "unknown");
          setText("discRoundResult", JSON.stringify(s, null, 2));
        }
      }
    } catch (e) {
      if (progress) progress.textContent = "Error: " + e;
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = "Find matches"; }
      pollState(); pollLibrary();
    }
  }

  async function loadReviewQueue(idx) {
    const url = "/api/file?path=" + encodeURIComponent(`data/discovery/review_queue_round_${idx}.json`);
    try {
      const txt = await (await fetch(url)).text();
      queueState = JSON.parse(txt);
    } catch (e) { queueState = {round_index: idx, items: []}; }
    const root = $id("discReviewQueue");
    while (root.firstChild) root.removeChild(root.firstChild);
    queueState.items.forEach(it => {
      const card = document.createElement("div");
      card.style.border = "1px solid #ccc"; card.style.borderRadius = "4px";
      card.style.padding = "10px"; card.style.margin = "8px 0"; card.style.background = "white";
      const head = document.createElement("div"); head.style.fontWeight = "bold";
      head.textContent = `${it.doc_id}  —  agent: ${it.verdict} (${(it.confidence||0).toFixed(2)})  —  reason: ${it.reason}`;
      card.appendChild(head);
      const ev = it.evidence_per_clause_type || {};
      Object.keys(ev).forEach(k => {
        const row = document.createElement("div");
        row.style.fontSize = "11px"; row.style.color = "#555"; row.style.marginTop = "4px";
        row.style.paddingLeft = "12px"; row.style.borderLeft = "2px solid #ddd";
        row.textContent = `[${k}]  ${(ev[k] || "(none)").slice(0, 240)}`;
        card.appendChild(row);
      });
      const btns = document.createElement("div"); btns.style.marginTop = "8px";
      ["yes","no","borderline"].forEach(v => {
        const b = document.createElement("button");
        b.textContent = v; b.style.marginRight = "6px";
        b.addEventListener("click", () => {
          it.userVerdict = v;
          [...btns.children].forEach(x => x.style.background = "");
          b.style.background = v === "yes" ? "#cfc" : v === "no" ? "#fcc" : "#ffc";
        });
        btns.appendChild(b);
      });
      card.appendChild(btns);
      root.appendChild(card);
    });
  }

  async function submitLabels() {
    const labels = queueState.items.filter(it => it.userVerdict).map(it => ({
      doc_id: it.doc_id, verdict: it.userVerdict,
    }));
    if (!labels.length) { alert("no labels selected"); return; }
    const r = await pj("/api/discovery/submit-labels",
                       {round_index: queueState.round_index, labels});
    alert(`Submitted ${r.labels_received}. Corrections: ${r.corrections}. Library grew by ${r.library_growth} variations.`);
    pollState(); pollLibrary();
  }

  async function finalize() {
    const idx = parseInt($id("discRoundIdx").value || "0", 10);
    const r = await pj("/api/discovery/finalize",
                       {round_index: idx, borderline_threshold: 0.7});
    setText("discFinalResult", JSON.stringify(r, null, 2));
    pollState();
  }

  async function pollState() {
    try { setText("discStateJson", JSON.stringify(await gj("/api/discovery/state"), null, 2)); }
    catch (e) { setText("discStateJson", "error: " + e); }
  }

  async function pollLibrary() {
    const root = $id("discLibrary"); if (!root) return;
    while (root.firstChild) root.removeChild(root.firstChild);
    let lib;
    try { lib = await gj("/api/discovery/library"); }
    catch (e) { return; }
    if (!lib.clause_types || !lib.clause_types.length) {
      const p = document.createElement("p"); p.textContent = "(library not yet seeded)";
      root.appendChild(p); return;
    }
    const h = document.createElement("h4"); h.textContent = `Target class: ${lib.target_class}`;
    root.appendChild(h);
    lib.clause_types.forEach(ct => {
      const div = document.createElement("div");
      div.style.border = "1px solid #ddd"; div.style.borderRadius = "4px";
      div.style.padding = "10px"; div.style.margin = "8px 0";
      div.style.background = ct.is_must_have ? "#f4faf4" : "#faf4f4";
      const head = document.createElement("div"); head.style.fontWeight = "bold";
      const flag = ct.is_must_have ? "[MUST HAVE]" : "[MUST NOT HAVE]";
      head.textContent = `${flag} ${ct.type}  (${ct.variations.length} variations)`;
      div.appendChild(head);
      const desc = document.createElement("div"); desc.style.fontSize = "11px"; desc.style.color = "#666";
      desc.textContent = ct.description; div.appendChild(desc);
      ct.variations.forEach(v => {
        const item = document.createElement("div");
        item.style.fontSize = "11px"; item.style.marginTop = "6px"; item.style.paddingLeft = "12px";
        item.style.borderLeft = "2px solid #ccc";
        item.textContent = `• "${v.text}"   — from ${v.source_doc_id} (${v.confirmed_by})`;
        div.appendChild(item);
      });
      root.appendChild(div);
    });
  }

  function bind() {
    const fi = $id("discFolderInput");
    if (fi) fi.addEventListener("change", e => uploadAndIngest(e.target.files));
    const dz = $id("discDropzone");
    if (dz) {
      ["dragover","dragenter"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.style.background = "#eef9ee"; }));
      ["dragleave","drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.style.background = "#fafafa"; }));
      dz.addEventListener("drop", e => { e.preventDefault(); uploadAndIngest(e.dataTransfer.files); });
    }
    const eb = $id("discEmbedBtn"); if (eb) eb.addEventListener("click", embed);
    const cs = $id("discChatSend"); if (cs) cs.addEventListener("click", chatSend);
    const ci = $id("discChatInput"); if (ci) ci.addEventListener("keydown", e => { if (e.key === "Enter") chatSend(); });
    const ss = $id("discSaveSig"); if (ss) ss.addEventListener("click", saveSig);
    const ac = $id("discAddClauseBtn"); if (ac) ac.addEventListener("click", addBlankClause);
    const tcEl = $id("discEditTargetClass");
    const tdEl = $id("discEditTargetDescription");
    if (tcEl) tcEl.addEventListener("input", () => { currentSig.target_class = tcEl.value; });
    if (tdEl) tdEl.addEventListener("input", () => { currentSig.target_description = tdEl.value; });
    const rr = $id("discRunRoundBtn"); if (rr) rr.addEventListener("click", runRound);
    const sl = $id("discSubmitLabels"); if (sl) sl.addEventListener("click", submitLabels);
    const fb = $id("discFinalizeBtn"); if (fb) fb.addEventListener("click", finalize);
    showOpening(); renderSig(); pollState(); pollLibrary();
    setInterval(pollState, 5000); setInterval(pollLibrary, 8000);
  }
  if (document.readyState !== "loading") bind();
  else document.addEventListener("DOMContentLoaded", bind);
})();
