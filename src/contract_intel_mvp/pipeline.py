"""Pipeline primitives for the standalone contract intelligence MVP."""

from __future__ import annotations

import ast
import csv
import html
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .prompts import build_agent_analysis_prompt, build_extraction_prompt


DATA_DIRS = [
    "data/raw_contracts",
    "data/corpus",
    "data/runs",
    "data/reviews",
    "data/memory",
    "data/training",
    "data/gold",
]

SEC_USER_AGENT = "contract-intelligence-mvp/0.1 gui@example.com"
LOCAL_CUAD_DIR = Path("/home/gui/archive/lexi-ai/Knowledge/Dataset/CUAD_v1")

EDGAR_SAMPLE_COMPANIES = [
    ("Apple", "0000320193"),
    ("Microsoft", "0000789019"),
    ("NVIDIA", "0001045810"),
    ("Tesla", "0001318605"),
    ("Netflix", "0001065280"),
    ("Adobe", "0000796343"),
    ("Salesforce", "0001108524"),
    ("Palantir", "0001321655"),
    ("ServiceNow", "0001373715"),
    ("Snowflake", "0001640147"),
    ("Uber", "0001543151"),
    ("Airbnb", "0001559720"),
    ("Roku", "0001428439"),
    ("DoorDash", "0001792789"),
]

EDGAR_EXHIBIT_PATTERNS = [
    "agreement",
    "license",
    "service",
    "supply",
    "distribution",
    "lease",
    "collaboration",
    "reseller",
    "partner",
    "customer",
    "vendor",
    "ex10",
    "ex-10",
]

CUAD_COVERSHEET_COLUMNS = {
    "Parties": "parties",
    "Agreement Date": "agreement_date",
    "Effective Date": "effective_date",
    "Expiration Date": "expiration_date",
    "Governing Law": "governing_law",
}

CUAD_CLAUSE_FAMILY_MAP = {
    "Governing Law": "governing_law",
    "Most Favored Nation": "most_favored_nation",
    "Non-Compete": "non_compete",
    "Exclusivity": "exclusivity",
    "No-Solicit Of Customers": "no_solicit_customers",
    "No-Solicit Of Employees": "no_solicit_employees",
    "Non-Disparagement": "non_disparagement",
    "Termination For Convenience": "term_and_termination",
    "Rofr/Rofo/Rofn": "rights_of_first_refusal",
    "Change Of Control": "change_of_control",
    "Anti-Assignment": "anti_assignment",
    "Revenue/Profit Sharing": "payment_or_royalty",
    "Price Restrictions": "payment_or_royalty",
    "Minimum Commitment": "minimum_commitment",
    "Volume Restriction": "volume_restriction",
    "IP Ownership Assignment": "ip_ownership",
    "Joint IP Ownership": "ip_ownership",
    "License Grant": "grant_of_rights",
    "Non-Transferable License": "grant_of_rights",
    "Affiliate License-Licensor": "affiliate_license",
    "Affiliate License-Licensee": "affiliate_license",
    "Unlimited/All-You-Can-Eat-License": "grant_of_rights",
    "Irrevocable Or Perpetual License": "grant_of_rights",
    "Source Code Escrow": "source_code_escrow",
    "Post-Termination Services": "post_termination_services",
    "Audit Rights": "audit_rights",
    "Uncapped Liability": "liability",
    "Cap On Liability": "liability",
    "Liquidated Damages": "liability",
    "Warranty Duration": "warranty",
    "Insurance": "insurance",
    "Covenant Not To Sue": "covenant_not_to_sue",
    "Third Party Beneficiary": "third_party_beneficiary",
}


@dataclass
class Document:
    doc_id: str
    source_path: str
    title: str
    text: str


def init_project(root: Path) -> list[Path]:
    created: list[Path] = []
    for rel in DATA_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)

    interview_src = root / "config" / "interview.example.json"
    interview_dst = root / "data" / "memory" / "interview.json"
    if interview_src.exists() and not interview_dst.exists():
        interview_dst.write_text(interview_src.read_text(), encoding="utf-8")
        created.append(interview_dst)

    taxonomy = root / "data" / "memory" / "taxonomy.json"
    if not taxonomy.exists():
        taxonomy.write_text(json.dumps(_empty_taxonomy(), indent=2), encoding="utf-8")
        created.append(taxonomy)

    if interview_dst.exists():
        current_taxonomy = _load_json(taxonomy, _empty_taxonomy())
        if _merge_interview_taxonomy(current_taxonomy, _load_json(interview_dst, {})):
            taxonomy.write_text(json.dumps(current_taxonomy, indent=2), encoding="utf-8")
    return created


def reset_project(root: Path, *, keep_raw: bool = False) -> dict[str, Any]:
    data_dir = root / "data"
    preserved_raw = None
    if keep_raw and (data_dir / "raw_contracts").exists():
        preserved_raw = root / ".contract_intel_raw_backup"
        if preserved_raw.exists():
            shutil.rmtree(preserved_raw)
        shutil.move(str(data_dir / "raw_contracts"), str(preserved_raw))
    if data_dir.exists():
        shutil.rmtree(data_dir)
    created = init_project(root)
    if preserved_raw and preserved_raw.exists():
        target = data_dir / "raw_contracts"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(preserved_raw), str(target))
    return {
        "reset": str(data_dir),
        "keep_raw": keep_raw,
        "created": [str(path) for path in created],
    }


def save_interview(root: Path, config_path: Path) -> dict[str, Any]:
    init_project(root)
    source = config_path if config_path.is_absolute() else root / config_path
    interview = _load_json(source, None)
    if not isinstance(interview, dict):
        raise ValueError(f"Interview config must be a JSON object: {source}")
    return save_interview_payload(root, interview)


def save_interview_payload(root: Path, interview: dict[str, Any]) -> dict[str, Any]:
    init_project(root)

    interview_path = root / "data" / "memory" / "interview.json"
    interview_path.write_text(json.dumps(interview, indent=2), encoding="utf-8")

    taxonomy_path = root / "data" / "memory" / "taxonomy.json"
    taxonomy = _load_json(taxonomy_path, _empty_taxonomy())
    _merge_interview_taxonomy(taxonomy, interview)
    taxonomy_path.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")

    return {
        "interview": str(interview_path),
        "taxonomy": str(taxonomy_path),
        "expected_contract_types": len(interview.get("expected_contract_types", [])),
        "key_clause_families": len(interview.get("key_clause_families", [])),
    }


def ingest_folder(input_dir: Path, root: Path) -> int:
    init_project(root)
    docs: list[Document] = []
    manifest: dict[str, Any] = {
        "input_dir": str(input_dir),
        "scanned_files": 0,
        "ingested_files": 0,
        "skipped_files": 0,
        "extensions": {},
        "files": [],
        "supported_inputs": [".txt", ".md", ".html", ".htm", ".docx", ".pdf with pdftotext"],
        "pdf_ocr_boundary": "PDF OCR is intentionally out of scope for this MVP. PDFs are read only when pdftotext can extract embedded text.",
    }
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        manifest["scanned_files"] += 1
        suffix = path.suffix.lower() or "(none)"
        manifest["extensions"][suffix] = manifest["extensions"].get(suffix, 0) + 1
        text = extract_text(path)
        if not text.strip():
            manifest["skipped_files"] += 1
            manifest["files"].append({
                "path": str(path),
                "extension": suffix,
                "status": "skipped",
                "reason": _ingest_skip_reason(path),
            })
            continue
        doc = Document(
            doc_id=_doc_id(path),
            source_path=str(path),
            title=_title_from_text(text, path),
            text=_normalize_text(text),
        )
        docs.append(doc)
        manifest["ingested_files"] += 1
        manifest["files"].append({
            "path": str(path),
            "extension": suffix,
            "status": "ingested",
            "doc_id": doc.doc_id,
            "title": doc.title,
            "characters": len(doc.text),
        })

    out = root / "data" / "corpus" / "documents.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.__dict__, ensure_ascii=True) + "\n")
    (root / "data" / "corpus" / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(docs)


def fetch_edgar_samples(root: Path, *, limit: int = 10, user_agent: str | None = None) -> dict[str, Any]:
    init_project(root)
    output_dir = root / "data" / "raw_contracts" / "edgar_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    ua = user_agent or SEC_USER_AGENT
    downloaded: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for company, cik in EDGAR_SAMPLE_COMPANIES:
        if len(downloaded) >= limit:
            break
        submissions = _sec_json(f"https://data.sec.gov/submissions/CIK{cik}.json", ua)
        recent = submissions.get("filings", {}).get("recent", {})
        filings = zip(
            recent.get("form", []),
            recent.get("accessionNumber", []),
            recent.get("filingDate", []),
        )
        for form, accession, filing_date in list(filings)[:90]:
            if len(downloaded) >= limit:
                break
            if form not in {"8-K", "10-K", "10-Q", "S-1", "S-1/A"}:
                continue
            cik_number = str(int(cik))
            accession_compact = accession.replace("-", "")
            index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_number}/{accession_compact}/index.json"
            try:
                index = _sec_json(index_url, ua)
            except urllib.error.URLError:
                continue
            for item in index.get("directory", {}).get("item", []):
                if len(downloaded) >= limit:
                    break
                filename = item.get("name", "")
                if not _looks_like_contract_exhibit(filename):
                    continue
                file_url = f"https://www.sec.gov/Archives/edgar/data/{cik_number}/{accession_compact}/{filename}"
                if file_url in seen_urls:
                    continue
                seen_urls.add(file_url)
                local_name = f"{filing_date}_{_slug(company)}_{accession_compact}_{_safe_filename(filename)}"
                local_path = output_dir / local_name
                try:
                    content = _sec_bytes(file_url, ua)
                except urllib.error.URLError:
                    continue
                local_path.write_bytes(content)
                downloaded.append({
                    "company": company,
                    "cik": cik,
                    "form": form,
                    "filing_date": filing_date,
                    "accession": accession,
                    "source_url": file_url,
                    "local_path": str(local_path),
                    "bytes": len(content),
                })
                time.sleep(0.12)
            time.sleep(0.12)

    manifest_path = output_dir / "edgar_manifest.json"
    manifest = {
        "source": "SEC EDGAR public archives",
        "downloaded": downloaded,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "downloaded": len(downloaded),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
    }


def prepare_cuad_sample(
    root: Path,
    *,
    source_dir: Path | None = None,
    limit: int = 12,
    contains: str | None = None,
) -> dict[str, Any]:
    init_project(root)
    source = source_dir or LOCAL_CUAD_DIR
    csv_path = source / "master_clauses.csv"
    text_dir = source / "full_contract_txt"
    if not csv_path.exists() or not text_dir.exists():
        raise FileNotFoundError(f"CUAD_v1 source must contain master_clauses.csv and full_contract_txt: {source}")

    output_dir = root / "data" / "raw_contracts" / "cuad_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    gold: dict[str, Any] = {
        "source": "CUAD v1 / The Atticus Project",
        "source_dir": str(source),
        "items": [],
    }
    selected = 0
    needle = contains.lower() if contains else None

    with csv_path.open(encoding="utf-8-sig", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            filename = row.get("Filename", "")
            contract_type = _contract_type_from_cuad_filename(filename)
            if needle and needle not in filename.lower() and needle not in contract_type.lower():
                continue
            source_txt = text_dir / f"{Path(filename).stem}.txt"
            if not source_txt.exists():
                continue
            local_path = output_dir / _safe_filename(source_txt.name)
            shutil.copy2(source_txt, local_path)
            doc_id = _doc_id(local_path)
            gold["items"].append({
                "doc_id": doc_id,
                "filename": filename,
                "source_path": str(local_path),
                "contract_type": contract_type,
                "coversheet": _cuad_coversheet(row),
                "key_clauses": _cuad_key_clauses(row),
            })
            selected += 1
            if selected >= limit:
                break

    gold_path = root / "data" / "gold" / "cuad_review_labels.json"
    manifest_path = output_dir / "cuad_manifest.json"
    gold_path.write_text(json.dumps(gold, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "source": gold["source"],
        "source_dir": str(source),
        "contracts": gold["items"],
    }, indent=2), encoding="utf-8")
    return {
        "prepared": len(gold["items"]),
        "output_dir": str(output_dir),
        "gold_labels": str(gold_path),
        "manifest": str(manifest_path),
    }


def run_cuad_demo(
    root: Path,
    *,
    model: str = "qwen3:4b",
    limit: int = 4,
    contains: str | None = "license",
    source_dir: Path | None = None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

    steps.append({"step": "reset", "result": reset_project(root)})
    steps.append({"step": "interview", "result": save_interview(root, Path("config/interview.example.json"))})
    sample = prepare_cuad_sample(root, source_dir=source_dir, limit=limit, contains=contains)
    steps.append({"step": "cuad_sample", "result": sample})
    documents_ingested = ingest_folder(root / "data" / "raw_contracts" / "cuad_samples", root)
    steps.append({"step": "ingest", "result": {"documents_ingested": documents_ingested}})
    steps.append({"step": "baseline", "result": run_extraction(root, model=model, mode="baseline")})
    steps.append({"step": "agent_analyze", "result": run_agent_analysis(root, model=model, run="baseline")})
    review_packet = generate_review_packet(root, limit=limit)
    steps.append({"step": "review_packet", "result": {"review_packet": str(review_packet)}})
    steps.append({"step": "cuad_apply_gold", "result": apply_cuad_gold_review(root)})
    steps.append({"step": "apply_review", "result": apply_review(root, root / "data" / "reviews" / "review_packet.reviewed.json")})
    steps.append({"step": "second_run", "result": run_extraction(root, model=model, mode="second")})
    benchmark = generate_benchmark(root)
    steps.append({"step": "benchmark", "result": benchmark})
    report = generate_demo_report(root)
    steps.append({"step": "demo_report", "result": report})

    manifest = {
        "demo": "cuad_reviewed_context",
        "created_at": _now_iso(),
        "model": model,
        "limit": limit,
        "contains": contains,
        "documents_ingested": documents_ingested,
        "baseline_contract_type_accuracy": benchmark.get("baseline_contract_type_accuracy"),
        "second_run_contract_type_accuracy": benchmark.get("second_run_contract_type_accuracy"),
        "training_pairs": report.get("training_pairs"),
        "report_markdown": report.get("report_markdown"),
        "steps": steps,
    }
    manifest_path = root / "data" / "runs" / "demo_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        return _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".docx":
        return _docx_to_text(path)
    if suffix == ".pdf":
        return _pdf_to_text(path)
    return ""


def run_extraction(root: Path, *, model: str, mode: str) -> dict[str, Any]:
    init_project(root)
    docs = _load_documents(root)
    interview = _load_json(root / "data" / "memory" / "interview.json", {})
    taxonomy = _load_json(root / "data" / "memory" / "taxonomy.json", _empty_taxonomy())
    use_memory = mode == "second"

    outputs = []
    for doc in docs:
        prompt = build_extraction_prompt(
            interview=interview,
            taxonomy=_compact_reviewed_context(taxonomy) if use_memory else {},
            doc_title=doc.title,
            doc_text=doc.text[:12000],
            use_memory=use_memory,
        )
        extracted = _call_ollama_json(model=model, prompt=prompt)
        engine = "ollama"
        if not _valid_extraction(extracted):
            extracted = _heuristic_extract(doc, interview, taxonomy if use_memory else {})
            engine = "heuristic_fallback"
        extracted["doc_id"] = doc.doc_id
        extracted["title"] = doc.title
        extracted["source_path"] = doc.source_path
        extracted["mode"] = mode
        extracted["model"] = model
        extracted["engine"] = engine
        outputs.append(extracted)

    out_path = root / "data" / "runs" / ("baseline_results.json" if mode == "baseline" else "second_run_results.json")
    out_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return {"mode": mode, "model": model, "documents": len(outputs), "output": str(out_path)}


def run_agent_analysis(root: Path, *, model: str, run: str = "baseline") -> dict[str, Any]:
    init_project(root)
    docs = _load_documents(root)
    interview = _load_json(root / "data" / "memory" / "interview.json", {})
    taxonomy = _load_json(root / "data" / "memory" / "taxonomy.json", _empty_taxonomy())
    run_path = root / "data" / "runs" / ("baseline_results.json" if run == "baseline" else "second_run_results.json")
    extractions = {item.get("doc_id"): item for item in _load_json(run_path, [])}

    outputs = []
    for doc in docs:
        extraction = extractions.get(doc.doc_id, {})
        prompt = build_agent_analysis_prompt(
            interview=interview,
            taxonomy=_compact_reviewed_context(taxonomy),
            doc_title=doc.title,
            doc_text=doc.text[:12000],
            extraction=extraction,
        )
        analysis = _call_ollama_json(model=model, prompt=prompt)
        engine = "ollama"
        if not _valid_agent_analysis(analysis):
            analysis = _heuristic_agent_analysis(doc, extraction, interview, taxonomy)
            engine = "heuristic_fallback"
        analysis["doc_id"] = doc.doc_id
        analysis["title"] = doc.title
        analysis["source_path"] = doc.source_path
        analysis["run"] = run
        analysis["model"] = model
        analysis["engine"] = engine
        outputs.append(analysis)

    out_path = root / "data" / "runs" / "agent_analysis.json"
    out_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return {"run": run, "model": model, "documents": len(outputs), "output": str(out_path)}


def generate_review_packet(root: Path, *, limit: int = 50) -> Path:
    baseline_path = root / "data" / "runs" / "baseline_results.json"
    results = _load_json(baseline_path, [])
    analyses = {item.get("doc_id"): item for item in _load_json(root / "data" / "runs" / "agent_analysis.json", [])}
    packet = {
        "instructions": "Review business-level outputs only. Confirm, edit, or reject contract type, coversheet fields, key clauses, and evidence.",
        "items": [],
    }
    for item in results[:limit]:
        analysis = analyses.get(item.get("doc_id"), {})
        packet["items"].append({
            "doc_id": item["doc_id"],
            "title": item.get("title", ""),
            "status": "pending",
            "proposed_contract_type": item.get("contract_type"),
            "accepted_contract_type": item.get("contract_type"),
            "contract_type_correct": None,
            "coversheet": item.get("coversheet", {}),
            "accepted_coversheet": item.get("coversheet", {}),
            "key_clauses": item.get("key_clauses", []),
            "accepted_key_clauses": item.get("key_clauses", []),
            "evidence_sufficient": None,
            "agent_analysis": analysis,
            "alternative_contract_types": analysis.get("alternative_contract_types", []),
            "missing_expected_elements": analysis.get("missing_expected_elements", []),
            "evidence_gaps": analysis.get("evidence_gaps", []),
            "reviewer_questions": analysis.get("reviewer_questions", []),
            "taxonomy_suggestions": _taxonomy_suggestions(item) + analysis.get("taxonomy_suggestions", []),
            "playbook_suggestions": _playbook_suggestions(item) + analysis.get("playbook_suggestions", []),
            "reviewer_authority": "sme",
            "review_notes": "",
        })

    json_path = root / "data" / "reviews" / "review_packet.pending.json"
    md_path = root / "data" / "reviews" / "review_packet.pending.md"
    json_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    md_path.write_text(_packet_markdown(packet), encoding="utf-8")
    return json_path


def apply_review(root: Path, review_path: Path) -> dict[str, Any]:
    review = _load_json(review_path, {})
    taxonomy = _load_json(root / "data" / "memory" / "taxonomy.json", _empty_taxonomy())
    baseline = {r["doc_id"]: r for r in _load_json(root / "data" / "runs" / "baseline_results.json", [])}
    docs = {doc.doc_id: doc for doc in _load_documents(root)}
    pairs_path = root / "data" / "training" / "training_pairs.jsonl"

    reviewed = 0
    with pairs_path.open("w", encoding="utf-8") as pairs:
        for item in review.get("items", []):
            if item.get("status") not in {"accepted", "edited", "rejected"}:
                continue
            reviewed += 1
            doc_id = item["doc_id"]
            baseline_output = baseline.get(doc_id, {})
            doc = docs.get(doc_id)
            accepted_type = item.get("accepted_contract_type") or item.get("proposed_contract_type")
            if item.get("status") == "rejected":
                _taxonomy_add(taxonomy, "rejected_patterns", {
                    "doc_id": doc_id,
                    "title": item.get("title"),
                    "rejected_contract_type": item.get("proposed_contract_type"),
                    "reason": item.get("review_notes", ""),
                })
            elif accepted_type:
                _taxonomy_add(taxonomy, "contract_types", accepted_type)
                _add_reviewed_example(taxonomy, item, baseline_output, accepted_type)
                _update_playbook(taxonomy, item, accepted_type)
            for clause in item.get("accepted_key_clauses", []):
                family = clause.get("family") if isinstance(clause, dict) else str(clause)
                if family:
                    _taxonomy_add(taxonomy, "clause_families", family)
            pair = {
                "instruction": "Classify the contract and produce a coversheet with evidence.",
                "input": {
                    "contract": {
                        "doc_id": doc_id,
                        "title": item.get("title"),
                        "source_path": baseline_output.get("source_path") or (doc.source_path if doc else ""),
                        "text_excerpt": (doc.text[:6000] if doc else ""),
                    },
                    "baseline_model_output": baseline_output,
                },
                "output": {
                    "reviewed_answer": {
                        "contract_type": accepted_type,
                        "coversheet": item.get("accepted_coversheet", {}),
                        "key_clauses": item.get("accepted_key_clauses", []),
                        "reviewer_authority": item.get("reviewer_authority", "sme"),
                        "review_notes": item.get("review_notes", ""),
                    }
                },
                "label_source": "human_review",
                "example_type": "reviewed_correction",
            }
            pairs.write(json.dumps(pair, ensure_ascii=True) + "\n")

    (root / "data" / "memory" / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    return {"reviewed_items": reviewed, "taxonomy": str(root / "data" / "memory" / "taxonomy.json"), "training_pairs": str(pairs_path)}


def accept_pending_review(root: Path, *, note: str) -> dict[str, Any]:
    pending_path = root / "data" / "reviews" / "review_packet.pending.json"
    reviewed_path = root / "data" / "reviews" / "review_packet.reviewed.json"
    packet = _load_json(pending_path, None)
    if not isinstance(packet, dict):
        raise ValueError(f"Pending review packet not found or invalid: {pending_path}")
    for item in packet.get("items", []):
        item["status"] = "accepted"
        item["contract_type_correct"] = True
        item["evidence_sufficient"] = True
        item["reviewer_authority"] = "demo_sme"
        item["review_notes"] = note
    reviewed_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return {"reviewed_packet": str(reviewed_path), "items": len(packet.get("items", [])), "demo_only": True}


def apply_cuad_gold_review(root: Path) -> dict[str, Any]:
    pending_path = root / "data" / "reviews" / "review_packet.pending.json"
    reviewed_path = root / "data" / "reviews" / "review_packet.reviewed.json"
    gold_path = root / "data" / "gold" / "cuad_review_labels.json"
    packet = _load_json(pending_path, None)
    gold = _load_json(gold_path, None)
    if not isinstance(packet, dict):
        raise ValueError(f"Pending review packet not found or invalid: {pending_path}")
    if not isinstance(gold, dict):
        raise ValueError(f"CUAD gold labels not found or invalid: {gold_path}")
    gold_by_doc = {item["doc_id"]: item for item in gold.get("items", [])}
    applied = 0
    missing = 0
    for item in packet.get("items", []):
        label = gold_by_doc.get(item.get("doc_id"))
        if not label:
            missing += 1
            continue
        item["status"] = "edited" if item.get("proposed_contract_type") != label["contract_type"] else "accepted"
        item["accepted_contract_type"] = label["contract_type"]
        item["contract_type_correct"] = item.get("proposed_contract_type") == label["contract_type"]
        item["accepted_coversheet"] = _merge_coversheet(item.get("coversheet", {}), label.get("coversheet", {}))
        item["accepted_key_clauses"] = label.get("key_clauses", item.get("key_clauses", []))
        item["evidence_sufficient"] = True
        item["reviewer_authority"] = "cuad_expert_annotation"
        item["review_notes"] = "Applied CUAD v1 expert annotation as reviewed label."
        applied += 1
    reviewed_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return {"reviewed_packet": str(reviewed_path), "applied": applied, "missing_gold_labels": missing}


def generate_benchmark(root: Path) -> dict[str, Any]:
    review_path = root / "data" / "reviews" / "review_packet.reviewed.json"
    baseline = {r["doc_id"]: r for r in _load_json(root / "data" / "runs" / "baseline_results.json", [])}
    second = {r["doc_id"]: r for r in _load_json(root / "data" / "runs" / "second_run_results.json", [])}
    review = _load_json(review_path, {"items": []})

    rows = []
    for item in review.get("items", []):
        if item.get("status") not in {"accepted", "edited", "rejected"}:
            continue
        doc_id = item["doc_id"]
        gold = item.get("accepted_contract_type")
        baseline_item = baseline.get(doc_id, {})
        second_item = second.get(doc_id, {})
        b = baseline_item.get("contract_type")
        s = second_item.get("contract_type")
        baseline_coversheet = _coversheet_score(baseline_item.get("coversheet", {}), item.get("accepted_coversheet", {}))
        second_coversheet = _coversheet_score(second_item.get("coversheet", {}), item.get("accepted_coversheet", {}))
        baseline_clauses = _clause_family_score(baseline_item.get("key_clauses", []), item.get("accepted_key_clauses", []))
        second_clauses = _clause_family_score(second_item.get("key_clauses", []), item.get("accepted_key_clauses", []))
        rows.append({
            "doc_id": doc_id,
            "gold_contract_type": gold,
            "baseline_contract_type": b,
            "second_run_contract_type": s,
            "baseline_correct": b == gold,
            "second_run_correct": s == gold,
            "baseline_coversheet_field_accuracy": baseline_coversheet["accuracy"],
            "second_run_coversheet_field_accuracy": second_coversheet["accuracy"],
            "coversheet_fields_compared": baseline_coversheet["fields_compared"],
            "baseline_clause_family_f1": baseline_clauses["f1"],
            "second_run_clause_family_f1": second_clauses["f1"],
            "baseline_clause_family_precision": baseline_clauses["precision"],
            "second_run_clause_family_precision": second_clauses["precision"],
            "baseline_clause_family_recall": baseline_clauses["recall"],
            "second_run_clause_family_recall": second_clauses["recall"],
            "gold_clause_families": sorted(baseline_clauses["gold_families"]),
            "baseline_clause_families": sorted(baseline_clauses["predicted_families"]),
            "second_run_clause_families": sorted(second_clauses["predicted_families"]),
        })

    benchmark = {
        "reviewed_documents": len(rows),
        "baseline_contract_type_accuracy": _accuracy(row["baseline_correct"] for row in rows),
        "second_run_contract_type_accuracy": _accuracy(row["second_run_correct"] for row in rows),
        "baseline_coversheet_field_accuracy": _mean(row["baseline_coversheet_field_accuracy"] for row in rows),
        "second_run_coversheet_field_accuracy": _mean(row["second_run_coversheet_field_accuracy"] for row in rows),
        "baseline_clause_family_f1": _mean(row["baseline_clause_family_f1"] for row in rows),
        "second_run_clause_family_f1": _mean(row["second_run_clause_family_f1"] for row in rows),
        "baseline_clause_family_precision": _mean(row["baseline_clause_family_precision"] for row in rows),
        "second_run_clause_family_precision": _mean(row["second_run_clause_family_precision"] for row in rows),
        "baseline_clause_family_recall": _mean(row["baseline_clause_family_recall"] for row in rows),
        "second_run_clause_family_recall": _mean(row["second_run_clause_family_recall"] for row in rows),
        "rows": rows,
    }
    out = root / "data" / "runs" / "benchmark.json"
    out.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    return benchmark


def generate_demo_report(root: Path) -> dict[str, Any]:
    docs = _load_documents(root)
    baseline = _load_json(root / "data" / "runs" / "baseline_results.json", [])
    second = _load_json(root / "data" / "runs" / "second_run_results.json", [])
    benchmark = _load_json(root / "data" / "runs" / "benchmark.json", {})
    taxonomy = _load_json(root / "data" / "memory" / "taxonomy.json", _empty_taxonomy())
    edgar_manifest = _load_json(root / "data" / "raw_contracts" / "edgar_samples" / "edgar_manifest.json", {"downloaded": []})
    cuad_manifest = _load_json(root / "data" / "raw_contracts" / "cuad_samples" / "cuad_manifest.json", {"contracts": []})
    training_pairs = _count_lines(root / "data" / "training" / "training_pairs.jsonl")

    summary = {
        "documents_ingested": len(docs),
        "edgar_files_downloaded": len(edgar_manifest.get("downloaded", [])),
        "cuad_contracts_prepared": len(cuad_manifest.get("contracts", [])),
        "baseline_documents": len(baseline),
        "second_run_documents": len(second),
        "training_pairs": training_pairs,
        "baseline_engines": sorted({item.get("engine", "unknown") for item in baseline}),
        "second_run_engines": sorted({item.get("engine", "unknown") for item in second}),
        "baseline_contract_type_accuracy": benchmark.get("baseline_contract_type_accuracy"),
        "second_run_contract_type_accuracy": benchmark.get("second_run_contract_type_accuracy"),
        "baseline_coversheet_field_accuracy": benchmark.get("baseline_coversheet_field_accuracy"),
        "second_run_coversheet_field_accuracy": benchmark.get("second_run_coversheet_field_accuracy"),
        "baseline_clause_family_f1": benchmark.get("baseline_clause_family_f1"),
        "second_run_clause_family_f1": benchmark.get("second_run_clause_family_f1"),
        "contract_types_in_memory": len(taxonomy.get("contract_types", [])),
        "reviewed_examples_in_memory": len(taxonomy.get("reviewed_examples", [])),
    }

    report_json = root / "data" / "runs" / "demo_report.json"
    report_md = root / "data" / "runs" / "demo_report.md"
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_md.write_text(_demo_report_markdown(summary, baseline, taxonomy, edgar_manifest, cuad_manifest), encoding="utf-8")
    return {"report_json": str(report_json), "report_markdown": str(report_md), **summary}


def _call_ollama_json(*, model: str, prompt: str) -> dict[str, Any] | None:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    response = payload.get("response", "") or payload.get("thinking", "")
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _valid_extraction(value: dict[str, Any] | None) -> bool:
    return isinstance(value, dict) and isinstance(value.get("contract_type"), str) and bool(value["contract_type"].strip())


def _valid_agent_analysis(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("review_priority") not in {"low", "medium", "high"}:
        return False
    return isinstance(value.get("challenge_summary"), str) and bool(value["challenge_summary"].strip())


def _compact_reviewed_context(taxonomy: dict[str, Any]) -> dict[str, Any]:
    playbook = taxonomy.get("playbook", {})
    return {
        "contract_types": taxonomy.get("contract_types", [])[:40],
        "contract_type_aliases": taxonomy.get("contract_type_aliases", {}),
        "clause_families": taxonomy.get("clause_families", [])[:60],
        "accepted_patterns": taxonomy.get("accepted_patterns", [])[-12:],
        "rejected_patterns": taxonomy.get("rejected_patterns", [])[-12:],
        "reviewed_examples": [
            {
                "title": item.get("title"),
                "contract_type": item.get("contract_type"),
                "key_clause_families": [
                    clause.get("family")
                    for clause in item.get("key_clauses", [])[:8]
                    if isinstance(clause, dict) and clause.get("family")
                ],
            }
            for item in taxonomy.get("reviewed_examples", [])[-8:]
        ],
        "playbook": {
            "contract_types": {
                name: {
                    "reviewed_count": info.get("reviewed_count", 0),
                    "expected_clause_families": info.get("expected_clause_families", [])[:12],
                }
                for name, info in playbook.get("contract_types", {}).items()
            }
        },
    }


def _sec_json(url: str, user_agent: str) -> dict[str, Any]:
    return json.loads(_sec_bytes(url, user_agent).decode("utf-8"))


def _sec_bytes(url: str, user_agent: str) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": user_agent,
        "Accept": "application/json,text/html,text/plain,*/*",
    })
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def _looks_like_contract_exhibit(filename: str) -> bool:
    lower = filename.lower()
    if not lower.endswith((".htm", ".html", ".txt")):
        return False
    if "index" in lower:
        return False
    return any(pattern in lower for pattern in EDGAR_EXHIBIT_PATTERNS)


def _contract_type_from_cuad_filename(filename: str) -> str:
    stem = Path(filename).stem
    agreement_parts = [
        part for part in re.split(r"_+", stem)
        if re.search(r"\b(agreement|contract|lease|amendment|addendum)\b", part, re.I)
    ]
    tail = agreement_parts[-1] if agreement_parts else stem
    tail = re.sub(r"^\d+[-_\s]*", "", tail)
    tail = re.sub(r"\d+$", "", tail)
    tail = tail.replace("_", " ").replace("-", " ")
    tail = re.sub(r"\s+", " ", tail).strip(" .")
    if not tail:
        return "Unknown Agreement"
    return tail.title().replace(" And ", " and ").replace(" Of ", " of ")


def _cuad_coversheet(row: dict[str, str]) -> dict[str, Any]:
    coversheet: dict[str, Any] = {}
    for column, key in CUAD_COVERSHEET_COLUMNS.items():
        answer = row.get(f"{column}-Answer", "").strip()
        context = _first_cuad_context(row.get(column, ""))
        coversheet[key] = {
            "accepted_value": answer,
            "evidence": context,
            "authority": "cuad_expert_annotation",
        }
    return coversheet


def _cuad_key_clauses(row: dict[str, str]) -> list[dict[str, Any]]:
    clauses = []
    for column, family in CUAD_CLAUSE_FAMILY_MAP.items():
        context = _first_cuad_context(row.get(column, ""))
        answer = row.get(f"{column}-Answer", "").strip()
        if not context and not answer:
            continue
        clauses.append({
            "family": family,
            "cuad_category": column,
            "answer": answer,
            "evidence": context,
            "confidence": 1.0,
            "authority": "cuad_expert_annotation",
        })
    return clauses


def _first_cuad_context(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw or raw == "[]":
        return ""
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if isinstance(parsed, list) and parsed:
        return str(parsed[0]).strip()
    if isinstance(parsed, str):
        return parsed.strip()
    return ""


def _merge_coversheet(model_fields: dict[str, Any], gold_fields: dict[str, Any]) -> dict[str, Any]:
    merged = dict(model_fields or {})
    for key, value in gold_fields.items():
        if isinstance(value, dict) and not value.get("accepted_value") and not value.get("evidence"):
            continue
        merged[key] = value
    return merged


def _heuristic_extract(doc: Document, interview: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
    text = doc.text.lower()
    candidates = list(interview.get("expected_contract_types", [])) + list(taxonomy.get("contract_types", []))
    scores: dict[str, int] = {}
    for candidate in candidates:
        words = re.findall(r"[a-z]+", candidate.lower())
        scores[candidate] = sum(1 for word in words if word in text)
    contract_type = max(scores, key=scores.get) if scores else "Unknown Agreement"
    if scores and scores[contract_type] == 0:
        contract_type = _fallback_contract_type(text)

    clauses = []
    for family, keywords in _clause_keywords().items():
        evidence = _first_sentence_with(text=doc.text, keywords=keywords)
        if evidence:
            clauses.append({"family": family, "evidence": evidence, "confidence": 0.7})

    return {
        "contract_type": contract_type,
        "confidence": 0.62,
        "rationale": "Heuristic fallback used because Ollama was unavailable or returned invalid JSON.",
        "coversheet": {
            "parties": _guess_parties(doc.text),
            "effective_date": _first_date(doc.text),
            "territory": _first_sentence_with(doc.text, ["united states", "california", "oregon", "washington"]) or "",
            "governing_law": _first_sentence_with(doc.text, ["governed by", "laws of"]) or "",
        },
        "key_clauses": clauses,
        "evidence": [c["evidence"] for c in clauses[:3]],
    }


def _heuristic_agent_analysis(
    doc: Document,
    extraction: dict[str, Any],
    interview: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    current_type = str(extraction.get("contract_type") or "Unknown Agreement")
    title_and_text = f"{doc.title}\n{doc.text[:5000]}"
    lower = title_and_text.lower()
    alternatives = _agent_type_alternatives(doc, current_type, lower)
    extracted_families = {
        clause.get("family")
        for clause in extraction.get("key_clauses", [])
        if isinstance(clause, dict) and clause.get("family")
    }
    expected_families = _expected_clause_families(current_type, interview, taxonomy)
    missing = [
        {
            "element": family,
            "why_it_matters": f"{family} is expected for this playbook/corpus and was not supported by first-pass evidence.",
        }
        for family in expected_families
        if family not in extracted_families
    ][:8]

    evidence_gaps = []
    if not extraction.get("evidence"):
        evidence_gaps.append("First-pass output did not include top-level evidence spans for the contract type.")
    coversheet = extraction.get("coversheet", {}) if isinstance(extraction.get("coversheet"), dict) else {}
    sparse_fields = [key for key, value in coversheet.items() if not value]
    if sparse_fields:
        evidence_gaps.append(f"Coversheet has empty or unsupported fields: {', '.join(sparse_fields[:6])}.")
    if alternatives:
        evidence_gaps.append("Title or opening text points to a more specific subtype than the first-pass label.")

    taxonomy_suggestions = []
    for alt in alternatives:
        taxonomy_suggestions.append({
            "suggestion_type": "canonical_type",
            "suggested_value": alt["label"],
            "reason": alt["reason"],
        })
        if current_type and current_type != alt["label"]:
            taxonomy_suggestions.append({
                "suggestion_type": "relationship",
                "suggested_value": f"{alt['label']} is more specific than {current_type}",
                "reason": "Use reviewed labels to distinguish broad labels from corpus-specific subtypes.",
            })

    playbook_suggestions = [
        {
            "suggestion_type": "expected_clause",
            "suggested_value": item["element"],
            "reason": item["why_it_matters"],
        }
        for item in missing[:5]
    ]
    if alternatives:
        playbook_suggestions.append({
            "suggestion_type": "review_rule",
            "suggested_value": "When the title contains a specific agreement subtype, ask the reviewer to validate that subtype instead of accepting a broad label.",
            "reason": "This prevents generic labels such as License Agreement from hiding business-useful distinctions.",
        })

    questions = []
    if alternatives:
        questions.append(f"Should this corpus label this document as {alternatives[0]['label']} instead of {current_type}?")
    if missing:
        questions.append("Which missing expected clause families should be required in future review packets?")
    if not questions:
        questions.append("Does the proposed contract type and evidence meet the business review standard?")

    if alternatives:
        priority = "high"
        summary = f"First-pass type '{current_type}' may be too broad; strongest alternative is '{alternatives[0]['label']}'."
    elif len(missing) >= 4 or len(evidence_gaps) >= 2:
        priority = "medium"
        summary = "First-pass type is plausible, but expected business evidence is incomplete."
    else:
        priority = "low"
        summary = "No major challenge found; reviewer should confirm the business label and evidence."

    return {
        "review_priority": priority,
        "challenge_summary": summary,
        "alternative_contract_types": alternatives,
        "missing_expected_elements": missing,
        "evidence_gaps": evidence_gaps,
        "taxonomy_suggestions": taxonomy_suggestions,
        "playbook_suggestions": playbook_suggestions,
        "reviewer_questions": questions,
    }


def _agent_type_alternatives(doc: Document, current_type: str, lower_text: str) -> list[dict[str, str]]:
    patterns = [
        (
            "Content License Agreement",
            ["content license agreement", "website content license", "digital content license"],
            "The title or opening text names a content-license subtype.",
        ),
        (
            "Software License Agreement",
            ["software license agreement", "source code license", "object code license"],
            "The evidence points to a software-specific license.",
        ),
        (
            "Distribution Agreement",
            ["distribution agreement", "distributor agreement", "distributorship agreement"],
            "The evidence points to distribution/resale obligations.",
        ),
        (
            "Reseller Agreement",
            ["reseller agreement", "resell", "value added reseller"],
            "The evidence points to reseller rights or channel sales.",
        ),
        (
            "Master Services Agreement",
            ["master services agreement", "statement of work", "professional services"],
            "The evidence points to services and statement-of-work governance.",
        ),
        (
            "Supply Agreement",
            ["supply agreement", "purchase and supply", "manufacturing and supply"],
            "The evidence points to supply-chain obligations.",
        ),
    ]
    current_norm = current_type.lower().strip()
    alternatives = []
    for label, needles, reason in patterns:
        if label.lower() == current_norm:
            continue
        evidence = next((needle for needle in needles if needle in lower_text), "")
        if evidence:
            alternatives.append({
                "label": label,
                "reason": reason,
                "evidence": _first_sentence_with(doc.text, [evidence]) or doc.title,
            })
    return alternatives[:3]


def _expected_clause_families(
    contract_type: str,
    interview: dict[str, Any],
    taxonomy: dict[str, Any],
) -> list[str]:
    playbook = taxonomy.get("playbook", {}).get("contract_types", {})
    exact = playbook.get(contract_type, {}).get("expected_clause_families", [])
    if exact:
        return list(dict.fromkeys(exact))

    expected = list(interview.get("key_clause_families", []))
    if not expected:
        return []
    lower_type = contract_type.lower()
    license_families = [
        "grant_of_rights",
        "territory",
        "term_and_termination",
        "payment_or_royalty",
        "ip_ownership",
        "confidentiality",
        "governing_law",
    ]
    service_families = [
        "services_scope",
        "term_and_termination",
        "payment_or_royalty",
        "confidentiality",
        "liability",
        "governing_law",
    ]
    if "license" in lower_type:
        preferred = license_families
    elif "service" in lower_type:
        preferred = service_families
    else:
        preferred = expected
    return [family for family in preferred if family in expected] or expected[:8]


def _empty_taxonomy() -> dict[str, Any]:
    return {
        "contract_types": [],
        "contract_type_aliases": {},
        "clause_families": [],
        "accepted_patterns": [],
        "rejected_patterns": [],
        "reviewed_examples": [],
        "playbook": {
            "contract_types": {},
            "clause_families": {},
        },
    }


def _merge_interview_taxonomy(taxonomy: dict[str, Any], interview: dict[str, Any]) -> bool:
    changed = False
    for contract_type in interview.get("expected_contract_types", []):
        before = json.dumps(taxonomy, sort_keys=True)
        _taxonomy_add(taxonomy, "contract_types", contract_type)
        aliases = interview.get("contract_type_aliases", {}).get(contract_type, [])
        taxonomy.setdefault("contract_type_aliases", {}).setdefault(contract_type, [])
        for alias in aliases:
            if alias not in taxonomy["contract_type_aliases"][contract_type]:
                taxonomy["contract_type_aliases"][contract_type].append(alias)
        changed = changed or before != json.dumps(taxonomy, sort_keys=True)
    for family in interview.get("key_clause_families", []):
        before = json.dumps(taxonomy, sort_keys=True)
        _taxonomy_add(taxonomy, "clause_families", family)
        changed = changed or before != json.dumps(taxonomy, sort_keys=True)
    return changed


def _load_documents(root: Path) -> list[Document]:
    path = root / "data" / "corpus" / "documents.jsonl"
    docs = []
    if not path.exists():
        return docs
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(Document(**json.loads(line)))
    return docs


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _doc_id(path: Path) -> str:
    return f"doc_{uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())).hex[:12]}"


def _title_from_text(text: str, path: Path) -> str:
    filename_type = _contract_type_from_cuad_filename(path.name)
    if filename_type != "Unknown Agreement":
        filename_title = filename_type
    else:
        filename_title = path.stem.replace("_", " ").replace("-", " ").strip()

    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("EXHIBIT") or _is_boilerplate_heading(line):
            continue
        candidates.append(line)
        if _looks_like_contract_title(line):
            return _clean_title(line)
    if filename_title:
        return _clean_title(filename_title)
    if candidates:
        return _clean_title(candidates[0])
    return _clean_title(path.stem)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_boilerplate_heading(line: str) -> bool:
    normalized = re.sub(r"[^A-Z]", "", line.upper())
    return normalized in {
        "CONFIDENTIAL",
        "STRICTLYCONFIDENTIAL",
        "EXECUTIONCOPY",
        "FINAL",
        "DRAFT",
        "CERTAINCONFIDENTIALINFORMATIONCONTAINEDINTHISDOCUMENT",
    }


def _looks_like_contract_title(line: str) -> bool:
    lower = line.lower()
    return any(token in lower for token in ["agreement", "contract", "license", "amendment", "addendum", "lease"])


def _clean_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:140]


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return html.unescape(re.sub(r"\s+", " ", raw))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:120]


def _ingest_skip_reason(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in {".txt", ".md", ".html", ".htm", ".docx", ".pdf"}:
        return "unsupported file extension"
    if suffix == ".pdf" and not shutil.which("pdftotext"):
        return "PDF text extraction requires pdftotext; OCR is out of scope for this MVP"
    return "no extractable text"


def _docx_to_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = [node.text or "" for node in root.findall(".//w:t", namespace)]
    return "\n".join(parts)


def _pdf_to_text(path: Path) -> str:
    if not shutil.which("pdftotext"):
        return ""
    proc = subprocess.run(["pdftotext", str(path), "-"], check=False, capture_output=True, text=True, timeout=60)
    return proc.stdout if proc.returncode == 0 else ""


def _packet_markdown(packet: dict[str, Any]) -> str:
    lines = ["# Contract Review Packet", "", packet["instructions"], ""]
    for item in packet["items"]:
        lines.extend([
            f"## {item['title']}",
            f"- Doc ID: `{item['doc_id']}`",
            f"- Proposed type: `{item.get('proposed_contract_type')}`",
            f"- Status: `{item.get('status')}`",
            "- Key clauses:",
        ])
        for clause in item.get("key_clauses", []):
            lines.append(f"  - {clause.get('family')}: {clause.get('evidence')}")
        lines.append("")
    return "\n".join(lines)


def _demo_report_markdown(
    summary: dict[str, Any],
    baseline: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    edgar_manifest: dict[str, Any],
    cuad_manifest: dict[str, Any],
) -> str:
    lines = [
        "# Contract Intelligence Demo Report",
        "",
        "## Run Summary",
        "",
        f"- Documents ingested: {summary['documents_ingested']}",
        f"- Public EDGAR files downloaded: {summary['edgar_files_downloaded']}",
        f"- CUAD contracts prepared: {summary['cuad_contracts_prepared']}",
        f"- Baseline engine(s): {', '.join(summary['baseline_engines']) or 'none'}",
        f"- Second-run engine(s): {', '.join(summary['second_run_engines']) or 'none'}",
        f"- Training pairs exported: {summary['training_pairs']}",
        f"- Baseline contract type accuracy: {summary['baseline_contract_type_accuracy']}",
        f"- Second-run contract type accuracy: {summary['second_run_contract_type_accuracy']}",
        f"- Baseline coversheet field accuracy: {summary.get('baseline_coversheet_field_accuracy')}",
        f"- Second-run coversheet field accuracy: {summary.get('second_run_coversheet_field_accuracy')}",
        f"- Baseline clause-family F1: {summary.get('baseline_clause_family_f1')}",
        f"- Second-run clause-family F1: {summary.get('second_run_clause_family_f1')}",
        "",
        "## Learned Memory",
        "",
        f"- Contract types in taxonomy: {summary['contract_types_in_memory']}",
        f"- Reviewed examples in memory: {summary['reviewed_examples_in_memory']}",
        "",
        "## Baseline Classifications",
        "",
    ]
    for item in baseline:
        lines.append(f"- {item.get('title', item.get('doc_id'))}: {item.get('contract_type')} ({item.get('engine')}, confidence {item.get('confidence')})")
    lines.extend([
        "",
        "## Public Source Files",
        "",
    ])
    for item in cuad_manifest.get("contracts", []):
        lines.append(f"- CUAD: {item.get('filename')} ({item.get('contract_type')})")
    for item in edgar_manifest.get("downloaded", []):
        lines.append(f"- {item.get('company')} {item.get('form')} {item.get('filing_date')}: {item.get('source_url')}")
    lines.extend([
        "",
        "## Demo Claim Boundary",
        "",
        "This run does not fine-tune model weights. Reviewed labels update taxonomy, playbook, examples, and rejected-pattern memory, then the same model is rerun with that reviewed context.",
    ])
    return "\n".join(lines)


def _taxonomy_add(taxonomy: dict[str, Any], key: str, value: Any) -> None:
    taxonomy.setdefault(key, [])
    if value and value not in taxonomy[key]:
        taxonomy[key].append(value)


def _taxonomy_suggestions(item: dict[str, Any]) -> list[dict[str, str]]:
    value = item.get("contract_type")
    if not value:
        return []
    return [{
        "suggestion_type": "contract_type",
        "suggested_value": value,
        "review_action": "confirm, edit, or reject canonical type for this corpus",
    }]


def _playbook_suggestions(item: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions = []
    for clause in item.get("key_clauses", []):
        family = clause.get("family") if isinstance(clause, dict) else None
        if family:
            suggestions.append({
                "suggestion_type": "clause_family_evidence",
                "suggested_value": family,
                "evidence": clause.get("evidence", ""),
                "review_action": "confirm whether this evidence belongs in the playbook",
            })
    return suggestions


def _add_reviewed_example(
    taxonomy: dict[str, Any],
    item: dict[str, Any],
    baseline_output: dict[str, Any],
    accepted_type: str,
) -> None:
    example = {
        "doc_id": item.get("doc_id"),
        "title": item.get("title"),
        "contract_type": accepted_type,
        "coversheet": item.get("accepted_coversheet", {}),
        "key_clauses": item.get("accepted_key_clauses", []),
        "source_model": baseline_output.get("model"),
        "source_engine": baseline_output.get("engine"),
        "reviewer_authority": item.get("reviewer_authority", "sme"),
    }
    _taxonomy_add(taxonomy, "reviewed_examples", example)
    _taxonomy_add(taxonomy, "accepted_patterns", {
        "contract_type": accepted_type,
        "evidence": baseline_output.get("evidence", []),
        "doc_id": item.get("doc_id"),
    })


def _update_playbook(taxonomy: dict[str, Any], item: dict[str, Any], accepted_type: str) -> None:
    playbook = taxonomy.setdefault("playbook", {}).setdefault("contract_types", {})
    entry = playbook.setdefault(accepted_type, {
        "reviewed_count": 0,
        "evidence_examples": [],
        "expected_clause_families": [],
        "coversheet_fields_seen": [],
    })
    entry["reviewed_count"] += 1
    for clause in item.get("accepted_key_clauses", []):
        if not isinstance(clause, dict):
            continue
        family = clause.get("family")
        evidence = clause.get("evidence")
        if family and family not in entry["expected_clause_families"]:
            entry["expected_clause_families"].append(family)
        if evidence:
            _taxonomy_add(entry, "evidence_examples", evidence)
        taxonomy.setdefault("playbook", {}).setdefault("clause_families", {}).setdefault(family or "other", {
            "reviewed_count": 0,
            "evidence_examples": [],
        })
        family_entry = taxonomy["playbook"]["clause_families"][family or "other"]
        family_entry["reviewed_count"] += 1
        if evidence:
            _taxonomy_add(family_entry, "evidence_examples", evidence)
    for field in item.get("accepted_coversheet", {}).keys():
        if field not in entry["coversheet_fields_seen"]:
            entry["coversheet_fields_seen"].append(field)


def _accuracy(values: Any) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def _mean(values: Any) -> float | None:
    numbers = [value for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def _coversheet_score(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    predicted = predicted if isinstance(predicted, dict) else {}
    gold = gold if isinstance(gold, dict) else {}
    compared = 0
    correct = 0
    field_rows = []
    for field, gold_value in gold.items():
        gold_norm = _normalize_benchmark_value(gold_value)
        if not gold_norm:
            continue
        compared += 1
        predicted_norm = _normalize_benchmark_value(predicted.get(field))
        match = bool(predicted_norm) and (predicted_norm in gold_norm or gold_norm in predicted_norm)
        correct += 1 if match else 0
        field_rows.append({
            "field": field,
            "predicted": predicted_norm,
            "gold": gold_norm,
            "correct": match,
        })
    return {
        "accuracy": round(correct / compared, 4) if compared else None,
        "fields_compared": compared,
        "fields_correct": correct,
        "fields": field_rows,
    }


def _clause_family_score(predicted: list[Any], gold: list[Any]) -> dict[str, Any]:
    predicted_families = _clause_families(predicted)
    gold_families = _clause_families(gold)
    if not predicted_families and not gold_families:
        precision = recall = f1 = None
    elif not predicted_families or not gold_families:
        precision = recall = f1 = 0.0
    else:
        overlap = len(predicted_families & gold_families)
        precision = round(overlap / len(predicted_families), 4)
        recall = round(overlap / len(gold_families), 4)
        f1 = round((2 * precision * recall) / (precision + recall), 4) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_families": predicted_families,
        "gold_families": gold_families,
    }


def _clause_families(clauses: list[Any]) -> set[str]:
    families = set()
    for clause in clauses or []:
        if isinstance(clause, dict) and clause.get("family"):
            families.add(str(clause["family"]))
    return families


def _normalize_benchmark_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("accepted_value") or value.get("value") or value.get("answer") or value.get("text") or ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fallback_contract_type(text: str) -> str:
    if "distributor" in text or "resell" in text:
        return "Distribution Agreement"
    if "license" in text or "royalty" in text:
        return "License Agreement"
    if "services" in text or "statement of work" in text:
        return "Service Agreement"
    return "Unknown Agreement"


def _clause_keywords() -> dict[str, list[str]]:
    return {
        "grant_of_rights": ["grants", "appoints", "license", "right to"],
        "territory": ["territory", "united states", "california", "oregon", "washington"],
        "term_and_termination": ["term", "terminate", "termination", "renew"],
        "payment_or_royalty": ["pay", "royalty", "fees", "pricing"],
        "confidentiality": ["confidential"],
        "governing_law": ["governed by", "laws of"],
    }


def _first_sentence_with(text: str, keywords: list[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            return sentence.strip()
    return ""


def _guess_parties(text: str) -> list[str]:
    match = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\.|,|\n)", text, re.I)
    if not match:
        match = re.search(r"by\s+(.+?)\s+and\s+(.+?)(?:\.|,|\n)", text, re.I)
    if not match:
        return []
    return [match.group(1).strip(), match.group(2).strip()]


def _first_date(text: str) -> str:
    match = re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b", text)
    return match.group(0) if match else ""


def extract_split(root: Path, *, split: str, primary_model: str, shadow_model: str
                  ) -> dict[str, Any]:
    """Extract one corpus split (review or holdout) with primary+shadow + verifier."""
    import asyncio
    from contract_intel_mvp.agent.shadow import run_shadow_pair
    from contract_intel_mvp.agent.verifier import extract_with_verification
    from contract_intel_mvp.splits import load_splits

    splits = load_splits(root)
    target_ids = set(splits[f"{split}_set"])
    docs = [d for d in _load_documents(root) if d.doc_id in target_ids]
    interview = _load_json(root / "data" / "memory" / "interview.json", {})
    taxonomy = _load_json(root / "data" / "memory" / "taxonomy.json", _empty_taxonomy())
    use_memory = split == "holdout"

    primary_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []

    for doc in docs:
        prompt = build_extraction_prompt(
            interview=interview, taxonomy=taxonomy, doc_title=doc.title,
            doc_text=doc.text[:8000], use_memory=use_memory)
        pair = asyncio.run(run_shadow_pair(
            prompt=prompt, primary_model=primary_model, shadow_model=shadow_model))
        primary_verified, _ = extract_with_verification(
            source_text=doc.text, model=primary_model, initial_extraction=pair.primary)
        primary_verified["doc_id"] = doc.doc_id
        primary_verified["engine"] = pair.primary_engine
        primary_verified["role"] = "primary"
        primary_rows.append(primary_verified)
        shadow_verified, _ = extract_with_verification(
            source_text=doc.text, model=shadow_model, initial_extraction=pair.shadow)
        shadow_verified["doc_id"] = doc.doc_id
        shadow_verified["engine"] = pair.shadow_engine
        shadow_verified["role"] = "shadow"
        shadow_rows.append(shadow_verified)

    runs = root / "data" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if split == "review":
        (runs / "baseline_results.json").write_text(
            json.dumps(primary_rows, indent=2), encoding="utf-8")
        (runs / "shadow_review_results.json").write_text(
            json.dumps(shadow_rows, indent=2), encoding="utf-8")
    else:
        (runs / "second_run_primary_holdout.json").write_text(
            json.dumps(primary_rows, indent=2), encoding="utf-8")
        (runs / "second_run_results.json").write_text(
            json.dumps(shadow_rows, indent=2), encoding="utf-8")
    return {"split": split, "n": len(docs)}


def extract_holdout_cold(root: Path, *, shadow_model: str) -> dict:
    """Run the small model on the holdout WITHOUT reviewed taxonomy."""
    from contract_intel_mvp.splits import load_splits
    splits = load_splits(root)
    holdout_ids = set(splits["holdout_set"])
    interview = _load_json(root / "data" / "memory" / "interview.json", {})
    rows: list[dict] = []
    for doc in _load_documents(root):
        if doc.doc_id not in holdout_ids:
            continue
        prompt = build_extraction_prompt(
            interview=interview, taxonomy={}, doc_title=doc.title,
            doc_text=doc.text[:8000], use_memory=False)
        result = _call_ollama_json(model=shadow_model, prompt=prompt)
        engine = "ollama" if result else "heuristic_fallback"
        if not result:
            result = _heuristic_extract(doc, interview, {})
        result["doc_id"] = doc.doc_id
        result["engine"] = engine
        result["role"] = "shadow_cold"
        rows.append(result)
    (root / "data" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "runs" / "shadow_holdout_cold_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    return {"n": len(rows)}


def cuad_apply_holdout_gold(root: Path) -> dict[str, Any]:
    """Project CUAD gold labels onto holdout doc_ids in the shape three_way expects."""
    from contract_intel_mvp.splits import load_splits
    splits = load_splits(root)
    holdout_ids = set(splits["holdout_set"])
    gold = _load_json(root / "data" / "gold" / "cuad_review_labels.json", {"items": []})
    rows = []
    for item in gold.get("items", []):
        if item["doc_id"] not in holdout_ids:
            continue
        rows.append({
            "doc_id": item["doc_id"],
            "accepted_contract_type": item["contract_type"],
            "accepted_coversheet": item.get("coversheet", {}),
            "accepted_key_clauses": item.get("key_clauses", []),
        })
    out = root / "data" / "reviews" / "holdout_gold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"holdout_gold": str(out), "n": len(rows)}
