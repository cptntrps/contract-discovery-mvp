"""Prompt builders for contract extraction."""

from __future__ import annotations

from typing import Any


REGROUND_PROMPT = """You previously extracted clauses from this contract, but the following clause families had evidence quotes that do NOT appear verbatim in the source: {missing_families}.

Re-extract ONLY those clause families. Each evidence_snippet MUST be an exact substring of the source. If you cannot find a verbatim quote, return null for that family.

Source text:
{source_text}

Return JSON: {{"key_clauses": [{{"family": "<family>", "evidence_snippet": "<exact substring or null>"}}]}}
"""


def build_extraction_prompt(
    *,
    interview: dict[str, Any],
    taxonomy: dict[str, Any],
    doc_title: str,
    doc_text: str,
    use_memory: bool,
) -> str:
    memory_block = ""
    if use_memory:
        memory_block = f"""
Reviewed taxonomy and playbook context:
{_reviewed_context_text(taxonomy)}

Use this compact reviewed context as guidance. If a reviewed subtype such as
"Content License Agreement" matches the document evidence, prefer that specific
reviewed label over a broader parent label such as "License Agreement".
Only return labels supported by evidence in the contract.
"""

    return f"""
You are a contract corpus intelligence agent.

Goal and scope:
{_interview_text(interview)}

{memory_block}
Analyze the contract below and return strict JSON with this shape:
{{
  "contract_type": "string",
  "confidence": 0.0,
  "rationale": "short explanation",
  "coversheet": {{
    "parties": ["party 1", "party 2"],
    "effective_date": "string or empty",
    "territory": "string or empty",
    "governing_law": "string or empty"
  }},
  "key_clauses": [
    {{
      "family": "grant_of_rights|territory|term_and_termination|payment_or_royalty|exclusivity|confidentiality|governing_law|other",
      "evidence": "exact supporting text span",
      "confidence": 0.0
    }}
  ],
  "evidence": ["top 1-3 exact text spans that justify the contract_type"]
}}

Rules:
- Do not invent facts not grounded in the contract text.
- Prefer the user's taxonomy when it matches evidence.
- If uncertain, lower confidence and explain what evidence is missing.
- The human reviewer should validate business-level outputs, not raw entities.

Document title: {doc_title}

Contract text:
{doc_text}
""".strip()


def build_agent_analysis_prompt(
    *,
    interview: dict[str, Any],
    taxonomy: dict[str, Any],
    doc_title: str,
    doc_text: str,
    extraction: dict[str, Any],
) -> str:
    return f"""
You are a contract intelligence review agent. Your job is to challenge the first-pass extraction and prepare the human reviewer.

Goal and scope:
{_interview_text(interview)}

Reviewed context:
{_reviewed_context_text(taxonomy)}

First-pass extraction:
- Contract type: {extraction.get('contract_type')}
- Confidence: {extraction.get('confidence')}
- Rationale: {extraction.get('rationale')}
- Evidence: {' | '.join(extraction.get('evidence', []))}

Return strict JSON with this shape:
{{
  "review_priority": "low|medium|high",
  "challenge_summary": "short business-facing summary",
  "alternative_contract_types": [
    {{"label": "string", "reason": "string", "evidence": "exact text span"}}
  ],
  "missing_expected_elements": [
    {{"element": "string", "why_it_matters": "string"}}
  ],
  "evidence_gaps": ["string"],
  "taxonomy_suggestions": [
    {{"suggestion_type": "canonical_type|alias|scope|relationship", "suggested_value": "string", "reason": "string"}}
  ],
  "playbook_suggestions": [
    {{"suggestion_type": "expected_clause|disconfirming_evidence|review_rule", "suggested_value": "string", "reason": "string"}}
  ],
  "reviewer_questions": ["string"]
}}

Rules:
- Be specific and evidence-backed.
- Prefer business-level outputs. Do not ask the reviewer to validate raw entities.
- If the model used a broad parent label but the document supports a narrower subtype, call that out.
- If evidence is missing for expected clauses, ask targeted reviewer questions.

Document title: {doc_title}

Contract text:
{doc_text[:9000]}
""".strip()


def _interview_text(interview: dict[str, Any]) -> str:
    return "\n".join([
        f"- Goal: {interview.get('goal', '')}",
        f"- Business unit: {interview.get('business_unit', '')}",
        f"- Region: {interview.get('region', '')}",
        f"- Expected contract types: {', '.join(interview.get('expected_contract_types', []))}",
        f"- Key clause families: {', '.join(interview.get('key_clause_families', []))}",
        f"- Not expected: {', '.join(interview.get('not_expected', []))}",
    ])


def _reviewed_context_text(taxonomy: dict[str, Any]) -> str:
    lines = [
        f"- Reviewed contract types: {', '.join(taxonomy.get('contract_types', []))}",
        f"- Reviewed clause families: {', '.join(taxonomy.get('clause_families', []))}",
    ]
    aliases = taxonomy.get("contract_type_aliases", {})
    for canonical, values in aliases.items():
        if values:
            lines.append(f"- Aliases for {canonical}: {', '.join(values)}")
    for example in taxonomy.get("reviewed_examples", []):
        families = ", ".join(example.get("key_clause_families", []))
        lines.append(f"- Reviewed example: {example.get('title')} => {example.get('contract_type')} [{families}]")
    for contract_type, info in taxonomy.get("playbook", {}).get("contract_types", {}).items():
        families = ", ".join(info.get("expected_clause_families", []))
        lines.append(f"- Playbook for {contract_type}: expected clause families include {families}")
    for pattern in taxonomy.get("rejected_patterns", []):
        lines.append(f"- Rejected pattern: {pattern}")
    return "\n".join(lines)
