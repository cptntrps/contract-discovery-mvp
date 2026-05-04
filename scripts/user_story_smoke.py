#!/usr/bin/env python3
"""Exercise executable end-to-end UI user stories against a running server."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


INTERVIEW_PAYLOAD = {
    "goal": "E2E smoke: classify contracts and prioritize evidence gaps.",
    "business_unit": "E2E Test Business Unit",
    "region": "US",
    "expected_contract_types": [
        "Master Services Agreement",
        "Data Processing Addendum",
    ],
    "contract_type_aliases": {
        "Master Services Agreement": ["MSA", "Services Agreement"],
        "Data Processing Addendum": ["DPA"],
    },
    "key_clause_families": [
        "data_protection",
        "limitation_of_liability",
        "termination",
    ],
    "not_expected": [
        "employment offer letters",
        "consumer terms of service",
    ],
    "review_priorities": [
        "missing governing law",
        "data protection obligations",
        "unusual termination rights",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8788", help="Running Contract Intelligence UI base URL.")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        before = get_json(f"{base_url}/api/state")
        draft: dict[str, object] = {"contract_type_aliases": INTERVIEW_PAYLOAD["contract_type_aliases"]}
        chat_turns = [
            ("goal", INTERVIEW_PAYLOAD["goal"]),
            ("business_unit", INTERVIEW_PAYLOAD["business_unit"]),
            ("region", INTERVIEW_PAYLOAD["region"]),
            ("expected_contract_types", ", ".join(INTERVIEW_PAYLOAD["expected_contract_types"])),
            ("key_clause_families", ", ".join(INTERVIEW_PAYLOAD["key_clause_families"])),
            ("not_expected", ", ".join(INTERVIEW_PAYLOAD["not_expected"])),
            ("review_priorities", ", ".join(INTERVIEW_PAYLOAD["review_priorities"])),
        ]
        chat_results = []
        for field, message in chat_turns:
            result = post_json(f"{base_url}/api/interview/chat", {"interview": draft, "field": field, "message": message})
            draft = result["interview"]
            chat_results.append({
                "field": field,
                "ready": result.get("ready"),
                "next_field": result.get("next_field"),
                "engine": result.get("engine"),
                "model": result.get("model"),
            })
        save_result = post_json(f"{base_url}/api/interview/save", {"interview": draft})
        post_completion = post_json(
            f"{base_url}/api/interview/chat",
            {"interview": draft, "field": "review_priorities", "message": "what can you do"},
        )
        after = get_json(f"{base_url}/api/state")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    taxonomy = after.get("taxonomy", {})
    output_files = {item.get("label"): item for item in after.get("output_files", [])}
    checks = [
        {
            "id": "US-1",
            "story": "Save interview goals and scope.",
            "passed": after.get("interview", {}).get("goal") == INTERVIEW_PAYLOAD["goal"]
            and after.get("interview", {}).get("business_unit") == INTERVIEW_PAYLOAD["business_unit"]
            and after.get("interview", {}).get("region") == INTERVIEW_PAYLOAD["region"],
        },
        {
            "id": "US-2",
            "story": "Seed expected contract types for extraction.",
            "passed": contains_all(after.get("interview", {}).get("expected_contract_types", []), INTERVIEW_PAYLOAD["expected_contract_types"]),
        },
        {
            "id": "US-3",
            "story": "Seed clause families for evidence review.",
            "passed": contains_all(after.get("interview", {}).get("key_clause_families", []), INTERVIEW_PAYLOAD["key_clause_families"]),
        },
        {
            "id": "US-4",
            "story": "Saved interview context updates taxonomy memory.",
            "passed": contains_all(taxonomy.get("contract_types", []), INTERVIEW_PAYLOAD["expected_contract_types"])
            and contains_all(taxonomy.get("clause_families", []), INTERVIEW_PAYLOAD["key_clause_families"]),
        },
        {
            "id": "US-5",
            "story": "Pipeline can proceed after interview setup.",
            "passed": bool(output_files.get("Interview Memory", {}).get("exists"))
            and after.get("summary", {}).get("documents", 0) >= before.get("summary", {}).get("documents", 0),
        },
        {
            "id": "US-6",
            "story": "Completed interview answers help without corrupting memory.",
            "passed": "update the interview memory" in str(post_completion.get("assistant", "")).lower()
            and post_completion.get("interview", {}).get("review_priorities") == draft.get("review_priorities"),
        },
    ]
    ok = all(check["passed"] for check in checks)
    result = {
        "ok": ok,
        "base_url": base_url,
        "saved": {
            "expected_contract_types": save_result.get("expected_contract_types"),
            "key_clause_families": save_result.get("key_clause_families"),
        },
        "chat_turns": chat_results,
        "post_completion": {
            "engine": post_completion.get("engine"),
            "assistant": post_completion.get("assistant"),
        },
        "documents": after.get("summary", {}).get("documents", 0),
        "user_stories": checks,
    }
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


def get_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def contains_all(actual: object, expected: list[str]) -> bool:
    return set(expected).issubset({str(item) for item in actual if str(item)})


if __name__ == "__main__":
    sys.exit(main())
