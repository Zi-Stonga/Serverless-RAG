#!/usr/bin/env python3
"""
Post-deploy smoke test for Ask My Docs.
Reads CDK output JSON, calls the /query endpoint with a test question,
and asserts the response shape is correct.

Usage:
    python scripts/smoke_test.py --outputs-file outputs.json [--auth-token TOKEN]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def load_outputs(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # CDK outputs are nested: { "StackName": { "OutputKey": "value" } }
    for stack_name, outputs in data.items():
        if "AskMyDocsStack" in stack_name:
            return outputs
    raise ValueError(f"AskMyDocsStack outputs not found in {path}")


def call_query(api_url: str, question: str, token: str | None = None) -> tuple[int, dict]:
    url     = f"{api_url.rstrip('/')}/query"
    payload = json.dumps({"question": question}).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-file", required=True)
    parser.add_argument("--auth-token",   default=None)
    args = parser.parse_args()

    outputs = load_outputs(args.outputs_file)
    api_url = outputs.get("ApiUrl")
    if not api_url:
        print("FAIL: ApiUrl not found in CDK outputs")
        sys.exit(1)

    print(f"Smoke testing: {api_url}")

    # Test 1: Empty index returns helpful message (no auth needed for 401 test)
    print("\n[1/3] Testing no-document response…")
    status, body = call_query(api_url, "What is the main topic?", args.auth_token)
    if args.auth_token:
        assert status == 200, f"Expected 200, got {status}: {body}"
        assert "answer" in body, f"Missing 'answer' field: {body}"
        print(f"     OK – answer field present, sources: {body.get('sources', [])}")
    else:
        # Without token, Cognito authorizer should return 401
        assert status in (401, 403), f"Expected 401/403 without token, got {status}"
        print(f"     OK – correctly rejected unauthenticated request ({status})")

    # Test 2: Input validation rejects oversized question
    print("\n[2/3] Testing input validation (oversized question)…")
    status, body = call_query(api_url, "A" * 1000, args.auth_token)
    # API Gateway request validator should return 400
    assert status in (400, 401, 403), f"Expected 400 or 401/403 for oversized question, got {status}"
    print(f"     OK – oversized question rejected ({status})")

    # Test 3: Health check – API responds at all
    print("\n[3/3] Testing API Gateway reachability…")
    status, _ = call_query(api_url, "test", args.auth_token)
    assert status != 0, "API Gateway unreachable"
    print(f"     OK – API Gateway responded ({status})")

    print("\n✓ All smoke tests passed")


if __name__ == "__main__":
    main()
