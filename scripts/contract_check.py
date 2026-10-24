"""Check a deployed API against the organizers' API contract (run it on the public ngrok URL).

    python scripts/contract_check.py https://abcd1234.ngrok-free.app --collection train \
        --question "What was Banque Saudi Fransi's NPL coverage ratio at the end of 2025?"
    python scripts/contract_check.py http://127.0.0.1:8765 --zip docs.zip --collection contract-check

Exits non-zero on the first contract violation.
"""

import argparse
from pathlib import Path
import sys
import time

import httpx

HEADERS = {"ngrok-skip-browser-warning": "true"}


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def json_body(response, route):
    if "application/json" not in response.headers.get("content-type", ""):
        fail(f"{route} returned {response.headers.get('content-type')!r}, not JSON (wrong URL or ngrok page?)")
    return response.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_url", help="Base URL without a route, e.g. https://abcd1234.ngrok-free.app")
    parser.add_argument("--collection", default="train")
    parser.add_argument("--question", default="What was Banque Saudi Fransi's NPL coverage ratio at the end of 2025?")
    parser.add_argument("--zip", type=Path, help="Also test /index with this ZIP (indexes into --collection)")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if base.endswith(("/index", "/query")):
        fail("submit the base URL only; the harness appends /index and /query")

    with httpx.Client(timeout=args.timeout, headers=HEADERS, follow_redirects=False) as client:
        if args.zip:
            started = time.perf_counter()
            with args.zip.open("rb") as stream:
                response = client.post(f"{base}/index", data={"collection_name": args.collection},
                                       files={"file": (args.zip.name, stream, "application/zip")})
            body = json_body(response, "/index")
            if response.status_code != 200 or body.get("status") != "success":
                fail(f"/index returned {response.status_code}: {body}")
            print(f"OK /index in {time.perf_counter() - started:.0f}s: {body.get('documents_indexed')} documents")

        response = client.post(f"{base}/query", json={"collection_name": "no-such-collection-xyz", "query": "x"})
        body = json_body(response, "/query")
        if response.status_code != 404 or "detail" not in body:
            fail(f"unknown collection should be HTTP 404 with a detail, got {response.status_code}: {body}")
        print("OK unknown collection -> 404", body)

        started = time.perf_counter()
        response = client.post(f"{base}/query", json={"collection_name": args.collection, "query": args.question})
        body = json_body(response, "/query")
        if response.status_code != 200:
            fail(f"/query returned {response.status_code}: {body}")
        if not isinstance(body.get("answer"), str) or not body["answer"].strip():
            fail("answer must be a non-empty string")
        if not isinstance(body.get("chunks"), list):
            fail("chunks must be an array")
        for i, chunk in enumerate(body["chunks"]):
            meta = chunk.get("metadata") if isinstance(chunk, dict) else None
            if not isinstance(chunk.get("content"), str) or not isinstance(meta, dict):
                fail(f"chunks[{i}] needs content (string) and metadata (object)")
            if not isinstance(meta.get("file_name"), str) or "/" in meta["file_name"]:
                fail(f"chunks[{i}].metadata.file_name must be the document's file name: {meta.get('file_name')!r}")
            pages = meta.get("page_numbers")
            if not isinstance(pages, list) or not pages or not all(isinstance(p, int) and p >= 1 for p in pages):
                fail(f"chunks[{i}].metadata.page_numbers must be 1-based integers: {pages!r}")
        trace = body.get("trace")
        if not isinstance(trace, list) or not trace:
            fail("trace must be a non-empty array")
        for i, step in enumerate(trace):
            if not isinstance(step, dict) or set(step) != {"step_name", "description", "input", "output"}:
                fail(f"trace[{i}] must have exactly step_name, description, input, output: {sorted(step)}")
        files = sorted({c["metadata"]["file_name"] for c in body["chunks"]})
        print(f"OK /query in {time.perf_counter() - started:.0f}s: {len(body['chunks'])} chunks from {files}, "
              f"{len(trace)} trace steps")
        print("Answer:", body["answer"][:400])
    print("PASS: responses match the API contract")


if __name__ == "__main__":
    main()
