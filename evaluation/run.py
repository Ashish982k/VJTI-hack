"""Benchmark the pipeline. Gold answers and chunks are only read after each response is produced."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time

from knowledge_base.config import Settings
from knowledge_base.pipeline import KnowledgeBase
from knowledge_base.text import terms
from .official_adapter import load_official, OFFICIAL_SHA256


def gold_chunks(row):
    return [
        chunk for document_chunks in row["benchmark_chunks_w_metadata"].values() for chunk in document_chunks
    ]


def retrieval_diagnostics(retrieved, gold):
    """Lexical/page overlap with gold chunks. Diagnostics only, NEVER the official LLM-judged metrics."""
    expected_pages = {
        (c["metadata"]["document"], int(p)) for c in gold for p in c["metadata"].get("page_numbers", [])
    }
    actual_pages = {(c["document"], p) for c in retrieved for p in c["page_numbers"]}
    gold_terms = [set(terms(c["content"])) for c in gold]
    retrieved_terms = [set(terms(c["text"])) for c in retrieved]
    covered = sum(
        1 for g in gold_terms if g and any(len(g & r) / len(g) >= 0.6 for r in retrieved_terms)
    )
    supported = sum(
        1 for r in retrieved_terms if r and any(len(g & r) / len(r) >= 0.5 for g in gold_terms)
    )
    return {
        "gold_chunks": len(gold),
        "returned_chunks": len(retrieved),
        "page_recall_proxy": len(expected_pages & actual_pages) / len(expected_pages) if expected_pages else 0,
        "gold_coverage_proxy": covered / len(gold) if gold else 0,
        "returned_precision_proxy": supported / len(retrieved) if retrieved else 0,
        # Mirrors the official recall formula: relevant returned / gold count (can exceed 1).
        "count_recall_proxy": supported / len(gold) if gold else 0,
    }


def batched_evaluation(evaluator, item, batch):
    """Same metrics as evaluate_single, but retrieved chunks are judged in batches by the unchanged
    context_precision_recall; relevance is per chunk, so summing relevant counts keeps the formulas."""
    precision, recall, detail = evaluator.answer_precision_recall(item["ai_answer"], item["benchmark_answer"])
    chunks, relevant = item["retrieved_chunks"], 0.0
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        part_precision, _ = evaluator.context_precision_recall(item["query"], part, len(item["benchmark_chunks"]))
        relevant += part_precision * len(part)
    return {
        "answer_precision": precision,
        "answer_recall": recall,
        "context_precision": relevant / len(chunks) if chunks else 0.0,
        "context_recall": relevant / len(item["benchmark_chunks"]) if chunks and item["benchmark_chunks"] else 0.0,
        "answer_detail": detail,
        "query": item["query"],
        "scoring": f"batched context judging ({batch} chunks per call)",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="train")
    parser.add_argument("--benchmark", type=Path, default=Path("Train/Benchmarks/benchmark_100.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/runs/latest"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--ids", help="Comma-separated benchmark ids to run (e.g. A01,B07)")
    parser.add_argument("--offline", action="store_true", help="Rule-based planning/judging, no LLM calls")
    parser.add_argument("--official", action="store_true", help="Run the unchanged notebook judge (paid calls)")
    parser.add_argument("--workers", type=int, default=4, help="Questions processed in parallel")
    parser.add_argument("--judge-model", help="Evaluator model override (proxy scores; default: notebook's)")
    parser.add_argument("--judge-provider", choices=["opencode", "llm"], default="opencode",
                        help="llm: score with the pipeline's LLM_* provider (e.g. Groq); proxy scores")
    parser.add_argument("--context-batch", type=int, default=0,
                        help="Judge retrieved chunks in batches of N with the unchanged method (small providers)")
    parser.add_argument("--official-attempts", type=int, default=2, help="Evaluator attempts per question")
    parser.add_argument("--reuse", type=Path, help="Judge responses.jsonl from an earlier run instead of querying")
    args = parser.parse_args()
    if args.official and args.offline:
        parser.error("Official answer scores require LLM answers; remove --offline")
    settings = Settings.from_env()
    if args.offline:
        settings = replace(settings, llm_backend="offline")
    judge = {}
    if args.judge_provider == "llm":
        base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        judge = {
            "base_url": base if base.endswith("/chat/completions") else base + "/chat/completions",
            "api_key": os.environ.get("LLM_API_KEY"),
        }
    evaluator, evaluate_dataset = (
        load_official(model=args.judge_model, **judge) if args.official else (None, None)
    )
    reused = {}
    if args.reuse:
        for line in (args.reuse / "responses.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            reused[record["id"]] = record["response"]
    rows = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    rows = rows[args.offset : args.offset + args.limit]
    kb = KnowledgeBase(settings)
    payload, _ = kb.load(args.collection)
    args.output.mkdir(parents=True, exist_ok=True)
    responses_path = args.output / "responses.jsonl"
    responses_path.write_text("", encoding="utf-8")
    official_path = args.output / "official.jsonl"
    official_path.write_text("", encoding="utf-8")
    details, judged, failures = {}, {}, []
    lock = threading.Lock()
    started = time.perf_counter()

    def process(row):
        response = reused[row["id"]] if row["id"] in reused else kb.query(args.collection, row["question"])
        gold = gold_chunks(row)
        diagnostic = retrieval_diagnostics(response["chunks"], gold)
        item = {
            "query": row["question"],
            "ai_answer": response["answer"],
            "benchmark_answer": row["benchmark_answer"],
            "retrieved_chunks": [c["text"] for c in response["chunks"]],
            "benchmark_chunks": [c["content"] for c in gold],
        }
        with lock:
            # Saved before judging so a failed official evaluation never loses the answer.
            details[row["id"]] = {"id": row["id"], **diagnostic, "statistics": response["statistics"]}
            with responses_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": row["id"], "response": response}) + "\n")
        result = None
        if args.official:
            # The unmodified evaluate_dataset, called on one item so questions can run in parallel. Its own
            # judge calls can fail transiently (e.g. a reasoning model hitting max_tokens): retry once.
            for attempt in range(args.official_attempts):
                try:
                    if args.context_batch:
                        result = batched_evaluation(evaluator, item, args.context_batch)
                    else:
                        result = evaluate_dataset(evaluator, [item], verbose=False)["per_question"][0]
                    break
                except RuntimeError:
                    if attempt == args.official_attempts - 1:
                        raise
        with lock:
            if result is not None:
                judged[row["id"]] = result
                with official_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"id": row["id"], **result}) + "\n")
        return diagnostic, response["statistics"], result

    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process, row): row for row in rows}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                diagnostic, stats, result = future.result()
            except Exception as e:
                failures.append({"id": row["id"], "error": f"{type(e).__name__}: {e}"})
                print(f"[{done}/{len(rows)}] {row['id']} FAILED: {e}", flush=True)
                continue
            scores = (
                " AnsP={answer_precision:.2f} AnsR={answer_recall:.2f} CtxP={context_precision:.2f} "
                "CtxR={context_recall:.2f}".format(**result) if result else ""
            )
            print(
                f"[{done}/{len(rows)}] {row['id']} returned={diagnostic['returned_chunks']} "
                f"gold={diagnostic['gold_chunks']} {stats['query_seconds']:.0f}s{scores}",
                flush=True,
            )
    order = {r["id"]: n for n, r in enumerate(rows)}
    details = sorted(details.values(), key=lambda d: order[d["id"]])
    summary = {
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "official_evaluator_sha256": OFFICIAL_SHA256,
        "official_scores_run": args.official,
        "judge_model": args.judge_model or "notebook default (deepseek-v4.1-flash)",
        "judge_provider": args.judge_provider,
        "context_batch": args.context_batch or None,
        "reused_responses": str(args.reuse) if args.reuse else None,
        "settings": settings.json(),
        "index_statistics": payload["statistics"],
        "questions": len(rows),
        "failures": failures,
        "wall_seconds": time.perf_counter() - started,
        "diagnostics_note": "Lexical/page overlap with gold chunks; not the official LLM-judged metrics.",
    }
    if details:
        for field in ("page_recall_proxy", "gold_coverage_proxy", "returned_precision_proxy",
                      "count_recall_proxy", "returned_chunks", "gold_chunks"):
            summary["mean_" + field] = statistics.mean(d[field] for d in details)
        for field in ("query_seconds", "llm_calls", "llm_input_tokens", "llm_output_tokens"):
            summary["mean_" + field] = statistics.mean(d["statistics"][field] for d in details)
        costs = [d["statistics"]["estimated_cost"]["total_usd"] for d in details]
        summary["total_query_api_cost_usd"] = sum(costs) if all(c is not None for c in costs) else None
    if judged:
        # Same averaging as evaluate_dataset: plain mean over the judged questions.
        summary["official_questions_judged"] = len(judged)
        for metric in ("answer_precision", "answer_recall", "context_precision", "context_recall"):
            summary[metric] = statistics.mean(r[metric] for r in judged.values())
    (args.output / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    kb.telemetry.flush()
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("mean_", "answer_", "context_"))},
                     indent=2))
    print(f"Reports: {args.output.resolve()}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
