"""Reproducible A-F comparisons. Gold data is only used after retrieval."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from knowledge_base.config import Settings
from knowledge_base.pipeline import KnowledgeBase
from knowledge_base.text import terms, term_set
from knowledge_base.selection import protected_facts
from .official_adapter import load_official, OFFICIAL_SHA256


def experiments(settings):
    base = replace(settings, mode="improved", metadata_routing=False, compact_embeddings=False,
                   hybrid_search=False, multi_section=False, coverage_selection=False)
    return {
        "A": ("Baseline", replace(base, mode="baseline")),
        "B": ("Metadata routing", replace(base, metadata_routing=True)),
        "C": ("Compact + BM25", replace(base, compact_embeddings=True, hybrid_search=True)),
        "D": ("Multi-section", replace(base, multi_section=True)),
        "E": ("Coverage selection", replace(base, coverage_selection=True)),
        "F": ("All improvements", replace(base, metadata_routing=True, compact_embeddings=True,
                                        hybrid_search=True, multi_section=True, coverage_selection=True)),
    }


def gold_chunks(row):
    return [chunk for document_chunks in row["benchmark_chunks_w_metadata"].values() for chunk in document_chunks]


def retrieval_diagnostics(retrieved, gold, k=6):
    """Page overlap and lexical coverage are diagnostics, NEVER official metrics."""
    expected_pages = {(c["metadata"]["document"], int(p)) for c in gold for p in c["metadata"].get("page_numbers", [])}
    actual_pages = {(c["document"], p) for c in retrieved for p in c["page_numbers"]}
    hit_pages = expected_pages & actual_pages
    retrieval_words = [(c, set(terms(c["text"]))) for c in retrieved[:k]]
    hits = 0
    for chunk in gold:
        expected = set(terms(chunk["content"]))
        if not expected:
            continue
        for candidate, words in retrieval_words:
            if candidate["document"] == chunk["metadata"]["document"] and len(expected & words) / len(expected) >= .6:
                hits += 1
                break
    return {"page_recall_proxy": len(hit_pages) / len(expected_pages) if expected_pages else 0,
            "page_precision_proxy": len(hit_pages) / len(actual_pages) if actual_pages else 0,
            "lexical_recall_at_k_proxy": hits / len(gold) if gold else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="train")
    parser.add_argument("--benchmark", type=Path, default=Path("Train/Benchmarks/benchmark_100.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/runs/latest"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--experiments", default="ABCDEF")
    parser.add_argument("--official", action="store_true", help="Run the unchanged notebook judge (paid API calls)")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip answer generation, report only retrieval diagnostics")
    args = parser.parse_args()
    if args.limit <= 0 or not args.experiments or set(args.experiments) - set("ABCDEF"):
        parser.error("Use a positive limit and experiment letters A-F")
    if args.official and args.retrieval_only:
        parser.error("Official answer scores require real answer generation; remove --retrieval-only")
    settings = Settings.from_env()
    if args.retrieval_only:
        settings = replace(settings, answer_backend="extractive")
    if args.official and settings.answer_backend != "opencode":
        parser.error("Official scoring requires KB_ANSWER_BACKEND=opencode")
    evaluator, evaluate_dataset = load_official() if args.official else (None, None)
    rows = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    kb = KnowledgeBase(settings)
    payload, _ = kb.load(args.collection)
    # All experiments get identical warm embedding/model/index caches. Warmup spend is reported.
    started = time.perf_counter()
    _, warmup_usage = kb.embedder.encode([r["question"] for r in rows])
    for chunk in payload["chunks"]:
        term_set(chunk["text"])
        protected_facts(chunk["text"])
    warmup_seconds = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    all_summary = []
    baseline = None
    for letter, (name, config) in experiments(settings).items():
        if letter not in args.experiments:
            continue
        evaluated, details, failures = [], [], []
        # Explicitly replace per-experiment output; reruns never append duplicate results.
        (args.output / f"{letter}-responses.jsonl").write_text("", encoding="utf-8")
        for i, row in enumerate(rows):
            try:
                response = kb.query(args.collection, row["question"], config)
                gold = gold_chunks(row)
                diagnostic = retrieval_diagnostics(response["chunks"], gold, config.top_k)
                details.append({"id": row["id"], **diagnostic, "statistics": response["statistics"],
                                "selected_chunk_ids": [c["chunk_id"] for c in response["chunks"]]})
                evaluated.append({"query": row["question"], "ai_answer": response["answer"],
                                  "benchmark_answer": row["benchmark_answer"],
                                  "retrieved_chunks": [c["text"] for c in response["chunks"]],
                                  "benchmark_chunks": [c["content"] for c in gold]})
                with (args.output / f"{letter}-responses.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"id": row["id"], "response": response}) + "\n")
            except Exception as e:
                failures.append({"id": row["id"], "error": str(e)})
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"{letter} {i + 1}/{len(rows)} complete; failures={len(failures)}", flush=True)
        if failures:
            # Never hide failed questions by publishing success-only averages.
            summary = {"experiment": letter, "name": name, "status": "incomplete", "questions": len(rows), "failures": failures}
        else:
            summary = {"experiment": letter, "name": name, "status": "complete", "questions": len(rows)}
            for field in ("page_recall_proxy", "page_precision_proxy", "lexical_recall_at_k_proxy"):
                summary[field] = statistics.mean(d[field] for d in details)
            for field in ("query_seconds", "retrieval_seconds", "generation_seconds", "context_tokens", "llm_input_tokens", "llm_output_tokens"):
                summary["mean_" + field] = statistics.mean(d["statistics"][field] for d in details)
            summary["mean_query_embedding_logical_tokens"] = statistics.mean(d["statistics"]["query_embedding"]["logical_input_tokens"] for d in details)
            costs = [d["statistics"]["estimated_cost"]["total_usd"] for d in details]
            summary["total_query_api_cost_usd"] = sum(costs) if all(c is not None for c in costs) else None
            if args.official:
                judged = evaluate_dataset(evaluator, evaluated)
                (args.output / f"{letter}-official.json").write_text(json.dumps(judged, indent=2), encoding="utf-8")
                for metric in ("answer_precision", "answer_recall", "context_precision", "context_recall"):
                    summary[metric] = judged["avg_" + metric]
            else:
                for metric in ("answer_precision", "answer_recall", "context_precision", "context_recall"):
                    summary[metric] = None
            if letter == "A":
                baseline = dict(summary)
            if baseline is not None:
                for key, value in list(summary.items()):
                    if isinstance(value, (int, float)) and key != "questions" and isinstance(baseline.get(key), (int, float)):
                        summary["delta_" + key] = value - baseline[key]
        all_summary.append(summary)
        (args.output / f"{letter}-details.json").write_text(json.dumps({"settings": config.json(), "rows": details, "failures": failures}, indent=2), encoding="utf-8")
        report = {"benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
                  "official_evaluator_sha256": OFFICIAL_SHA256, "official_scores_run": args.official,
                  "retrieval_only": args.retrieval_only, "embedding_backend": settings.embedding_backend,
                  "embedding_model": settings.embedding_model, "answer_backend": settings.answer_backend,
                  "answer_model": os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash"),
                  "temperature": 0, "index_statistics": payload["statistics"],
                  "warmup": {"seconds": warmup_seconds, "embedding": warmup_usage},
                  "cache_policy": "shared warm model/index/query embedding caches; warmup separately charged",
                  "diagnostics_note": "Physical-page overlap and >=60% lexical gold-chunk coverage at top_k; not official metrics. Printed page offsets may affect page proxies.",
                  "budget_note": f"A/B/C/E top_k={settings.top_k}; D/F broad_k={settings.broad_k}; all context budget={settings.context_tokens} lexical tokens",
                  "experiments": all_summary}
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        keys = list(dict.fromkeys(k for row in all_summary for k in row))
        with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_summary)
    kb.telemetry.flush()
    print(f"Reports: {args.output.resolve()}")
    if any(s["status"] != "complete" for s in all_summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
