"""Failure buckets for a benchmark run. Gold data is read here only, after all answers were produced.

Every gold chunk is mapped to the indexed unit containing most of its words, then traced through the
pipeline: not in the index -> wrong document routed -> not retrieved -> rejected by the judge -> returned.
Official/proxy judge results, when given, add answer-point buckets.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

from knowledge_base.config import Settings
from knowledge_base.storage import Store
from knowledge_base.text import terms

METRICS = ("answer_precision", "answer_recall", "context_precision", "context_recall")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def step_output(response, name):
    return next((s["output"] for s in response["trace"] if s["step_name"] == name), {})


def point_kind(text):
    lowered = text.lower()
    if re.search(r"\bcit(ed|es|ation)|\bsource\b|\bp\.\s*\d|\bpage\b", lowered):
        return "citation"
    if re.search(r"do(es)? not (state|provide|specify|include)|not (stated|provided|available)|excerpt", lowered):
        return "limitation statement"
    return "fact"


class GoldMapper:
    def __init__(self, chunks):
        self.by_doc = defaultdict(list)
        for chunk in chunks:
            self.by_doc[chunk["document"]].append((chunk["chunk_id"], set(terms(chunk["text"]))))
        self.aliases = {}

    def _best(self, words, document):
        best, score = None, 0.0
        for chunk_id, unit in self.by_doc.get(document, []):
            overlap = len(words & unit) / len(words)
            if overlap > score:
                best, score = chunk_id, overlap
        return best, score

    def _cover(self, words, document):
        """Up to three units that together contain at least half the gold chunk's words."""
        ranked = sorted(self.by_doc.get(document, []), key=lambda u: -len(words & u[1]))[:3]
        chosen, covered = [], set()
        for chunk_id, unit in ranked:
            if words & unit - covered:
                chosen.append(chunk_id)
                covered |= words & unit
            if len(covered) / len(words) >= 0.5:
                return chosen
        return []

    def resolve(self, name, samples):
        """Gold file names can differ from the indexed ones; pick the document whose text matches best."""
        if name in self.by_doc:
            return name
        if name not in self.aliases:
            scores = Counter()
            for words in samples[:3]:
                for document in self.by_doc:
                    scores[document] += self._best(words, document)[1]
            self.aliases[name] = scores.most_common(1)[0][0] if scores else name
        return self.aliases[name]

    def map(self, gold_by_doc):
        """Return ([(document, unit ids)] per gold chunk, number of gold chunks not found in the index)."""
        mapped, unmapped = [], 0
        for name, chunks in gold_by_doc.items():
            samples = [set(terms(c["content"])) for c in chunks]
            document = self.resolve(name, [s for s in samples if len(s) >= 5])
            for words in samples:
                if len(words) < 5:
                    continue
                units = self._cover(words, document)
                if units:
                    mapped.append((document, units))
                else:
                    unmapped += 1
        return mapped, unmapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Folder with responses.jsonl")
    parser.add_argument("--official", type=Path, help="Folder with official.jsonl/summary.json (official judge)")
    parser.add_argument("--proxy", type=Path, help="Folder with official.jsonl/summary.json (proxy judge)")
    parser.add_argument("--benchmark", type=Path, default=Path("Train/Benchmarks/benchmark_100.jsonl"))
    parser.add_argument("--collection", default="train")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = Settings.from_env()
    _, payload, _ = Store(settings.data_dir).load(args.collection)
    mapper = GoldMapper(payload["chunks"])
    gold = {r["id"]: r for r in load_jsonl(args.benchmark)}
    responses = {r["id"]: r["response"] for r in load_jsonl(args.run / "responses.jsonl")}

    buckets, per_question = Counter(), []
    for qid, response in sorted(responses.items()):
        mapped, unmapped = mapper.map(gold[qid]["benchmark_chunks_w_metadata"])
        planned = set(step_output(response, "query_planning").get("documents", []))
        candidates = set(response["statistics"].get("candidate_chunk_ids")
                         or step_output(response, "candidate_retrieval").get("candidate_chunk_ids", []))
        returned = {c.get("metadata", {}).get("chunk_id") or c.get("chunk_id") for c in response["chunks"]}
        counts = Counter(unmapped=unmapped)
        gold_units = set()
        for document, units in mapped:
            gold_units.update(units)
            if returned & set(units):
                counts["returned"] += 1
            elif document not in planned:
                counts["routing miss"] += 1
            elif not candidates & set(units):
                counts["retrieval miss"] += 1
            else:
                counts["judge rejected"] += 1
        buckets.update(counts)
        gold_docs = {document for document, _ in mapped}
        per_question.append({
            "id": qid,
            "gold_chunks": sum(len(v) for v in gold[qid]["benchmark_chunks_w_metadata"].values()),
            "returned_chunks": len(returned),
            "returned_gold": counts["returned"],
            "extra_returned": len(returned - gold_units),
            "routing_recall": len(gold_docs & planned) / len(gold_docs) if gold_docs else None,
            "breadth": step_output(response, "query_planning").get("breadth"),
            "answer_words": len(response["answer"].split()),
            "seconds": response["statistics"]["query_seconds"],
            **{k: counts[k] for k in ("unmapped", "routing miss", "retrieval miss", "judge rejected")},
        })

    lines = [f"# Failure analysis: {args.run}", ""]
    total = sum(buckets.values())
    lines += ["## Where gold evidence is lost", "", "| Bucket | Gold chunks | Share |", "|---|---:|---:|"]
    for name in ("returned", "judge rejected", "retrieval miss", "routing miss", "unmapped"):
        lines.append(f"| {name} | {buckets[name]} | {buckets[name] / max(1, total):.0%} |")
    returned_total = sum(q["returned_chunks"] for q in per_question)
    extra_total = sum(q["extra_returned"] for q in per_question)
    lines += ["", f"Returned chunks: {returned_total}; matching a gold chunk: {returned_total - extra_total} "
              f"({(returned_total - extra_total) / max(1, returned_total):.0%}).", ""]
    routing = [q["routing_recall"] for q in per_question if q["routing_recall"] is not None]
    lines.append(f"Mean routing recall (gold documents planned): {statistics.mean(routing):.2f}; "
                 f"questions with a routing miss: {sum(r < 1 for r in routing)}.")
    lines.append(f"Mean query time: {statistics.mean(q['seconds'] for q in per_question):.0f} s; "
                 f"mean answer length: {statistics.mean(q['answer_words'] for q in per_question):.0f} words.")

    for label, folder in (("Official judge", args.official), ("Proxy judge", args.proxy)):
        if not folder or not (folder / "official.jsonl").exists():
            continue
        judged = load_jsonl(folder / "official.jsonl")
        summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else {}
        lines += ["", f"## {label} ({folder})", "",
                  f"Scored {len(judged)} of {len(responses)} questions; evaluator failures: "
                  f"{len(summary.get('failures', []))}.", "", "| Group | n | " + " | ".join(METRICS) + " |",
                  "|---|---:|" + "---:|" * len(METRICS)]
        groups = {"all": judged}
        for prefix in "ABC":
            groups[prefix] = [j for j in judged if j["id"].startswith(prefix)]
        for name, rows in groups.items():
            if rows:
                lines.append(f"| {name} | {len(rows)} | " + " | ".join(
                    f"{statistics.mean(r[m] for r in rows):.2f}" for m in METRICS) + " |")
        extra, missing = Counter(), Counter()
        for row in judged:
            detail = row["answer_detail"]
            extra.update(point_kind(detail["agent_points"].get(i, "")) for i in detail["extra_ids"])
            missing.update(point_kind(detail["benchmark_points"].get(i, "")) for i in detail["missing_ids"])
        lines += ["", "| Point bucket | Extra (hurts precision) | Missing (hurts recall) |", "|---|---:|---:|"]
        for kind in ("fact", "citation", "limitation statement"):
            lines.append(f"| {kind} | {extra[kind]} | {missing[kind]} |")
        failures = Counter(
            "length limit" if "length" in f["error"] else "truncated JSON" if "parse" in f["error"]
            else "rate limit" if "429" in f["error"] else "other"
            for f in summary.get("failures", [])
        )
        if failures:
            lines += ["", "Evaluator failures: " + ", ".join(f"{k} {v}" for k, v in failures.most_common())]

    lines += ["", "## Per question", "", "| id | breadth | gold | returned | gold returned | routing miss | "
              "retrieval miss | judge rejected | unmapped | words |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for q in per_question:
        lines.append(f"| {q['id']} | {q['breadth']} | {q['gold_chunks']} | {q['returned_chunks']} | "
                     f"{q['returned_gold']} | {q['routing miss']} | {q['retrieval miss']} | "
                     f"{q['judge rejected']} | {q['unmapped']} | {q['answer_words']} |")
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
