"""Bounded, deterministic evidence plans derived from the question and corpus structure."""

from collections import defaultdict
import re

import numpy as np

from .text import terms, term_set


def classify(question):
    q = question.lower()
    if re.search(r"\b(all|every|exhaustive|comprehensive|complete list|pull together)\b", q):
        return "exhaustive"
    if re.search(r"\b(compare|comparison|versus|vs\.?|across|difference|changes? between)\b", q):
        return "comparison"
    if re.search(r"\b(analy[sz]e|assess|explain|discuss|risks|red flags|overview|how serious)\b", q):
        return "broad_analysis"
    if len(re.split(r";|\band\b", q)) >= 3:
        return "multi_section"
    if re.search(r"\b(how much|how many|amount|ratio|revenue|profit|percentage|total|million|crore)\b", q):
        return "numerical"
    return "factual"


def evidence_requirements(question, chunks, allowed, dense_scores, settings):
    query_terms = set(terms(question))
    grouped = defaultdict(list)
    for i, chunk in enumerate(chunks):
        if allowed[i]:
            grouped[(chunk["document_id"], chunk["section_heading"])].append(i)
    requirements = []
    # User-stated clauses become independent requirements; there is no fixed risk taxonomy.
    clauses = re.split(r"[,;]|\s+[\u2013\u2014]\s+|\band\b", question)
    if len(clauses) > 1:
        for clause in clauses:
            words = set(terms(clause))
            if len(words) >= 2 and words != query_terms:
                requirements.append(
                    {
                        "id": f"clause:{len(requirements)}",
                        "label": clause.strip(),
                        "terms": sorted(words),
                        "kind": "clause",
                    }
                )
    # Comparison questions must have a chance to draw evidence from every routed document.
    if classify(question) == "comparison":
        documents = {
            chunks[i]["document_id"]: chunks[i]["document"] for i in range(len(chunks)) if allowed[i]
        }
        for doc_id, filename in sorted(documents.items()):
            requirements.append(
                {
                    "id": f"document:{doc_id}",
                    "label": filename,
                    "document_id": doc_id,
                    "terms": [],
                    "kind": "document",
                }
            )
    ranked = []
    for (doc_id, heading), indices in grouped.items():
        heading_terms = set(terms(heading))
        score = max(float(dense_scores[i]) for i in indices) + 0.15 * len(query_terms & heading_terms)
        ranked.append((score, doc_id, heading))
    for _, doc_id, heading in sorted(ranked, reverse=True):
        if heading == "Document":
            continue
        requirements.append(
            {
                "id": f"section:{doc_id}:{heading}",
                "label": heading,
                "document_id": doc_id,
                "section_heading": heading,
                "terms": sorted(set(terms(heading))),
                "kind": "section",
            }
        )
    return requirements[: settings.max_requirements]


def covers(chunk, requirement):
    if requirement.get("document_id") and chunk["document_id"] != requirement["document_id"]:
        return False
    if requirement["kind"] == "document":
        return True
    if requirement["kind"] == "section":
        return chunk["section_heading"] == requirement["section_heading"]
    needed = set(requirement["terms"])
    return len(needed & term_set(chunk["text"])) >= max(1, min(3, len(needed) // 2))


def expand_candidates(question, chunks, dense, sparse_index, allowed, initial, requirements, settings):
    """Search each section independently, then bounded missing-clause BM25 recovery."""
    candidates = dict(initial)
    seed_ids, searches = [], []
    for req in requirements:
        mask = np.array([bool(allowed[i]) and covers(c, req) for i, c in enumerate(chunks)])
        if not mask.any():
            continue
        lexical = sparse_index.scores(question + " " + req["label"])
        max_lexical = max(float(lexical.max()), 1e-9)
        score = np.where(mask, dense + lexical / max_lexical, -np.inf)
        picks = [
            int(i) for i in np.argsort(-score, kind="stable")[:2] if np.isfinite(score[i]) and score[i] > 0
        ]
        for i in picks:
            candidates[i] = max(candidates.get(i, 0), 1 / 61 + max(0, float(dense[i])) / 61)
        if picks:
            seed_ids.append(picks[0])
        searches.append({"requirement": req["id"], "chunk_ids": [chunks[i]["chunk_id"] for i in picks]})
    covered = {r["id"] for r in requirements if any(covers(chunks[i], r) for i in candidates)}
    missing = [r for r in requirements if r["id"] not in covered]
    for req in missing[: settings.max_followups]:
        lexical = np.where(allowed, sparse_index.scores(req["label"]), -np.inf)
        picks = [int(i) for i in np.argsort(-lexical, kind="stable")[:2] if lexical[i] > 0]
        for i in picks:
            candidates[i] = max(candidates.get(i, 0), 1 / 61)
        searches.append(
            {"requirement": req["id"], "followup": True, "chunk_ids": [chunks[i]["chunk_id"] for i in picks]}
        )
    # Round-robin section seeds prevent a global ranking from undoing section coverage.
    ordered = list(dict.fromkeys(seed_ids + sorted(candidates, key=lambda i: (-candidates[i], i))))
    ordered = ordered[: max(settings.candidate_k, settings.max_requirements * 2)]
    return [(i, candidates[i]) for i in ordered], searches
