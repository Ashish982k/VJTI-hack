"""Transparent greedy marginal coverage under a context budget."""

import re
from functools import lru_cache

from .planning import covers
from .text import term_set, tokens


def evidence_cost(chunk):
    # Includes the same identifier/document/page label sent with the original passage.
    return len(
        tokens(f"[{chunk['chunk_id']}] {chunk['document']} pages {chunk['page_numbers']}\n{chunk['text']}")
    )


@lru_cache(maxsize=32768)
def protected_facts(text):
    numbers = tuple(re.findall(r"\d[\d,.]*(?:%|\b)", text))
    entities = frozenset(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z]{2,}\b", text))
    return numbers, entities


def redundancy(a, b):
    # Period/document identity and financial facts are hard guards against conflation.
    if a["document_id"] != b["document_id"] or protected_facts(a["text"]) != protected_facts(b["text"]):
        return 0.0
    if a["text"] == b["text"]:
        return 1.0
    first, second = term_set(a["text"]), term_set(b["text"])
    return len(first & second) / max(1, len(first | second))


def select_evidence(candidates, requirements, budget, limit):
    selected, remaining, rejected = [], list(candidates), []
    used, covered, seen_terms = 0, set(), set()
    relevance_scale = max((c["score"] for c in remaining), default=1) or 1
    while remaining and len(selected) < limit:
        choices = []
        for pos, chunk in enumerate(remaining):
            cost = evidence_cost(chunk)
            if used + cost > budget:
                continue
            coverage = {r["id"] for r in requirements if covers(chunk, r)}
            duplicate = max((redundancy(chunk, previous) for previous in selected), default=0)
            words = term_set(chunk["text"])
            novelty = len(words - seen_terms) / max(1, len(words))
            marginal = (
                chunk["score"] / relevance_scale
                + 1.5 * len(coverage - covered)
                + 0.4 * novelty
                - 0.8 * duplicate
            )
            if duplicate >= 0.94 and not coverage - covered:
                continue
            choices.append((marginal / max(1, cost) ** 0.35, -pos, chunk, coverage, words, cost))
        if not choices:
            break
        _, negative_pos, chunk, coverage, words, cost = max(choices, key=lambda c: (c[0], c[1]))
        selected.append(dict(chunk))
        remaining.pop(-negative_pos)
        used += cost
        covered |= coverage
        seen_terms |= words
    for chunk in remaining:
        exact = next(
            (s for s in selected if s["document_id"] == chunk["document_id"] and s["text"] == chunk["text"]),
            None,
        )
        if exact:
            exact.setdefault("duplicate_sources", []).append(
                {k: chunk[k] for k in ("chunk_id", "document", "document_id", "page_numbers", "periods")}
            )
        rejected.append(
            {
                "chunk_id": chunk["chunk_id"],
                "reason": "exact_duplicate" if exact else "redundancy_or_budget_or_limit",
            }
        )
    return selected, {
        "context_tokens": used,
        "budget": budget,
        "selected": len(selected),
        "rejected": rejected,
        "covered_requirements": sorted(covered),
        "tokenizer": "lexical estimate including evidence labels",
    }
