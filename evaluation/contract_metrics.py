"""Context precision/recall exactly as the organizers' API contract (section 3.2) describes them.

A returned chunk is relevant when its file matches a benchmark document and its pages overlap that benchmark
passage's pages. Precision = relevant chunks / returned chunks. Recall = share of benchmark (document, page)
pairs covered by the returned chunks. File names match ignoring case, extension and punctuation.
"""

from pathlib import PurePosixPath
import re


def normalize_name(name):
    stem = PurePosixPath(str(name)).name
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", stem)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def gold_passages(row, aliases=None):
    """[(normalized document, {pages})] from a benchmark row; aliases map gold names to corpus names."""
    aliases = aliases or {}
    passages = []
    for document, chunks in row["benchmark_chunks_w_metadata"].items():
        for chunk in chunks:
            name = chunk.get("metadata", {}).get("document", document)
            pages = {int(p) for p in chunk.get("metadata", {}).get("page_numbers", [])}
            passages.append((normalize_name(aliases.get(name, name)), pages))
    return passages


def context_metrics(chunks, passages):
    """chunks: [{"metadata": {"file_name", "page_numbers"}}] in the contract shape."""
    gold_pairs = {(doc, page) for doc, pages in passages for page in pages}
    relevant, covered = 0, set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        doc, pages = normalize_name(meta.get("file_name", "")), set(meta.get("page_numbers", []))
        if any(doc == gold_doc and pages & gold_pages for gold_doc, gold_pages in passages):
            relevant += 1
        covered |= {(doc, page) for page in pages} & gold_pairs
    return {
        "context_precision": relevant / len(chunks) if chunks else 0.0,
        "context_recall": len(covered) / len(gold_pairs) if gold_pairs else 0.0,
        "relevant_chunks": relevant,
        "gold_pairs": len(gold_pairs),
    }
