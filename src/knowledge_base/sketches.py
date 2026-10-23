"""Lossy search sketches; original passages remain the only generation evidence."""
from collections import Counter
import math
import re

from .text import terms, tokens, truncate


def repetitive_margins(chunks):
    by_doc = {}
    for c in chunks:
        pages = by_doc.setdefault(c["document_id"], {})
        pages.setdefault(c["page_numbers"][0], []).append(c["text"])
    result = {}
    for doc, pages in by_doc.items():
        counts = Counter()
        for pieces in pages.values():
            lines = "\n".join(pieces).splitlines()
            # Restrict removal to non-numeric page margins with strong repetition.
            counts.update({s.strip() for s in lines[:2] + lines[-2:] if 3 <= len(s.strip()) <= 90
                           and not re.search(r"\d|[%$\u20b9\u20ac]", s)})
        result[doc] = {s for s, n in counts.items() if n >= max(3, math.ceil(len(pages) * .6))}
    return result


def make_sketches(chunks, limit):
    frequencies = Counter(t for c in chunks for t in set(terms(c["text"])))
    margins = repetitive_margins(chunks)
    n = len(chunks)
    for c in chunks:
        original = c["text"]
        cleaned = "\n".join(line for line in original.splitlines() if line.strip() not in margins[c["document_id"]])
        prefix = " | ".join(filter(None, [c["company"], ", ".join(c["periods"]), c["section_heading"]]))
        prefix = truncate(prefix, min(35, max(1, limit // 3)))
        counts = Counter(terms(cleaned))
        weight = {t: (1 + math.log(tf)) * math.log(1 + n / frequencies[t]) for t, tf in counts.items()}
        keywords = sorted(weight, key=lambda t: (-weight[t], t))[:12]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", cleaned) if s.strip()]
        scored = sorted(enumerate(sentences), key=lambda pair: (
            -sum(weight.get(t, 0) for t in set(terms(pair[1]))) / max(1, len(tokens(pair[1]))) ** .5, pair[0]))
        # Reserve most space for salient original sentences, including values and units.
        sketch = prefix + " | " + " ".join(keywords) + " | "
        remaining = max(1, limit - len(tokens(sketch)))
        picked = []
        for position, sentence in scored:
            if remaining <= 0:
                break
            part = truncate(sentence, remaining)
            picked.append((position, part))
            remaining -= len(tokens(part))
        sketch += " ".join(s for _, s in sorted(picked))
        sketch = truncate(sketch, limit)
        c["sketch"] = sketch if len(tokens(sketch)) < len(tokens(original)) else original
        c["sketch_method"] = "deterministic_idf_sentences_v1"

