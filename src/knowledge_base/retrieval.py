"""Scoped candidate generation: the question and each planned aspect search only routed documents."""

import numpy as np

from .search import GroupedBM25, rank
from .vectorstore import NumpyStore


def unit_search_text(chunk, title):
    # Contextual header lets near-duplicate passages from different documents/years separate.
    return f"{title} | {chunk['section']}\n{chunk['text']}"


class CandidateSearch:
    def __init__(self, chunks, unit_vectors, bm25, embedder, settings, stores=None, collection="", generation=""):
        self.chunks = chunks
        self.vectors = unit_vectors
        self.bm25 = bm25
        self.embedder = embedder
        self.settings = settings
        self.stores, self.collection, self.generation = stores, collection, generation
        self.document_ids = np.array([c["document_id"] for c in chunks])
        self.unit_store = self._store("units", unit_vectors, self.document_ids)
        self._build_sections()

    def _store(self, kind, vectors, document_ids, variant=""):
        if self.stores is None:
            return NumpyStore(vectors, document_ids)
        return self.stores.create(kind, self.collection, self.generation, vectors, document_ids, variant)

    def _section_starts(self):
        """Coarse sections in document order. Detected headings are too fine (median one paragraph), so
        'merged' joins heading runs until a section holds section_min_units, and 'pages' uses fixed page
        windows. A section never crosses a document and never exceeds section_cap_units."""
        s = self.settings
        starts, size, first_page = [], 0, None
        for i, c in enumerate(self.chunks):
            previous = self.chunks[i - 1] if i else None
            page = c["page_numbers"][0]
            if previous is None or previous["document_id"] != c["document_id"] or size >= s.section_cap_units:
                new = True
            elif s.section_mode == "pages":
                new = page - first_page >= s.section_pages
            else:
                new = c["section"] != previous["section"] and size >= s.section_min_units
            if new:
                starts.append(i)
                size, first_page = 0, page
            size += 1
        return starts

    def _build_sections(self):
        starts = self._section_starts()
        self.section_starts = np.array(starts, dtype=np.int64)
        self.section_ends = np.append(self.section_starts[1:], len(self.chunks))
        self.section_of_unit = np.repeat(np.arange(len(starts)), self.section_ends - self.section_starts)
        self.section_docs = self.document_ids[self.section_starts]
        centroids = np.add.reduceat(self.vectors, self.section_starts, axis=0)
        self.section_vectors = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        self.section_bm25 = GroupedBM25(self.bm25, self.section_of_unit, len(starts))
        s = self.settings
        variant = f"{s.section_mode}:{s.section_min_units}:{s.section_cap_units}:{s.section_pages}"
        self.section_store = self._store("sections", self.section_vectors, self.section_docs, variant)

    def _sections(self, text, vector, dense, sparse, document_ids):
        """Pick whole sections for one query, then their units (capped to the best-matching ones)."""
        s = self.settings
        allowed = np.isin(self.section_docs, list(document_ids))
        chosen = rank(self.section_store.dense(vector, document_ids), self.section_bm25.scores(text), allowed,
                      s.sections_per_query, depth=50)
        picked = []
        top = max(float(sparse.max()), 1e-9)
        for section, score in chosen:
            units = np.arange(self.section_starts[section], self.section_ends[section])
            if len(units) > s.section_max_units:
                fit = dense[units] + sparse[units] / top
                units = units[np.argsort(-fit, kind="stable")[: s.section_max_units]]
            picked.append((int(section), float(score), [int(u) for u in units]))
        return picked

    def mask(self, document_ids):
        return np.isin(self.document_ids, list(document_ids))

    def _neighbors(self, index):
        chunk = self.chunks[index]
        for offset in range(1, self.settings.neighbor_window + 1):
            for j in (index - offset, index + offset):
                if 0 <= j < len(self.chunks):
                    other = self.chunks[j]
                    if other["document_id"] == chunk["document_id"] and other["section"] == chunk["section"]:
                        yield j

    def gather(self, question, plan):
        """Return ranked candidates [{index, score, queries}] and a trace summary."""
        s = self.settings
        allowed = self.mask(plan["document_ids"])
        queries = [("question", question, s.main_query_k)]
        for i, aspect in enumerate(plan["aspects"]):
            text = aspect["label"] + ": " + "; ".join(aspect["queries"])
            queries.append((f"aspect:{i}", text, s.aspect_query_k))
        vectors, usage = self.embedder.encode_queries([q[1] for q in queries])
        found, searches, main_scores = {}, [], None
        # Structure-first for broad questions: choose sections per sub-question and take their units,
        # instead of hoping chunk-level hits land on every relevant paragraph.
        use_sections = s.section_retrieval and plan.get("breadth") != "narrow"
        for (name, text, limit), vector in zip(queries, vectors):
            dense = self.unit_store.dense(vector, plan["document_ids"])
            sparse = self.bm25.scores(text)
            if name == "question":
                main_scores = (dense, sparse)
            sections = self._sections(text, vector, dense, sparse, plan["document_ids"]) if use_sections else []
            for section, score, units in sections:
                for index in units:
                    entry = found.setdefault(index, {"index": index, "score": 0.0, "queries": []})
                    entry["score"] = max(entry["score"], score * 0.9)
                    entry["queries"].append(f"{name}:section")
            ranked = rank(dense, sparse, allowed, limit, depth=max(limit, 100))
            for position, (index, score) in enumerate(ranked):
                entry = found.setdefault(index, {"index": index, "score": 0.0, "queries": []})
                entry["score"] = max(entry["score"], score)
                entry["queries"].append(name)
                # The best hits pull in adjacent paragraphs of the same section (continuations).
                if position < 5:
                    for j in self._neighbors(index):
                        if allowed[j] and j not in found:
                            found[j] = {"index": j, "score": score * 0.5, "queries": [f"{name}:neighbor"]}
            searches.append({
                "query": name, "text": text, "hits": len(ranked),
                "sections": [f"{self.chunks[self.section_starts[sec]]['document']} :: "
                             f"{self.chunks[self.section_starts[sec]]['section']} ({len(units)} units)"
                             for sec, _, units in sections],
            })
        # Every routed document gets a fair share, so one verbose document cannot crowd out the rest.
        dense, sparse = main_scores
        for doc_id in plan["document_ids"]:
            doc_mask = allowed & self.mask([doc_id])
            for index, score in rank(dense, sparse, doc_mask, s.per_document_k, depth=100):
                found.setdefault(index, {"index": index, "score": score, "queries": ["per_document"]})
        ranked = sorted(found.values(), key=lambda c: (-c["score"], c["index"]))[: s.max_candidates]
        return ranked, {"searches": searches, "candidates": len(ranked), "allowed_units": int(allowed.sum()),
                        "vector_store": self.unit_store.name}, usage

    def deeper(self, text, document_ids, exclude, k):
        """Bounded recovery search for one uncovered aspect, skipping already-judged units."""
        allowed = self.mask(document_ids)
        vectors, usage = self.embedder.encode_queries([text])
        dense = self.unit_store.dense(vectors[0], document_ids)
        ranked = rank(dense, self.bm25.scores(text), allowed, k + len(exclude), depth=200)
        return [i for i, _ in ranked if i not in exclude][:k], usage
