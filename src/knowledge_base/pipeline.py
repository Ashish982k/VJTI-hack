import io
import time
from threading import RLock

import numpy as np

from .config import Settings
from .embedding import Embedder
from .generation import check_citations, generate
from .ingestion import extract_document, read_zip
from .metadata import route
from .planning import classify, covers, evidence_requirements, expand_candidates
from .search import BM25, rank
from .sketches import make_sketches
from .storage import Store, validate_collection
from .text import tokens


def step(name, description, input_value, output):
    return {"step_name": name, "description": description, "input": input_value, "output": output}


class KnowledgeBase:
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()
        self.embedder = Embedder(self.settings)
        self.store = Store(self.settings.data_dir)
        self.cache = {}
        self.lock = RLock()

    def index_zip(self, content, collection):
        if len(content) > self.settings.max_zip_bytes:
            raise ValueError("ZIP exceeds upload limit")
        return self.index_documents(read_zip(io.BytesIO(content), self.settings), collection)

    def index_documents(self, documents, collection):
        validate_collection(collection)
        started = time.perf_counter()
        chunks, metadata, errors, raw = [], [], [], {}
        for name, content in documents:
            try:
                doc, parts = extract_document(name, content, collection, self.settings)
                metadata.append(doc)
                chunks.extend(parts)
                raw[doc["document_id"]] = content
            except Exception as e:
                errors.append({"document": name, "error": str(e)})
        if not chunks:
            raise ValueError(f"No readable PDFs: {errors}")
        make_sketches(chunks, self.settings.sketch_tokens)
        vectors, usages = {}, {}
        variants = ("full", "compact") if self.settings.index_variants == "both" else (self.settings.index_variants,)
        for variant in variants:
            field = "text" if variant == "full" else "sketch"
            vectors[variant], usages[variant] = self.embedder.encode([c[field] for c in chunks])
        full_tokens = sum(self.embedder.count(c["text"]) for c in chunks)
        compact_tokens = sum(self.embedder.count(c["sketch"]) for c in chunks)
        stats = {"documents": len(metadata), "chunks": len(chunks), "errors": errors,
                 "full_embedding": usages.get("full"), "compact_embedding": usages.get("compact"),
                 "full_logical_tokens": full_tokens, "compact_logical_tokens": compact_tokens,
                 "embedding_token_reduction_percent": 100 * (1 - compact_tokens / max(1, full_tokens)),
                 "actual_computed_embedding_tokens": sum(u["computed_input_tokens"] for u in usages.values()),
                 "index_variants": list(variants), "sketch_llm_input_tokens": 0, "sketch_llm_output_tokens": 0,
                 "indexing_seconds": time.perf_counter() - started}
        payload = {"schema_version": 1, "embedding_identity": self.embedder.identity,
                   "documents": metadata, "chunks": chunks, "bm25": BM25.build(chunks), "statistics": stats,
                   "settings": self.settings.json()}
        self.store.save(collection, payload, vectors, raw)
        return {"status": "success", "collection_name": collection, "statistics": stats}

    def load(self, collection):
        with self.lock:
            generation = self.store.generation(collection)
            cached = self.cache.get(collection)
            if cached is None or cached[0] != generation:
                cached = self.store.load(collection)
                if cached[1]["embedding_identity"] != self.embedder.identity:
                    raise ValueError("Embedding configuration differs from index; reindex the collection")
                self.cache[collection] = cached
            return cached[1], cached[2]

    def retrieve(self, question, payload, vectors, settings, trace):
        query, usage = self.embedder.encode([question])
        variant = "compact" if settings.compact_embeddings else "full"
        if variant not in vectors:
            raise ValueError(f"Index has no {variant} vectors; reindex with KB_INDEX_VARIANTS=both")
        scores = vectors[variant] @ query[0]
        allowed = np.ones(len(scores), dtype=bool)
        if settings.metadata_routing:
            routing = route(question, payload["documents"])
            doc_ids = set(routing["document_ids"])
            allowed = np.array([c["document_id"] in doc_ids for c in payload["chunks"]])
            trace.append(step("metadata_filter", "Route by entity, reporting period and document type", {"question": question}, routing))
        sparse_index = BM25(payload["bm25"])
        lexical = sparse_index.scores(question) if settings.hybrid_search else np.zeros(len(scores))
        ranked = rank(scores, lexical, allowed, settings.candidate_k, settings.hybrid_search)
        query_type = classify(question)
        broad = query_type in {"exhaustive", "comparison", "broad_analysis", "multi_section"}
        requirements = []
        trace.append(step("query_classification", "Rule-based question category", {"question": question}, {"category": query_type}))
        if settings.multi_section and broad:
            requirements = evidence_requirements(question, payload["chunks"], allowed, scores, settings)
            ranked, searches = expand_candidates(question, payload["chunks"], scores, sparse_index, allowed, ranked, requirements, settings)
            trace.append(step("multi_section_retrieval", "Independent section searches and bounded missing-evidence recovery", {"requirements": requirements, "max_followups": settings.max_followups}, {"searches": searches}))
        limit = settings.broad_k if settings.multi_section and broad else settings.top_k
        ranked = ranked[:limit]
        selected, used = [], 0
        for i, score in ranked:
            chunk = payload["chunks"][int(i)]
            cost = len(tokens(chunk["text"]))
            if used + cost <= settings.context_tokens:
                selected.append({**chunk, "score": score})
                used += cost
        trace.append(step("similarity_search", "Vector search with optional original-text BM25 fusion", {"top_k": settings.top_k, "variant": variant, "hybrid": settings.hybrid_search},
                          {"chunk_ids": [c["chunk_id"] for c in selected]}))
        covered = [r["id"] for r in requirements if any(covers(c, r) for c in selected)]
        trace.append(step("evidence_coverage", "Heuristic coverage of the evidence plan, not a proof of completeness", {"requirements": requirements}, {"covered": covered, "missing": [r["id"] for r in requirements if r["id"] not in covered]}))
        return selected, {"query_embedding": usage, "context_tokens": used, "query_type": query_type,
                          "requirements": len(requirements), "covered_requirements": len(covered), "retrieval_limit": limit}

    def query(self, collection, question, settings=None):
        if not question.strip() or len(question) > 20000:
            raise ValueError("Question must contain 1-20000 characters")
        started = time.perf_counter()
        settings = (settings or self.settings).effective()
        payload, vectors = self.load(collection)
        trace = []
        chunks, stats = self.retrieve(question, payload, vectors, settings, trace)
        answer, generation_usage = generate(question, chunks, settings)
        trace.append(step("llm_answer_generation", "Answer from original cited passages", {"chunk_ids": [c["chunk_id"] for c in chunks]}, {"answer": answer}))
        checks = check_citations(answer, chunks)
        trace.append(step("citation_checks", "Validate citation identifiers (not a semantic proof)", {"answer": answer}, checks))
        stats.update(generation_usage)
        stats.update({"query_seconds": time.perf_counter() - started, "mode": settings.mode,
                      "planning_llm_input_tokens": 0, "planning_llm_output_tokens": 0})
        return {"answer": answer, "chunks": chunks, "trace": trace, "statistics": stats}
