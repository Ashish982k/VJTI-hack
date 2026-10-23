import io
import os
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
from .selection import evidence_cost, select_evidence
from .telemetry import Telemetry, estimate_cost
from .text import digest
from .storage import Store, validate_collection


def step(name, description, input_value, output):
    return {"step_name": name, "description": description, "input": input_value, "output": output}


class KnowledgeBase:
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()
        self.embedder = Embedder(self.settings)
        self.store = Store(self.settings.data_dir)
        self.cache = {}
        self.lock = RLock()
        self.telemetry = Telemetry(self.settings.data_dir)

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
        stats["estimated_cost"] = estimate_cost(stats, self.settings.embedding_backend)
        self.telemetry.record({"operation": "index", "collection": collection, "statistics": stats})
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
        if (settings.multi_section or settings.coverage_selection) and broad:
            requirements = evidence_requirements(question, payload["chunks"], allowed, scores, settings)
        if settings.multi_section and broad:
            ranked, searches = expand_candidates(question, payload["chunks"], scores, sparse_index, allowed, ranked, requirements, settings)
            trace.append(step("multi_section_retrieval", "Independent section searches and bounded missing-evidence recovery", {"requirements": requirements, "max_followups": settings.max_followups}, {"searches": searches}))
        limit = settings.broad_k if settings.multi_section and broad else settings.top_k
        selected, used = [], 0
        candidates = [{**payload["chunks"][i], "score": score} for i, score in ranked]
        if settings.coverage_selection:
            selected, selection = select_evidence(candidates, requirements, settings.context_tokens, limit)
            used = selection["context_tokens"]
            trace.append(step("coverage_selection", "Greedy relevance, new coverage, redundancy and token cost", {"candidate_ids": [c["chunk_id"] for c in candidates]}, selection))
        else:
            for chunk in candidates[:limit]:
                cost = evidence_cost(chunk)
                if used + cost <= settings.context_tokens:
                    selected.append(chunk)
                    used += cost
        trace.append(step("similarity_search", "Vector search with optional original-text BM25 fusion", {"top_k": settings.top_k, "variant": variant, "hybrid": settings.hybrid_search},
                          {"chunk_ids": [c["chunk_id"] for c in selected]}))
        covered = [r["id"] for r in requirements if any(covers(c, r) for c in selected)]
        trace.append(step("evidence_coverage", "Heuristic coverage of the evidence plan, not a proof of completeness", {"requirements": requirements}, {"covered": covered, "missing": [r["id"] for r in requirements if r["id"] not in covered]}))
        return selected, {"query_embedding": usage, "context_tokens": used, "context_tokenizer": "lexical estimate including evidence labels", "query_type": query_type,
                          "requirements": len(requirements), "covered_requirements": len(covered), "retrieval_limit": limit}

    def query(self, collection, question, settings=None):
        started = time.perf_counter()
        with self.telemetry.span("knowledge_base.query", input={"collection": collection, "query": question}) as observation:
            try:
                result = self._query(collection, question, settings)
                stats = result["statistics"]
                stats["estimated_cost"] = estimate_cost(stats, self.settings.embedding_backend)
                stats["langfuse_enabled"] = self.telemetry.client is not None
                stats["total_computed_tokens"] = (stats["query_embedding"]["computed_input_tokens"]
                                                   + stats["llm_input_tokens"] + stats["llm_output_tokens"])
                self.telemetry.record({"operation": "query", "collection": collection,
                                       "query_hash": digest(question), "statistics": stats})
                if observation is not None:
                    observation.update(output={"answer": result["answer"], "trace": result["trace"]}, metadata=stats)
                return result
            except Exception as e:
                self.telemetry.record({"operation": "query", "collection": collection, "query_hash": digest(question),
                                       "error_type": type(e).__name__, "query_seconds": time.perf_counter() - started})
                if observation is not None:
                    observation.update(level="ERROR", status_message=type(e).__name__)
                raise

    def _query(self, collection, question, settings=None):
        if not question.strip() or len(question) > 20000:
            raise ValueError("Question must contain 1-20000 characters")
        started = time.perf_counter()
        settings = (settings or self.settings).effective()
        payload, vectors = self.load(collection)
        trace = []
        retrieval_started = time.perf_counter()
        with self.telemetry.span("retrieval", as_type="retriever", input={"question": question}) as observation:
            chunks, stats = self.retrieve(question, payload, vectors, settings, trace)
            if observation is not None:
                observation.update(output={"chunk_ids": [c["chunk_id"] for c in chunks]}, metadata=stats)
        stats["retrieval_seconds"] = time.perf_counter() - retrieval_started
        generation_started = time.perf_counter()
        with self.telemetry.span("answer", as_type="generation", model=os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash") if settings.answer_backend == "opencode" else "extractive",
                                 input={"question": question, "chunk_ids": [c["chunk_id"] for c in chunks]}) as observation:
            answer, generation_usage = generate(question, chunks, settings)
            if observation is not None:
                costs = estimate_cost(generation_usage, settings.embedding_backend)
                observation.update(output=answer, usage_details={"input": generation_usage["llm_input_tokens"], "output": generation_usage["llm_output_tokens"]},
                                   cost_details={k: v for k, v in {"input": costs["llm_input_usd"], "output": costs["llm_output_usd"]}.items() if v is not None})
        stats["generation_seconds"] = time.perf_counter() - generation_started
        trace.append(step("llm_answer_generation", "Answer from original cited passages", {"chunk_ids": [c["chunk_id"] for c in chunks]}, {"answer": answer}))
        checks = check_citations(answer, chunks)
        trace.append(step("citation_checks", "Validate citation identifiers (not a semantic proof)", {"answer": answer}, checks))
        stats.update(generation_usage)
        stats.update({"query_seconds": time.perf_counter() - started, "mode": settings.mode,
                      "planning_llm_input_tokens": 0, "planning_llm_output_tokens": 0})
        return {"answer": answer, "chunks": chunks, "trace": trace, "statistics": stats}
