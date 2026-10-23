import io
import time
from threading import RLock

import numpy as np

from .config import Settings
from .embedding import Embedder
from .generation import check_citations, generate
from .ingestion import extract_document, read_zip
from .metadata import route
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
        vectors, usage = self.embedder.encode([c["text"] for c in chunks])
        stats = {"documents": len(metadata), "chunks": len(chunks), "errors": errors,
                 "full_embedding": usage, "indexing_seconds": time.perf_counter() - started}
        payload = {"schema_version": 1, "embedding_identity": self.embedder.identity,
                   "documents": metadata, "chunks": chunks, "statistics": stats,
                   "settings": self.settings.json()}
        self.store.save(collection, payload, {"full": vectors}, raw)
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
        scores = vectors["full"] @ query[0]
        if settings.metadata_routing:
            routing = route(question, payload["documents"])
            allowed = set(routing["document_ids"])
            scores = np.where([c["document_id"] in allowed for c in payload["chunks"]], scores, -np.inf)
            trace.append(step("metadata_filter", "Route by entity, reporting period and document type", {"question": question}, routing))
        ranked = np.argsort(-scores, kind="stable")[:settings.top_k]
        selected, used = [], 0
        for i in ranked:
            chunk = payload["chunks"][int(i)]
            cost = len(tokens(chunk["text"]))
            if scores[i] > 0 and used + cost <= settings.context_tokens:
                selected.append({**chunk, "score": float(scores[i])})
                used += cost
        trace.append(step("similarity_search", "Full-chunk vector similarity search", {"top_k": settings.top_k},
                          {"chunk_ids": [c["chunk_id"] for c in selected]}))
        return selected, {"query_embedding": usage, "context_tokens": used}

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
