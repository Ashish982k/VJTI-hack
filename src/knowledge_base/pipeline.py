"""Catalog-routed, aspect-planned, judge-filtered evidence retrieval with verified cited answers."""

from contextlib import contextmanager
import io
import time
from threading import RLock

from .catalog import display_name, profile_documents
from .config import Settings
from .embedding import Embedder
from .generation import edit_answer, generate, render_citations, verify
from .ingestion import extract_document, read_zip
from .judging import judge_candidates
from .llm import LLMClient, merge_usage
from .ocr import create_ocr
from .planning import plan_query
from .retrieval import CandidateSearch, unit_search_text
from .search import BM25
from .storage import Store, validate_collection
from .telemetry import Telemetry, estimate_cost
from .text import digest, tokens

SCHEMA_VERSION = 2


def step(name, description, input_value, output):
    return {"step_name": name, "description": description, "input": input_value, "output": output}


class KnowledgeBase:
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()
        self.embedder = Embedder(self.settings)
        self.store = Store(self.settings.data_dir)
        self.llm = LLMClient(self.settings)
        self.telemetry = Telemetry(self.settings.data_dir)
        self.cache = {}
        self.lock = RLock()
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = create_ocr(self.settings) or False
        return self._ocr or None

    # ------------------------------------------------------------------ indexing

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
                doc, parts = extract_document(name, content, collection, self.settings, self.ocr)
                if not parts:
                    raise ValueError("No text units extracted")
                metadata.append(doc)
                chunks.extend(parts)
                raw[doc["document_id"]] = content
            except Exception as e:
                errors.append({"document": name, "error": str(e)})
        if not chunks:
            raise ValueError(f"No readable PDFs: {errors}")
        parsed_seconds = time.perf_counter() - started
        profile_usage = merge_usage(profile_documents(metadata, self.llm))
        titles = {d["document_id"]: display_name(d) for d in metadata}
        texts = [unit_search_text(c, titles[c["document_id"]]) for c in chunks]
        vectors, embedding_usage = self.embedder.encode(texts)
        for doc in metadata:
            doc.pop("preview", None)
        stats = {
            "documents": len(metadata),
            "units": len(chunks),
            "errors": errors,
            "ocr_pages": sum(len(d["ocr_pages"]) for d in metadata),
            "embedding": embedding_usage,
            "profiling_llm": profile_usage,
            "llm_input_tokens": profile_usage["input_tokens"],
            "llm_output_tokens": profile_usage["output_tokens"],
            "parse_seconds": parsed_seconds,
            "indexing_seconds": time.perf_counter() - started,
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "embedding_identity": self.embedder.identity,
            "documents": metadata,
            "chunks": chunks,
            "bm25": BM25.build(texts),
            "statistics": stats,
            "settings": self.settings.json(),
        }
        self.store.save(collection, payload, {"units": vectors}, raw)
        stats["estimated_cost"] = estimate_cost(stats, self.settings.embedding_backend)
        self.telemetry.record({"operation": "index", "collection": collection, "statistics": stats})
        return {"status": "success", "collection_name": collection, "statistics": stats}

    def load(self, collection):
        with self.lock:
            generation = self.store.generation(collection)
            cached = self.cache.get(collection)
            if cached is None or cached[0] != generation:
                _, payload, vectors = self.store.load(collection)
                if payload.get("schema_version") != SCHEMA_VERSION or "units" not in vectors:
                    raise ValueError("Index was built by an older version; reindex the collection")
                if payload["embedding_identity"] != self.embedder.identity:
                    raise ValueError("Embedding configuration differs from index; reindex the collection")
                search = CandidateSearch(
                    payload["chunks"], vectors["units"], BM25(payload["bm25"]), self.embedder, self.settings
                )
                cached = (generation, payload, search)
                self.cache[collection] = cached
            return cached[1], cached[2]

    # ------------------------------------------------------------------ querying

    @contextmanager
    def stage(self, run, name, description, input_value, as_type="span"):
        """One pipeline step: required trace entry, Langfuse observation, latency and LLM usage."""
        record = {"output": None, "usages": []}
        started = time.perf_counter()
        with self.telemetry.span(name, as_type=as_type, input=input_value) as observation:
            yield record
            usage = merge_usage(u for u in record["usages"] if u)
            seconds = time.perf_counter() - started
            if observation is not None:
                extra = {}
                if usage["calls"]:
                    costs = estimate_cost(
                        {"llm_input_tokens": usage["input_tokens"], "llm_output_tokens": usage["output_tokens"]},
                        self.settings.embedding_backend,
                    )
                    extra["usage_details"] = {"input": usage["input_tokens"], "output": usage["output_tokens"]}
                    extra["cost_details"] = {
                        k: costs[f"llm_{k}_usd"] for k in ("input", "output") if costs[f"llm_{k}_usd"] is not None
                    }
                    if as_type == "generation":
                        extra["model"] = ", ".join(usage["models"])
                observation.update(output=record["output"], metadata={"seconds": seconds}, **extra)
        run["trace"].append(step(name, description, input_value, record["output"]))
        run["stages"][name] = {"seconds": round(seconds, 3), "llm": usage}

    def query(self, collection, question):
        started = time.perf_counter()
        with self.telemetry.span(
            "knowledge_base.query", as_type="chain", input={"collection": collection, "query": question}
        ) as observation:
            try:
                result = self._query(collection, question)
                stats = result["statistics"]
                stats["estimated_cost"] = estimate_cost(stats, self.settings.embedding_backend)
                stats["langfuse_enabled"] = self.telemetry.client is not None
                self.telemetry.record(
                    {"operation": "query", "collection": collection, "query_hash": digest(question),
                     "statistics": stats}
                )
                if observation is not None:
                    observation.update(output={"answer": result["answer"]}, metadata=stats)
                return result
            except Exception as e:
                self.telemetry.record(
                    {"operation": "query", "collection": collection, "query_hash": digest(question),
                     "error_type": type(e).__name__, "query_seconds": time.perf_counter() - started}
                )
                if observation is not None:
                    observation.update(level="ERROR", status_message=type(e).__name__)
                raise

    def _items(self, payload, indices, scores, prefix_start=1):
        titles = {d["document_id"]: display_name(d) for d in payload["documents"]}
        items = []
        for n, index in enumerate(indices, prefix_start):
            chunk = payload["chunks"][index]
            title = titles[chunk["document_id"]]
            items.append({
                "label": f"P{n}",
                "index": index,
                "score": scores.get(index, 0.0),
                "title": title,
                "source": f"{title}, page {chunk['page_numbers'][0]}, {chunk['section']}",
                "text": chunk["text"],
            })
        return items

    def _query(self, collection, question):
        if not question.strip() or len(question) > 20000:
            raise ValueError("Question must contain 1-20000 characters")
        started = time.perf_counter()
        settings = self.settings
        payload, search = self.load(collection)
        run = {"trace": [], "stages": {}}
        documents = {d["document_id"]: d for d in payload["documents"]}

        with self.stage(run, "query_planning", "Route the question to catalog documents (entity, period, "
                        "type) and decompose it into searchable aspects", {"question": question},
                        as_type="generation" if self.llm.enabled else "span") as record:
            plan, usage = plan_query(question, payload["documents"], self.llm, settings)
            record["usages"].append(usage)
            record["output"] = {
                "documents": [documents[i]["document"] for i in plan["document_ids"]],
                "aspects": plan["aspects"],
                "entities_periods": plan["entities_periods"],
                "breadth": plan.get("breadth", "broad"),
                "source": plan["source"],
            }

        with self.stage(run, "candidate_retrieval", "Hybrid dense+BM25 search for the question and each aspect, "
                        "restricted to routed documents, with same-section neighbours and per-document quotas",
                        {"documents": len(plan["document_ids"]), "aspects": len(plan["aspects"])},
                        as_type="retriever") as record:
            candidates, summary, _ = search.gather(question, plan)
            scores = {c["index"]: c["score"] for c in candidates}
            record["output"] = {
                **summary,
                "top": [
                    {"chunk_id": payload["chunks"][c["index"]]["chunk_id"], "score": round(c["score"], 5),
                     "matched": c["queries"]}
                    for c in candidates[:50]
                ],
                "candidate_chunk_ids": [payload["chunks"][c["index"]]["chunk_id"] for c in candidates],
            }

        items = self._items(payload, [c["index"] for c in candidates], scores)
        with self.stage(run, "evidence_judgement", "LLM judges each candidate: kept only if a complete answer "
                        "would use it (right entity, period and topic)",
                        {"candidates": len(items), "batch_size": settings.judge_batch},
                        as_type="evaluator") as record:
            verdicts, usages, failures = judge_candidates(question, plan, items, self.llm, settings)
            record["usages"].extend(usages)
            relevant = [i for i in items if verdicts[i["label"]]["relevant"]]
            record["output"] = {
                "relevant": len(relevant),
                "rejected": len(items) - len(relevant),
                "failed_batches": failures,
                "relevant_chunk_ids": [payload["chunks"][i["index"]]["chunk_id"] for i in relevant],
            }

        if self.llm.enabled and plan["aspects"] and settings.recovery_k:
            covered = {a for i in relevant for a in verdicts[i["label"]]["aspects"]}
            missing = [n for n in range(1, len(plan["aspects"]) + 1) if n not in covered][:3]
            with self.stage(run, "coverage_recovery", "Deeper scoped search and judging, once, for aspects "
                            "with no accepted evidence", {"uncovered_aspects": [plan["aspects"][n - 1]["label"]
                                                                                for n in missing]},
                            as_type="retriever") as record:
                judged = {i["index"] for i in items}
                extra = []
                for n in missing:
                    aspect = plan["aspects"][n - 1]
                    found, _ = search.deeper(aspect["label"] + ": " + "; ".join(aspect["queries"]),
                                             plan["document_ids"], judged, settings.recovery_k)
                    judged |= set(found)
                    extra.extend(found)
                new_items = self._items(payload, extra, {}, prefix_start=len(items) + 1)
                more, usages, failures = judge_candidates(question, plan, new_items, self.llm, settings)
                record["usages"].extend(usages)
                verdicts.update(more)
                recovered = [i for i in new_items if more[i["label"]]["relevant"]]
                relevant.extend(recovered)
                record["output"] = {"searched": len(new_items), "recovered": len(recovered),
                                    "failed_batches": failures}

        if self.llm.enabled and not relevant and items:
            # Nothing matched: often the question asks for a period or version the corpus lacks (e.g. a 2025
            # report when only 2024 exists). Re-judge the strongest candidates allowing the nearest period;
            # the answer then states the mismatch instead of returning nothing.
            plan = {**plan, "period_fallback": True}
            with self.stage(run, "period_fallback", "No evidence matched the exact period/version asked; "
                            "re-judge top candidates allowing the nearest available period",
                            {"candidates": min(len(items), 2 * settings.judge_batch)},
                            as_type="evaluator") as record:
                top = sorted(items, key=lambda i: -i["score"])[: 2 * settings.judge_batch]
                more, usages, failures = judge_candidates(question, plan, top, self.llm, settings)
                record["usages"].extend(usages)
                verdicts.update(more)
                relevant = [i for i in top if more[i["label"]]["relevant"]]
                record["output"] = {"accepted": len(relevant), "failed_batches": failures}

        with self.stage(run, "evidence_assembly", "Return every accepted passage (ordered by document and "
                        "page); give the answer model the strongest ones within the context budget",
                        {"accepted": len(relevant)}) as record:
            relevant.sort(key=lambda i: -i["score"])
            relevant = relevant[: settings.max_evidence]
            order = {doc_id: n for n, doc_id in enumerate(plan["document_ids"])}
            ordered = sorted(
                relevant, key=lambda i: (order.get(payload["chunks"][i["index"]]["document_id"], 99), i["index"])
            )
            evidence = []
            for n, item in enumerate(ordered, 1):
                chunk = payload["chunks"][item["index"]]
                evidence.append({**chunk, "label": f"E{n}", "title": item["title"], "page": chunk["page_numbers"][0],
                                 "score": item["score"], "grade": verdicts[item["label"]].get("grade"),
                                 "aspects": verdicts[item["label"]].get("aspects") or []})
            # The writer sees passages that directly answer (grade 3). Broad questions also get grade-2
            # facts tied to a planned sub-question; untied grade-2 support is returned as context only,
            # because feeding it to the writer produced correct-but-unasked points.
            narrow = plan.get("breadth") == "narrow"
            direct = [e for e in evidence if e["grade"] == 3 or (not narrow and e["grade"] == 2 and e["aspects"])]
            prompt_labels, used = set(), 0
            for item in sorted(direct or evidence, key=lambda e: (-(e["grade"] or 0), -e["score"])):
                cost = len(tokens(item["text"])) + 20
                if used + cost <= settings.context_tokens:
                    prompt_labels.add(item["label"])
                    used += cost
            answer_evidence = [e for e in evidence if e["label"] in prompt_labels]
            record["output"] = {"evidence": len(evidence), "direct_answer_passages": len(direct),
                                "answer_context_passages": len(answer_evidence),
                                "context_tokens": used, "chunk_ids": [e["chunk_id"] for e in evidence]}

        with self.stage(run, "llm_answer_generation", "Synthesize a structured answer from the evidence only, "
                        "citing passage ids", {"evidence_ids": [e["label"] for e in answer_evidence]},
                        as_type="generation" if self.llm.enabled else "span") as record:
            draft, usage = generate(question, answer_evidence, self.llm, settings, plan)
            record["usages"].append(usage)
            record["output"] = {"draft_characters": len(draft)}

        # Some models (e.g. gpt-oss) emit 【E1】 instead of [E1].
        draft = draft.replace("\u3010", "[").replace("\u3011", "]")
        if self.llm.enabled and evidence:
            with self.stage(run, "answer_editing", "Fast LLM marks lines that state something the question did "
                            "not ask for; those lines are deleted (it cannot add or rewrite text)",
                            {"draft_lines": len(draft.splitlines())}, as_type="generation") as record:
                draft, edits, usage = edit_answer(question, draft, self.llm)
                record["usages"].append(usage)
                record["output"] = edits

        with self.stage(run, "answer_verification", "Remove lines whose figures are absent from the evidence, "
                        "then render document/page citations", {"draft_characters": len(draft)}) as record:
            checked, checks = verify(draft, answer_evidence, question)
            answer, cited = render_citations(checked, answer_evidence)
            record["output"] = {**checks, "cited_evidence": cited}

        cited_set = set(cited)
        chunks = [
            {
                "chunk_id": e["chunk_id"],
                "evidence_id": e["label"],
                "text": e["text"],
                "document": e["document"],
                "document_id": e["document_id"],
                "page_numbers": e["page_numbers"],
                "page_label": e["page_label"],
                "section": e["section"],
                "collection_name": e["collection_name"],
                "cited_in_answer": e["label"] in cited_set,
                "retrieval_score": round(e["score"], 6),
                "judge_grade": e["grade"],
            }
            for e in evidence
        ]
        totals = {
            "llm_calls": sum(s["llm"]["calls"] for s in run["stages"].values()),
            "llm_input_tokens": sum(s["llm"]["input_tokens"] for s in run["stages"].values()),
            "llm_output_tokens": sum(s["llm"]["output_tokens"] for s in run["stages"].values()),
        }
        statistics = {
            **totals,
            "stages": run["stages"],
            "candidates": len(items),
            "evidence_chunks": len(chunks),
            "cited_chunks": len(cited_set),
            "query_seconds": time.perf_counter() - started,
            "llm_backend": settings.llm_backend,
        }
        return {"answer": answer, "chunks": chunks, "trace": run["trace"], "statistics": statistics}
