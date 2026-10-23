# Architecture and design decisions

The application follows `1st PS.docx`: Python, separate collections, exactly `POST /index` and `POST /query`, document/page citations, four-field inference traces, configurable OpenCode generation, and Langfuse query monitoring. The PS sets a 0.95 target; this implementation does not claim that result. The official evaluator remains byte-identical to the supplied notebook.

## Ingestion and storage

ZIP validation rejects traversal, symlinks, duplicate document paths, excessive file counts, and oversized archives. PyMuPDF extracts each physical page independently, retaining layout-oriented reading order, page labels and original bytes. Chunks never cross page boundaries. Extraction is not a lossless table parser: complex multi-column tables can lose alignment. Empty or image-only pages are reported; there is no OCR dependency. Unreadable files are listed in indexing statistics and do not prevent readable PDFs from being indexed.

Filename patterns, cover text and PDF outlines produce document identity, company aliases, reporting periods, type and section headings. Unknown metadata remains unknown. A deterministic sketch uses document identity, headings, inverse-document-frequency terms and salient original sentences. Highly repeated, nonnumeric margin lines are excluded only from sketches. Original passages, numerical values and dates remain intact. Sketches consume no LLM calls.

Local sentence-transformer embeddings are stored in NumPy arrays, an exact vector index appropriate for this corpus size. Large inputs are embedded in model-length windows and their vectors averaged and normalized. SQLite caches exact embedding inputs by backend/model identity and content hash. Different sketch metadata gets a different embedding key. Duplicate inputs share computation, while every chunk retains its own source identity. BM25 postings index original passage text.

Each collection has immutable version directories containing JSON metadata, BM25 postings, compressed vector arrays, and original PDF bytes. Publishing a new `CURRENT` pointer is atomic; failed indexing preserves the previous generation. Readers invalidate their in-memory cache when the pointer changes. Old snapshots are retained for recovery and require explicit operator cleanup. Cache databases are shared computation stores; retrieval always reads collection-scoped snapshots, so a cache hit cannot expose another collection's evidence.

## Query processing

1. Classify the question using deterministic rules.
2. Route by company aliases, reporting end-year and document type. Multiple matches remain eligible; missing metadata or absent requested years broaden the search and appear in the trace. These rules are heuristics and can misidentify ambiguous entities.
3. Search either full-chunk or sketch vectors. Optionally combine original-text BM25 with reciprocal-rank fusion.
4. For broad questions, derive evidence requirements from question clauses and relevant document headings. Comparison questions can also require evidence from each candidate document. Independently retrieve section candidates, merge them, and perform bounded BM25 follow-ups for unmet requirements. The planner has no list of predefined financial risks and never reads benchmark answers.
5. Greedily select evidence using normalized relevance, new requirement coverage, new terms, redundancy and token cost. Documents and financial values are protected from unsafe deduplication. Source references from exact duplicate passages are retained. Full passages must fit the context budget; they are never truncated to fit.
6. Give the answer model original passages and citation IDs. Validate citation identifiers, paragraph citation presence, and whether stated numbers occur in cited evidence. A failed check causes abstention. This does not prove semantic entailment; supported numbers can still be misinterpreted, and valid derived calculations may be rejected.

The architectural distinction from ordinary hybrid RAG is the document identity layer plus an explicit, bounded evidence-coverage plan. It addresses which report and which sections must supply evidence before final selection. Coverage is heuristic, not a guarantee that every relevant page has been found. The default 12 requirements and 24 broad-query passages will not necessarily cover questions whose gold evidence spans hundreds of passages.

## Measurement and limits

The baseline uses fixed page-aware chunks, full embeddings, vector top-k and the same answer model, prompt, temperature and validation. Ablations independently enable routing, sketches/BM25, multi-section retrieval and coverage selection. D/F can use a larger passage budget than A/B/C/E; reports explicitly disclose that difference. A comparison with a matched larger baseline budget is an important next experiment.

Embedding token counts use the embedding model tokenizer with window special tokens. Context packing uses a lexical token estimate, not the answer model tokenizer. Provider usage is used for actual answer input/output tokens when available; otherwise counts are marked estimated. Prices must be configured; unknown costs remain null. Local embedding API charges are zero, but local compute is not free and is not priced by this harness. Comparison-mode indexing builds both representations and therefore costs more work than either representation alone.

Langfuse wraps query, retrieval and generation operations, recording usage and configured API cost. A local JSONL ledger records indexing/query statistics and failures even without credentials. Official judge calls are a separate evaluation expense; the unchanged notebook does not expose judge token usage. No answer precision/recall or official context precision/recall is claimed without executing that notebook with credentials.

Known limitations include English-only MiniLM embeddings, heuristic metadata/section extraction, no OCR, approximate token packing, lexical requirement coverage, no calibrated relevance/abstention model, and no production authentication/rate limiting. Deploy behind the organizer's access-controlled gateway. No server credentials were provided, so deployment is supplied as commands and a Dockerfile rather than a claimed live service.
