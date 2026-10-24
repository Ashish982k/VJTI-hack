# Architecture: catalog-routed, judge-filtered evidence retrieval

One pipeline, designed around how the task is scored. The official evaluator:

- judges each returned chunk for relevance to the query (context precision = relevant / returned);
- divides the number of relevant returned chunks by the number of gold chunks (context recall). The gold
  set for the training questions has a median of 81 paragraph-sized chunks (about 63 words each), and up
  to 352;
- splits our answer and the gold answer into atomic points. Points we add that are not in the gold answer
  lower precision; gold points we miss lower recall. Gold answers are structured analyst reports with a
  median of about 750 words.

A top-k retriever that returns 6–24 passages cannot reach high context recall on these questions, and a
long top-k list loses precision. So the pipeline returns a variable-size set of paragraph units, and every
unit has been checked by an LLM against the same relevance test the evaluator applies.

```
INDEX (POST /index)
 PDF ─▶ PyMuPDF layout blocks ──(page has no text)──▶ PaddleOCR PP-OCRv5 lines → paragraphs
     ─▶ drop repeated headers/footers, page numbers, TOC dot leaders
     ─▶ headings (font size / bold + PDF outline) give every unit a section path
     ─▶ paragraph units: 60–220 words, never crossing a page or section
     ─▶ document profile (LLM, one call per document): title, issuer, type, period,
        jurisdiction, summary, topics, "what distinguishes it from similar documents"
     ─▶ embeddings (BGE, with "title | section" header) + BM25 over the same text

QUERY (POST /query)
 1 query_planning        LLM reads the compact catalog: picks the documents (entity / period / version)
                         and splits the question into 2–8 aspects with document-style search queries
 2 candidate_retrieval   dense+BM25 fusion for the question and for every aspect, restricted to the
                         routed documents; adjacent same-section paragraphs of top hits; a per-document
                         quota (≤ 240 candidates)
 3 evidence_judgement    LLM judges candidates in parallel batches of 30: relevant only if a complete
                         answer would use it (right entity, period and topic); tags supported aspects
 4 coverage_recovery     aspects with no accepted evidence get one deeper scoped search + judging
 5 evidence_assembly     accepted units, ordered by document and page, within the context budget
 6 llm_answer_generation structured analyst answer from the evidence only, every line cites [E#]
 7 answer_verification   lines with figures absent from the evidence are removed; [E#] becomes
                         [document, p.N]
 → answer, chunks (all accepted evidence, with document, page, section, cited flag), trace
```

## Why each piece exists

| Failure mode | Mechanism |
|---|---|
| Near-duplicate documents (same issuer, different year/version) | Planner routes on an LLM-built catalog with an explicit "distinguishing" field; the judge rejects passages from the wrong entity or period; units embed their document title |
| Answer spread over many pages | Aspect decomposition retrieves each part separately; neighbour expansion pulls continuation paragraphs; per-document quotas; the returned set size follows the evidence, not k |
| Irrelevant chunks lowering context precision | Every returned chunk passed the judge, which mirrors the evaluator's relevance question |
| Chunk granularity | Paragraph units sized like the gold chunks, so relevant evidence is counted at the same grain |
| Wrong-but-confident figures | Answer uses only evidence; any line with a number that appears in no evidence passage is removed |
| Extra or missing answer points | Prompt requires covering every part with specific figures, and forbids outside knowledge, generic caveats and advice |
| Scanned PDFs | PaddleOCR fallback per page, cached by PDF hash, page and DPI |

Vector search is used, but only inside the routed scope and only to propose candidates. Inclusion is
decided by the judge, and the final set size depends on the evidence.

## Cost and latency per query

With the LLM backend: 1 planning call, ⌈candidates / 30⌉ judge calls (usually 4–8, run in parallel,
`KB_LLM_CONCURRENCY`), at most one recovery judge round, and 1 answer call. Judge prompts carry about
80–100 words per candidate, so a broad question uses roughly 25–40k judge input tokens. Set
`LLM_FAST_MODEL` to a cheaper model (or a comma-separated pool) for planning, profiling and judging. Every stage reports its
latency and token usage in `statistics.stages`, the trace, and Langfuse (generation observations with
usage and configured cost).

To trade accuracy for cost, lower `KB_MAX_CANDIDATES` and `KB_ASPECT_QUERY_K`. To raise recall on very broad
questions, raise them.

## Storage and safety

Each collection has immutable version directories (JSON metadata, BM25 postings, NumPy vectors, original
PDFs) published by an atomic `CURRENT` pointer; a failed re-index keeps the previous version. The
embedding and OCR caches are keyed by content hash and model identity and are shared across collections;
retrieval only reads the collection's own snapshot. ZIP validation rejects traversal, symlinks, duplicate
paths and oversized archives. Document text reaches the LLM as quoted data and every prompt says it is
never instructions.

## Offline mode and limits

`KB_LLM_BACKEND=offline` replaces planning with filename/cover-date routing, judging with the top
`KB_OFFLINE_EVIDENCE` ranked candidates, and the answer with an extractive list. It exists for tests and
retrieval diagnostics. It is not the scored configuration.

Limits: the judge and the answer model can still be wrong; figure verification checks presence, not
meaning; tables are kept as text blocks rather than reconstructed; OCR runs about 2–17 s per page on CPU.
No official scores are claimed until the unchanged evaluator is run with credentials.
