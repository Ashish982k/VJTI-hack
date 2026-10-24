# API contract

Implements the organizers' *Hackathon API Contract* (supplement to section 11.3). Exactly two routes;
OpenAPI/Swagger routes are disabled. No authentication. All responses are `application/json`.

## POST /index

`multipart/form-data` with `file` (ZIP of PDFs; sub-folders allowed, `__MACOSX` ignored) and
`collection_name` (1–80 letters, digits, `_` or `-`).

```bash
curl -X POST https://<base-url>/index -H "ngrok-skip-browser-warning: true" \
  -F "file=@documents.zip" -F "collection_name=eval-2026"
```

Synchronous: returns only after the collection is queryable.

```json
{"status": "success", "collection_name": "eval-2026", "documents_indexed": 50, "statistics": {"...": "..."}}
```

Collections are independent; re-indexing a name atomically replaces it, and a failed re-index keeps the
previous version. Errors: 400 for an invalid archive or name (`{"detail": "..."}`), 503 for processing errors.
Unreadable PDFs are listed in `statistics.errors` without failing the whole archive.

## POST /query

```json
{"collection_name": "eval-2026", "query": "What are the duties of the administrative body?"}
```

```json
{
  "answer": "Markdown answer, grounded in the chunks below",
  "chunks": [
    {
      "content": "original passage text",
      "metadata": {
        "file_name": "SolvencyII-Directive-2009-138.pdf",
        "page_numbers": [33],
        "document_path": "folder/SolvencyII-Directive-2009-138.pdf",
        "section": "Title I > Article 41",
        "page_label": "33",
        "chunk_id": "…",
        "evidence_id": "E4",
        "judge_grade": 3,
        "cited_in_answer": true,
        "retrieval_score": 0.0312,
        "collection_name": "eval-2026"
      }
    }
  ],
  "trace": [{"step_name": "query_planning", "description": "…", "input": {}, "output": {}}],
  "statistics": {"query_seconds": 21.4, "llm_calls": 7, "...": "..."}
}
```

- `chunks[].content`, `metadata.file_name` (the PDF's file name, never a folder path) and
  `metadata.page_numbers` (physical PDF pages, **1-based**) are the contract fields. Other `metadata` keys and
  `statistics` are extra fields, which the harness ignores.
- `chunks` holds every passage the evidence judge accepted. Context is scored by page overlap, and returning
  all judged evidence scored best on the training set (see `reports/RESULTS.md`).
- `trace`: one entry per pipeline step in execution order, each with exactly `step_name`, `description`,
  `input`, `output`; large lists are summarised (full candidate ids are in `statistics.candidate_chunk_ids`).
- Unanswerable question: HTTP 200, an honest answer ("The indexed documents do not contain evidence that
  answers this question."), empty `chunks`, and a full trace.
- Unknown collection: HTTP 404 `{"detail": "unknown collection"}`. Malformed body: 422. Provider or
  processing failure: 503 (the harness retries 5xx).

## Self-check before submitting

```bash
python scripts/contract_check.py https://<your-ngrok-url> --collection train
python scripts/contract_check.py https://<your-ngrok-url> --zip docs.zip --collection contract-check
```

The script sends the `ngrok-skip-browser-warning` header, rejects HTML responses, and validates the 404,
the chunk fields (file names, 1-based integer pages) and the four-field trace.
