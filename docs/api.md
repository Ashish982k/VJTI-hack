# API contract

The application exposes exactly two routes. OpenAPI/Swagger routes are disabled to respect the PS. The PS specifies logical inputs but no stricter serialized transport field names; this implementation uses `file`, `collection_name`, and `query`.

## POST /index

Multipart form: `file` is a ZIP containing PDFs; `collection_name` is 1–80 letters, digits, underscores or hyphens, beginning with a letter or digit. Nested PDFs are supported. Non-PDF files are ignored, so benchmarks cannot accidentally enter the index.

```powershell
curl.exe -X POST http://localhost:8000/index -F "collection_name=train" -F "file=@documents.zip"
```

Success returns `status`, `collection_name` and `statistics`: document and unit counts, extraction errors, OCR page count, embedding tokens, profiling LLM usage and indexing latency. At least one readable PDF is required. Reusing a collection name atomically replaces that collection; other collections are unaffected. Partial ingestion is explicit in `statistics.errors`. A wholly unreadable archive fails without replacing the collection.

## POST /query

```json
{"collection_name":"train","query":"Compare the reporting years for Acme."}
```

Response fields:

- `answer`: Markdown analyst answer from the evidence only, with `[document, p.N]` citations. Offline mode returns a labelled extractive list.
- `chunks`: every evidence paragraph the judge accepted, ordered by document and page: `text` (original wording), `document`, `page_numbers` (physical PDF pages from 1), `page_label`, `section`, `chunk_id`, `evidence_id`, `document_id`, `collection_name`, `cited_in_answer` and `retrieval_score`.
- `trace`: ordered steps `query_planning`, `candidate_retrieval`, `evidence_judgement`, `coverage_recovery` (LLM mode), `evidence_assembly`, `llm_answer_generation`, `answer_verification`. Each contains exactly `step_name`, `description`, `input`, `output`.
- `statistics`: LLM calls and tokens, per-stage latency and usage (`stages`), candidate and evidence counts, estimated API cost and tracing status.

Errors: 400 for invalid archives/collection names or incompatible indexes, 404 for missing collections, 422 for malformed request bodies, and 503 for unavailable model/configuration or processing errors. Logs retain failure details; API error responses do not include provider credentials.

The printed page label may differ from the PDF page index. Citations always include the physical page index to allow verification against the original PDF; benchmark page-overlap diagnostics disclose potential printed-page offsets.
