# VJTI hack — Smart Knowledge Base for Agents

Python backend for the Smart Knowledge Base problem statement. It runs one pipeline: **catalog-routed,
judge-filtered evidence retrieval**. Source PDFs and benchmark data stay in `Train/` locally and are
excluded from Git, as are secrets, models and generated indexes.

```
PDF → paragraph units (+ PaddleOCR for scanned pages) → LLM document catalog
query → plan (documents + aspects) → scoped dense+BM25 candidates → LLM evidence judge
      → bounded coverage recovery → cited analyst answer → figure verification
```

See [docs/architecture.md](docs/architecture.md) for the design and why each piece exists, and
[docs/api.md](docs/api.md) for the HTTP contract.

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[semantic,observability,ocr,dev]"
cp .env.example .env   # set LLM_API_KEY (+ LLM_BASE_URL/LLM_MODEL) and the Langfuse keys
.venv/bin/python -m knowledge_base.cli index ../Train/Documents --collection train
.venv/bin/python -m knowledge_base.cli query "What are Gammon India's FY2026 payment defaults?" --collection train
.venv/bin/python -m knowledge_base.cli serve
```

On Windows use `.venv\Scripts\python` and `Copy-Item .env.example .env`. On macOS, if
`import knowledge_base` fails after an editable install, run `chflags nohidden .venv/lib/python3.11/site-packages/*.pth`
(Python skips hidden `.pth` files).

`POST /index`: multipart `file` (ZIP of PDFs) and `collection_name`. `POST /query`: JSON
`{"collection_name": "train", "query": "..."}`. The response has `answer`, `chunks` (every accepted
evidence paragraph with document, page and section), `trace` (each entry has exactly `step_name`,
`description`, `input`, `output`) and `statistics`. These are the only two routes.

## Ask questions in the browser

```bash
.venv/bin/python -m knowledge_base.cli serve                                  # API on :8000
.venv/bin/python -m http.server 8080 --bind 127.0.0.1 --directory frontend    # page on :8080
```

Open http://127.0.0.1:8080, ask a question, and read the cited answer, the evidence paragraphs and the
pipeline trace. The page can also upload a ZIP to index a new collection. The API allows that page's
origin via CORS (`KB_CORS_ORIGINS`); it still exposes only `/index` and `/query`.

**Groq free tier:** each model allows about 8,000 tokens per minute, so `.env.example` lists smaller
budgets and spreads calls over three models. Broad questions then take a minute or more, and the answer
sees fewer passages than with a paid provider. The official evaluator itself still calls OpenCode.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `KB_LLM_BACKEND` | `llm` | `offline` swaps planning, judging and answering for rules (tests and diagnostics only) |
| `LLM_API_KEY` / `LLM_BASE_URL` | — / OpenCode | Any OpenAI-compatible provider (OpenCode, Groq, ...); `OPENCODE_*` names also work |
| `LLM_MODEL` / `LLM_FAST_MODEL` | `deepseek-v4.1-flash` / same | Answer model / cheaper model for profiling, planning and judging. Comma-separated lists rotate and skip rate-limited models |
| `KB_OCR_BACKEND` | `paddle` | `none` skips scanned pages |
| `KB_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (`KB_EMBEDDING_BACKEND=hash` for tests) |
| `KB_UNIT_MIN_WORDS` / `KB_UNIT_MAX_WORDS` | `60` / `220` | Paragraph unit size (re-index after changing) |
| `KB_MAX_DOCUMENTS` | `8` | Documents the planner may route to |
| `KB_MAIN_QUERY_K` / `KB_ASPECT_QUERY_K` / `KB_PER_DOCUMENT_K` | `40` / `20` / `8` | Candidate depth for the question, each aspect, each routed document |
| `KB_MAX_CANDIDATES` / `KB_JUDGE_BATCH` | `240` / `30` | Judge workload per query |
| `KB_RECOVERY_K` | `15` | Extra candidates per uncovered aspect (0 disables recovery) |
| `KB_MAX_EVIDENCE` / `KB_CONTEXT_TOKENS` | `160` / `60000` | Returned evidence cap / answer context budget |
| `KB_MAX_OUTPUT_TOKENS` | `8000` | Answer length limit |
| `KB_LLM_CONCURRENCY` | `6` | Parallel judge/profile calls |

## Evaluate

```bash
# Retrieval diagnostics only (no LLM calls; rule routing, top-ranked evidence):
.venv/bin/python -m evaluation.run --benchmark ../Train/Benchmarks/benchmark_100.jsonl --offline --output reports/runs/offline

# Full pipeline + the UNMODIFIED official notebook evaluator (paid calls):
.venv/bin/python -m evaluation.run --benchmark ../Train/Benchmarks/benchmark_100.jsonl --official --output reports/runs/official

# A subset while iterating:
.venv/bin/python -m evaluation.run --benchmark ../Train/Benchmarks/benchmark_100.jsonl --official --ids A01,B05,C10
```

`evaluation/official/Evaluation.ipynb` is a byte-identical copy of the supplied notebook; its SHA-256 is
checked before it is executed, and only its demo cell is skipped. Runs write `summary.json`,
`details.json`, `responses.jsonl` and, for official runs, `official.json` with the judge's point matching.
Gold answers and chunks are only read after each response is produced.

## Tokens, cost and tracing

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`. Each query is a Langfuse trace
with one observation per pipeline stage. LLM stages are generation observations with token usage and cost
(set `KB_LLM_INPUT_PRICE` / `KB_LLM_OUTPUT_PRICE` in USD per million tokens; unknown prices stay `null`).
`statistics.stages` in every response repeats per-stage latency and tokens, and `data/telemetry.jsonl`
keeps a local ledger even without Langfuse.

## Verification and deployment

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ruff check src tests evaluation scripts
.venv/bin/python scripts/api_smoke.py
docker build -t vjti-kb .
docker run --env-file .env -p 8000:8000 -v vjti-data:/data -v vjti-models:/models vjti-kb
```

Tests use a hash embedder, offline rules and a scripted LLM, so CI needs no model downloads or secrets.
See [reports/RESULTS.md](reports/RESULTS.md) for measured indexing time and retrieval diagnostics.
