# VJTI hack — Smart Knowledge Base for Agents

Python backend for the supplied Smart Knowledge Base problem statement. Source PDFs and benchmark data remain in `Train/` locally and are excluded from Git. Secrets, models and generated indexes are also excluded.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools
.venv\Scripts\python -m pip install -e ".[semantic,observability]"
Copy-Item .env.example .env
# Configure OPENCODE_API_KEY and Langfuse credentials in .env.
.venv\Scripts\python -m knowledge_base.cli index Train/Documents --collection train
.venv\Scripts\python -m knowledge_base.cli --mode baseline query "What is the revenue?" --collection train
.venv\Scripts\python -m knowledge_base.cli serve
```

`POST /index`: multipart fields `file` (ZIP) and `collection_name`. `POST /query`: JSON `{"collection_name":"train","query":"your question"}`. The PS names these logical fields but supplies no stricter transport schema. The query response includes `answer`, original `chunks`, `trace`, and optional `statistics`. Every trace entry contains exactly `step_name`, `description`, `input`, `output`. These are the only two HTTP routes.

For tests without downloads/API calls set `KB_EMBEDDING_BACKEND=hash` and `KB_ANSWER_BACKEND=extractive`. This is a lexical smoke-test mode, not a semantic RAG quality result. Real RAG uses local sentence-transformer embeddings plus the configured OpenCode answer model. Long embedding inputs are windowed, avoiding silent truncation. The default API gateway/model follows the supplied notebook and can be configured.

Run tests: `python -m unittest discover -s tests -v` after installing the package.

## Baseline, improvements and ablations

Use `KB_MODE=baseline` or `kb --mode baseline ...` for the full-chunk vector baseline. Use `KB_MODE=improved` for feature-controlled retrieval. The answer model, prompt, temperature (0), and validation are shared across experiments.

| Environment variable | Default | Purpose |
|---|---|---|
| `KB_METADATA_ROUTING` | `true` | Route by document identity and reporting period |
| `KB_COMPACT_EMBEDDINGS` | `true` | Search compact-description vectors |
| `KB_HYBRID_SEARCH` | `true` | Fuse vector search with original-text BM25 |
| `KB_MULTI_SECTION` | `true` | Independent section retrieval and bounded recovery |
| `KB_COVERAGE_SELECTION` | `true` | Prefer new coverage under a token budget |
| `KB_INDEX_VARIANTS` | `both` | Store `full`, `compact`, or `both` representations |
| `KB_CHUNK_TOKENS` / `KB_OVERLAP_TOKENS` | `400` / `60` | Lexical chunk-size estimates |
| `KB_SKETCH_TOKENS` | `120` | Lexical sketch-size cap; model token counts are measured separately |
| `KB_TOP_K` / `KB_CANDIDATE_K` | `6` / `40` | Narrow evidence and candidate budgets |
| `KB_BROAD_K` / `KB_CONTEXT_TOKENS` | `24` / `10000` | Broad evidence limit and lexical context budget |
| `KB_MAX_REQUIREMENTS` / `KB_MAX_FOLLOWUPS` | `12` / `4` | Bound section planning and recovery |
| `KB_MAX_OUTPUT_TOKENS` | `3000` | Answer-generation output limit |

Boolean variables accept `true`/`false` or `1`/`0`. Baseline mode disables all five feature switches regardless of their environment values. Index with `both` for comparisons; compact-only production indexing avoids computing full-chunk vectors. Changes to chunk/sketch settings require reindexing. A changed embedding backend/model is rejected against incompatible stored indexes.

The observed strongest precision/recall proxy balance was experiment C. To reproduce its settings in the API, use improved mode with compact embeddings and hybrid search enabled and the other three switches disabled. This observation is limited to this training set and is not an official score or a hidden-test guarantee.

## Evaluate

The six experiments are A baseline, B routing only, C sketches + BM25, D multi-section only, E coverage selection only, and F all improvements. Only document PDFs are indexed; benchmark answers/evidence are consumed exclusively by the evaluation harness after retrieval.

```powershell
# All 100 questions, without answer-model or judge API calls:
.venv\Scripts\python -m evaluation.run --collection train --retrieval-only --limit 100 --output reports/runs/local

# Generate answers and run the UNMODIFIED official notebook evaluator:
$env:KB_ANSWER_BACKEND='opencode'
.venv\Scripts\python -m evaluation.run --collection train --official --limit 100 --output reports/runs/official

# Focused run; still the same benchmark and model settings:
.venv\Scripts\python -m evaluation.run --collection train --experiments AC --official --limit 10 --output reports/runs/small
```

Official evaluation needs the organizer's `OPENCODE_API_KEY` and makes paid answer and judge calls. `evaluation/official/Evaluation.ipynb` is an exact copy of the supplied notebook; its SHA-256 is checked before execution. The adapter runs its original definitions and skips only the example invocation. Scoring definitions are not changed. In particular, the official context-recall formula can exceed one; the harness does not clamp or redefine it.

Each run exports `comparison.csv`, `comparison.json`, per-experiment details, and response JSONL. Official runs additionally export the complete judge results and point-matching explanations. Failures are reported and prevent publication of success-only averages. Separate output directories preserve earlier experiments.

Read [observed results](reports/RESULTS.md), [architecture](docs/architecture.md), and [API details](docs/api.md). The checked-in summary contains actual 100-question retrieval measurements, not generated official scores.

## Tokens, cost and tracing

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` to enable hosted traces. See [Langfuse SDK documentation](https://langfuse.com/docs/observability/sdk/instrumentation) and [usage/cost tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking). Query/retrieval/generation spans record stage latency, token usage and configured API prices. Each query also emits the required four-field trace in its HTTP response. Local summaries are appended to `data/telemetry.jsonl` even without Langfuse credentials.

Set `KB_LLM_INPUT_PRICE` and `KB_LLM_OUTPUT_PRICE` in USD per million tokens. Unknown prices remain `null`, not zero. Local embeddings incur no provider API charge, but compute cost is excluded. Final-model tokens come from the provider's usage response where available. Planning and sketch generation use no LLM. Official-judge spend is separate and is not measured by the supplied evaluator.

The default OpenCode endpoint/model follows the official notebook. [Current OpenCode documentation](https://docs.opencode.ai/docs/zen/) also describes a Zen endpoint; configure `OPENCODE_BASE_URL` for the gateway supplied by your organizers. The official evaluator deliberately retains its original endpoint and model.

## Verification and deployment

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m ruff check src tests evaluation scripts
.venv\Scripts\python scripts/api_smoke.py

# Docker deployment, once Docker and organizer server access are available:
docker build -t vjti-kb .
docker run --env-file .env -p 8000:8000 -v vjti-data:/data -v vjti-models:/models vjti-kb
```

The smoke test starts a temporary localhost server, indexes an in-memory PDF ZIP, verifies a query/citations/trace, and shuts it down. GitHub Actions runs lint, unit/integration tests and the HTTP smoke test without downloading a model or using secrets.

The first real local index took about five minutes for 34 readable PDFs, excluding model download. Treat that as an observation on this machine, not a deployment timeout guarantee. One supplied PDF requires OCR. Model loading/cache warmup, cold embeddings, network/model generation and judge latency must be measured on the organizer's server before committing to a timeout.

No API credentials or deployment access were supplied: real OpenCode answer generation, official metrics, hosted Langfuse delivery and deployment are not verified. Provider integration and support checks are covered by mocked tests; local semantic retrieval and real HTTP ingestion/querying were exercised.

## Layout

`src/knowledge_base/` contains ingestion, metadata, sketches, embeddings, search, planning, selection, generation, storage, telemetry and API modules. `evaluation/` contains the independent ablation runner and unchanged official evaluator. `tests/`, `scripts/`, `docs/`, and `reports/` contain verification, the HTTP smoke test, design/API documentation and measured summaries. `Train/`, model downloads, generated indexes, full response dumps and `.env` stay local and are Git-ignored.
