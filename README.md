# VJTI hack — Smart Knowledge Base for Agents

Python backend for the supplied Smart Knowledge Base problem statement. Source PDFs and benchmark data remain in `Train/` locally and are excluded from Git. Secrets, models and generated indexes are also excluded.

## Quick start

```powershell
python -m venv --system-site-packages .venv
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
