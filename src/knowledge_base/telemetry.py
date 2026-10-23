"""Langfuse observations plus durable local usage summaries; no invented price defaults."""
from contextlib import contextmanager
import json
import logging
import os
from threading import Lock
import time

logger = logging.getLogger(__name__)


def estimate_cost(statistics, embedding_backend):
    def charge(env, count, local=False):
        if count == 0 or local:
            return 0.0
        price = os.getenv(env)
        return float(price) * count / 1_000_000 if price else None

    costs = {
        "embedding_usd": charge("KB_EMBEDDING_PRICE", statistics.get("query_embedding", {}).get("computed_input_tokens", statistics.get("actual_computed_embedding_tokens", 0)), local=embedding_backend in {"hash", "sentence-transformers"}),
        "llm_input_usd": charge("KB_LLM_INPUT_PRICE", statistics.get("llm_input_tokens", 0)),
        "llm_output_usd": charge("KB_LLM_OUTPUT_PRICE", statistics.get("llm_output_tokens", 0)),
    }
    costs["total_usd"] = sum(costs.values()) if all(v is not None for v in costs.values()) else None
    costs["scope"] = "API charges only; excludes local compute and official judge calls"
    return costs


class Telemetry:
    def __init__(self, root):
        self.path = root / "telemetry.jsonl"
        self.lock = Lock()
        self.client = None
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse
                self.client = Langfuse()
            except Exception:
                logger.exception("Langfuse initialization failed; local telemetry remains active")

    @contextmanager
    def span(self, name, **kwargs):
        if self.client is None:
            yield None
            return
        manager = self.client.start_as_current_observation(name=name, **kwargs)
        with manager as observation:
            yield observation

    def record(self, record):
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp_unix": time.time(), **record}) + "\n")

    def flush(self):
        if self.client is not None:
            self.client.flush()
