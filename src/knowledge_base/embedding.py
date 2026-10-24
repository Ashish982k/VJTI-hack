"""Local embeddings with persistent exact-input cache and explicit token accounting."""

import hashlib
import sqlite3
from contextlib import closing
from threading import RLock

import numpy as np

from .text import digest, terms, tokens


class Embedder:
    def __init__(self, settings):
        self.settings = settings
        self.identity = f"v2:{settings.embedding_backend}:{settings.embedding_model}"
        self.model = None
        self.lock = RLock()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = settings.data_dir / "embedding-cache.sqlite3"
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB)")

    def _load(self):
        if self.settings.embedding_backend == "sentence-transformers" and self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.settings.embedding_model)

    def count(self, text):
        self._load()
        if self.model is not None:
            ids = self.model.tokenizer.encode(text, add_special_tokens=False)
            window = self.model.max_seq_length - 2
            return len(ids) + 2 * max(1, (len(ids) + window - 1) // window)
        return len(tokens(text))

    def _encode(self, texts):
        if self.model is not None:
            # Window long inputs instead of silently truncating complete baseline chunks.
            windows, owners = [], []
            width = self.model.max_seq_length - 2
            for i, text in enumerate(texts):
                ids = self.model.tokenizer.encode(text, add_special_tokens=False)
                for j in range(0, max(1, len(ids)), width):
                    windows.append(self.model.tokenizer.decode(ids[j : j + width]))
                    owners.append(i)
            encoded = self.model.encode(
                windows, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
            result = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
            for owner, vector in zip(owners, encoded):
                result[owner] += vector
            return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
        # Explicit test backend, never described as semantic retrieval.
        result = np.zeros((len(texts), 512), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in terms(text):
                value = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
                result[row, value % 512] += 1
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)

    def encode_queries(self, texts):
        # Asymmetric retrieval models (e.g. BGE) expect an instruction prefix on queries only.
        prefix = self.settings.embedding_query_prefix if self.settings.embedding_backend != "hash" else ""
        return self.encode([prefix + t for t in texts])

    def encode(self, texts):
        with self.lock:
            self._load()
            keys = [digest(self.identity + "\0" + t) for t in texts]
            total = sum(self.count(t) for t in texts)
            vectors, missing = {}, {}
            with closing(sqlite3.connect(self.db, timeout=60)) as db, db:
                for key, text in zip(keys, texts):
                    row = db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                    if row:
                        vectors[key] = np.frombuffer(row[0], dtype=np.float32)
                    else:
                        missing[key] = text
                if missing:
                    computed = self._encode(list(missing.values()))
                    for key, vector in zip(missing, computed):
                        vector = np.asarray(vector, dtype=np.float32)
                        vectors[key] = vector
                        db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?)", (key, vector.tobytes()))
            return np.stack([vectors[k] for k in keys]), {
                "logical_input_tokens": total,
                "computed_input_tokens": sum(self.count(t) for t in missing.values()),
                "cache_or_duplicate_hits": len(texts) - len(missing),
                "tokenizer": "model tokenizer incl. window specials" if self.model else "lexical estimate",
                "embedding_backend": self.settings.embedding_backend,
            }
