"""Vector stores for dense retrieval: Qdrant (default) or plain NumPy. Both return exact cosine scores.

Retrieval needs a score for every unit of the routed documents (rank fusion, section fits), so a store
answers "score this query against all vectors of these documents" and leaves everything else at -inf.
"""

import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


class NumpyStore:
    name = "numpy"

    def __init__(self, vectors, document_ids):
        self.vectors = vectors
        self.document_ids = document_ids

    def dense(self, vector, document_ids):
        allowed = np.isin(self.document_ids, list(document_ids))
        return np.where(allowed, self.vectors @ vector, -np.inf).astype(np.float32)


class QdrantStore:
    """Qdrant collections partitioned by document: one collection per document of an index version.

    Document routing selects which collections to search, so no payload filter is scanned (in embedded
    mode a filter is checked point by point, ~185 ms per search; a per-document search takes a few ms).
    Embedded mode (no URL) rebuilds the collections from the saved vectors on load; with a server URL they
    are created once and reused across restarts.
    """

    name = "qdrant"

    def __init__(self, client, prefix, vectors, document_ids, exact):
        from qdrant_client import models

        self.models = models
        self.client = client
        self.size = len(document_ids)
        self.search_params = models.SearchParams(exact=True) if exact else None
        self.parts = {}
        for n, document in enumerate(sorted(set(document_ids.tolist()))):
            positions = np.flatnonzero(document_ids == document)
            collection = f"{prefix}_{n}"
            if not self._ready(collection, len(positions)):
                self._build(collection, vectors[positions], document)
            self.parts[document] = (collection, positions)

    def _ready(self, collection, count):
        try:
            return self.client.collection_exists(collection) and \
                self.client.count(collection, exact=True).count == count
        except Exception:
            return False

    def _build(self, collection, vectors, document):
        models = self.models
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
        )
        self.client.upload_collection(
            collection,
            vectors=np.asarray(vectors, dtype=np.float32),
            payload=[{"document_id": str(document)}] * len(vectors),
            ids=list(range(len(vectors))),
            batch_size=512,
        )

    def dense(self, vector, document_ids):
        scores = np.full(self.size, -np.inf, dtype=np.float32)
        query = np.asarray(vector, dtype=np.float32).tolist()
        for document in document_ids:
            if document not in self.parts:
                continue
            collection, positions = self.parts[document]
            result = self.client.query_points(
                collection, query=query, limit=len(positions), with_payload=False,
                search_params=self.search_params,
            )
            for point in result.points:
                scores[positions[int(point.id)]] = point.score
        return scores


class StoreFactory:
    """Creates stores for a loaded collection; one Qdrant client per process."""

    def __init__(self, settings):
        self.settings = settings
        self.client = None

    def _qdrant(self):
        if self.client is None:
            from qdrant_client import QdrantClient

            if self.settings.qdrant_url:
                self.client = QdrantClient(url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key or None)
            else:
                self.client = QdrantClient(location=":memory:")
        return self.client

    def create(self, kind, collection, generation, vectors, document_ids, variant=""):
        if self.settings.vector_store == "numpy":
            return NumpyStore(vectors, document_ids)
        tag = hashlib.sha256(f"{generation}:{variant}".encode()).hexdigest()[:12]
        name = f"kb_{collection}_{kind}_{tag}"
        return QdrantStore(self._qdrant(), name, vectors, document_ids, exact=bool(self.settings.qdrant_url))
