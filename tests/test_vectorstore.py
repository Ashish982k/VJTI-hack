from pathlib import Path
import tempfile
import unittest

import numpy as np

from knowledge_base.config import Settings
from knowledge_base.vectorstore import NumpyStore, QdrantStore, StoreFactory


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(60, 16)).astype(np.float32)
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.docs = np.array(["a"] * 20 + ["b"] * 25 + ["c"] * 15)
        self.query = self.vectors[3]

    def test_qdrant_matches_numpy_scores_within_routed_documents(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        factory = StoreFactory(Settings(data_dir=Path(temp.name), embedding_backend="hash"))
        qdrant = factory.create("units", "test", "gen1", self.vectors, self.docs)
        self.assertIsInstance(qdrant, QdrantStore)
        expected = NumpyStore(self.vectors, self.docs).dense(self.query, ["a", "c"])
        actual = qdrant.dense(self.query, ["a", "c"])
        self.assertTrue(np.all(np.isinf(actual[self.docs == "b"])))
        np.testing.assert_allclose(actual[self.docs != "b"], expected[self.docs != "b"], atol=1e-5)

    def test_collections_are_reused_for_the_same_index_version(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        factory = StoreFactory(Settings(data_dir=Path(temp.name), embedding_backend="hash"))
        first = factory.create("units", "test", "gen1", self.vectors, self.docs)
        second = factory.create("units", "test", "gen1", self.vectors, self.docs)
        self.assertEqual(first.parts.keys(), second.parts.keys())
        self.assertEqual(len(factory.client.get_collections().collections), 3)  # one per document

    def test_numpy_backend_is_selectable(self):
        settings = Settings(embedding_backend="hash", vector_store="numpy")
        store = StoreFactory(settings).create("units", "test", "gen1", self.vectors, self.docs)
        self.assertEqual(store.name, "numpy")


if __name__ == "__main__":
    unittest.main()
