import unittest

from fastapi.testclient import TestClient

from knowledge_base.api import create_app
from knowledge_base.pipeline import KnowledgeBase
import test_api
from test_api import pdf_bytes, zip_bytes


class IntegrityTests(unittest.TestCase):
    setUp = test_api.ApiTests.setUp

    def test_failed_reindex_preserves_previous_generation(self):
        kb = KnowledgeBase(self.settings)
        kb.index_zip(self.content, "safe")
        generation = kb.store.generation("safe")
        with self.assertRaises(ValueError):
            kb.index_zip(zip_bytes({"broken.pdf": b"not pdf"}), "safe")
        self.assertEqual(kb.store.generation("safe"), generation)
        self.assertIn("100", kb.query("safe", "revenue")["answer"])

    def test_no_cross_collection_leak(self):
        with TestClient(create_app(self.settings)) as client:
            client.post("/index", data={"collection_name": "one"}, files={"file": ("a.zip", self.content)})
            second = zip_bytes({"Beta_2026.pdf": pdf_bytes("Beta profit 700 million.")})
            client.post("/index", data={"collection_name": "two"}, files={"file": ("b.zip", second)})
            response = client.post(
                "/query", json={"collection_name": "two", "query": "List all Beta profit figures"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result["chunks"])
            self.assertTrue(all(c["metadata"]["collection_name"] == "two" for c in result["chunks"]))
            self.assertNotIn("Acme", result["answer"])


if __name__ == "__main__":
    unittest.main()
