import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from fastapi.testclient import TestClient
import pymupdf

from knowledge_base.api import create_app
from knowledge_base.config import Settings
from knowledge_base.pipeline import KnowledgeBase


def pdf_bytes(*pages):
    with pymupdf.open() as pdf:
        for text in pages:
            pdf.new_page().insert_text((40, 50), text)
        return pdf.tobytes()


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(
            data_dir=Path(self.temp.name),
            mode="baseline",
            embedding_backend="hash",
            answer_backend="extractive",
        )
        self.content = zip_bytes(
            {"Acme_2025.pdf": pdf_bytes("Acme revenue is 100 million in 2025.", "Debt was 20 million.")}
        )

    def test_api_and_restart_isolation(self):
        with TestClient(create_app(self.settings)) as client:
            indexed = client.post(
                "/index", data={"collection_name": "acme"}, files={"file": ("docs.zip", self.content)}
            )
            self.assertEqual(indexed.status_code, 200, indexed.text)
            self.assertEqual(
                client.post("/query", json={"collection_name": "absent", "query": "revenue"}).status_code, 404
            )
        result = KnowledgeBase(self.settings).query("acme", "Acme revenue")
        self.assertIn("100", result["answer"])
        self.assertEqual(result["chunks"][0]["page_numbers"], [1])
        self.assertTrue(
            all(set(s) == {"step_name", "description", "input", "output"} for s in result["trace"])
        )

    def test_reject_unsafe_zip_and_collection(self):
        kb = KnowledgeBase(self.settings)
        with self.assertRaises(ValueError):
            kb.index_zip(zip_bytes({"../escape.pdf": pdf_bytes("bad")}), "safe")
        with self.assertRaises(ValueError):
            kb.index_zip(self.content, "../escape")

    def test_cache_and_baseline_switches(self):
        kb = KnowledgeBase(self.settings)
        first = kb.index_zip(self.content, "one")
        second = kb.index_zip(self.content, "two")
        self.assertGreater(first["statistics"]["full_embedding"]["computed_input_tokens"], 0)
        self.assertEqual(second["statistics"]["full_embedding"]["computed_input_tokens"], 0)
        self.assertFalse(self.settings.effective().compact_embeddings)


if __name__ == "__main__":
    unittest.main()
