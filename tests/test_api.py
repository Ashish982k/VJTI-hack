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
            page = pdf.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=10)
        return pdf.tobytes()


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def offline_settings(directory):
    return Settings(
        data_dir=Path(directory), embedding_backend="hash", llm_backend="offline", ocr_backend="none"
    )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = offline_settings(self.temp.name)
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
            self.assertEqual({r.path for r in client.app.routes}, {"/index", "/query"})
        result = KnowledgeBase(self.settings).query("acme", "Acme revenue")
        self.assertIn("100", result["answer"])
        self.assertEqual(result["chunks"][0]["page_numbers"], [1])
        self.assertTrue(
            all(set(s) == {"step_name", "description", "input", "output"} for s in result["trace"])
        )
        self.assertIn("Acme_2025, p.1", result["answer"])

    def test_reject_unsafe_zip_and_collection(self):
        kb = KnowledgeBase(self.settings)
        with self.assertRaises(ValueError):
            kb.index_zip(zip_bytes({"../escape.pdf": pdf_bytes("bad")}), "safe")
        with self.assertRaises(ValueError):
            kb.index_zip(self.content, "../escape")

    def test_embedding_cache_reused_across_collections(self):
        kb = KnowledgeBase(self.settings)
        first = kb.index_zip(self.content, "one")
        second = kb.index_zip(self.content, "two")
        self.assertGreater(first["statistics"]["embedding"]["computed_input_tokens"], 0)
        self.assertEqual(second["statistics"]["embedding"]["computed_input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
