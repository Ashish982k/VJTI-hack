from pathlib import Path
import tempfile
import unittest

import pymupdf

from knowledge_base.config import Settings
from knowledge_base.ingestion import build_units, extract_document
from knowledge_base.ocr import group_lines
from knowledge_base.text import split_long


def settings(directory):
    return Settings(data_dir=Path(directory), embedding_backend="hash", llm_backend="offline",
                    ocr_backend="none", unit_min_words=10, unit_max_words=40)


def block(text, size=10.0, bold=False):
    return {"text": text, "size": size, "bold": bold}


class FakeOCR:
    def __init__(self):
        self.calls = []

    def page_blocks(self, document_digest, page):
        self.calls.append(page.number)
        return [block("Scanned guideline text about problem loans and watch list triggers for banks.")]


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = settings(self.temp.name)

    def test_units_follow_headings_and_drop_margins(self):
        body = "Lenders recalled all facilities and the account remains a non performing asset today."
        pages = [
            {"number": n, "label": str(n), "blocks": [block("ACME LIMITED")]
             + ([block("RISK FACTORS", size=14)] if n == 1 else []) + [block(body + " " + "abcd"[n - 1] * 3), block(str(n))]}
            for n in range(1, 5)
        ]
        units, outline = build_units(pages, [], self.settings)
        self.assertEqual(len(units), 4)
        self.assertTrue(all(u["section"] == "RISK FACTORS" for u in units))
        self.assertTrue(all("ACME LIMITED" not in u["text"] for u in units))
        self.assertEqual(outline, ["RISK FACTORS"])

    def test_short_blocks_merge_and_long_blocks_split(self):
        short = [block("Revenue rose."), block("Margins fell sharply this year."), block("Debt was flat.")]
        pages = [{"number": 1, "label": "1", "blocks": short + [block("Long sentence here. " * 30)]}]
        units, _ = build_units(pages, [], self.settings)
        self.assertIn("Revenue rose.\nMargins fell", units[0]["text"])
        self.assertTrue(all(u["words"] <= 40 + 30 for u in units))
        self.assertEqual(len(split_long("One two three. Four five six.", 3)), 2)

    def test_scanned_page_uses_ocr(self):
        with pymupdf.open() as pdf:
            pdf.new_page()
            content = pdf.tobytes()
        with self.assertRaises(ValueError):
            extract_document("scan.pdf", content, "c", self.settings, None)
        ocr = FakeOCR()
        doc, chunks = extract_document("scan.pdf", content, "c", self.settings, ocr)
        self.assertEqual(ocr.calls, [0])
        self.assertEqual(doc["ocr_pages"], [1])
        self.assertIn("watch list", chunks[0]["text"])

    def test_ocr_lines_grouped_by_vertical_gap(self):
        lines = [("First line of a", (0, 0, 100, 10)), ("paragraph.", (0, 12, 60, 22)),
                 ("Second paragraph.", (0, 60, 90, 70))]
        self.assertEqual([b["text"] for b in group_lines(lines)],
                         ["First line of a paragraph.", "Second paragraph."])


if __name__ == "__main__":
    unittest.main()
