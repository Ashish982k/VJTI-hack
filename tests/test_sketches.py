import unittest
import numpy as np

from knowledge_base.search import BM25, rank
from knowledge_base.sketches import make_sketches
from knowledge_base.text import tokens


class SketchTests(unittest.TestCase):
    def test_budget_original_and_figures(self):
        original = "Boilerplate operating review. " * 100 + "Distinct litigation liability INR 42.75 crore in 2026."
        chunks = [{"text": original, "company": "Acme", "periods": ["2026"], "document_id": "d", "page_numbers": [1], "section_heading": "Litigation"}]
        make_sketches(chunks, 100)
        self.assertEqual(chunks[0]["text"], original)
        self.assertLessEqual(len(tokens(chunks[0]["sketch"])), 100)
        bm25 = BM25(BM25.build(chunks))
        self.assertGreater(bm25.scores("42.75 litigation")[0], 0)

    def test_sparse_recovers_missing_dense_evidence_and_obeys_filter(self):
        bm25 = BM25(BM25.build([{"text": "debt"}, {"text": "rare guarantee covenant"}, {"text": "rare covenant"}]))
        ranked = rank(np.array([1., 0., .2]), bm25.scores("rare covenant"), np.array([True, True, False]), 3, True)
        self.assertIn(1, [i for i, _ in ranked])
        self.assertNotIn(2, [i for i, _ in ranked])
