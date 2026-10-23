import unittest
import numpy as np

from knowledge_base.config import Settings
from knowledge_base.planning import classify, evidence_requirements, expand_candidates
from knowledge_base.search import BM25


class PlanningTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(classify("List every operational issue"), "exhaustive")
        self.assertEqual(classify("Compare Acme and Beta"), "comparison")
        self.assertEqual(classify("What is revenue?"), "numerical")

    def test_independent_sections_recover_diverse_evidence(self):
        chunks = [
            {
                "chunk_id": str(i),
                "document_id": "a",
                "document": "a.pdf",
                "section_heading": title,
                "text": body,
            }
            for i, (title, body) in enumerate(
                [
                    ("Borrowings", "risk from unpaid debt"),
                    ("Borrowings", "risk from overdue debt"),
                    ("Litigation", "risk from court proceedings"),
                ]
            )
        ]
        settings = Settings(max_requirements=3)
        scores = np.array([0.9, 0.8, 0.6])
        allowed = np.ones(3, dtype=bool)
        reqs = evidence_requirements("List every risk", chunks, allowed, scores, settings)
        ranked, _ = expand_candidates(
            "List every risk",
            chunks,
            scores,
            BM25(BM25.build(chunks)),
            allowed,
            [(0, 0.1), (1, 0.09)],
            reqs,
            settings,
        )
        self.assertEqual({chunks[i]["section_heading"] for i, _ in ranked[:2]}, {"Borrowings", "Litigation"})
        self.assertTrue(all(r["label"] in {"Borrowings", "Litigation"} for r in reqs))
