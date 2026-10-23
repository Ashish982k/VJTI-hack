import unittest

from knowledge_base.selection import evidence_cost, select_evidence


def chunk(identifier, text, score=1, doc="d", section="Debt"):
    return {
        "chunk_id": identifier,
        "text": text,
        "score": score,
        "document_id": doc,
        "document": doc + ".pdf",
        "page_numbers": [1],
        "periods": ["2026"],
        "section_heading": section,
    }


class SelectionTests(unittest.TestCase):
    def test_coverage_duplicate_and_budget(self):
        chunks = [
            chunk("1", "Debt default INR 100 crore."),
            chunk("2", "Debt default INR 100 crore."),
            chunk("3", "Court litigation could cost INR 50 crore.", 0.8, section="Litigation"),
        ]
        req = [{"id": "legal", "kind": "section", "document_id": "d", "section_heading": "Litigation"}]
        selected, stats = select_evidence(chunks, req, 100, 2)
        self.assertIn("3", [c["chunk_id"] for c in selected])
        self.assertEqual(len(selected), 2)
        self.assertLessEqual(sum(evidence_cost(c) for c in selected), 100)
        self.assertEqual(stats["covered_requirements"], ["legal"])
        self.assertEqual(sum(len(c.get("duplicate_sources", [])) for c in selected), 1)

    def test_distinct_figures_and_documents_survive(self):
        chunks = [
            chunk("1", "Debt default INR 100 crore."),
            chunk("2", "Debt default INR 101 crore."),
            chunk("3", "Debt default INR 100 crore.", doc="other"),
        ]
        selected, _ = select_evidence(chunks, [], 1000, 3)
        self.assertEqual(len(selected), 3)

    def test_tiny_budget_returns_no_truncated_financial_passage(self):
        selected, _ = select_evidence([chunk("1", "Amount INR 100 crore")], [], 1, 4)
        self.assertEqual(selected, [])
