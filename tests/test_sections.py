from pathlib import Path
import tempfile
import unittest

import numpy as np

from knowledge_base.config import Settings
from knowledge_base.embedding import Embedder
from knowledge_base.retrieval import CandidateSearch, unit_search_text
from knowledge_base.search import BM25, GroupedBM25


def unit(doc, section, text, page):
    return {"document_id": doc, "document": f"{doc}.pdf", "section": section, "text": text,
            "page_numbers": [page], "chunk_id": f"{doc}:{page}:{text[:5]}"}


class SectionRetrievalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.settings = Settings(data_dir=Path(temp.name), embedding_backend="hash", llm_backend="offline",
                                 ocr_backend="none", main_query_k=1, aspect_query_k=1, per_document_k=1,
                                 neighbor_window=0, sections_per_query=1, section_min_units=1)
        self.chunks = [
            unit("a", "Overview", "The company builds bridges and roads across the region.", 1),
            unit("a", "Lender actions", "Lenders recalled all facilities in June.", 2),
            unit("a", "Lender actions", "Banks filed recovery suits at the tribunal.", 3),
            unit("a", "Lender actions", "Secured creditors took possession of the head office.", 4),
            unit("a", "Outlook", "Management expects growth in new orders.", 5),
        ]
        embedder = Embedder(self.settings)
        texts = [unit_search_text(c, "Report") for c in self.chunks]
        vectors, _ = embedder.encode(texts)
        self.search = CandidateSearch(self.chunks, vectors, BM25(BM25.build(texts)), embedder, self.settings)

    def test_sections_are_heading_runs(self):
        self.assertEqual(self.search.section_starts.tolist(), [0, 1, 4])
        self.assertEqual(self.search.section_of_unit.tolist(), [0, 1, 1, 1, 2])

    def test_merged_sections_join_small_heading_runs(self):
        from dataclasses import replace
        self.search.settings = replace(self.settings, section_min_units=3, section_cap_units=4)
        self.search._build_sections()
        # Overview (1) + Lender actions (3) merge until 3 units; the cap of 4 then starts a new section.
        self.assertEqual(self.search.section_starts.tolist(), [0, 4])

    def test_grouped_bm25_sums_unit_counts(self):
        units = BM25(BM25.build(["lenders recalled", "lenders sued", "orders grew"]))
        grouped = GroupedBM25(units, np.array([0, 0, 1]), 2)
        scores = grouped.scores("lenders")
        self.assertGreater(scores[0], 0)
        self.assertEqual(scores[1], 0)
        self.assertEqual(grouped.lengths.tolist(), [4.0, 2.0])

    def test_broad_question_takes_whole_section(self):
        plan = {"document_ids": ["a"], "aspects": [], "breadth": "broad"}
        found, summary, _ = self.search.gather("lenders recalled facilities", plan)
        indices = {c["index"] for c in found}
        self.assertTrue({1, 2, 3} <= indices)  # all lender-action paragraphs, not just the matching one
        narrow, _, _ = self.search.gather("lenders recalled facilities", {**plan, "breadth": "narrow"})
        self.assertLess(len(narrow), len(found))


if __name__ == "__main__":
    unittest.main()
