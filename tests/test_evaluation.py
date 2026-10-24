import hashlib
import os
import unittest
from unittest.mock import patch

from evaluation.official_adapter import DEFAULT_NOTEBOOK, OFFICIAL_SHA256, load_official
from evaluation.contract_metrics import normalize_name
from evaluation.run import retrieval_diagnostics
from knowledge_base.telemetry import estimate_cost


class EvaluationTests(unittest.TestCase):
    def test_official_evaluator_unchanged(self):
        self.assertEqual(hashlib.sha256(DEFAULT_NOTEBOOK.read_bytes()).hexdigest(), OFFICIAL_SHA256)
        with patch.dict(os.environ, {"OPENCODE_API_KEY": "test-only"}):
            evaluator, run = load_official()
        self.assertEqual(type(evaluator).__name__, "RAGMetricsEvaluator")
        self.assertTrue(callable(run))

    def test_unknown_prices_not_reported_as_zero(self):
        with patch.dict(os.environ, {}, clear=True):
            cost = estimate_cost({"llm_input_tokens": 100, "llm_output_tokens": 10}, "sentence-transformers")
        self.assertIsNone(cost["total_usd"])
        self.assertEqual(cost["embedding_usd"], 0)

    def test_contract_context_metrics_need_document_and_page_overlap(self):
        row = {"benchmark_chunks_w_metadata": {"Report_2023.pdf": [
            {"content": "debt rose", "metadata": {"document": "Report_2023.pdf", "page_numbers": [3, 4]}}]}}
        chunks = [
            {"content": "debt rose", "metadata": {"file_name": "report-2023.PDF", "page_numbers": [4]}},
            {"content": "debt rose", "metadata": {"file_name": "other.pdf", "page_numbers": [3]}},
        ]
        diagnostics = retrieval_diagnostics(chunks, row)
        self.assertEqual(diagnostics["contract_context_precision"], 0.5)
        self.assertEqual(diagnostics["contract_context_recall"], 0.5)
        self.assertEqual(normalize_name("docs/Report_2023.pdf"), "report2023")
