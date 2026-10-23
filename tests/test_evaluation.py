import hashlib
import os
import unittest
from unittest.mock import patch

from evaluation.official_adapter import DEFAULT_NOTEBOOK, OFFICIAL_SHA256, load_official
from evaluation.run import experiments, retrieval_diagnostics
from knowledge_base.config import Settings
from knowledge_base.telemetry import estimate_cost


class EvaluationTests(unittest.TestCase):
    def test_official_evaluator_unchanged(self):
        self.assertEqual(hashlib.sha256(DEFAULT_NOTEBOOK.read_bytes()).hexdigest(), OFFICIAL_SHA256)
        with patch.dict(os.environ, {"OPENCODE_API_KEY": "test-only"}):
            evaluator, run = load_official()
        self.assertEqual(type(evaluator).__name__, "RAGMetricsEvaluator")
        self.assertTrue(callable(run))

    def test_ablation_flags_are_independent(self):
        configs = experiments(Settings())
        self.assertFalse(configs["A"][1].effective().metadata_routing)
        self.assertTrue(configs["B"][1].metadata_routing)
        self.assertFalse(configs["B"][1].compact_embeddings)
        self.assertFalse(configs["D"][1].metadata_routing)
        self.assertFalse(configs["E"][1].multi_section)
        self.assertTrue(configs["F"][1].coverage_selection)

    def test_unknown_prices_not_reported_as_zero(self):
        with patch.dict(os.environ, {}, clear=True):
            cost = estimate_cost({"llm_input_tokens": 100, "llm_output_tokens": 10}, "sentence-transformers")
        self.assertIsNone(cost["total_usd"])
        self.assertEqual(cost["embedding_usd"], 0)

    def test_page_proxy_matches_document_as_well_as_page(self):
        gold = [{"content": "debt", "metadata": {"document": "a.pdf", "page_numbers": [1]}}]
        predicted = [{"text": "debt", "document": "b.pdf", "page_numbers": [1]}]
        self.assertEqual(retrieval_diagnostics(predicted, gold)["page_recall_proxy"], 0)
