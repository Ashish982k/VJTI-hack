from dataclasses import replace
import os
import unittest
from unittest.mock import MagicMock, patch

from knowledge_base.config import Settings
from knowledge_base.generation import generate, verified_answer


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"chunk_id": "abc:1:0", "document": "Acme2026.pdf", "page_numbers": [1], "text": "Debt was INR 100 crore in 2026."}]

    def test_reject_fabricated_numbers_and_citations(self):
        answer, checks = verified_answer("Debt was INR 900 crore [abc:1:0].", self.chunks, "opencode")
        self.assertIn("Insufficient", answer)
        self.assertEqual(checks["unsupported_numbers"], ["900"])
        answer, checks = verified_answer("Debt was INR 100 crore [fake].", self.chunks, "opencode")
        self.assertFalse(checks["passed"])
        self.assertIn("fake", checks["unknown_citations"])

    def test_valid_number_and_source_retained(self):
        text = "Debt was INR 100 crore in 2026 [abc:1:0]."
        answer, checks = verified_answer(text, self.chunks, "opencode")
        self.assertEqual(answer, text)
        self.assertTrue(checks["passed"])

    def test_actual_provider_usage_and_original_evidence(self):
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": "Debt was INR 100 crore [abc:1:0]."}}], "usage": {"prompt_tokens": 123, "completion_tokens": 12}}
        client = MagicMock()
        client.__enter__.return_value.post.return_value = response
        with patch("knowledge_base.generation.httpx.Client", return_value=client), patch.dict(os.environ, {"OPENCODE_API_KEY": "test-only"}):
            _, usage = generate("What is debt?", self.chunks, Settings())
        self.assertEqual(usage["llm_input_tokens"], 123)
        request = client.__enter__.return_value.post.call_args.kwargs["json"]
        self.assertIn(self.chunks[0]["text"], request["messages"][1]["content"])
        self.assertEqual(request["temperature"], 0)

    def test_empty_evidence_does_not_call_model(self):
        with patch("knowledge_base.generation.httpx.Client") as client:
            answer, usage = generate("Anything", [], replace(Settings(), answer_backend="opencode"))
        client.assert_not_called()
        self.assertIn("Insufficient", answer)
        self.assertEqual(usage["llm_input_tokens"], 0)
