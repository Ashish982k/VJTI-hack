import os
import unittest
from unittest.mock import MagicMock, patch

from knowledge_base.config import Settings
from knowledge_base.generation import edit_answer, evidence_prompt, generate, render_citations, verify
from knowledge_base.llm import LLMClient, parse_json


def evidence():
    return [
        {"label": "E1", "document": "Acme_FY2026.pdf", "page": 4, "section": "Debt",
         "text": "Principal default was Rs 4,053.90 crore as at 31 March 2026.", "title": "Acme"},
        {"label": "E2", "document": "Acme_FY2026.pdf", "page": 9, "section": "Audit",
         "text": "The auditor reported a material uncertainty on going concern.", "title": "Acme"},
    ]


class GenerationTests(unittest.TestCase):
    def test_unsupported_figures_removed_and_supported_kept(self):
        draft = (
            "## Bottom line\n"
            "Principal default was Rs 4053.9 crore [E1].\n"
            "Interest default was Rs 9,999 crore [E1].\n"
            "- Auditors flagged going concern [E2][E7]."
        )
        answer, checks = verify(draft, evidence(), "What are the defaults?")
        self.assertIn("4053.9", answer)
        self.assertNotIn("9,999", answer)
        self.assertEqual(checks["removed_lines"][0]["unsupported_numbers"], ["9999"])
        self.assertEqual(checks["unknown_citations"], ["E7"])
        self.assertIn("going concern", answer)

    def test_render_page_citations(self):
        text, cited = render_citations("Default [E1][E2]. Again [E1, E2].", evidence())
        self.assertEqual(text, "Default [Acme_FY2026, pp.4, 9]. Again [Acme_FY2026, pp.4, 9].")
        self.assertEqual(cited, ["E1", "E2"])

    def test_provider_usage_and_evidence_in_prompt(self):
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": "Default was Rs 4,053.90 crore [E1]."}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 12},
        }
        with patch.dict(os.environ, {"OPENCODE_API_KEY": "test-only"}):
            llm = LLMClient(Settings())
        llm.client = MagicMock()
        llm.client.post.return_value = response
        answer, usage = generate("What is the default?", evidence(), llm, Settings())
        self.assertEqual(usage["input_tokens"], 123)
        request = llm.client.post.call_args.kwargs["json"]
        self.assertIn(evidence()[0]["text"], request["messages"][1]["content"])
        self.assertEqual(request["temperature"], 0)

    def test_truncated_reasoning_retries_with_larger_budget(self):
        cut = MagicMock(status_code=200)
        cut.json.return_value = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        with patch.dict(os.environ, {"LLM_API_KEY": "test-only"}):
            llm = LLMClient(Settings(reasoning_tokens=1000))
        llm.client = MagicMock()
        llm.client.post.side_effect = [cut, ok]
        with patch("knowledge_base.llm.time.sleep"):
            result, _ = llm.chat("s", "u", json_mode=True, max_tokens=500)
        self.assertEqual(result, {"a": 1})
        budgets = [c.kwargs["json"]["max_tokens"] for c in llm.client.post.call_args_list]
        self.assertEqual(budgets, [1500, 3000])

    def test_empty_evidence_does_not_call_model(self):
        llm = MagicMock(enabled=True)
        answer, usage = generate("Anything", [], llm, Settings())
        llm.chat.assert_not_called()
        self.assertIsNone(usage)
        self.assertIn("do not contain", answer)

    def test_editor_only_deletes_lines_and_drops_empty_sections(self):
        llm = MagicMock()
        llm.chat.return_value = ({"delete": [4, "5"]}, {"model": "m", "input_tokens": 1, "output_tokens": 1})
        draft = "## Bottom line\nDefault was Rs 4,053.90 crore [E1].\n## Background\nAcme is old [E2].\nIt has staff."
        answer, edits, _ = edit_answer("What is the default?", draft, llm)
        self.assertEqual(answer, "## Bottom line\nDefault was Rs 4,053.90 crore [E1].")
        self.assertEqual(len(edits["deleted_lines"]), 2)
        llm.chat.return_value = ({"delete": [1, 2]}, {"model": "m", "input_tokens": 1, "output_tokens": 1})
        self.assertIn("Default", edit_answer("q", "## Bottom line\nDefault [E1].", llm)[0])

    def test_editor_never_deletes_not_stated_lines(self):
        llm = MagicMock()
        llm.chat.return_value = ({"delete": [2, 3]}, {"model": "m", "input_tokens": 1, "output_tokens": 1})
        draft = "Default was Rs 4,053.90 crore [E1].\nThe provided documents do not state the rating.\nFiller."
        answer, _, _ = edit_answer("q", draft, llm)
        self.assertIn("do not state the rating", answer)
        self.assertNotIn("Filler", answer)

    def test_outline_only_for_broad_questions(self):
        broad = {"breadth": "broad", "aspects": [{"label": "lender actions"}, {"label": "auditor concerns"}]}
        self.assertIn("1. lender actions\n2. auditor concerns", evidence_prompt("q", evidence(), broad))
        narrow = evidence_prompt("q", evidence(), {"breadth": "narrow", "aspects": broad["aspects"]})
        self.assertIn("narrow question", narrow)
        self.assertNotIn("lender actions", narrow)

    def test_parse_fenced_json(self):
        self.assertEqual(parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json('Here: {"a": [1]} done'), {"a": [1]})


if __name__ == "__main__":
    unittest.main()
