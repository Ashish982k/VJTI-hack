"""End-to-end pipeline with a scripted LLM: routing, judging, recovery, verification, citations."""

import json
import re
import tempfile
import unittest
from dataclasses import replace

from knowledge_base.catalog import PROFILE_SYSTEM
from knowledge_base.generation import ANSWER_SYSTEM, EDIT_SYSTEM
from knowledge_base.judging import JUDGE_SYSTEM
from knowledge_base.pipeline import KnowledgeBase
from knowledge_base.planning import PLAN_SYSTEM
from test_api import offline_settings, pdf_bytes


def usage():
    return {"model": "fake", "input_tokens": 10, "output_tokens": 5, "tokens_estimated": False, "seconds": 0}


class ScriptedLLM:
    enabled = True

    def __init__(self, settings):
        self.settings = settings
        self.prompts = []

    def chat(self, system, user, *, json_mode=False, fast=False, max_tokens=4000, retries=3):
        self.prompts.append((system, user))
        if system == PROFILE_SYSTEM:
            year = "2026" if "2026" in user else "2025"
            return {"title": f"Acme Annual Report FY{year}", "issuer": "Acme", "period": f"FY{year}",
                    "topics": ["debt"], "summary": "Annual report"}, usage()
        if system == PLAN_SYSTEM:
            key = next(line.split(":")[0] for line in user.splitlines() if "FY2026" in line)
            return {"documents": [key], "aspects": [{"label": "debt defaults", "queries": ["default"]},
                                                    {"label": "auditor opinion", "queries": ["auditor"]}],
                    "entities_periods": ["Acme FY2026"], "breadth": "broad"}, usage()
        if system == JUDGE_SYSTEM:
            passages = json.loads(user.split("Passages:\n", 1)[1].split("\n\nGrade every passage")[0])
            verdicts = []
            for p in passages:
                grade = 3 if "default" in p["text"] else 2 if "auditor" in p["text"] else 0
                aspects = [1] if grade == 3 else [2] if grade == 2 else []
                verdicts.append({"id": p["id"], "grade": grade, "aspects": aspects})
            return {"verdicts": verdicts}, usage()
        if system == ANSWER_SYSTEM:
            labels = re.findall(r"\[(E\d+)\] page", user)
            self.answer_labels = labels
            return (f"## Bottom line\nAcme defaulted on Rs 4,053.90 crore [{labels[0]}].\n"
                    f"It also defaulted on Rs 777 crore [{labels[0]}].\n"
                    f"Acme was founded long ago [{labels[0]}]."), usage()
        if system == EDIT_SYSTEM:
            line = next(n for n, text in (x.split(": ", 1) for x in user.splitlines() if ": " in x and x[0].isdigit())
                        if "founded" in text)
            return {"delete": [int(line)]}, usage()
        raise AssertionError(system)


class LLMPipelineTests(unittest.TestCase):
    def test_routes_to_requested_year_and_returns_judged_evidence(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        settings = replace(offline_settings(temp.name), llm_backend="opencode")
        kb = KnowledgeBase(settings)
        kb.llm = ScriptedLLM(settings)
        filler = "The company operates construction projects across several states in India."
        docs = [
            ("Acme_Annual Report FY 2025.pdf",
             pdf_bytes("Acme annual report FY2025. Principal default was Rs 3,000 crore.", filler)),
            ("Acme_Annual Report FY 2026.pdf",
             pdf_bytes("Acme annual report FY2026. Principal default was Rs 4,053.90 crore.", filler,
                       "The auditor issued a qualified opinion with material uncertainty.")),
        ]
        stats = kb.index_documents(docs, "acme")["statistics"]
        self.assertEqual(stats["profiling_llm"]["calls"], 2)
        result = kb.query("acme", "How serious are Acme's FY2026 debt defaults?")
        documents = {c["document"] for c in result["chunks"]}
        self.assertEqual(documents, {"Acme_Annual Report FY 2026.pdf"})
        self.assertTrue(all("default" in c["text"] or "auditor" in c["text"] for c in result["chunks"]))
        self.assertIn("4,053.90", result["answer"])
        self.assertNotIn("777", result["answer"])  # unsupported figure removed by verification
        self.assertNotIn("founded", result["answer"])  # unasked line removed by the editor
        # Broad question: grade-3 plus the aspect-linked grade-2 passage reach the writer.
        self.assertEqual(len(kb.llm.answer_labels), 2)
        self.assertIn("[Acme_Annual Report FY 2026, p.1]", result["answer"])
        steps = [s["step_name"] for s in result["trace"]]
        self.assertEqual(steps, ["query_planning", "candidate_retrieval", "evidence_judgement",
                                 "coverage_recovery", "evidence_assembly", "llm_answer_generation",
                                 "answer_editing", "answer_verification"])
        self.assertGreater(result["statistics"]["llm_calls"], 2)


if __name__ == "__main__":
    unittest.main()


class PeriodFallbackLLM(ScriptedLLM):
    """Rejects everything as the wrong period unless the judge is told to accept the nearest period."""

    def chat(self, system, user, **kwargs):
        if system == JUDGE_SYSTEM:
            passages = json.loads(user.split("Passages:\n", 1)[1].split("\n\nGrade every passage")[0])
            fallback = "No passage matched the exact period" in user
            return {"verdicts": [{"id": p["id"], "grade": 3 if fallback and "default" in p["text"] else 0,
                                  "aspects": [1]} for p in passages]}, usage()
        if system == ANSWER_SYSTEM:
            self.answer_prompt = user
            label = re.findall(r"\[(E\d+)\] page", user)[0]
            return f"The documents cover FY2026, not FY2030. Default was Rs 4,053.90 crore [{label}].", usage()
        return super().chat(system, user, **kwargs)


class PeriodFallbackTests(unittest.TestCase):
    def test_missing_period_answers_from_nearest_with_mismatch_note(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        settings = replace(offline_settings(temp.name), llm_backend="opencode")
        kb = KnowledgeBase(settings)
        kb.llm = PeriodFallbackLLM(settings)
        kb.index_documents([("Acme_Annual Report FY 2026.pdf",
                             pdf_bytes("Acme FY2026. Principal default was Rs 4,053.90 crore."))], "acme")
        result = kb.query("acme", "What were Acme's FY2030 defaults?")
        self.assertIn("period_fallback", [s["step_name"] for s in result["trace"]])
        self.assertTrue(result["chunks"])
        self.assertIn("do not cover the requested period", kb.llm.answer_prompt)
