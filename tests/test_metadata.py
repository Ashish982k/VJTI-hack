import unittest

from knowledge_base.metadata import document_metadata, periods, route


def doc(name, identifier):
    return {"document_id": identifier, **document_metadata(name, "")}


class MetadataTests(unittest.TestCase):
    def test_fiscal_formats(self):
        self.assertEqual(periods("FY202526 and 2024-25 and FY2026"), [(2024, 2025), (2025, 2026), (2026, 2026)])
        self.assertEqual(document_metadata("Acme_Annual Report FY 202526.pdf", "")["company"], "Acme")

    def test_route_year_and_cross_company(self):
        documents = [doc("Acme_Annual Report FY 2025.pdf", "old"), doc("Acme_Annual Report FY 2026.pdf", "new"), doc("Beta_Annual Report FY 2026.pdf", "beta")]
        self.assertEqual(route("Acme revenue in 2026", documents)["document_ids"], ["new"])
        self.assertEqual(set(route("Compare Acme and Beta in 2026", documents)["document_ids"]), {"new", "beta"})
        self.assertEqual(set(route("Compare Acme 2025 and 2026", documents)["document_ids"]), {"new", "old"})

    def test_uncertainty_does_not_force_one_document(self):
        documents = [doc("Acme_Annual Report FY 2025.pdf", "old"), doc("Beta handbook.pdf", "unknown")]
        self.assertEqual(len(route("What are the risks?", documents)["document_ids"]), 2)
        self.assertEqual(route("Acme 2030 debt", documents)["document_ids"], ["old"])
