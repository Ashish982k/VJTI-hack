"""Deterministic document identity and conservative routing, independent of benchmark data."""
from pathlib import PurePosixPath
import re

from .text import terms

YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:\s*[-\u2013\u2014/]\s*((?:19|20)?\d{2})|(\d{2})(?!\d))?")


def periods(text):
    result = []
    for match in YEAR.finditer(text):
        start = int(match[1])
        suffix = match[2] or match[3]
        end = int(suffix) if suffix and len(suffix) == 4 else start // 100 * 100 + int(suffix) if suffix else start
        if suffix and end < start:
            end += 100
        if end - start <= 1:
            result.append((start, end))
    return sorted(set(result))


def document_metadata(filename, cover_text):
    stem = PurePosixPath(filename).stem
    expanded = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", stem)
    name = expanded.replace("_", " ")
    identified = periods(name)
    if not identified:
        short_year = re.search(r"\b(?:FY|AR)[ -]?(\d{2})(?!\d)", name, re.I)
        identified = [(2000 + int(short_year[1]), 2000 + int(short_year[1]))] if short_year else periods(cover_text[:1800])
    combined = name + " " + cover_text[:1500]
    doc_type = next((kind for pattern, kind in [
        (r"annual\s*report|\bar[- ]?\d", "annual report"), (r"handbook", "handbook"),
        (r"standard|framework|committee", "regulatory standard"), (r"memo", "memorandum"),
        (r"technical|model|paper", "technical paper")]
        if re.search(pattern, combined, re.I)), "document")
    company = ""
    if doc_type == "annual report":
        company = re.sub(r"(?i)integrated|annual\s*report|\bFY\b|\bar\b|\bre\b", " ", name)
        company = YEAR.sub(" ", company)
        company = re.sub(r"\d+|[-()]", " ", company)
        company = re.sub(r"\s+", " ", company).strip()
    if not company:
        # A title-like organization line, never arbitrary body entities.
        for line in cover_text.splitlines()[:25]:
            if re.search(r"\b(limited|ltd\.?|corporation|committee|authority|bank)\b", line, re.I) and 2 <= len(line.split()) <= 12:
                company = line.strip()
                break
    aliases = sorted(set(terms(company)) - {"india", "industries", "inc", "ltd", "group", "corporation"})
    return {"company": company, "periods": [f"{a}-{b}" if a != b else str(a) for a, b in identified],
            "reporting_years": sorted({b for _, b in identified}), "document_type": doc_type,
            "aliases": aliases, "metadata_source": "filename_and_cover_rules"}


def page_heading(text, fallback):
    for line in text.splitlines()[:16]:
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 12 and not re.search(r"\d{2,}|https?://|\.{3}", line):
            if line.isupper() or re.match(r"^\d{1,2}[.)]\s+[A-Z]", line):
                return line
    return fallback


def route(question, documents):
    query_terms = set(terms(question))
    requested = periods(question)
    years = {b for _, b in requested}
    company_matches = [d for d in documents if set(d.get("aliases", [])) & query_terms]
    candidates = company_matches or list(documents)
    reasons = ["company alias match"] if company_matches else ["no confident entity match: retain candidates"]
    if years:
        matched = [d for d in candidates if years & set(d.get("reporting_years", []))]
        # Missing metadata is uncertainty, not evidence that a document is wrong.
        unknown = [d for d in candidates if not d.get("reporting_years")]
        covered = set().union(*(set(d.get("reporting_years", [])) for d in matched)) if matched else set()
        if matched and years <= covered:
            candidates = matched + unknown
            reasons.append("requested reporting end-year(s); retain unknown periods")
        else:
            reasons.append("requested year unavailable or uncertain: retain entity candidates")
    for kind in ("annual report", "handbook", "memorandum", "technical paper", "regulatory standard"):
        if kind in question.lower():
            matched = [d for d in candidates if d["document_type"] == kind]
            if matched:
                candidates = matched
                reasons.append("document type")
    return {"document_ids": [d["document_id"] for d in candidates],
            "requested_years": sorted(years), "matched_companies": sorted({d["company"] for d in company_matches}),
            "confidence": "high" if company_matches else "uncertain", "reasons": reasons}
