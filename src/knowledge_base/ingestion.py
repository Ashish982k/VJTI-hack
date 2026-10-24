"""Page-preserving ingestion into paragraph-sized evidence units. Only PDFs enter the index."""

from collections import Counter
import hashlib
import logging
from pathlib import PurePosixPath
import re
import statistics
import zipfile

import pymupdf

from .metadata import document_metadata
from .text import digest, split_long, words

logger = logging.getLogger(__name__)

PAGE_NUMBER = re.compile(r"(?i)^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$")


def read_zip(source, settings):
    documents = []
    seen = set()
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if len(members) > settings.max_files:
            raise ValueError("ZIP has too many entries")
        if sum(i.file_size for i in members) > settings.max_unpacked_bytes:
            raise ValueError("ZIP exceeds uncompressed size limit")
        for info in members:
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in info.filename:
                raise ValueError("Unsafe ZIP path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are not supported")
            if path.parts and path.parts[0] == "__MACOSX":
                continue
            if not info.is_dir() and path.suffix.lower() == ".pdf":
                if str(path).casefold() in seen:
                    raise ValueError("ZIP contains duplicate PDF paths")
                seen.add(str(path).casefold())
                documents.append((str(path), archive.read(info)))
    if not documents:
        raise ValueError("ZIP contains no PDFs")
    return documents


def _text_blocks(page):
    blocks = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue
        lines, sizes, bold = [], [], True
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            lines.append("".join(s["text"] for s in line["spans"]).strip())
            sizes.extend(s["size"] for s in spans)
            bold &= all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
        text = re.sub(r"[ \t]+", " ", " ".join(lines)).strip()
        if text:
            blocks.append({"text": text, "size": max(sizes), "bold": bold})
    return blocks


def _margin_signature(text):
    return re.sub(r"\d+", "#", text.lower()).strip()


def _is_noise(text):
    if PAGE_NUMBER.match(text):
        return True
    dots = text.count(".")
    return dots >= 8 and dots / max(1, len(text)) > 0.25  # table-of-contents dot leaders


def _is_heading(block, body_size):
    text = block["text"]
    count = words(text)
    if not 1 <= count <= 14 or not re.search(r"[A-Za-z]{3}", text) or text.endswith((",", ";")):
        return False
    if text.endswith(".") and count > 6:
        return False
    return block["size"] >= body_size * 1.15 or (block["bold"] and count <= 10)


def extract_pages(pdf, document_digest, ocr, settings):
    pages, ocr_pages, ocr_errors = [], [], []
    for page in pdf:
        blocks = _text_blocks(page)
        if sum(len(b["text"]) for b in blocks) < settings.ocr_min_chars and ocr is not None:
            if page.get_images() or not blocks:
                try:
                    recognized = ocr.page_blocks(document_digest, page)
                    if recognized:
                        blocks = recognized
                        ocr_pages.append(page.number + 1)
                except Exception as e:  # OCR failure must not discard the rest of the document
                    logger.exception("OCR failed on page %s", page.number + 1)
                    ocr_errors.append({"page": page.number + 1, "error": str(e)})
        pages.append({"number": page.number + 1, "label": page.get_label() or str(page.number + 1),
                      "blocks": blocks})
    return pages, ocr_pages, ocr_errors


def build_units(pages, toc, settings):
    """Merge layout blocks into paragraph-sized units without crossing page or section boundaries."""
    margins = Counter()
    for page in pages:
        edge = page["blocks"][:2] + page["blocks"][-2:]
        margins.update({_margin_signature(b["text"]) for b in edge if len(b["text"]) <= 120})
    repeated = {s for s, n in margins.items() if n >= max(3, 0.4 * len(pages))}
    sizes = [b["size"] for p in pages for b in p["blocks"] for _ in range(min(50, words(b["text"])))]
    body_size = statistics.median(sizes) if sizes else 10.0
    toc_titles = {}
    for _, title, number in toc:
        if not title.lower().endswith(".pdf"):
            toc_titles.setdefault(number, title.strip())
    units, chapter, heading, outline = [], "", "", []
    for page in pages:
        chapter = toc_titles.get(page["number"], chapter)
        buffer = []

        def flush():
            text = "\n".join(buffer).strip()
            buffer.clear()
            if words(text) < 4:
                return
            section = " > ".join(dict.fromkeys(filter(None, [chapter, heading]))) or "Document"
            previous = units[-1] if units else None
            if (
                previous
                and words(text) < settings.unit_min_words // 3
                and previous["page"] == page["number"]
                and previous["section"] == section
                and previous["words"] + words(text) <= settings.unit_max_words + 30
            ):
                previous["text"] += "\n" + text
                previous["words"] += words(text)
                return
            units.append({"text": text, "page": page["number"], "label": page["label"],
                          "section": section, "words": words(text)})

        for block in page["blocks"]:
            text = block["text"]
            if _margin_signature(text) in repeated or _is_noise(text):
                continue
            if _is_heading(block, body_size):
                flush()
                heading = text
                if len(outline) < 80 and text not in outline:
                    outline.append(text)
                continue
            count = words(text)
            if count > settings.unit_max_words:
                flush()
                for piece in split_long(text, settings.unit_max_words):
                    buffer.append(piece)
                    flush()
                continue
            if buffer and words("\n".join(buffer)) + count > settings.unit_max_words:
                flush()
            buffer.append(text)
            if words("\n".join(buffer)) >= settings.unit_min_words:
                flush()
        flush()
    return units, outline


def extract_document(filename, content, collection, settings, ocr=None):
    document_digest = hashlib.sha256(content).hexdigest()
    doc_id = digest(filename + "\0" + document_digest)[:24]
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("Encrypted PDF requires a password")
        pages, ocr_pages, ocr_errors = extract_pages(pdf, document_digest, ocr, settings)
        toc = pdf.get_toc()
    if not any(p["blocks"] for p in pages):
        raise ValueError("PDF has no extractable text and OCR is unavailable or found nothing")
    cover = "\n".join(b["text"] for p in pages[:3] for b in p["blocks"])
    metadata = {
        "document_id": doc_id,
        "document": filename,
        "collection_name": collection,
        **document_metadata(filename, cover),
    }
    units, outline = build_units(pages, toc, settings)
    chunks = [
        {
            "chunk_id": f"{doc_id}:{u['page']}:{i}",
            "document_id": doc_id,
            "document": filename,
            "collection_name": collection,
            "text": u["text"],
            "page_numbers": [u["page"]],
            "page_label": u["label"],
            "section": u["section"],
        }
        for i, u in enumerate(units)
    ]
    preview, total = [], 0
    for chunk in chunks:
        if total > 900:
            break
        preview.append(chunk["text"])
        total += words(chunk["text"])
    return {
        **metadata,
        "pages": len(pages),
        "units": len(chunks),
        "ocr_pages": ocr_pages,
        "ocr_errors": ocr_errors,
        "empty_pages": [p["number"] for p in pages if not p["blocks"]],
        "outline": outline,
        "preview": "\n".join(preview),
    }, chunks
