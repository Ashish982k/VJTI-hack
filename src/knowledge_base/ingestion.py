"""Page-preserving ingestion. Only PDFs enter the index, never benchmark files."""
from pathlib import PurePosixPath
import zipfile

import pymupdf

from .text import chunk_text, digest


def read_zip(source, settings):
    documents = []
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
            if not info.is_dir() and path.suffix.lower() == ".pdf":
                documents.append((str(path), archive.read(info)))
    if not documents:
        raise ValueError("ZIP contains no PDFs")
    return documents


def extract_document(filename, content, collection, settings):
    doc_id = digest(filename + "\0" + digest(content.hex()))[:24]
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("Encrypted PDF requires a password")
        pages = [{"number": i + 1, "label": page.get_label() or str(i + 1),
                  "text": page.get_text("text", sort=True)} for i, page in enumerate(pdf)]
        toc = pdf.get_toc()
    if not any(p["text"].strip() for p in pages):
        raise ValueError("PDF has no extractable text; OCR is required")
    metadata = {"document_id": doc_id, "document": filename, "company": "", "periods": [],
                "document_type": "document", "collection_name": collection}
    chunks = []
    heading = "Document"
    for page in pages:
        for _, title, number in toc:
            if number == page["number"]:
                heading = title
        for index, text in enumerate(chunk_text(page["text"], settings.chunk_tokens, settings.overlap_tokens)):
            chunks.append({**metadata, "chunk_id": f"{doc_id}:{page['number']}:{index}", "text": text,
                           "page_numbers": [page["number"]], "page_label": page["label"],
                           "section_heading": heading})
    return {**metadata, "pages": len(pages), "empty_pages": [p["number"] for p in pages if not p["text"].strip()]}, chunks
