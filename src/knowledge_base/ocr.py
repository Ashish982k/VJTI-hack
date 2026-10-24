"""OCR fallback for scanned pages: PaddleOCR lines regrouped into paragraph blocks, cached on disk."""

from contextlib import closing
import json
import logging
import os
import sqlite3
import statistics
from threading import Lock

import numpy as np

from .text import digest

logger = logging.getLogger(__name__)


def group_lines(lines):
    """Turn OCR lines [(text, (x1, y1, x2, y2))] into paragraph blocks using vertical gaps."""
    lines = [(t.strip(), box) for t, box in lines if t and t.strip()]
    if not lines:
        return []
    lines.sort(key=lambda item: (round(item[1][1] / 8), item[1][0]))
    heights = [max(1.0, box[3] - box[1]) for _, box in lines]
    typical = statistics.median(heights)
    blocks, current, previous = [], [], None
    for (text, box), height in zip(lines, heights):
        if previous is not None and box[1] - previous[3] > 0.8 * typical:
            blocks.append(current)
            current = []
        current.append((text, height))
        previous = box
    blocks.append(current)
    result = []
    for block in blocks:
        text = " ".join(t for t, _ in block)
        size = max(h for _, h in block)
        result.append({"text": text, "size": size, "bold": False})
    return result


class PaddleOCREngine:
    """Lazy PaddleOCR (PP-OCRv5 mobile) wrapper. Results are cached by PDF digest, page and DPI."""

    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.lock = Lock()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = settings.data_dir / "ocr-cache.sqlite3"
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS pages (key TEXT PRIMARY KEY, blocks TEXT)")

    @property
    def identity(self):
        return f"paddle:PP-OCRv5_mobile:{self.settings.ocr_dpi}"

    def _load(self):
        if self.model is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR

            self.model = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            )

    def _recognize(self, page):
        pixmap = page.get_pixmap(dpi=self.settings.ocr_dpi)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)
        image = np.ascontiguousarray(image[:, :, :3])
        lines = []
        for result in self.model.predict(image):
            for text, score, box in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"]):
                if score >= 0.5:
                    lines.append((text, tuple(float(v) for v in box)))
        scale = 72 / self.settings.ocr_dpi
        return [
            {**block, "size": block["size"] * scale} for block in group_lines(lines)
        ]

    def page_blocks(self, document_digest, page):
        key = digest(f"{self.identity}\0{document_digest}\0{page.number}")
        with closing(sqlite3.connect(self.db, timeout=60)) as db:
            row = db.execute("SELECT blocks FROM pages WHERE key=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        with self.lock:
            self._load()
            blocks = self._recognize(page)
        with closing(sqlite3.connect(self.db, timeout=60)) as db, db:
            db.execute("INSERT OR REPLACE INTO pages VALUES (?,?)", (key, json.dumps(blocks)))
        return blocks


def create_ocr(settings):
    if settings.ocr_backend == "none":
        return None
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        logger.warning("PaddleOCR is not installed; scanned pages will be reported as unreadable")
        return None
    return PaddleOCREngine(settings)
