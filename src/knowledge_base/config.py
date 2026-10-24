from dataclasses import asdict, dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    # sentence-transformers is semantic; hash is an offline smoke-test backend only.
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    # opencode runs profiling, query analysis, evidence judging and answers; offline uses rules.
    llm_backend: str = "llm"
    ocr_backend: str = "paddle"
    ocr_dpi: int = 150
    ocr_min_chars: int = 40
    # Evidence units mirror paragraph-sized benchmark chunks rather than fixed token windows.
    unit_min_words: int = 60
    unit_max_words: int = 220
    max_documents: int = 8
    # Planner catalog size; lower it for providers with small per-request limits (e.g. Groq free tier).
    catalog_chars: int = 20000
    main_query_k: int = 40
    aspect_query_k: int = 20
    per_document_k: int = 8
    neighbor_window: int = 1
    # Structure-first retrieval: whole sections per sub-question for broad questions.
    section_retrieval: bool = True
    sections_per_query: int = 3
    section_max_units: int = 20
    section_mode: str = "merged"
    section_min_units: int = 8
    section_cap_units: int = 40
    section_pages: int = 3
    max_candidates: int = 240
    judge_batch: int = 15
    recovery_k: int = 15
    max_evidence: int = 160
    offline_evidence: int = 12
    context_tokens: int = 60000
    max_output_tokens: int = 8000
    llm_concurrency: int = 12
    llm_timeout: int = 180
    # Extra output budget for hidden reasoning tokens (reasoning models such as deepseek-v4.1-flash).
    reasoning_tokens: int = 6000
    max_zip_bytes: int = 300_000_000
    max_unpacked_bytes: int = 1_000_000_000
    max_files: int = 200

    def __post_init__(self):
        if self.embedding_backend not in {"hash", "sentence-transformers"}:
            raise ValueError("Unsupported embedding backend")
        if self.llm_backend not in {"offline", "llm", "opencode"}:
            raise ValueError("llm_backend must be llm (alias opencode) or offline")
        if self.section_mode not in {"merged", "pages"}:
            raise ValueError("section_mode must be merged or pages")
        if self.ocr_backend not in {"none", "paddle"}:
            raise ValueError("ocr_backend must be none or paddle")
        if not 0 < self.unit_min_words < self.unit_max_words:
            raise ValueError("unit_min_words must be positive and below unit_max_words")
        for name in (
            "ocr_dpi",
            "max_documents",
            "catalog_chars",
            "main_query_k",
            "aspect_query_k",
            "per_document_k",
            "max_candidates",
            "sections_per_query",
            "section_max_units",
            "section_min_units",
            "section_cap_units",
            "section_pages",
            "judge_batch",
            "max_evidence",
            "offline_evidence",
            "context_tokens",
            "max_output_tokens",
            "llm_concurrency",
            "llm_timeout",
            "max_zip_bytes",
            "max_unpacked_bytes",
            "max_files",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("ocr_min_chars", "neighbor_window", "recovery_k", "reasoning_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    @classmethod
    def from_env(cls):
        load_dotenv()
        values = {}
        defaults = cls()
        for name, default in asdict(defaults).items():
            value = os.getenv("KB_" + name.upper())
            if value is None:
                continue
            if isinstance(default, bool):
                if value.lower() not in {"true", "false", "1", "0"}:
                    raise ValueError(f"Invalid boolean KB_{name.upper()}")
                value = value.lower() in {"true", "1"}
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, Path):
                value = Path(value)
            values[name] = value
        return cls(**values)

    def json(self):
        return {**asdict(self), "data_dir": str(self.data_dir)}
