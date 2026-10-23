from dataclasses import asdict, dataclass, replace
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    mode: str = "improved"
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_variants: str = "both"
    answer_backend: str = "opencode"
    chunk_tokens: int = 400
    overlap_tokens: int = 60
    sketch_tokens: int = 120
    top_k: int = 6
    candidate_k: int = 40
    broad_k: int = 24
    context_tokens: int = 10000
    max_requirements: int = 12
    max_followups: int = 4
    max_output_tokens: int = 3000
    metadata_routing: bool = True
    compact_embeddings: bool = True
    hybrid_search: bool = True
    multi_section: bool = True
    coverage_selection: bool = True
    max_zip_bytes: int = 300_000_000
    max_unpacked_bytes: int = 1_000_000_000
    max_files: int = 200

    def __post_init__(self):
        if self.mode not in {"baseline", "improved"}:
            raise ValueError("mode must be baseline or improved")
        if self.embedding_backend not in {"hash", "sentence-transformers"}:
            raise ValueError("Unsupported embedding backend")
        if self.answer_backend not in {"extractive", "opencode"}:
            raise ValueError("Unsupported answer backend")
        if self.index_variants not in {"both", "full", "compact"}:
            raise ValueError("index_variants must be both, full or compact")
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("overlap_tokens must be less than chunk_tokens")
        for name in ("chunk_tokens", "sketch_tokens", "top_k", "candidate_k", "broad_k", "context_tokens",
                     "max_requirements", "max_output_tokens", "max_zip_bytes", "max_unpacked_bytes", "max_files"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_followups < 0:
            raise ValueError("max_followups cannot be negative")

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

    def effective(self):
        if self.mode == "baseline":
            return replace(self, metadata_routing=False, compact_embeddings=False, hybrid_search=False,
                           multi_section=False, coverage_selection=False)
        return self

    def json(self):
        return {**asdict(self), "data_dir": str(self.data_dir)}
