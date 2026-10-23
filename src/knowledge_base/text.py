import hashlib
import re
from functools import lru_cache

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
STOP = set(
    "the a an of and or to in for on with by from is are was were as at this that it its be has have how what which all every please report annual company limited year".split()
)


def tokens(text):
    """Deterministic lexical tokens, NOT a provider's billable tokenizer."""
    return TOKEN_RE.findall(text)


def terms(text):
    return [t.lower() for t in tokens(text) if len(t) > 1 and t.isalnum() and t.lower() not in STOP]


@lru_cache(maxsize=32768)
def term_set(text):
    return frozenset(terms(text))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text, size, overlap):
    spans = list(TOKEN_RE.finditer(text))
    for start in range(0, len(spans), size - overlap):
        end = min(start + size, len(spans))
        yield text[spans[start].start() : spans[end - 1].end()]
        if end == len(spans):
            break


def truncate(text, limit):
    spans = list(TOKEN_RE.finditer(text))
    return text if len(spans) <= limit else text[: spans[limit - 1].end()]
