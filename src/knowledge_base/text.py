import hashlib
import re

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
STOP = set(
    "the a an of and or to in for on with by from is are was were as at this that it its be has have how what which all every please report annual company limited year".split()
)


def tokens(text):
    """Deterministic lexical tokens, NOT a provider's billable tokenizer."""
    return TOKEN_RE.findall(text)


def terms(text):
    return [t.lower() for t in tokens(text) if len(t) > 1 and t.isalnum() and t.lower() not in STOP]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[\"'(\[A-Z0-9•-])")


def words(text):
    return len(text.split())


def split_long(text, limit):
    """Split an over-long paragraph on sentence boundaries into pieces of at most ~limit words."""
    pieces, current = [], []
    for sentence in SENTENCE_RE.split(text):
        sentence_words = sentence.split()
        while len(sentence_words) > limit:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(sentence_words[:limit]))
            sentence_words = sentence_words[limit:]
        if current and len(current) + len(sentence_words) > limit:
            pieces.append(" ".join(current))
            current = []
        current.extend(sentence_words)
    if current:
        pieces.append(" ".join(current))
    return pieces
