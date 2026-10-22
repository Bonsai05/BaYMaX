"""Small shared text helpers (tokens, stopwords, a light stemmer)."""
from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "of", "to", "my", "me", "and", "or", "in", "on", "for", "is", "it", "i",
    "what", "was", "do", "did", "with", "about", "that", "this", "from", "at", "be", "how",
    "can", "each", "all", "are", "has", "have", "so", "if", "when", "where", "which", "why",
    "does", "by", "as", "its", "their", "them", "than", "then", "there", "using", "use",
}


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_terms(text: str) -> list[str]:
    """Lower-cased tokens without stopwords or one-letter tokens (not stemmed)."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]
