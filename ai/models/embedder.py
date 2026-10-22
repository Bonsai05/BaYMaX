"""Embedding interface plus a dependency-free placeholder.

The placeholder (HashingEmbedder) only captures word and character overlap, so it
cannot match paraphrases. It exists so the whole pipeline and the benchmark run
without downloading a model. Replace it with a real model (for example bge-small
or MiniLM through ONNX Runtime) behind the same Embedder protocol, then re-run the
benchmark: the gap between the two is a result worth reporting.

Every vector is stored with embedder.model_id. If the model changes, the store
refuses mixed vectors and a full re-embed is required (see AIModule.rebuild_index).
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    model_id: str   # include the version, for example "bge-small-en-v1.5@onnx-1"
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), rows L2-normalised."""
        ...

    # Optional: embedders whose model uses a query instruction (for example bge) may also
    # define `embed_query(query) -> array of shape (1, dim)`. The retriever uses it if present.


class HashingEmbedder:
    """Signed feature hashing of words and character trigrams. Deterministic."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_id = f"hashing-v1-{dim}"

    def _features(self, text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        feats = list(words)
        for w in words:
            padded = f"^{w}$"
            feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self.dim] += sign
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out
