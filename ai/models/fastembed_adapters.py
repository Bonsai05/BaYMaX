"""Real embedding and reranking models through the `fastembed` package (ONNX Runtime).

    pip install fastembed

The first use downloads the model weights (bge-small is about 130 MB), so it needs
internet access once; after that it runs offline from the cache. Both classes import
fastembed lazily, so the rest of the module works without it installed.

Wiring is covered by unit tests with a stub package. Retrieval quality of the real
models is measured by the benchmark, not asserted here:

    python -m ai.eval.run --embedder fastembed --reranker fastembed
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..contracts import RetrievedItem

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def _l2(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(n == 0, 1.0, n)


class FastEmbedEmbedder:
    """Embedder protocol implementation. Queries use the model's query instruction
    (bge models are trained to expect it); documents are embedded as plain text."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, cache_dir: Optional[str] = None):
        from fastembed import TextEmbedding   # lazy: optional dependency

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        info = {m["model"]: m for m in TextEmbedding.list_supported_models()}
        self.dim = int(info[model_name]["dim"])
        self.model_id = f"fastembed:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return _l2(np.asarray(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, query: str) -> np.ndarray:
        return _l2(np.asarray(list(self._model.query_embed(query)), dtype=np.float32))


class FastEmbedReranker:
    """Cross-encoder reranker. Scores are raw model logits (higher is better) and replace
    the fused retrieval score. Run it on the Windows Core, not on the phone."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, cache_dir: Optional[str] = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder   # lazy

        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
        self.model_id = f"fastembed:{model_name}"

    def rerank(self, query: str, items: list[RetrievedItem]) -> list[RetrievedItem]:
        if not items:
            return items
        scores = list(self._model.rerank(query, [i.content for i in items]))
        for it, s in zip(items, scores):
            it.stage_scores["cross_encoder"] = float(s)
            it.score = float(s)
        return sorted(items, key=lambda i: -i.score)
