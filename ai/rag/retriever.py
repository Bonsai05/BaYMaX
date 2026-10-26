"""Hybrid retrieval: FTS5 (BM25) + vector similarity, fused with reciprocal rank
fusion, optional recency boost, optional rerank.

Modes (these are the rows of the ablation table in the benchmark):
    lexical        BM25 only
    vector         embedding similarity only
    hybrid         BM25 + vector, reciprocal rank fusion
    hybrid_rerank  hybrid, then rerank the top candidates

Timings are reported per stage so the Core can compute T_retrieve honestly.
"""
from __future__ import annotations

import math
import time
from typing import Protocol

from ..contracts import RetrievalRequest, RetrievalResult, RetrievedItem
from ..memory.store import MemoryStore
from ..models.embedder import Embedder
from ..text import TOKEN, content_terms, stem

RRF_K = 60


class IndexStaleError(RuntimeError):
    """Vectors for the current embedding model are missing. Run rebuild_index()."""


def query_terms(text: str) -> list[str]:
    return [stem(t) for t in content_terms(text)]


def coverage(query: str, content: str) -> float:
    """Share of query terms that appear in the content (0..1)."""
    q = set(query_terms(query))
    if not q:
        return 0.0
    c = {stem(t) for t in TOKEN.findall(content.lower())}
    return len(q & c) / len(q)


class Reranker(Protocol):
    def rerank(self, query: str, items: list[RetrievedItem]) -> list[RetrievedItem]:
        """Return items re-ordered best first, with scores updated."""
        ...


class CoverageReranker:
    """Placeholder reranker: query-term coverage blended with the fused score.

    Swap for a cross-encoder (run on the Windows Core, it is too heavy for the phone)
    behind the same Reranker protocol and compare in the benchmark.
    """

    def __init__(self, weight: float = 0.7):
        self.weight = weight

    def rerank(self, query: str, items: list[RetrievedItem]) -> list[RetrievedItem]:
        if not items:
            return items
        top = max(i.score for i in items) or 1.0
        for it in items:
            cov = coverage(query, it.content)
            it.stage_scores["rerank_coverage"] = cov
            it.score = self.weight * cov + (1 - self.weight) * (it.score / top)
        return sorted(items, key=lambda i: -i.score)


class HybridRetriever:
    def __init__(self, store: MemoryStore, embedder: Embedder, reranker: Reranker | None = None,
                 recency_half_life_days: float | None = 30.0, recency_weight: float = 0.1,
                 rerank_candidates: int = 20):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or CoverageReranker()
        self.half_life = recency_half_life_days
        self.recency_weight = recency_weight
        self.rerank_candidates = rerank_candidates

    def _filters(self, req: RetrievalRequest) -> dict:
        return {
            "project": req.project, "source_device": req.source_device, "kinds": req.kinds,
            "include_quarantined": req.include_quarantined, "for_cloud": req.for_cloud,
            "since": req.since,
        }

    def retrieve(self, req: RetrievalRequest, now: float | None = None) -> RetrievalResult:
        t_all = time.perf_counter()
        now = now if now is not None else time.time()
        timings = {"lexical": 0.0, "vector": 0.0, "fuse": 0.0, "rerank": 0.0}
        pool = max(self.rerank_candidates, req.top_k * 4)
        filters = self._filters(req)
        use_lex = req.mode in ("lexical", "hybrid", "hybrid_rerank")
        use_vec = req.mode in ("vector", "hybrid", "hybrid_rerank")
        if req.mode not in ("lexical", "vector", "hybrid", "hybrid_rerank"):
            raise ValueError(f"unknown retrieval mode: {req.mode}")

        lex: list[tuple[str, float]] = []
        vec: list[tuple[str, float]] = []
        if use_lex:
            t = time.perf_counter()
            lex = self.store.lexical_search(req.query, pool, filters)
            timings["lexical"] = (time.perf_counter() - t) * 1000
        if use_vec:
            t = time.perf_counter()
            if self.embedder.model_id not in self.store.vector_models() and self.store.count() > 0:
                raise IndexStaleError(
                    f"no vectors for model {self.embedder.model_id}; call rebuild_index()")
            embed_query = getattr(self.embedder, "embed_query", None)   # optional: query instruction
            qv = (embed_query(req.query) if embed_query else self.embedder.embed([req.query]))[0]
            vec = self.store.vector_search(qv, self.embedder.model_id, pool, filters)
            timings["vector"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        fused: dict[str, float] = {}
        stage: dict[str, dict[str, float]] = {}
        for name, ranked in (("lexical", lex), ("vector", vec)):
            for rank, (mid, raw) in enumerate(ranked):
                fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)
                stage.setdefault(mid, {})[name] = raw
        rows = self.store.rows_by_ids(fused)
        if self.half_life:
            for mid in fused:
                age_days = max(0.0, (now - rows[mid]["created_at"]) / 86400)
                fused[mid] *= 1 + self.recency_weight * math.exp(-age_days * math.log(2) / self.half_life)
        ordered = sorted(fused, key=lambda m: -fused[m])
        timings["fuse"] = (time.perf_counter() - t) * 1000

        items = []
        for mid in ordered[: (self.rerank_candidates if req.mode == "hybrid_rerank" else req.top_k)]:
            r = rows[mid]
            ss = dict(stage.get(mid, {}))
            ss["coverage"] = coverage(req.query, r["content"])
            items.append(RetrievedItem(
                memory_id=mid, canonical_source=r["canonical_source"], content=r["content"],
                score=fused[mid], trust_tier=r["trust_tier"], quarantined=bool(r["quarantined"]),
                kind=r["kind"], project=r["project"], source_device=r["source_device"],
                source_artifact=r["source_artifact"], stage_scores=ss,
            ))

        if req.mode == "hybrid_rerank":
            t = time.perf_counter()
            items = self.reranker.rerank(req.query, items)
            timings["rerank"] = (time.perf_counter() - t) * 1000
        items = items[: req.top_k]
        timings["total"] = (time.perf_counter() - t_all) * 1000
        return RetrievalResult(items=items, mode=req.mode, timings_ms=timings)
