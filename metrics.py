"""Ranking metrics on binary relevance. Results are de-duplicated by source event
(several chunks of one event count once, at the rank of the first chunk)."""
from __future__ import annotations

import math
import statistics


def dedupe(ranked_sources: list[str]) -> list[str]:
    seen, out = set(), []
    for s in ranked_sources:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(dedupe(ranked)[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = dedupe(ranked)[:k]
    return (sum(1 for s in top if s in relevant) / len(top)) if top else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, s in enumerate(dedupe(ranked), 1):
        if s in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = dedupe(ranked)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(top) if s in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, int(round(p / 100 * (len(vs) - 1)))))
    return vs[idx]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0
