"""Bounded map-reduce for long context (Phase A of the RLM plan).

This is the "long-context" path the router sees. A true recursive REPL-style RLM is
a gated research branch (Phase B) and would sit behind this same function signature.

Bounds: depth, number of model calls, wall-clock time. If a bound is hit, the result
is marked `truncated` instead of silently dropping text.

Protected values (file paths, hashes, ids, permission scopes) are carried through
as typed data next to the summary. They are never trusted to survive inside a
natural-language summary; `missing_protected` reports any that did not.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

# summarize(texts, instruction) -> one summary string. In production this wraps the
# local LLM; in tests it is a deterministic function.
Summarizer = Callable[[list[str], str], str]


@dataclass
class MapReduceConfig:
    fan_in: int = 4
    max_depth: int = 2
    max_calls: int = 24
    time_budget_s: float = 60.0


@dataclass
class MapReduceResult:
    text: str
    protected: list[str]
    calls: int
    depth_used: int
    truncated: bool
    elapsed_s: float
    missing_protected: list[str] = field(default_factory=list)


def bounded_map_reduce(chunks: list[str], summarize: Summarizer, instruction: str,
                       protected: list[str] | None = None,
                       cfg: MapReduceConfig | None = None) -> MapReduceResult:
    cfg = cfg or MapReduceConfig()
    protected = list(protected or [])
    start = time.monotonic()
    calls, passes, truncated = 0, 0, False
    level = [c for c in chunks if c.strip()]

    def out_of_budget() -> bool:
        return calls >= cfg.max_calls or (time.monotonic() - start) > cfg.time_budget_s

    # Map/reduce passes: shrink the list by a factor of fan_in each pass, at most max_depth times.
    while len(level) > cfg.fan_in and passes < cfg.max_depth:
        nxt = []
        for i in range(0, len(level), cfg.fan_in):
            group = level[i : i + cfg.fan_in]
            if out_of_budget():
                truncated = True
                nxt.append("\n".join(group)[:2000])   # budget hit: keep text but cut hard
            else:
                nxt.append(summarize(group, instruction))
                calls += 1
        level = nxt
        passes += 1

    # Final synthesis over whatever is left (one call), unless a single summary already remains.
    if not level:
        final = ""
    elif len(level) == 1 and passes > 0:
        final = level[0]
    elif out_of_budget():
        truncated = True
        final = "\n".join(level)[:4000]
    else:
        final = summarize(level, instruction)
        calls += 1

    missing = [p for p in protected if p not in final]
    return MapReduceResult(final, protected, calls, passes, truncated,
                           time.monotonic() - start, missing)
