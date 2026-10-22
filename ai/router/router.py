"""Three-stage router from the v3/v5 architecture.

Stage 1  rules, zero LLM: slash commands and explicit file lookups go to a Direct tool.
Stage 2  light classifier: cheap features -> RAG or long-context. Default on
         uncertainty is RAG, never Direct.
Stage 3  escalation after retrieval: if the context will not fit the token budget,
         switch to long-context (map-reduce); if retrieval looks weak, flag low
         confidence. Escalation happens at most once.

Every decision is logged so routing accuracy can be scored against the golden set.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Optional, Protocol

from ..contracts import RetrievalResult, Route, RouteDecision, RouteRequest
from ..memory.store import MemoryStore

SLASH_TOOLS = {"/find": "SEARCH_FILES", "/read": "READ_FILE"}
NL_FIND = re.compile(r"^\s*(?:find|search for|look for)\s+(?:the\s+)?files?\s+(?:named\s+|called\s+)?(?P<q>.+?)\s*$", re.I)
CONTINUE_CUES = re.compile(r"\b(continue|resume|pick up|where (?:did )?i (?:leave|stop)|what was i (?:doing|working))\b", re.I)
BROAD_CUES = re.compile(r"\b(summari[sz]e (?:all|everything|every)|across all|everything (?:about|on)|full history|all my (?:notes|files|messages))\b", re.I)
TEMPORAL_CUES = re.compile(r"\b(yesterday|today|last (?:week|month|night)|this (?:week|morning)|recent(?:ly)?)\b", re.I)
_ENTITY = re.compile(r'"[^"]+"|\b[A-Z][A-Za-z0-9]{2,}\b')


class Classifier(Protocol):
    def predict(self, features: dict[str, Any]) -> tuple[Route, float]: ...


class RuleClassifier:
    """Hand-set thresholds. Replace with a small model fitted on the golden task set
    (features are already logged with every decision, so training data accumulates)."""

    def predict(self, f: dict[str, Any]) -> tuple[Route, float]:
        if f["broad_cue"] or f["n_words"] > 80:
            return Route.LONG_CONTEXT, 0.7
        return Route.RAG, 0.9 if (f["continue_cue"] or f["fts_hits"] > 0) else 0.6


def features_for(query: str, store: Optional[MemoryStore]) -> dict[str, Any]:
    return {
        "n_words": len(query.split()),
        "n_entities": len(_ENTITY.findall(query)),
        "continue_cue": bool(CONTINUE_CUES.search(query)),
        "broad_cue": bool(BROAD_CUES.search(query)),
        "temporal_cue": bool(TEMPORAL_CUES.search(query)),
        "fts_hits": store.lexical_hit_count(query) if store else 0,
    }


class Router:
    def __init__(self, store: Optional[MemoryStore] = None, classifier: Classifier | None = None,
                 log_path: Optional[str] = None, min_coverage: float = 0.34):
        self.store = store
        self.classifier = classifier or RuleClassifier()
        self.log_path = log_path
        self.min_coverage = min_coverage
        self.log: list[dict[str, Any]] = []

    # ---- stage 1 + 2 -----------------------------------------------------
    def route(self, req: RouteRequest) -> RouteDecision:
        q = req.query.strip()
        d = self._stage1(q, req)
        if d is None:
            feats = features_for(q, self.store)
            route, conf = self.classifier.predict(feats)
            if conf < 0.5 or route == Route.DIRECT:
                route, conf = Route.RAG, max(conf, 0.5)   # default on uncertainty is RAG
            d = RouteDecision(route, 2, f"classifier:{route.value}", conf, feats)
        self._log("route", q, d)
        return d

    def _stage1(self, q: str, req: RouteRequest) -> Optional[RouteDecision]:
        tools = set(req.available_tools)
        if q.startswith("/"):
            cmd, _, arg = q.partition(" ")
            tool = SLASH_TOOLS.get(cmd.lower())
            if tool and arg.strip() and (not tools or tool in tools):
                key = "query" if tool == "SEARCH_FILES" else "path"
                return RouteDecision(Route.DIRECT, 1, f"slash_command:{cmd}", 1.0,
                                     tool=tool, tool_args={key: arg.strip()})
        m = NL_FIND.match(q)
        if m and (not tools or "SEARCH_FILES" in tools):
            return RouteDecision(Route.DIRECT, 1, "explicit_file_lookup", 0.95,
                                 tool="SEARCH_FILES", tool_args={"query": m.group("q")})
        return None

    # ---- stage 3 ---------------------------------------------------------
    def escalate(self, decision: RouteDecision, retrieval: RetrievalResult,
                 estimated_tokens: int, token_budget: int) -> RouteDecision:
        """Call after retrieval. At most one escalation per request."""
        if decision.stage == 3:
            return decision
        new = replace(decision, stage=3, features=dict(decision.features))
        new.features.update({"estimated_tokens": estimated_tokens, "token_budget": token_budget})
        if estimated_tokens > token_budget and decision.route != Route.LONG_CONTEXT:
            new.route, new.reason = Route.LONG_CONTEXT, "context_exceeds_budget"
        top_cov = retrieval.items[0].stage_scores.get("coverage", 0.0) if retrieval.items else 0.0
        new.features["top_coverage"] = top_cov
        if not retrieval.items or top_cov < self.min_coverage:
            new.low_confidence = True
            new.reason = (new.reason if new.reason != decision.reason else "weak_retrieval") + "+low_confidence"
        if new.route == decision.route and not new.low_confidence:
            return decision      # nothing to escalate
        self._log("escalate", "", new)
        return new

    def _log(self, kind: str, query: str, d: RouteDecision) -> None:
        entry = {"kind": kind, "query": query, **d.to_dict()}
        self.log.append(entry)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
