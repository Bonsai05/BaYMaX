"""The single entry point Member 1's Core calls.

    ai = AIModule(db_path="vault/memory.db")
    ai.admit(event)                 # after the Core commits the event
    ai.index_pending()              # background job; keeps admission off the request path
    ai.route(RouteRequest(...))     # fast router
    ai.retrieve(RetrievalRequest(...))
    ai.long_context(chunks, summarize, instruction)

Nothing in ai/ opens a socket. The Core owns networking, auth and the event log.
"""
from __future__ import annotations

from typing import Optional

from .contracts import (
    AdmissionResult, Event, RetrievalRequest, RetrievalResult, RouteDecision, RouteRequest,
)
from .memory.admission import AdmissionGate
from .memory.store import MemoryStore
from .models.embedder import Embedder, HashingEmbedder
from .rag.retriever import HybridRetriever, Reranker
from .rlm.mapreduce import MapReduceConfig, MapReduceResult, Summarizer, bounded_map_reduce
from .router.router import Router


class AIModule:
    def __init__(self, db_path: str = ":memory:", embedder: Optional[Embedder] = None,
                 reranker: Optional[Reranker] = None, router_log_path: Optional[str] = None):
        self.store = MemoryStore(db_path)
        self.embedder = embedder or HashingEmbedder()
        self.gate = AdmissionGate(self.store)
        self.retriever = HybridRetriever(self.store, self.embedder, reranker)
        self.router = Router(self.store, log_path=router_log_path)

    def close(self) -> None:
        """Release the database. Call on Core shutdown."""
        self.store.close()

    def __enter__(self) -> "AIModule":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- memory ----------------------------------------------------------
    def admit(self, event: Event, index_now: bool = True) -> AdmissionResult:
        result = self.gate.admit(event)
        if index_now and result.memory_ids:
            self.index_pending()
        return result

    def index_pending(self) -> int:
        """Embed records that have no vector for the current model. Returns how many."""
        pending = self.store.ids_without_vector(self.embedder.model_id)
        if not pending:
            return 0
        ids, texts = zip(*pending)
        self.store.put_vectors(list(ids), self.embedder.model_id, self.embedder.embed(list(texts)))
        return len(ids)

    def rebuild_index(self) -> int:
        """Drop every vector and re-embed from canonical records (model change or corruption)."""
        self.store.clear_vectors()
        return self.index_pending()

    def delete_memory(self, memory_id: str) -> None:
        self.store.delete(memory_id)

    # ---- lifecycle (deterministic; the LLM is never involved) -------------
    # The Core must pass confirmed=True only from a real user action in the UI. This module
    # cannot verify that, so it is the Core's responsibility (see docs/ai_module.md).
    def promote_memory(self, memory_id: str, confirmed: bool = False, actor: str = "user",
                       acknowledge_injection_flag: bool = False) -> None:
        """Promote quarantined evidence to trusted after explicit user confirmation."""
        self.store.promote(memory_id, confirmed=confirmed, actor=actor,
                           acknowledge_injection_flag=acknowledge_injection_flag)

    def resolve_contradiction(self, new_memory_id: str, old_memory_id: str,
                              confirmed: bool = False, actor: str = "user") -> None:
        """Make a confirmed new fact supersede an older conflicting one (old one is kept)."""
        self.store.resolve_contradiction(new_memory_id, old_memory_id, confirmed=confirmed, actor=actor)

    def revoke_source(self, canonical_source: str, actor: str = "user") -> int:
        """Hide every memory made from one event. Not erasure: use delete_memory for that."""
        return self.store.revoke_source(canonical_source, actor=actor)

    def revoke_artifact(self, source_artifact: str, actor: str = "user") -> int:
        """Hide every memory made from one file or URL, across all events that imported it."""
        return self.store.revoke_artifact(source_artifact, actor=actor)

    def restore_memory(self, memory_id: str, confirmed: bool = False, actor: str = "user") -> None:
        """Undo a supersede or revoke on one memory, after explicit confirmation."""
        self.store.restore(memory_id, confirmed=confirmed, actor=actor)

    def audit_trail(self, memory_id: str) -> list[dict]:
        return self.store.audit_trail(memory_id)

    # ---- retrieval and routing -------------------------------------------
    def retrieve(self, req: RetrievalRequest) -> RetrievalResult:
        return self.retriever.retrieve(req)

    def route(self, req: RouteRequest) -> RouteDecision:
        return self.router.route(req)

    def escalate(self, decision: RouteDecision, retrieval: RetrievalResult,
                 estimated_tokens: int, token_budget: int) -> RouteDecision:
        return self.router.escalate(decision, retrieval, estimated_tokens, token_budget)

    def long_context(self, chunks: list[str], summarize: Summarizer, instruction: str,
                     protected: Optional[list[str]] = None,
                     cfg: Optional[MapReduceConfig] = None) -> MapReduceResult:
        return bounded_map_reduce(chunks, summarize, instruction, protected, cfg)
