"""One call to assemble the whole AI side, for Member 1's Core.

    from ai.assistant import build_assistant, AssistantConfig

    assistant = build_assistant(AssistantConfig(db_path="vault/memory.db"))
    assistant.warm_up()                                   # optional: load the model before the first request
    turn = assistant.ask("When is the capstone report due?", session_id="phone-1")
    assistant.admit(event)                                # after the Core commits an event
    assistant.close()                                     # on Core shutdown

What it wires together: AIModule (memory, retrieval, router), the real embedder and reranker
(fastembed) or the offline hashing embedder, the Ollama model with the measured best settings, and
the AgentPipeline. It also keeps a short conversation history per session id, in memory only.

Requests are processed one at a time (a lock): a CPU-only local model answers one question at a
time anyway, and it keeps the SQLite store simple. The pipeline never executes a tool; a proposal
comes back as data for the policy engine (Member 4).
"""
from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Optional

from .api import AIModule
from .contracts import AdmissionResult, Event
from .gateway.provider import GenerationRequest, ModelProvider
from .models.embedder import HashingEmbedder
from .models.ollama_provider import OllamaProvider
from .pipeline.agent import AgentPipeline, AgentTurn


@dataclass
class AssistantConfig:
    db_path: str = ":memory:"                  # the Core passes the vault path, for example "vault/memory.db"
    model: str = "qwen2.5:3b"
    ollama_host: str = "http://localhost:11434"
    embedder: str = "fastembed"                # "fastembed" (real, needs the package) or "hashing" (offline stand-in)
    use_reranker: bool = True                  # only used with the fastembed embedder
    embed_cache_dir: Optional[str] = None
    request_timeout_s: float = 120.0
    top_k: int = 5
    max_sessions: int = 64                     # conversation histories kept in memory
    history_turns: int = 6                     # pairs kept per session


class Assistant:
    def __init__(self, ai: AIModule, provider: ModelProvider, pipeline: AgentPipeline, config: AssistantConfig):
        self.ai, self.provider, self.pipeline, self.config = ai, provider, pipeline, config
        self.lock = threading.RLock()
        self._sessions: "OrderedDict[str, deque]" = OrderedDict()

    # ---- conversation ----------------------------------------------------------------
    def ask(self, text: str, project: Optional[str] = None, session_id: str = "default",
            on_token: Optional[Callable[[str], None]] = None) -> AgentTurn:
        with self.lock:
            history = list(self._sessions.get(session_id, []))
            turn = self.pipeline.handle(text, project=project, history=history, on_token=on_token)
            if turn.kind == "answer":
                self._remember_turn(session_id, text, turn.text)
            return turn

    def _remember_turn(self, session_id: str, user: str, assistant: str) -> None:
        q = self._sessions.setdefault(session_id, deque(maxlen=self.config.history_turns))
        q.append((user, assistant))
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.config.max_sessions:
            self._sessions.popitem(last=False)

    def clear_session(self, session_id: str) -> None:
        with self.lock:
            self._sessions.pop(session_id, None)

    # ---- memory (thin, locked wrappers so the Core has one object to hold) -------------
    def admit(self, event: Event) -> AdmissionResult:
        with self.lock:
            return self.ai.admit(event)

    # ---- operations ------------------------------------------------------------------
    def health(self) -> dict:
        h = self.provider.health()
        with self.lock:
            counts = self.ai.store.db.execute(
                "SELECT COUNT(*), COALESCE(SUM(quarantined), 0) FROM memories").fetchone()
            pending = len(self.ai.store.ids_without_vector(self.ai.embedder.model_id))
        return {"ok": bool(h.ok), "model": {"ok": bool(h.ok), "detail": h.detail, "name": self.config.model},
                "embedder": self.ai.embedder.model_id, "memories": int(counts[0]),
                "quarantined": int(counts[1]), "unindexed": pending,
                "sessions": len(self._sessions)}

    def warm_up(self) -> float:
        """Load the model into memory with a tiny request. Returns seconds taken (0 on failure)."""
        res = self.provider.generate(GenerationRequest(
            [{"role": "user", "content": "Reply with the word ok."}], max_tokens=4, timeout_s=self.config.request_timeout_s))
        return res.timings.total_s if res.ok else 0.0

    def close(self) -> None:
        with self.lock:
            self.ai.close()


def build_assistant(config: Optional[AssistantConfig] = None, provider: Optional[ModelProvider] = None) -> Assistant:
    """Assemble everything. Pass `provider` to use a different model (tests use a scripted one)."""
    cfg = config or AssistantConfig()
    embedder, reranker, mode = HashingEmbedder(), None, "hybrid"
    if cfg.embedder == "fastembed":
        from .models.fastembed_adapters import FastEmbedEmbedder, FastEmbedReranker
        embedder = FastEmbedEmbedder(cache_dir=cfg.embed_cache_dir)
        if cfg.use_reranker:
            reranker, mode = FastEmbedReranker(cache_dir=cfg.embed_cache_dir), "hybrid_rerank"
        else:
            mode = "hybrid"
    elif cfg.embedder != "hashing":
        raise ValueError(f"unknown embedder {cfg.embedder!r}; use 'fastembed' or 'hashing'")
    ai = AIModule(db_path=cfg.db_path, embedder=embedder, reranker=reranker)
    provider = provider or OllamaProvider(model=cfg.model, host=cfg.ollama_host)
    pipeline = AgentPipeline(ai, provider, retrieval_mode=mode, top_k=cfg.top_k, timeout_s=cfg.request_timeout_s)
    return Assistant(ai, provider, pipeline, cfg)
