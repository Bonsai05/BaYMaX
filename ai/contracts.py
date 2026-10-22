"""Shared contracts for the BAYMAX AI/Memory module (Member 3).

These dataclasses are the only types that cross the boundary between the AI
module and the rest of BAYMAX. Member 1's FastAPI core builds Event objects and
calls AIModule.admit / retrieve / route; everything else stays inside ai/.

Field names follow the v5 architecture document (Appendix A). Changing a field
here is a contract change and must be agreed with Member 1.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
class TrustTier(str, Enum):
    T0 = "T0"  # typed or confirmed by the user
    T1 = "T1"  # derived from the user's own BAYMAX conversations
    T2 = "T2"  # files, tool output, web, anything third-party (untrusted)


class Decision(str, Enum):
    ADMIT = "admit"
    QUARANTINE = "quarantine"  # stored as untrusted evidence, never promoted
    DISCARD = "discard"


class MemoryKind(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROFILE = "profile"
    DEVICE = "device"
    ARTIFACT = "artifact"


class PrivacyClass(str, Enum):
    LOCAL_ONLY = "local_only"
    CLOUD_ALLOWED = "cloud_allowed"


class Route(str, Enum):
    DIRECT = "direct"            # deterministic tool, no RAG, no LLM
    RAG = "rag"                  # retrieve, then answer
    LONG_CONTEXT = "long_context"  # bounded map-reduce (RLM interface)


# Ingest paths. The CORE (Member 1) sets this from the code path that wrote the
# event. It must never be copied from event content or chosen by the LLM.
INGEST_USER_TYPED = "user_typed"
INGEST_USER_CONFIRMED = "user_confirmed"
INGEST_ASSISTANT_CONVERSATION = "assistant_conversation"
INGEST_FILE_IMPORT = "file_import"
INGEST_TOOL_OUTPUT = "tool_output"
INGEST_WEB = "web"
INGEST_DEVICE_SENSOR = "device_sensor"


# --------------------------------------------------------------------------
# 1. Event (input to admission)
# --------------------------------------------------------------------------
@dataclass
class Event:
    """A committed event from the Core's canonical event log."""

    event_id: str
    entity_id: str
    source_device_id: str
    device_sequence: int
    event_type: str                 # e.g. "note", "file", "task", "message"
    payload: dict[str, Any]         # must contain "text"; optional "title", "path", "project"
    ingest_path: str                # set by Core code, see INGEST_* above
    payload_hash: str = ""
    core_seq: Optional[int] = None
    base_version: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Event":
        return Event(**d)


# --------------------------------------------------------------------------
# 2. Memory record (what admission produces and the store keeps)
# --------------------------------------------------------------------------
@dataclass
class MemoryRecord:
    memory_id: str
    canonical_source: str           # event_id this record came from
    source_device: str
    source_artifact: str            # file path or "" if none
    trust_tier: TrustTier
    kind: MemoryKind
    content: str
    confidence: float               # 0..1, how sure admission is about kind/tier
    authority: str                  # "user" (T0) or "none"; evidence never grants authority
    privacy_class: PrivacyClass
    project: str = ""
    quarantined: bool = False
    flags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    supersedes: Optional[str] = None
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trust_tier"] = self.trust_tier.value
        d["kind"] = self.kind.value
        d["privacy_class"] = self.privacy_class.value
        return d


@dataclass
class AdmissionResult:
    decision: Decision
    reason: str
    memory_ids: list[str] = field(default_factory=list)   # one per stored chunk
    trust_tier: Optional[TrustTier] = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["trust_tier"] = self.trust_tier.value if self.trust_tier else None
        return d


# --------------------------------------------------------------------------
# 3. Retrieval request / response
# --------------------------------------------------------------------------
@dataclass
class RetrievalRequest:
    query: str
    top_k: int = 5
    mode: str = "hybrid_rerank"     # lexical | vector | hybrid | hybrid_rerank
    project: Optional[str] = None
    source_device: Optional[str] = None
    kinds: Optional[list[str]] = None
    include_quarantined: bool = True   # returned flagged as untrusted
    for_cloud: bool = False            # True: the result will be sent to a cloud model, so
                                       # LOCAL_ONLY items are dropped. False: nothing is filtered.
    since: Optional[float] = None      # unix time lower bound


@dataclass
class RetrievedItem:
    memory_id: str
    canonical_source: str
    content: str
    score: float
    trust_tier: str
    quarantined: bool
    kind: str
    project: str
    source_device: str
    source_artifact: str
    stage_scores: dict[str, float] = field(default_factory=dict)

    @property
    def untrusted(self) -> bool:
        return self.quarantined or self.trust_tier == TrustTier.T2.value


@dataclass
class RetrievalResult:
    items: list[RetrievedItem]
    mode: str
    timings_ms: dict[str, float]    # lexical, vector, fuse, rerank, total

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "timings_ms": self.timings_ms,
            "items": [asdict(i) | {"untrusted": i.untrusted} for i in self.items],
        }


# --------------------------------------------------------------------------
# 4. Router
# --------------------------------------------------------------------------
@dataclass
class RouteRequest:
    query: str
    available_tools: list[str] = field(default_factory=list)
    token_budget: int = 3000
    project: Optional[str] = None


@dataclass
class RouteDecision:
    route: Route
    stage: int                      # 1 rules, 2 classifier, 3 escalation
    reason: str
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)
    tool: Optional[str] = None      # set for DIRECT routes
    tool_args: dict[str, Any] = field(default_factory=dict)
    low_confidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["route"] = self.route.value
        return d
