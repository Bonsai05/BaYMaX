"""HTTP surface for the AI module, as a FastAPI router Member 1 mounts inside the Core.

    from fastapi import FastAPI, Depends
    from ai.assistant import build_assistant, AssistantConfig
    from ai.service import build_router

    assistant = build_assistant(AssistantConfig(db_path="vault/memory.db"))
    app = FastAPI()
    app.include_router(build_router(assistant), dependencies=[Depends(core_auth)])   # the Core owns auth

Rules this layer follows:
  * It adds no authentication and no authorization. The Core protects the routes (see Depends above).
  * `ingest_path` is accepted from the Core's code only. Never forward it from a client request.
  * `confirmed=true` on promote and restore must come from a real user action in the UI, never from
    the model or from text. The Core decides; this layer just passes it on.
  * A proposal in an answer is DATA, not an action. Member 4's policy engine decides what happens to it.
  * Handlers are plain `def`, so FastAPI runs them in a worker thread; the Assistant serializes them.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .assistant import Assistant
from .contracts import Event
from .pipeline.agent import AgentTurn


class AskIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    project: Optional[str] = None
    session_id: str = "default"


class EventIn(BaseModel):
    event_id: str
    entity_id: str
    source_device_id: str
    device_sequence: int
    event_type: str
    payload: dict[str, Any]
    ingest_path: str                     # set by the Core's own code path
    payload_hash: str = ""
    core_seq: Optional[int] = None
    base_version: int = 0
    created_at: Optional[float] = None


class ConfirmIn(BaseModel):
    confirmed: bool = False
    acknowledge_injection_flag: bool = False
    actor: str = "user"


class ArtifactIn(BaseModel):
    path: str = Field(min_length=1)
    actor: str = "user"


def turn_to_dict(turn: AgentTurn) -> dict[str, Any]:
    return {
        "kind": turn.kind,                                   # answer | proposal | error
        "text": turn.text,
        "proposal": ({"name": turn.proposal.name, "arguments": turn.proposal.arguments}
                     if turn.proposal else None),
        "problems": turn.problems,
        "route": turn.route.to_dict() if turn.route else None,
        "evidence_ids": turn.evidence_ids,
        "tainted_by": turn.tainted_by,                       # untrusted memories the model saw
        "degraded": turn.degraded,
        "model_calls": turn.model_calls,
        "fast_path": turn.fast_path,
        "timings_s": {k: round(v, 3) for k, v in turn.timings_s.items()},
    }


def build_router(assistant: Assistant) -> APIRouter:
    r = APIRouter(prefix="/ai", tags=["ai"])

    @r.get("/health")
    def health() -> dict[str, Any]:
        return assistant.health()

    @r.post("/ask")
    def ask(body: AskIn) -> dict[str, Any]:
        return turn_to_dict(assistant.ask(body.text, project=body.project, session_id=body.session_id))

    @r.post("/ask/stream")
    def ask_stream(body: AskIn) -> StreamingResponse:
        """Server-sent events: `event: token` lines while the answer is written, then one `event: done`
        with the same JSON as /ai/ask. A proposal or an error produces only the `done` event."""
        q: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        def work() -> None:
            try:
                turn = assistant.ask(body.text, project=body.project, session_id=body.session_id,
                                     on_token=lambda d: q.put(("token", d)))
                q.put(("done", turn_to_dict(turn)))
            except Exception as e:                           # never leave the client hanging
                q.put(("done", {"kind": "error", "text": f"server_error: {type(e).__name__}", "problems": []}))

        threading.Thread(target=work, daemon=True).start()

        def events():
            while True:
                name, data = q.get()
                yield f"event: {name}\ndata: {json.dumps(data)}\n\n"
                if name == "done":
                    return

        return StreamingResponse(events(), media_type="text/event-stream")

    @r.post("/session/{session_id}/clear")
    def clear(session_id: str) -> dict[str, bool]:
        assistant.clear_session(session_id)
        return {"ok": True}

    @r.post("/ingest")
    def ingest(body: EventIn) -> dict[str, Any]:
        data = body.model_dump(exclude_none=True)
        return assistant.admit(Event(**data)).to_dict()

    @r.post("/memory/{memory_id}/promote")
    def promote(memory_id: str, body: ConfirmIn) -> dict[str, Any]:
        try:
            with assistant.lock:
                assistant.ai.promote_memory(memory_id, confirmed=body.confirmed, actor=body.actor,
                                            acknowledge_injection_flag=body.acknowledge_injection_flag)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown_memory")
        except (ValueError, PermissionError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"ok": True}

    @r.post("/memory/{memory_id}/restore")
    def restore(memory_id: str, body: ConfirmIn) -> dict[str, Any]:
        try:
            with assistant.lock:
                assistant.ai.restore_memory(memory_id, confirmed=body.confirmed, actor=body.actor)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown_memory")
        except (ValueError, PermissionError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"ok": True}

    @r.post("/memory/revoke-artifact")
    def revoke_artifact(body: ArtifactIn) -> dict[str, Any]:
        with assistant.lock:
            return {"revoked": assistant.ai.revoke_artifact(body.path, actor=body.actor)}

    @r.get("/memory/{memory_id}/audit")
    def audit(memory_id: str) -> dict[str, Any]:
        with assistant.lock:
            return {"memory_id": memory_id, "trail": assistant.ai.audit_trail(memory_id)}

    return r
