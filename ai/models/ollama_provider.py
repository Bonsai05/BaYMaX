"""Ollama adapter (local server, default http://localhost:11434). Standard library only.

Always streams internally so time-to-first-token is measured the same way for
generate() and stream(). Never raises: failures become error or timeout results.

Three ways to get a tool call out of a small model, all supported here so the
benchmark can compare them:
  1. prompt only        (ask for JSON in the system prompt)
  2. schema constrained (json_schema -> Ollama `format`, the runtime forces valid JSON)
  3. native tools       (tools -> Ollama `tools`, for models that support tool calling)
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

from ..gateway.provider import (
    GenerationRequest, GenerationResult, Health, StreamChunk, Timings, ToolCall, error_result,
)


class OllamaProvider:
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434",
                 keep_alive: str = "10m", num_ctx: int = 4096, name: str = "ollama"):
        self.model, self.host, self.keep_alive, self.num_ctx = model, host.rstrip("/"), keep_alive, num_ctx
        self.name = name

    # ---- request ---------------------------------------------------------
    def _payload(self, req: GenerationRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model, "messages": req.messages, "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens,
                        "num_ctx": self.num_ctx},
        }
        if req.json_schema is not None:
            body["format"] = req.json_schema
        if req.tools:
            body["tools"] = req.tools
        return body

    def _fail(self, msg: str, t0: float, timeout: bool = False) -> GenerationResult:
        return error_result(self.name, self.model, msg, timeout, time.perf_counter() - t0)

    # ---- streaming -------------------------------------------------------
    def stream(self, req: GenerationRequest) -> Iterator[StreamChunk]:
        t0 = time.perf_counter()
        first: float | None = None
        text: list[str] = []
        calls: list[ToolCall] = []
        final: dict[str, Any] = {}
        try:
            http_req = urllib.request.Request(
                f"{self.host}/api/chat", data=json.dumps(self._payload(req)).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(http_req, timeout=req.timeout_s) as resp:
                for line in resp:
                    if time.perf_counter() - t0 > req.timeout_s:
                        raise TimeoutError("total time limit exceeded")
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if "error" in obj:
                        yield StreamChunk(result=self._fail(str(obj["error"]), t0))
                        return
                    msg = obj.get("message") or {}
                    delta = msg.get("content") or ""
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        args = fn.get("arguments") or {}
                        calls.append(ToolCall(fn.get("name", ""), json.loads(args) if isinstance(args, str) else args))
                    if (delta or msg.get("tool_calls")) and first is None:
                        first = time.perf_counter() - t0
                    if delta:
                        text.append(delta)
                        yield StreamChunk(delta=delta)
                    if obj.get("done"):
                        final = obj
                        break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            yield StreamChunk(result=self._fail(f"http {e.code}: {body}", t0))
            return
        except (socket.timeout, TimeoutError):
            yield StreamChunk(result=self._fail("timeout", t0, timeout=True))
            return
        except urllib.error.URLError as e:
            is_timeout = isinstance(e.reason, (socket.timeout, TimeoutError))
            yield StreamChunk(result=self._fail("timeout" if is_timeout else f"unreachable: {e.reason}", t0, is_timeout))
            return
        except (OSError, ValueError) as e:
            yield StreamChunk(result=self._fail(f"{type(e).__name__}: {e}", t0))
            return

        if not final:
            yield StreamChunk(result=self._fail("stream ended before completion", t0))
            return
        total = time.perf_counter() - t0
        yield StreamChunk(result=GenerationResult(
            text="".join(text), tool_calls=calls, model=self.model, provider=self.name,
            finish_reason="length" if final.get("done_reason") == "length" else "stop",
            timings=Timings(ttft_s=first, total_s=total,
                            prompt_tokens=int(final.get("prompt_eval_count") or 0),
                            output_tokens=int(final.get("eval_count") or 0),
                            load_s=(final.get("load_duration") or 0) / 1e9)))

    def generate(self, req: GenerationRequest) -> GenerationResult:
        result = None
        for chunk in self.stream(req):
            result = chunk.result or result
        return result

    # ---- health ----------------------------------------------------------
    def _get(self, path: str, timeout: float = 3.0) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.host}{path}", timeout=timeout) as r:
            return json.loads(r.read())

    def health(self) -> Health:
        try:
            names = [m.get("name", "") for m in self._get("/api/tags").get("models", [])]
        except (OSError, ValueError) as e:      # URLError is an OSError
            return Health(False, f"unreachable: {e}")
        wanted = {self.model, self.model if ":" in self.model else f"{self.model}:latest"}
        if not wanted & set(names):
            return Health(False, f"model '{self.model}' not installed (run: ollama pull {self.model})")
        try:
            loaded = [{"name": m.get("name"), "size": m.get("size"), "size_vram": m.get("size_vram")}
                      for m in self._get("/api/ps").get("models", [])]
        except (OSError, ValueError):
            loaded = []
        return Health(True, "ready", loaded)
