"""Stage 8: the model-independent interface.

Rules:
  * A provider never raises. Failures come back as a GenerationResult with
    finish_reason "error" or "timeout", so "model unavailable" is an ordinary value
    the agent pipeline can handle.
  * The model only proposes. It has no authority over state, permissions or security.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    messages: list[dict[str, str]]                 # [{"role": "system|user|assistant", "content": "..."}]
    json_schema: Optional[dict[str, Any]] = None   # constrain the output to this schema
    tools: Optional[list[dict[str, Any]]] = None   # native tool definitions (provider may ignore)
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 60.0


@dataclass
class Timings:
    ttft_s: Optional[float] = None    # request start to first content or tool call
    total_s: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    load_s: float = 0.0               # model load time reported by the runtime (cold start)

    @property
    def tokens_per_s(self) -> float:
        gen = self.total_s - (self.ttft_s or 0.0)
        return self.output_tokens / gen if gen > 0 and self.output_tokens else 0.0


@dataclass
class GenerationResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"       # stop | length | timeout | error
    error: Optional[str] = None
    model: str = ""
    provider: str = ""
    timings: Timings = field(default_factory=Timings)
    fallback_from: list[str] = field(default_factory=list)   # providers that failed first

    @property
    def ok(self) -> bool:
        return self.error is None and self.finish_reason in ("stop", "length")


@dataclass
class StreamChunk:
    delta: str = ""
    result: Optional[GenerationResult] = None   # set on the last chunk only


@dataclass
class Health:
    ok: bool
    detail: str = ""
    loaded_models: list[dict[str, Any]] = field(default_factory=list)   # name, size, size_vram


class ModelProvider(Protocol):
    name: str

    def generate(self, req: GenerationRequest) -> GenerationResult: ...
    def stream(self, req: GenerationRequest) -> Iterator[StreamChunk]: ...
    def health(self) -> Health: ...


def error_result(provider: str, model: str, message: str, timeout: bool = False,
                 total_s: float = 0.0) -> GenerationResult:
    return GenerationResult(finish_reason="timeout" if timeout else "error", error=message,
                            model=model, provider=provider, timings=Timings(total_s=total_s))


# ---- health metrics ---------------------------------------------------------
class ProviderMetrics:
    """Rolling record of calls, for the model health numbers the stage asks for."""

    def __init__(self) -> None:
        self.results: list[GenerationResult] = []

    def record(self, r: GenerationResult) -> None:
        self.results.append(r)

    def summary(self) -> dict[str, Any]:
        n = len(self.results)
        if not n:
            return {"calls": 0}
        totals = sorted(r.timings.total_s for r in self.results)
        ttfts = sorted(r.timings.ttft_s for r in self.results if r.timings.ttft_s is not None)
        pick = lambda xs, p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None
        return {
            "calls": n,
            "errors": sum(1 for r in self.results if r.finish_reason == "error"),
            "timeouts": sum(1 for r in self.results if r.finish_reason == "timeout"),
            "fallbacks": sum(1 for r in self.results if r.fallback_from),
            "total_s_median": statistics.median(totals), "total_s_p95": pick(totals, 0.95),
            "ttft_s_median": statistics.median(ttfts) if ttfts else None, "ttft_s_p95": pick(ttfts, 0.95),
        }


class MeteredProvider:
    """Wrap any provider so every generate() call is recorded in .metrics."""

    def __init__(self, inner: ModelProvider):
        self.inner, self.name, self.metrics = inner, inner.name, ProviderMetrics()

    def generate(self, req: GenerationRequest) -> GenerationResult:
        r = self.inner.generate(req)
        self.metrics.record(r)
        return r

    def stream(self, req: GenerationRequest) -> Iterator[StreamChunk]:
        for chunk in self.inner.stream(req):
            if chunk.result is not None:
                self.metrics.record(chunk.result)
            yield chunk

    def health(self) -> Health:
        return self.inner.health()


class FallbackProvider:
    """Try providers in order. generate() falls through on any failed result.
    stream() can only switch before the first token has been shown; once text has
    started to flow, a later failure is returned as-is (no mid-sentence switch)."""

    def __init__(self, providers: list[ModelProvider]):
        self.providers = providers
        self.name = "fallback(" + ",".join(p.name for p in providers) + ")"

    def generate(self, req: GenerationRequest) -> GenerationResult:
        failed: list[str] = []
        last = error_result(self.name, "", "no providers configured")
        for p in self.providers:
            last = p.generate(req)
            if last.ok:
                last.fallback_from = failed
                return last
            failed.append(p.name)
        last.fallback_from = failed[:-1]
        return last

    def stream(self, req: GenerationRequest) -> Iterator[StreamChunk]:
        failed: list[str] = []
        for i, p in enumerate(self.providers):
            started = False
            for chunk in p.stream(req):
                if chunk.result is not None:
                    if chunk.result.ok or started or i == len(self.providers) - 1:
                        chunk.result.fallback_from = failed
                        yield chunk
                        return
                    failed.append(p.name)       # failed before any text: try the next provider
                    break
                started = started or bool(chunk.delta)
                yield chunk
        else:
            yield StreamChunk(result=error_result(self.name, "", "no providers configured"))

    def health(self) -> Health:
        hs = [p.health() for p in self.providers]
        return Health(any(h.ok for h in hs), "; ".join(f"{p.name}: {h.detail}" for p, h in zip(self.providers, hs)))
