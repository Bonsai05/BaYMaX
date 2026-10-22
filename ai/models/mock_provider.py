"""Scripted provider so the pipeline and tests never wait on a real model."""
from __future__ import annotations

from typing import Callable, Iterator, Union

from ..gateway.provider import (
    GenerationRequest, GenerationResult, Health, StreamChunk, Timings, error_result,
)

Script = Union[list[str], Callable[[GenerationRequest], str]]


class MockProvider:
    def __init__(self, script: Script, name: str = "mock", model: str = "mock-1"):
        self.name, self.model, self._script, self._i = name, model, script, 0
        self.fail_next = 0          # failure injection: the next N calls return an error result
        self.timeout_next = 0
        self.calls: list[GenerationRequest] = []

    def _next_text(self, req: GenerationRequest) -> str:
        if callable(self._script):
            return self._script(req)
        text = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return text

    def stream(self, req: GenerationRequest) -> Iterator[StreamChunk]:
        self.calls.append(req)
        if self.timeout_next > 0:
            self.timeout_next -= 1
            yield StreamChunk(result=error_result(self.name, self.model, "mock timeout", timeout=True))
            return
        if self.fail_next > 0:
            self.fail_next -= 1
            yield StreamChunk(result=error_result(self.name, self.model, "mock failure"))
            return
        text = self._next_text(req)
        for piece in text.split(" "):
            yield StreamChunk(delta=piece + " ")
        words = len(text.split())
        yield StreamChunk(result=GenerationResult(
            text=text, model=self.model, provider=self.name,
            timings=Timings(ttft_s=0.0, total_s=0.0, output_tokens=words)))

    def generate(self, req: GenerationRequest) -> GenerationResult:
        last = None
        for chunk in self.stream(req):
            last = chunk.result or last
        return last

    def health(self) -> Health:
        return Health(True, "mock")
