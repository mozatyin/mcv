from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):

    def text(self, prompt: str, *,
             max_tokens: int = 512,
             temperature: float = 0.0,
             model_tier: str = "fast") -> str:
        ...

    def vision(self, prompt: str, images: list[bytes], *,
               max_tokens: int = 512,
               temperature: float = 0.0,
               model_tier: str = "smart") -> str:
        ...
