"""Method-owned text-generation dependency for the reference HiAgent."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence


class TextGenerationClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]], *, stage: str) -> str:
        """Return the raw completion for one method-owned generation request."""
