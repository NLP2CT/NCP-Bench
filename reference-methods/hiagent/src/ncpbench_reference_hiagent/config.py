"""Fixed runtime parameters for the paper HiAgent method."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HiAgentRuntimeConfig:
    memory_size: int = 100
    max_input_tokens: int = 12000
    summary_max_chars: int = 220
    max_retrieve_depth: int = 4
    system_message: str = "You are a helpful assistant."
