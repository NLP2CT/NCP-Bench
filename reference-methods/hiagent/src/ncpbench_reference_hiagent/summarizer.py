from __future__ import annotations

from typing import Sequence

from ncpbench.model_output import ModelOutputError, call_with_retries, require_text
from ncpbench_reference_hiagent.client import TextGenerationClient
from ncpbench_reference_hiagent.hiagent_prompts import build_hiagent_summary_messages

Pair = tuple[str, str]
SubgoalTrace = Sequence[Sequence[Pair]]


class TrajectorySummarizer:
    """Summarize completed subgoal trajectories into one concise line."""

    def __init__(self, client: TextGenerationClient, max_chars: int = 220) -> None:
        self.client = client
        self.max_chars = max(64, int(max_chars))

    def generate_summary(
        self, trajectories: list[SubgoalTrace], subgoals: list[Pair]
    ) -> list[str]:
        if len(trajectories) != len(subgoals):
            raise ValueError("HiAgent summaries require one subgoal per trajectory")
        return [
            self._llm_summary(trajectory, subgoal)
            for trajectory, subgoal in zip(trajectories, subgoals)
        ]

    def _llm_summary(self, trajectory: SubgoalTrace, subgoal: Pair) -> str:
        serialized = self._serialize_trajectory(trajectory)
        messages = build_hiagent_summary_messages(subgoal_text=subgoal[1], trajectory_text=serialized)
        content = call_with_retries(
            lambda: self.client.complete(messages, stage="hiagent_summary"),
            _one_line,
            stage="hiagent_summary",
        )
        return self._truncate(content)

    def _serialize_trajectory(self, trajectory: SubgoalTrace) -> str:
        lines: list[str] = []
        for chunk in trajectory:
            for role, text in chunk:
                if not text:
                    continue
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def _truncate(self, text: str) -> str:
        text = " ".join(text.split())
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars - 3].rstrip() + "..."


def _one_line(raw: str) -> str:
    text = require_text(raw).strip()
    if len(text.splitlines()) != 1:
        raise ModelOutputError("HiAgent summary must contain exactly one line")
    return text
