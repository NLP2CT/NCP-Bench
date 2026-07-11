"""The public boundary for a narrator under evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Sequence

from ncpbench.models import Commitment, Fact, PendingFactUpdates, StoryTurn, TrajectoryNode


@dataclass(frozen=True)
class EpisodeContext:
    """Benchmark state fixed for one independent story episode."""

    episode_id: str
    player_role: str
    trajectory: tuple[TrajectoryNode, ...]


@dataclass(frozen=True)
class OpeningRequest:
    """The initial story state supplied to a narrator before player input."""

    player_role: str
    active_facts: tuple[Fact, ...]
    commitments: tuple[Commitment, ...]
    trajectory: tuple[TrajectoryNode, ...]


@dataclass(frozen=True)
class NarratorRequest:
    """The complete runner-provided context for one narrator turn.

    This contains story state only. It intentionally contains no evaluator, audit
    prompt, conflict decision, or access to evaluator-model outputs.
    """

    turn_id: int
    player_role: str
    player_input: str
    history: tuple[StoryTurn, ...]
    active_facts: tuple[Fact, ...]
    commitments: tuple[Commitment, ...]
    trajectory: tuple[TrajectoryNode, ...]
    current_node_id: str | None
    pending_fact_updates: PendingFactUpdates = PendingFactUpdates()


@dataclass(frozen=True)
class NarratorResponse:
    """A narrator may only return the text of its next story response."""

    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Narrator responses must contain non-empty text")


class Narrator(ABC):
    """A method under evaluation. It has no access to the evaluator."""

    def start_episode(self, context: EpisodeContext) -> None:
        """Initialize optional method-local state for an episode."""

    @abstractmethod
    def open(self, request: OpeningRequest) -> NarratorResponse:
        """Generate the episode's initial narrative before any player input."""

    @abstractmethod
    def respond(self, request: NarratorRequest) -> NarratorResponse:
        """Generate one narrative response."""

    def close_episode(self) -> None:
        """Release optional method-local state after an episode."""

    def checkpoint_state(self) -> Mapping[str, object]:
        """Return the minimal JSON-compatible method state at a turn boundary."""

        return {}

    def restore_episode(
        self, context: EpisodeContext, state: Mapping[str, object]
    ) -> None:
        """Restore a method at a previously committed turn boundary."""

        if state:
            raise ValueError("This narrator does not define checkpoint state")
        self.start_episode(context)


def render_opening_prompt(request: OpeningRequest) -> str:
    """Render the paper's fixed opening prompt for a narrator implementation."""

    return files("ncpbench.prompts").joinpath("opening.txt").read_text(
        encoding="utf-8"
    ).format(
        player_role=request.player_role or "Player",
        current_node=_format_opening_node(request.trajectory),
        narrative_commitments=_format_opening_commitments(request.commitments),
        pre_turn_facts=_format_opening_facts(request.active_facts),
    )


def _format_opening_node(trajectory: Sequence[TrajectoryNode]) -> str:
    if not trajectory:
        return "(None)"
    node = trajectory[0]
    return "\n".join(
        (
            f"- node_id: {node.id}",
            f" | description: {node.description}",
            f" | trigger: {node.trigger_event}",
            f" | delta: {node.key_delta}",
        )
    )


def _format_opening_commitments(commitments: Sequence[Commitment]) -> str:
    lines: list[str] = []
    for commitment in commitments:
        lines.extend(
            (
                f"- {commitment.id}: type={commitment.kind} | status={commitment.status}",
                f"  description: {commitment.description}",
                f"  satisfaction_condition: {commitment.satisfaction_condition}",
                f"  violation_condition: {commitment.violation_condition}",
            )
        )
    return "\n".join(lines) or "(None)"


def _format_opening_facts(facts: Sequence[Fact]) -> str:
    lines = [f"- {fact.id}: {fact.text}" for fact in facts if fact.active]
    return "\n".join(lines) or "(none)"
