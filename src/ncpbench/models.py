"""Benchmark-owned state passed between the runner, methods, and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Fact:
    """An explicit fact, retained even after a later update negates it."""

    id: str
    text: str
    active: bool = True


@dataclass(frozen=True)
class Commitment:
    id: str
    kind: Literal["invariant", "achievement", "ordering"]
    description: str
    satisfaction_condition: str
    violation_condition: str
    status: Literal["pending", "satisfied"] = "pending"


@dataclass(frozen=True)
class TrajectoryNode:
    """One trajectory milestone and its runtime progress marker.

    ``occurred`` means that the episode has reached this node. The current node's
    trigger and key delta still need to occur before the runner advances to the
    next node.
    """

    id: str
    description: str
    trigger_event: str
    key_delta: str
    occurred: bool = False


@dataclass(frozen=True)
class StoryTurn:
    """One completed story interaction, or the opening narration at turn -1."""

    turn_id: int
    player_input: str | None
    narrator_response: str | None


@dataclass(frozen=True)
class PendingFactUpdates:
    """Runner-proposed state changes not yet committed to the world state."""

    add_facts: tuple[str, ...] = ()
    negate_fact_ids: tuple[str, ...] = ()
