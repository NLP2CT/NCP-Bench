"""Pure metric computation for NCP-Bench episode-result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ncpbench.results import EpisodeResult, JSONValue


@dataclass(frozen=True)
class EpisodeMetrics:
    """Paper metrics derived from one completed episode."""

    episode_id: str
    turn_count: int
    termination: str
    trajectory_reached: int
    trajectory_total: int
    trajectory_progress: float
    commitments_satisfied: int
    commitments_total: int
    commitment_satisfaction: float
    fact_conflict: bool
    commitment_conflict: bool
    player_input_conflict: bool


@dataclass(frozen=True)
class AggregateMetrics:
    """Macro-averaged paper metrics for a collection of episodes."""

    episode_count: int
    average_turns: float
    average_trajectory_progress: float
    average_commitment_satisfaction: float
    fact_conflict_rate: float
    commitment_conflict_rate: float
    player_input_conflict_rate: float
    conflict_count: int
    max_turns_count: int
    all_resolved_count: int


class MetricInputError(ValueError):
    """Raised when an episode result lacks the state required for metrics."""


def compute_episode_metrics(result: EpisodeResult) -> EpisodeMetrics:
    """Compute the metrics reported by the paper for one episode result.

    Trajectory progress follows the experiment artifacts: a node counts toward
    the numerator only after its own trigger and key delta have both been
    judged to have occurred, so progress starts at 0 and reaches 1 only when
    every node has completed. Commitment satisfaction uses all commitment
    types, matching the paper tables.
    """

    episode = _mapping(result.get("episode"), "episode")
    final_state = _mapping(result.get("final_state"), "final_state")
    turns = _list(result.get("turns"), "turns")
    trajectory = _list(final_state.get("trajectory"), "final_state.trajectory")
    commitments = _list(final_state.get("commitments"), "final_state.commitments")

    trajectory_reached = sum(
        1 for index, item in enumerate(trajectory) if _boolean(_mapping(item, f"trajectory[{index}]").get("occurred"))
    )
    commitments_satisfied = sum(
        1
        for index, item in enumerate(commitments)
        if _mapping(item, f"commitments[{index}]").get("status") == "satisfied"
    )

    fact_conflict = False
    commitment_conflict = False
    player_input_conflict = False
    for index, item in enumerate(turns):
        turn = _mapping(item, f"turns[{index}]")
        evaluation = _mapping(turn.get("evaluation"), f"turns[{index}].evaluation")
        conflict = _mapping(evaluation.get("conflict"), f"turns[{index}].evaluation.conflict")
        fact_conflict = fact_conflict or _positive_count(conflict.get("fact_count"))
        commitment_conflict = commitment_conflict or _positive_count(conflict.get("commitment_count"))
        player_input_conflict = player_input_conflict or _positive_count(conflict.get("player_input_count"))

    return EpisodeMetrics(
        episode_id=_string(episode.get("id"), "episode.id"),
        turn_count=len(turns),
        termination=_string(result.get("termination"), "termination"),
        trajectory_reached=trajectory_reached,
        trajectory_total=len(trajectory),
        trajectory_progress=_ratio(trajectory_reached, len(trajectory)),
        commitments_satisfied=commitments_satisfied,
        commitments_total=len(commitments),
        commitment_satisfaction=_ratio(commitments_satisfied, len(commitments)),
        fact_conflict=fact_conflict,
        commitment_conflict=commitment_conflict,
        player_input_conflict=player_input_conflict,
    )


def aggregate_metrics(episodes: Sequence[EpisodeMetrics]) -> AggregateMetrics:
    """Macro-average a non-empty collection of episode metrics."""

    if not episodes:
        raise ValueError("Cannot aggregate an empty episode collection")

    count = len(episodes)
    return AggregateMetrics(
        episode_count=count,
        average_turns=sum(item.turn_count for item in episodes) / count,
        average_trajectory_progress=sum(item.trajectory_progress for item in episodes) / count,
        average_commitment_satisfaction=sum(item.commitment_satisfaction for item in episodes) / count,
        fact_conflict_rate=sum(item.fact_conflict for item in episodes) / count,
        commitment_conflict_rate=sum(item.commitment_conflict for item in episodes) / count,
        player_input_conflict_rate=sum(item.player_input_conflict for item in episodes) / count,
        conflict_count=sum(item.termination == "conflict" for item in episodes),
        max_turns_count=sum(item.termination == "max_turns" for item in episodes),
        all_resolved_count=sum(item.termination == "all_resolved" for item in episodes),
    )


def survival_curve(episodes: Sequence[EpisodeMetrics], *, max_turns: int) -> tuple[float, ...]:
    """Return the fraction of episodes reaching each turn threshold from 0 onward."""

    if not episodes:
        raise ValueError("Cannot compute survival for an empty episode collection")
    if max_turns < 0:
        raise ValueError("max_turns must be non-negative")

    count = len(episodes)
    return tuple(
        sum(item.turn_count >= threshold for item in episodes) / count
        for threshold in range(max_turns + 1)
    )


def _mapping(value: JSONValue, label: str) -> Mapping[str, JSONValue]:
    if not isinstance(value, dict):
        raise MetricInputError(f"{label} must be an object")
    return value


def _list(value: JSONValue, label: str) -> list[JSONValue]:
    if not isinstance(value, list):
        raise MetricInputError(f"{label} must be an array")
    return value


def _string(value: JSONValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MetricInputError(f"{label} must be a non-empty string")
    return value


def _boolean(value: JSONValue) -> bool:
    return value is True


def _positive_count(value: JSONValue) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = [
    "AggregateMetrics",
    "EpisodeMetrics",
    "MetricInputError",
    "aggregate_metrics",
    "compute_episode_metrics",
    "survival_curve",
]
