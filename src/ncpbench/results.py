"""Stable, JSON-compatible artifacts for completed benchmark episodes."""

from __future__ import annotations

from typing import TypeAlias

from ncpbench.evaluator import ConflictDecision, TurnEvaluation
from ncpbench.models import Commitment, Fact, StoryTurn, TrajectoryNode
from ncpbench.runner import EpisodeState, EpisodeTrace, EpisodeTurnResult


JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
EpisodeResult: TypeAlias = dict[str, JSONValue]


def episode_trace_to_result(trace: EpisodeTrace) -> EpisodeResult:
    """Convert a terminal :class:`EpisodeTrace` into a stable JSON artifact.

    The returned mapping contains only benchmark observations and final state.
    It intentionally excludes evaluator prompts, raw provider responses, and
    narrator implementation details. Callers own any choice to write this
    mapping to disk or transmit it elsewhere.
    """

    return {
        "episode": {
            "id": trace.session.context.episode_id,
            "player_role": trace.session.context.player_role,
        },
        "termination": trace.termination.value,
        "turns": [_turn_result(turn) for turn in trace.turns],
        "final_state": _state(trace.session.state),
    }


def _turn_result(turn: EpisodeTurnResult) -> dict[str, JSONValue]:
    return {
        "turn_id": turn.completed_turn.turn_id,
        "player_input": turn.completed_turn.player_input,
        "narrator_response": turn.response.text,
        "evaluation": _evaluation(turn.evaluation),
        "trajectory_assessments": [
            {
                "target_node_id": assessment.target_node_id,
                "occurred": assessment.occurred,
                "reason": assessment.reason,
            }
            for assessment in turn.trajectory_assessments
        ],
        "commitment_assessments": [
            {
                "commitment_id": assessment.commitment_id,
                "status": assessment.status,
                "reason": assessment.reason,
            }
            for assessment in turn.commitment_assessments
        ],
        "newly_occurred_node_ids": list(turn.newly_occurred_node_ids),
        "new_satisfaction_ids": list(turn.new_satisfaction_ids),
    }


def _evaluation(evaluation: TurnEvaluation) -> dict[str, JSONValue]:
    return {
        "fact_update": {
            "add_facts": list(evaluation.fact_update.add_facts),
            "negate_fact_ids": list(evaluation.fact_update.negate_fact_ids),
            "reason": evaluation.fact_update.reason,
        },
        "conflict": _conflict_decision(evaluation.conflict_decision),
    }


def _conflict_decision(decision: ConflictDecision) -> dict[str, JSONValue]:
    return {
        "has_conflict": decision.has_conflict,
        "fact_count": decision.fact_count,
        "commitment_count": decision.commitment_count,
        "player_input_count": decision.player_input_count,
        "total_count": decision.total_count,
        "conflicts": [
            {
                "kind": conflict.kind,
                "target_id": conflict.target_id,
                "reason": conflict.reason,
            }
            for conflict in decision.conflicts
        ],
        "double_checked": decision.double_checked,
        "review_reason": decision.review_reason,
    }


def _state(state: EpisodeState) -> dict[str, JSONValue]:
    return {
        "facts": [_fact(fact) for fact in state.facts],
        "commitments": [_commitment(commitment) for commitment in state.commitments],
        "trajectory": [_trajectory_node(node) for node in state.trajectory],
        "history": [_story_turn(turn) for turn in state.history],
    }


def _fact(fact: Fact) -> dict[str, JSONValue]:
    return {"id": fact.id, "text": fact.text, "active": fact.active}


def _commitment(commitment: Commitment) -> dict[str, JSONValue]:
    return {
        "id": commitment.id,
        "kind": commitment.kind,
        "description": commitment.description,
        "satisfaction_condition": commitment.satisfaction_condition,
        "violation_condition": commitment.violation_condition,
        "status": commitment.status,
    }


def _trajectory_node(node: TrajectoryNode) -> dict[str, JSONValue]:
    return {
        "id": node.id,
        "description": node.description,
        "trigger_event": node.trigger_event,
        "key_delta": node.key_delta,
        "occurred": node.occurred,
    }


def _story_turn(turn: StoryTurn) -> dict[str, JSONValue]:
    return {
        "turn_id": turn.turn_id,
        "player_input": turn.player_input,
        "narrator_response": turn.narrator_response,
    }
