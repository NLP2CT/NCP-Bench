"""JSON checkpoints for committed episode boundaries."""

from __future__ import annotations

from collections.abc import Mapping

from ncpbench.dataset import StorySpec
from ncpbench.evaluator import (
    CommitmentAssessment,
    Conflict,
    ConflictDecision,
    FactUpdate,
    TrajectoryAssessment,
    TurnEvaluation,
)
from ncpbench.models import Commitment, Fact, PendingFactUpdates, StoryTurn, TrajectoryNode
from ncpbench.narrator import EpisodeContext, NarratorResponse
from ncpbench.results import JSONValue, episode_trace_to_result
from ncpbench.runner import (
    EpisodeCheckpoint,
    EpisodeSession,
    EpisodeState,
    EpisodeTermination,
    EpisodeTrace,
    EpisodeTurnResult,
)


def checkpoint_to_mapping(checkpoint: EpisodeCheckpoint) -> dict[str, JSONValue]:
    """Serialize one checkpoint without prompts, provider responses, or secrets."""

    progress = episode_trace_to_result(
        EpisodeTrace(
            checkpoint.session,
            checkpoint.turns,
            EpisodeTermination.MAX_TURNS,
        )
    )
    progress.pop("termination")
    progress["pending_fact_updates"] = {
        "add_facts": list(checkpoint.session.pending_fact_updates.add_facts),
        "negate_fact_ids": list(
            checkpoint.session.pending_fact_updates.negate_fact_ids
        ),
    }
    progress["narrator_state"] = _json_mapping(checkpoint.narrator_state)
    return progress


def checkpoint_from_mapping(
    value: Mapping[str, object], spec: StorySpec
) -> EpisodeCheckpoint:
    """Restore a checkpoint against its immutable story specification."""

    episode = _mapping(value.get("episode"), "episode")
    episode_id = _string(episode.get("id"), "episode.id")
    if episode_id != spec.id:
        raise ValueError(
            f"Checkpoint episode {episode_id!r} does not match {spec.id!r}"
        )
    player_role = _string(episode.get("player_role"), "episode.player_role")
    if player_role != spec.player_role:
        raise ValueError("Checkpoint player role does not match the specification")

    context = EpisodeContext(spec.id, spec.player_role, spec.trajectory)
    state = _state(_mapping(value.get("final_state"), "final_state"))
    pending = _mapping(value.get("pending_fact_updates"), "pending_fact_updates")
    session = EpisodeSession(
        context,
        state,
        PendingFactUpdates(
            _string_tuple(pending.get("add_facts"), "pending_fact_updates.add_facts"),
            _string_tuple(
                pending.get("negate_fact_ids"),
                "pending_fact_updates.negate_fact_ids",
            ),
        ),
    )
    turns = tuple(
        _turn_result(_mapping(item, f"turns[{index}]"), session)
        for index, item in enumerate(_list(value.get("turns"), "turns"))
    )
    narrator_state = _mapping(value.get("narrator_state"), "narrator_state")
    return EpisodeCheckpoint(session, turns, narrator_state)


def _state(value: Mapping[str, object]) -> EpisodeState:
    return EpisodeState(
        facts=tuple(
            Fact(
                _string(item.get("id"), f"facts[{index}].id"),
                _string(item.get("text"), f"facts[{index}].text"),
                _boolean(item.get("active"), f"facts[{index}].active"),
            )
            for index, raw in enumerate(_list(value.get("facts"), "facts"))
            for item in (_mapping(raw, f"facts[{index}]"),)
        ),
        commitments=tuple(
            Commitment(
                id=_string(item.get("id"), f"commitments[{index}].id"),
                kind=_commitment_kind(
                    item.get("kind"), f"commitments[{index}].kind"
                ),
                description=_string(
                    item.get("description"), f"commitments[{index}].description"
                ),
                satisfaction_condition=_string(
                    item.get("satisfaction_condition"),
                    f"commitments[{index}].satisfaction_condition",
                ),
                violation_condition=_string(
                    item.get("violation_condition"),
                    f"commitments[{index}].violation_condition",
                ),
                status=_commitment_status(
                    item.get("status"), f"commitments[{index}].status"
                ),
            )
            for index, raw in enumerate(
                _list(value.get("commitments"), "commitments")
            )
            for item in (_mapping(raw, f"commitments[{index}]"),)
        ),
        trajectory=tuple(
            TrajectoryNode(
                id=_string(item.get("id"), f"trajectory[{index}].id"),
                description=_string(
                    item.get("description"), f"trajectory[{index}].description"
                ),
                trigger_event=_string(
                    item.get("trigger_event"), f"trajectory[{index}].trigger_event"
                ),
                key_delta=_string(
                    item.get("key_delta"), f"trajectory[{index}].key_delta"
                ),
                occurred=_boolean(
                    item.get("occurred"), f"trajectory[{index}].occurred"
                ),
            )
            for index, raw in enumerate(_list(value.get("trajectory"), "trajectory"))
            for item in (_mapping(raw, f"trajectory[{index}]"),)
        ),
        history=tuple(
            StoryTurn(
                _integer(item.get("turn_id"), f"history[{index}].turn_id"),
                _optional_string(item.get("player_input"), f"history[{index}].player_input"),
                _optional_string(
                    item.get("narrator_response"),
                    f"history[{index}].narrator_response",
                ),
            )
            for index, raw in enumerate(_list(value.get("history"), "history"))
            for item in (_mapping(raw, f"history[{index}]"),)
        ),
    )


def _turn_result(
    value: Mapping[str, object], session: EpisodeSession
) -> EpisodeTurnResult:
    evaluation = _mapping(value.get("evaluation"), "turn.evaluation")
    fact_update = _mapping(evaluation.get("fact_update"), "turn.evaluation.fact_update")
    conflict = _mapping(evaluation.get("conflict"), "turn.evaluation.conflict")
    response = _string(value.get("narrator_response"), "turn.narrator_response")
    completed_turn = StoryTurn(
        _integer(value.get("turn_id"), "turn.turn_id"),
        _string(value.get("player_input"), "turn.player_input"),
        response,
    )
    turn_evaluation = TurnEvaluation(
        FactUpdate(
            _string_tuple(fact_update.get("add_facts"), "fact_update.add_facts"),
            _string_tuple(
                fact_update.get("negate_fact_ids"), "fact_update.negate_fact_ids"
            ),
            _optional_string(fact_update.get("reason"), "fact_update.reason"),
        ),
        ConflictDecision(
            fact_count=_integer(conflict.get("fact_count"), "conflict.fact_count"),
            commitment_count=_integer(
                conflict.get("commitment_count"), "conflict.commitment_count"
            ),
            player_input_count=_integer(
                conflict.get("player_input_count"), "conflict.player_input_count"
            ),
            total_count=_integer(conflict.get("total_count"), "conflict.total_count"),
            conflicts=tuple(
                Conflict(
                    _string(item.get("kind"), f"conflicts[{index}].kind"),
                    _optional_string(
                        item.get("target_id"), f"conflicts[{index}].target_id"
                    ),
                    _string(item.get("reason"), f"conflicts[{index}].reason"),
                )
                for index, raw in enumerate(
                    _list(conflict.get("conflicts"), "conflict.conflicts")
                )
                for item in (_mapping(raw, f"conflicts[{index}]"),)
            ),
            double_checked=_boolean(
                conflict.get("double_checked"), "conflict.double_checked"
            ),
            review_reason=_optional_string(
                conflict.get("review_reason"), "conflict.review_reason"
            ),
        ),
    )
    return EpisodeTurnResult(
        session=session,
        completed_turn=completed_turn,
        response=NarratorResponse(response),
        evaluation=turn_evaluation,
        trajectory_assessments=tuple(
            TrajectoryAssessment(
                _string(item.get("target_node_id"), "target_node_id"),
                _boolean(item.get("occurred"), "occurred"),
                _string(item.get("reason"), "reason"),
            )
            for raw in _list(
                value.get("trajectory_assessments"), "trajectory_assessments"
            )
            for item in (_mapping(raw, "trajectory_assessment"),)
        ),
        commitment_assessments=tuple(
            CommitmentAssessment(
                _string(item.get("commitment_id"), "commitment_id"),
                _string(item.get("status"), "status"),
                _string(item.get("reason"), "reason"),
            )
            for raw in _list(
                value.get("commitment_assessments"), "commitment_assessments"
            )
            for item in (_mapping(raw, "commitment_assessment"),)
        ),
        newly_occurred_node_ids=_string_tuple(
            value.get("newly_occurred_node_ids"), "newly_occurred_node_ids"
        ),
        new_satisfaction_ids=_string_tuple(
            value.get("new_satisfaction_ids"), "new_satisfaction_ids"
        ),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    items = _list(value, label)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{label} must contain non-empty strings")
    return tuple(items)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _commitment_kind(value: object, label: str):
    if value not in {"invariant", "achievement", "ordering"}:
        raise ValueError(f"{label} is invalid")
    return value


def _commitment_status(value: object, label: str):
    if value not in {"pending", "satisfied"}:
        raise ValueError(f"{label} is invalid")
    return value


def _json_mapping(value: Mapping[str, object]) -> dict[str, JSONValue]:
    import json

    encoded = json.dumps(value, ensure_ascii=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("Narrator checkpoint state must be a JSON object")
    return decoded
