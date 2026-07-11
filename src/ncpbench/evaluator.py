"""The external Fact Update and Conflict Check stages for one narrator turn."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Sequence, cast

from ncpbench.evaluator_prompts import (
    EvaluationPrompts,
    load_evaluation_prompts,
    render_conflict_check,
    render_conflict_double_check,
    render_fact_update,
    render_status_check,
    render_trajectory_check,
)
from ncpbench.models import Commitment, Fact, StoryTurn, TrajectoryNode
from ncpbench.model_output import (
    ModelOutputError,
    call_with_retries,
    parse_json_object,
)


ChatMessage = Mapping[str, str]


class AuditorClient(Protocol):
    """The fixed evaluator-model client. Narrator methods never receive it."""

    def complete(self, messages: Sequence[ChatMessage], *, stage: str) -> str:
        """Run one evaluator prompt and return the raw model response."""


@dataclass(frozen=True)
class TurnAuditContext:
    """All evaluator-owned material needed to inspect one narrator response."""

    player_role: str
    player_input: str
    history: Sequence[StoryTurn]
    active_facts: Sequence[Fact]
    commitments: Sequence[Commitment]
    trajectory: Sequence[TrajectoryNode]


@dataclass(frozen=True)
class FactUpdate:
    add_facts: tuple[str, ...]
    negate_fact_ids: tuple[str, ...]
    reason: str | None


@dataclass(frozen=True)
class Conflict:
    kind: str
    target_id: str | None
    reason: str


@dataclass(frozen=True)
class ConflictDecision:
    fact_count: int
    commitment_count: int
    player_input_count: int
    total_count: int
    conflicts: tuple[Conflict, ...]
    double_checked: bool
    review_reason: str | None = None

    @property
    def has_conflict(self) -> bool:
        return self.total_count > 0


@dataclass(frozen=True)
class TurnEvaluation:
    """Evaluator-owned result. Applying it to world state is the runner's job."""

    fact_update: FactUpdate
    conflict_decision: ConflictDecision


@dataclass(frozen=True)
class TrajectoryCheckRequest:
    """The runner-owned state needed to decide whether one node advances."""

    player_role: str
    active_facts: Sequence[Fact]
    history: Sequence[StoryTurn]
    trajectory: Sequence[TrajectoryNode]


@dataclass(frozen=True)
class TrajectoryAssessment:
    target_node_id: str
    occurred: bool
    reason: str


@dataclass(frozen=True)
class CommitmentCheckRequest:
    """The runner-owned state needed to audit all active commitments."""

    player_role: str
    active_facts: Sequence[Fact]
    history: Sequence[StoryTurn]
    trajectory: Sequence[TrajectoryNode]
    commitments: Sequence[Commitment]


@dataclass(frozen=True)
class CommitmentAssessment:
    commitment_id: str
    status: str
    reason: str


class TurnEvaluator:
    """Audits narrator text without owning or invoking a narrator implementation."""

    def __init__(self, auditor: AuditorClient, prompts: EvaluationPrompts | None = None) -> None:
        self._auditor = auditor
        self._prompts = prompts or load_evaluation_prompts()

    def evaluate(
        self, context: TurnAuditContext, narrator_text: str
    ) -> TurnEvaluation:
        if not narrator_text.strip():
            raise ValueError("Cannot evaluate empty narrator text")

        fact_update = self._extract_fact_update(context, narrator_text)
        conflict = self._check_conflicts(context, narrator_text, fact_update)
        return TurnEvaluation(
            fact_update=fact_update,
            conflict_decision=conflict,
        )

    def _extract_fact_update(
        self, context: TurnAuditContext, narrator_text: str
    ) -> FactUpdate:
        prompt = render_fact_update(self._prompts, context, narrator_text)
        return call_with_retries(
            lambda: self._auditor.complete(
                ({"role": "user", "content": prompt},),
                stage="method_response_fact_extract",
            ),
            _parse_fact_update,
            stage="method_response_fact_extract",
        )

    def _check_conflicts(
        self,
        context: TurnAuditContext,
        narrator_text: str,
        fact_update: FactUpdate,
    ) -> ConflictDecision:
        prompt = render_conflict_check(self._prompts, context, narrator_text, fact_update)
        initial_payload = call_with_retries(
            lambda: self._auditor.complete(
                ({"role": "user", "content": prompt},),
                stage="method_response_conflict_check",
            ),
            _parse_conflict_payload,
            stage="method_response_conflict_check",
        )
        initial = _parse_conflict_decision(initial_payload, double_checked=False)

        if not initial.has_conflict:
            return initial

        initial_conflicts = _conflicts_for_review(initial_payload, context)
        review_prompt = render_conflict_double_check(
            self._prompts,
            context,
            narrator_text,
            fact_update,
            json.dumps(initial_conflicts, ensure_ascii=False, indent=2),
        )
        review_payload = call_with_retries(
            lambda: self._auditor.complete(
                ({"role": "user", "content": review_prompt},),
                stage="conflict_double_check",
            ),
            _parse_conflict_review_payload,
            stage="conflict_double_check",
        )
        if review_payload.get("confirmed") is False:
            return ConflictDecision(0, 0, 0, 0, (), True, _optional_string(review_payload.get("review_reason")))

        reviewed = _parse_conflict_decision(review_payload, double_checked=True)
        return ConflictDecision(
            reviewed.fact_count,
            reviewed.commitment_count,
            reviewed.player_input_count,
            reviewed.total_count,
            reviewed.conflicts,
            True,
            _optional_string(review_payload.get("review_reason")),
        )

class TrajectoryChecker:
    """Advance one position when the current node's trigger and delta occur."""

    def __init__(self, auditor: AuditorClient, prompts: EvaluationPrompts | None = None) -> None:
        self._auditor = auditor
        self._prompts = prompts or load_evaluation_prompts()

    def check(self, request: TrajectoryCheckRequest) -> tuple[TrajectoryAssessment, ...]:
        current_index = _last_occurred_node_index(request.trajectory)
        if current_index is None or current_index >= len(request.trajectory) - 1:
            return ()

        prompt = render_trajectory_check(self._prompts, request)
        trigger_occurred, trigger_reason, delta_occurred, delta_reason = call_with_retries(
            lambda: self._auditor.complete(
                ({"role": "user", "content": prompt},), stage="trajectory_check"
            ),
            _parse_trajectory_response,
            stage="trajectory_check",
        )
        reason = f"trigger: {trigger_reason or 'missing'} | delta: {delta_reason or 'missing'}"
        return (
            TrajectoryAssessment(
                target_node_id=request.trajectory[current_index + 1].id,
                occurred=trigger_occurred and delta_occurred,
                reason=reason,
            ),
        )


class CommitmentChecker:
    """Evaluate commitment statuses without applying any state transition."""

    def __init__(self, auditor: AuditorClient, prompts: EvaluationPrompts | None = None) -> None:
        self._auditor = auditor
        self._prompts = prompts or load_evaluation_prompts()

    def check(self, request: CommitmentCheckRequest) -> tuple[CommitmentAssessment, ...]:
        if not request.commitments:
            return ()

        prompt = render_status_check(self._prompts, request)
        parsed = call_with_retries(
            lambda: self._auditor.complete(
                ({"role": "user", "content": prompt},), stage="status_check"
            ),
            lambda raw: _parse_commitment_statuses(raw, request.commitments),
            stage="status_check",
        )
        return tuple(
            CommitmentAssessment(
                commitment_id=commitment.id,
                status=parsed[commitment.id][0],
                reason=parsed[commitment.id][1],
            )
            for commitment in request.commitments
        )


def _last_occurred_node_index(trajectory: Sequence[TrajectoryNode]) -> int | None:
    current_index: int | None = None
    for index, node in enumerate(trajectory):
        if node.occurred:
            current_index = index
    return current_index


def _parse_fact_update(raw: str) -> FactUpdate:
    payload = parse_json_object(raw)
    return FactUpdate(
        add_facts=_required_string_list(payload, "add_facts"),
        negate_fact_ids=_required_string_list(payload, "negate_facts"),
        reason=_required_string(payload, "reason"),
    )


def _parse_conflict_payload(raw: str) -> dict[str, object]:
    payload = parse_json_object(raw)
    _validate_conflict_payload(payload)
    return payload


def _parse_conflict_review_payload(raw: str) -> dict[str, object]:
    payload = parse_json_object(raw)
    if not isinstance(payload.get("confirmed"), bool):
        raise ModelOutputError("confirmed must be boolean")
    _validate_conflict_payload(payload)
    _required_string(payload, "review_reason")
    return payload


def _validate_conflict_payload(payload: Mapping[str, object]) -> None:
    count_keys = (
        "fact_conflict_count",
        "commitment_conflict_count",
        "player_input_conflict_count",
        "total_conflict_count",
    )
    counts = {key: _required_nonnegative_int(payload, key) for key in count_keys}
    conflicts = payload.get("conflicts")
    if not isinstance(conflicts, list):
        raise ModelOutputError("conflicts must be an array")
    if counts["total_conflict_count"] != len(conflicts):
        raise ModelOutputError("total_conflict_count must equal len(conflicts)")
    if counts["total_conflict_count"] != sum(
        counts[key] for key in count_keys[:3]
    ):
        raise ModelOutputError("conflict category counts must sum to total_conflict_count")
    category_counts = {"fact": 0, "commitment": 0, "player_input": 0}
    for index, item in enumerate(conflicts):
        if not isinstance(item, Mapping):
            raise ModelOutputError(f"conflicts[{index}] must be an object")
        kind = item.get("type")
        if kind not in category_counts:
            raise ModelOutputError(f"conflicts[{index}].type is invalid")
        category_counts[str(kind)] += 1
        _required_string(item, "id", label=f"conflicts[{index}].id")
        _required_string(item, "reason", label=f"conflicts[{index}].reason")
    for kind, key in (
        ("fact", "fact_conflict_count"),
        ("commitment", "commitment_conflict_count"),
        ("player_input", "player_input_conflict_count"),
    ):
        if category_counts[kind] != counts[key]:
            raise ModelOutputError(f"{key} does not match conflicts")


def _parse_trajectory_response(raw: str) -> tuple[bool, str, bool, str]:
    payload = parse_json_object(raw)
    trigger = payload.get("trigger")
    delta = payload.get("delta")
    if not isinstance(trigger, Mapping) or not isinstance(delta, Mapping):
        raise ModelOutputError("trigger and delta must be objects")
    trigger_occurred = trigger.get("occurred")
    delta_occurred = delta.get("occurred")
    if not isinstance(trigger_occurred, bool) or not isinstance(delta_occurred, bool):
        raise ModelOutputError("trigger.occurred and delta.occurred must be boolean")
    return (
        trigger_occurred,
        _required_string(trigger, "reason", label="trigger.reason"),
        delta_occurred,
        _required_string(delta, "reason", label="delta.reason"),
    )


def _parse_commitment_statuses(
    raw: str, commitments: Sequence[Commitment]
) -> dict[str, tuple[str, str]]:
    payload = parse_json_object(raw)
    entries = payload.get("statuses")
    if not isinstance(entries, list):
        raise ModelOutputError("statuses must be an array")

    results: dict[str, tuple[str, str]] = {}
    expected_ids = {commitment.id for commitment in commitments}
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ModelOutputError(f"statuses[{index}] must be an object")
        commitment_id = _required_string(item, "id", label=f"statuses[{index}].id")
        if commitment_id not in expected_ids:
            raise ModelOutputError(f"statuses[{index}].id is unknown")
        if commitment_id in results:
            raise ModelOutputError(f"statuses contains duplicate id {commitment_id!r}")
        status_raw = _required_string(
            item, "status", label=f"statuses[{index}].status"
        ).upper()
        if status_raw not in {"SATISFIED", "PENDING"}:
            raise ModelOutputError(f"statuses[{index}].status is invalid")
        reason = _required_string(item, "reason", label=f"statuses[{index}].reason")
        results[commitment_id] = (status_raw.lower(), reason)
    missing_ids = expected_ids - results.keys()
    if missing_ids:
        raise ModelOutputError(
            f"statuses is missing ids: {', '.join(sorted(missing_ids))}"
        )
    return results


def _parse_conflict_decision(
    payload: Mapping[str, object], *, double_checked: bool
) -> ConflictDecision:
    fact_count = _required_nonnegative_int(payload, "fact_conflict_count")
    commitment_count = _required_nonnegative_int(
        payload, "commitment_conflict_count"
    )
    player_input_count = _required_nonnegative_int(
        payload, "player_input_conflict_count"
    )
    total_count = _required_nonnegative_int(payload, "total_conflict_count")
    conflicts = tuple(_parse_conflict(item) for item in _conflict_items(payload))
    return ConflictDecision(
        fact_count,
        commitment_count,
        player_input_count,
        total_count,
        conflicts,
        double_checked,
    )


def _parse_conflict(item: Mapping[str, object]) -> Conflict:
    return Conflict(
        kind=_required_string(item, "type"),
        target_id=_required_string(item, "id"),
        reason=_required_string(item, "reason"),
    )


def _conflict_items(
    payload: Mapping[str, object],
) -> list[Mapping[str, object]]:
    return cast(list[Mapping[str, object]], payload["conflicts"])


def _conflicts_for_review(
    payload: Mapping[str, object], context: TurnAuditContext
) -> list[dict[str, object | None]]:
    """Build the review payload with resolved target text."""

    fact_lookup = {fact.id: fact.text for fact in context.active_facts}
    commitment_lookup = {
        commitment.id: commitment.description for commitment in context.commitments
    }
    review_items: list[dict[str, object | None]] = []
    for item in _conflict_items(payload):
        conflict_type = item.get("type")
        conflict_id = item.get("id")
        content: object | None = None
        if conflict_id:
            identifier = str(conflict_id)
            if isinstance(conflict_type, str) and "fact" in conflict_type:
                content = fact_lookup.get(identifier)
            elif isinstance(conflict_type, str) and "commitment" in conflict_type:
                content = commitment_lookup.get(identifier)
            elif isinstance(conflict_type, str) and "player_input" in conflict_type:
                content = context.player_input
            else:
                content = fact_lookup.get(identifier) or commitment_lookup.get(identifier)
        review_items.append(
            {
                "type": conflict_type,
                "id": conflict_id,
                "reason": item.get("reason"),
                "content": content,
            }
        )
    return review_items


def _required_string_list(
    payload: Mapping[str, object], key: str
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ModelOutputError(f"{key} must be an array")
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ModelOutputError(f"{key}[{index}] must be a non-empty string")
        strings.append(item.strip())
    return tuple(strings)


def _required_string(
    payload: Mapping[str, object], key: str, *, label: str | None = None
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelOutputError(f"{label or key} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelOutputError(f"{key} must be a non-negative integer")
    return value


def _optional_string(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
