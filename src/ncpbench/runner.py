"""Method-neutral turn execution and deterministic benchmark state transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Sequence

from ncpbench.evaluator import (
    CommitmentChecker,
    CommitmentCheckRequest,
    CommitmentAssessment,
    FactUpdate,
    TrajectoryChecker,
    TrajectoryCheckRequest,
    TrajectoryAssessment,
    TurnAuditContext,
    TurnEvaluation,
    TurnEvaluator,
)
from ncpbench.models import Commitment, Fact, PendingFactUpdates, StoryTurn, TrajectoryNode
from ncpbench.narrator import EpisodeContext, Narrator, NarratorRequest, NarratorResponse, OpeningRequest
from ncpbench.opening import OpeningEvaluator

if TYPE_CHECKING:
    from ncpbench.dataset import StorySpec
    from ncpbench.input_conditions import PlayerInputCondition


@dataclass(frozen=True)
class EpisodeState:
    """The runner-owned state of one episode.

    ``facts`` retains inactive facts because fact identifiers are allocated
    from the complete ledger. Methods and evaluators receive
    :func:`active_facts` instead.
    """

    facts: tuple[Fact, ...]
    commitments: tuple[Commitment, ...]
    trajectory: tuple[TrajectoryNode, ...]
    history: tuple[StoryTurn, ...] = ()


@dataclass(frozen=True)
class EpisodeTransition:
    """One fully evaluated turn applied to an :class:`EpisodeState`."""

    state: EpisodeState
    newly_occurred_node_ids: tuple[str, ...]
    new_satisfaction_ids: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeSession:
    """A started episode with runner-owned state and one-time opening updates."""

    context: EpisodeContext
    state: EpisodeState
    pending_fact_updates: PendingFactUpdates = PendingFactUpdates()


@dataclass(frozen=True)
class EpisodeTurnResult:
    """The complete result of advancing a session by one player input."""

    session: EpisodeSession
    completed_turn: StoryTurn
    response: NarratorResponse
    evaluation: TurnEvaluation
    trajectory_assessments: tuple[TrajectoryAssessment, ...]
    commitment_assessments: tuple[CommitmentAssessment, ...]
    newly_occurred_node_ids: tuple[str, ...]
    new_satisfaction_ids: tuple[str, ...]

    @property
    def has_conflict(self) -> bool:
        """Whether the fixed evaluator rejected this narrator response."""

        return self.evaluation.conflict_decision.has_conflict


class EpisodeTermination(str, Enum):
    """The three terminal outcomes of an episode."""

    ALL_RESOLVED = "all_resolved"
    MAX_TURNS = "max_turns"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class EpisodeTrace:
    """The terminal result of running a narrator over a complete input sequence."""

    session: EpisodeSession
    turns: tuple[EpisodeTurnResult, ...]
    termination: EpisodeTermination


@dataclass(frozen=True)
class EpisodeCheckpoint:
    """The last committed episode boundary and narrator-owned state."""

    session: EpisodeSession
    turns: tuple[EpisodeTurnResult, ...]
    narrator_state: Mapping[str, object]


def initialize_episode_state(
    facts: Sequence[Fact],
    commitments: Sequence[Commitment],
    trajectory: Sequence[TrajectoryNode],
) -> EpisodeState:
    """Create the initial runner state with the first trajectory node reached.

    A reached node is the current milestone, not a claim that its trigger and
    delta have completed.
    """

    normalized_facts = _normalize_facts(facts)
    normalized_trajectory = tuple(
        replace(node, occurred=index == 0) for index, node in enumerate(trajectory)
    )
    return EpisodeState(
        facts=normalized_facts,
        commitments=tuple(commitments),
        trajectory=normalized_trajectory,
    )


def active_facts(state: EpisodeState) -> tuple[Fact, ...]:
    """Return the current visible fact ledger in stable insertion order."""

    return tuple(fact for fact in state.facts if fact.active)


def apply_fact_updates(
    facts: Sequence[Fact], updates: PendingFactUpdates | FactUpdate
) -> tuple[Fact, ...]:
    """Apply fact additions and negations with stable fact identifiers."""

    ledger = {fact.id: fact for fact in _normalize_facts(facts)}
    for content in updates.add_facts:
        normalized = str(content).strip()
        if not normalized:
            continue
        fact_id = _next_fact_id(ledger)
        ledger[fact_id] = Fact(id=fact_id, text=normalized)

    for fact_id in updates.negate_fact_ids:
        normalized_id = str(fact_id).strip()
        if not normalized_id or normalized_id not in ledger:
            continue
        ledger[normalized_id] = replace(ledger[normalized_id], active=False)

    return tuple(ledger.values())


def apply_turn_outcome(
    state: EpisodeState,
    completed_turn: StoryTurn,
    evaluation: TurnEvaluation,
    trajectory_assessments: Sequence[TrajectoryAssessment],
    commitment_assessments: Sequence[CommitmentAssessment],
    *,
    pending_fact_updates: PendingFactUpdates = PendingFactUpdates(),
) -> EpisodeTransition:
    """Commit one externally evaluated turn without invoking a method or model.

    Opening updates are projected with the first response, checked together,
    and then committed before that response's updates.
    """

    facts_after_pending = apply_fact_updates(state.facts, pending_fact_updates)
    updated_facts = apply_fact_updates(facts_after_pending, evaluation.fact_update)
    updated_trajectory, newly_occurred_node_ids = _apply_trajectory_assessments(
        state.trajectory, trajectory_assessments
    )
    updated_commitments, new_satisfaction_ids = _apply_commitment_assessments(
        state.commitments, commitment_assessments
    )
    return EpisodeTransition(
        state=EpisodeState(
            facts=updated_facts,
            commitments=updated_commitments,
            trajectory=updated_trajectory,
            history=(*state.history, completed_turn),
        ),
        newly_occurred_node_ids=newly_occurred_node_ids,
        new_satisfaction_ids=new_satisfaction_ids,
    )


def all_achievement_commitments_resolved(state: EpisodeState) -> bool:
    """Return whether every achievement commitment is satisfied."""

    achievement_commitments = [
        commitment for commitment in state.commitments if commitment.kind == "achievement"
    ]
    return bool(achievement_commitments) and all(
        commitment.status == "satisfied" for commitment in achievement_commitments
    )


def _normalize_facts(facts: Sequence[Fact]) -> tuple[Fact, ...]:
    ledger = {fact.id: fact for fact in facts}
    return tuple(ledger.values())


def _next_fact_id(ledger: dict[str, Fact]) -> str:
    base = f"f_{len(ledger)}"
    candidate = base
    counter = 1
    while candidate in ledger:
        counter += 1
        candidate = f"{base}_{counter}"
    return candidate


def _apply_trajectory_assessments(
    trajectory: Sequence[TrajectoryNode], assessments: Sequence[TrajectoryAssessment]
) -> tuple[tuple[TrajectoryNode, ...], tuple[str, ...]]:
    occurred_ids = {assessment.target_node_id for assessment in assessments if assessment.occurred}
    newly_occurred_ids: list[str] = []
    updated: list[TrajectoryNode] = []
    for node in trajectory:
        should_occur = node.occurred or node.id in occurred_ids
        if should_occur and not node.occurred:
            newly_occurred_ids.append(node.id)
        updated.append(replace(node, occurred=should_occur))
    return tuple(updated), tuple(newly_occurred_ids)


def _apply_commitment_assessments(
    commitments: Sequence[Commitment], assessments: Sequence[CommitmentAssessment]
) -> tuple[tuple[Commitment, ...], tuple[str, ...]]:
    statuses = {assessment.commitment_id: assessment.status for assessment in assessments}
    expected_ids = {commitment.id for commitment in commitments}
    missing_ids = expected_ids - statuses.keys()
    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise ValueError(f"Missing commitment assessments for: {missing}")

    newly_satisfied_ids: list[str] = []
    updated: list[Commitment] = []
    for commitment in commitments:
        status = statuses[commitment.id]
        if status not in {"pending", "satisfied"}:
            raise ValueError(f"Invalid status for commitment {commitment.id!r}: {status!r}")
        if status == "satisfied" and commitment.status != "satisfied":
            newly_satisfied_ids.append(commitment.id)
        updated.append(replace(commitment, status=status))
    return tuple(updated), tuple(newly_satisfied_ids)


class EpisodeRunner:
    """Advance any public narrator through one fixed benchmark episode turn.

    This class owns benchmark sequencing, not a narrator policy. It has no
    method-name checks and no reference-method imports, so a third-party
    ``Narrator`` follows the identical evaluator and checker path.
    """

    def __init__(
        self,
        evaluator: TurnEvaluator,
        trajectory_checker: TrajectoryChecker,
        commitment_checker: CommitmentChecker,
        opening_evaluator: OpeningEvaluator,
    ) -> None:
        self._evaluator = evaluator
        self._trajectory_checker = trajectory_checker
        self._commitment_checker = commitment_checker
        self._opening_evaluator = opening_evaluator

    def start_episode(
        self,
        narrator: Narrator,
        context: EpisodeContext,
        facts: Sequence[Fact],
        commitments: Sequence[Commitment],
    ) -> EpisodeSession:
        """Initialize a narrator, generate its opening, and create the session."""

        narrator.start_episode(context)
        state = initialize_episode_state(facts, commitments, context.trajectory)
        opening_request = OpeningRequest(
            player_role=context.player_role,
            active_facts=active_facts(state),
            commitments=state.commitments,
            trajectory=state.trajectory,
        )
        opening_response = narrator.open(opening_request)
        pending_fact_updates = self._opening_evaluator.evaluate(
            opening_request, opening_response
        )
        state = replace(
            state,
            history=(StoryTurn(-1, None, opening_response.text),),
        )
        return EpisodeSession(
            context=context,
            state=state,
            pending_fact_updates=pending_fact_updates,
        )

    def close_episode(self, narrator: Narrator) -> None:
        """Close method-local episode state after the caller stops advancing it."""

        narrator.close_episode()

    def run(
        self,
        narrator: Narrator,
        spec: "StorySpec",
        input_condition: "PlayerInputCondition",
        *,
        max_turns: int = 100,
        checkpoint: EpisodeCheckpoint | None = None,
        on_checkpoint: Callable[[EpisodeCheckpoint], None] | None = None,
    ) -> EpisodeTrace:
        """Run one specification from a fresh or committed episode boundary."""

        context = EpisodeContext(spec.id, spec.player_role, spec.trajectory)
        turns = list(checkpoint.turns) if checkpoint else []
        try:
            if checkpoint is None:
                session = self.start_episode(
                    narrator,
                    context,
                    spec.initial_facts,
                    spec.commitments,
                )
                if on_checkpoint is not None:
                    on_checkpoint(
                        EpisodeCheckpoint(
                            session,
                            (),
                            narrator.checkpoint_state(),
                        )
                    )
            else:
                if checkpoint.session.context.episode_id != spec.id:
                    raise ValueError("Checkpoint does not match the selected specification")
                session = checkpoint.session
                narrator.restore_episode(context, checkpoint.narrator_state)

            for _ in range(len(turns), max(0, max_turns)):
                player_input = input_condition.next_input(
                    player_role=context.player_role,
                    history=session.state.history,
                )
                if not isinstance(player_input, str) or not player_input.strip():
                    raise RuntimeError("Player input condition returned no player input")

                result = self.run_turn(narrator, session, player_input)
                turns.append(result)
                if result.has_conflict:
                    return EpisodeTrace(session, tuple(turns), EpisodeTermination.CONFLICT)

                session = result.session
                if on_checkpoint is not None:
                    on_checkpoint(
                        EpisodeCheckpoint(
                            session,
                            tuple(turns),
                            narrator.checkpoint_state(),
                        )
                    )
                if all_achievement_commitments_resolved(session.state):
                    return EpisodeTrace(session, tuple(turns), EpisodeTermination.ALL_RESOLVED)

            return EpisodeTrace(session, tuple(turns), EpisodeTermination.MAX_TURNS)
        finally:
            self.close_episode(narrator)

    def run_turn(
        self,
        narrator: Narrator,
        session: EpisodeSession,
        player_input: str,
    ) -> EpisodeTurnResult:
        """Run one input through the exact evaluator/checker/state sequence."""

        state = session.state
        turn_id = _next_turn_id(state.history)
        narrator_request = NarratorRequest(
            turn_id=turn_id,
            player_role=session.context.player_role,
            player_input=player_input,
            history=state.history,
            active_facts=active_facts(state),
            commitments=state.commitments,
            trajectory=state.trajectory,
            current_node_id=_current_node_id(state.trajectory),
            pending_fact_updates=session.pending_fact_updates,
        )
        projected_facts = apply_fact_updates(state.facts, session.pending_fact_updates)
        audit_context = _build_audit_context(
            session.context.player_role,
            player_input,
            state.history,
            projected_facts,
            state.commitments,
            state.trajectory,
        )
        response = narrator.respond(narrator_request)
        evaluation = self._evaluator.evaluate(audit_context, response.text)
        completed_turn = StoryTurn(turn_id, player_input, response.text)

        if evaluation.conflict_decision.has_conflict:
            return EpisodeTurnResult(
                session=session,
                completed_turn=completed_turn,
                response=response,
                evaluation=evaluation,
                trajectory_assessments=(),
                commitment_assessments=(),
                newly_occurred_node_ids=(),
                new_satisfaction_ids=(),
            )

        projected_facts = apply_fact_updates(projected_facts, evaluation.fact_update)
        checker_history = (*state.history, completed_turn)
        trajectory_assessments = self._trajectory_checker.check(
            TrajectoryCheckRequest(
                player_role=session.context.player_role,
                active_facts=tuple(fact for fact in projected_facts if fact.active),
                history=checker_history,
                trajectory=state.trajectory,
            )
        )
        projected_trajectory, _ = _apply_trajectory_assessments(
            state.trajectory, trajectory_assessments
        )
        commitment_assessments = self._commitment_checker.check(
            CommitmentCheckRequest(
                player_role=session.context.player_role,
                active_facts=tuple(fact for fact in projected_facts if fact.active),
                history=checker_history,
                trajectory=projected_trajectory,
                commitments=state.commitments,
            )
        )
        transition = apply_turn_outcome(
            state,
            completed_turn,
            evaluation,
            trajectory_assessments,
            commitment_assessments,
            pending_fact_updates=session.pending_fact_updates,
        )
        return EpisodeTurnResult(
            session=EpisodeSession(session.context, transition.state),
            completed_turn=completed_turn,
            response=response,
            evaluation=evaluation,
            trajectory_assessments=tuple(trajectory_assessments),
            commitment_assessments=tuple(commitment_assessments),
            newly_occurred_node_ids=transition.newly_occurred_node_ids,
            new_satisfaction_ids=transition.new_satisfaction_ids,
        )


def _build_audit_context(
    player_role: str,
    player_input: str,
    history: Sequence[StoryTurn],
    projected_facts: Sequence[Fact],
    commitments: Sequence[Commitment],
    trajectory: Sequence[TrajectoryNode],
) -> TurnAuditContext:
    active_projected_facts = tuple(fact for fact in projected_facts if fact.active)
    return TurnAuditContext(
        player_role=player_role,
        player_input=player_input,
        history=history,
        active_facts=active_projected_facts,
        commitments=commitments,
        trajectory=trajectory,
    )


def _next_turn_id(history: Sequence[StoryTurn]) -> int:
    completed_ids = (turn.turn_id for turn in history if turn.turn_id >= 0)
    return max(completed_ids, default=-1) + 1


def _current_node_id(trajectory: Sequence[TrajectoryNode]) -> str | None:
    current_node: str | None = None
    for node in trajectory:
        if node.occurred:
            current_node = node.id
    return current_node
