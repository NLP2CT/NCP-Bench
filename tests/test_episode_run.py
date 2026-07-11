from __future__ import annotations

import unittest
from typing import Sequence

from ncpbench.dataset import StorySpec
from ncpbench.evaluator import CommitmentChecker, TrajectoryChecker, TurnEvaluator
from ncpbench.evaluator_prompts import EvaluationPrompts
from ncpbench.models import Commitment, Fact, PendingFactUpdates, StoryTurn
from ncpbench.narrator import EpisodeContext, Narrator, NarratorRequest, NarratorResponse, OpeningRequest
from ncpbench.opening import OpeningEvaluator
from ncpbench.runner import EpisodeRunner, EpisodeTermination


class _Narrator(Narrator):
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.requests: list[NarratorRequest] = []
        self.started: list[EpisodeContext] = []
        self.opening_requests: list[OpeningRequest] = []
        self.closed = False

    def start_episode(self, context: EpisodeContext) -> None:
        self.started.append(context)

    def open(self, request: OpeningRequest) -> NarratorResponse:
        self.opening_requests.append(request)
        return NarratorResponse("Rain rattles the shutters.")

    def respond(self, request: NarratorRequest) -> NarratorResponse:
        self.requests.append(request)
        return NarratorResponse(next(self._responses))

    def close_episode(self) -> None:
        self.closed = True


class _InputCondition:
    name = "scripted"

    def __init__(self, inputs: Sequence[str]) -> None:
        self._inputs = iter(inputs)
        self.calls: list[tuple[str, tuple[StoryTurn, ...]]] = []

    def next_input(self, *, player_role: str, history: Sequence[StoryTurn]) -> str:
        self.calls.append((player_role, tuple(history)))
        return next(self._inputs)


class _Auditor:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.stages: list[str] = []

    def complete(self, messages, *, stage: str) -> str:
        self.stages.append(stage)
        return next(self._responses)


class EpisodeRunTests(unittest.TestCase):
    def test_drives_inputs_from_visible_history_through_the_existing_runner(self) -> None:
        narrator = _Narrator(("The lock remains closed.", "The door opens."))
        condition = _InputCondition(("I wait.", "I turn the key."))
        auditor = _Auditor(
            (
                '{"add_facts": ["A key is in the lock."], "negate_facts": [], "reason": "Opening state."}',
                '{"add_facts": [], "negate_facts": [], "reason": "No changes."}',
                '{"fact_conflict_count": 0, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 0, "conflicts": []}',
                '{"statuses": [{"id": "c_0", "status": "PENDING", "reason": "Not yet."}]}',
                '{"add_facts": [], "negate_facts": [], "reason": "No changes."}',
                '{"fact_conflict_count": 0, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 0, "conflicts": []}',
                '{"statuses": [{"id": "c_0", "status": "SATISFIED", "reason": "Done."}]}',
            )
        )

        trace = _runner(auditor).run(narrator, _spec(), condition)

        self.assertEqual(trace.termination, EpisodeTermination.ALL_RESOLVED)
        self.assertEqual([turn.completed_turn.player_input for turn in trace.turns], ["I wait.", "I turn the key."])
        self.assertEqual([role for role, _ in condition.calls], ["Player", "Player"])
        self.assertEqual(condition.calls[0][1], (StoryTurn(-1, None, "Rain rattles the shutters."),))
        self.assertEqual(
            condition.calls[1][1],
            (
                StoryTurn(-1, None, "Rain rattles the shutters."),
                StoryTurn(0, "I wait.", "The lock remains closed."),
            ),
        )
        self.assertEqual(narrator.requests[0].pending_fact_updates.add_facts, ("A key is in the lock.",))
        self.assertEqual(narrator.requests[1].pending_fact_updates, PendingFactUpdates())
        self.assertTrue(narrator.closed)

    def test_stops_before_requesting_another_input_after_a_conflict(self) -> None:
        narrator = _Narrator(("The door vanishes.",))
        condition = _InputCondition(("I erase the door.", "must not be consumed"))
        auditor = _Auditor(
            (
                '{"add_facts": [], "negate_facts": [], "reason": "No opening changes."}',
                '{"add_facts": [], "negate_facts": [], "reason": "No turn changes."}',
                '{"fact_conflict_count": 1, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 1, "conflicts": [{"type": "fact", "id": "f_0", "reason": "Impossible."}]}',
                '{"confirmed": true, "fact_conflict_count": 1, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 1, "conflicts": [{"type": "fact", "id": "f_0", "reason": "Confirmed."}], "review_reason": "Confirmed."}',
            )
        )

        trace = _runner(auditor).run(narrator, _spec(), condition)

        self.assertEqual(trace.termination, EpisodeTermination.CONFLICT)
        self.assertEqual(len(condition.calls), 1)
        self.assertEqual(trace.session.state.facts, (Fact("f_0", "The door is closed."),))
        self.assertTrue(narrator.closed)

    def test_rejects_a_missing_generated_input_and_closes_the_narrator(self) -> None:
        narrator = _Narrator(("must not be used",))
        condition = _InputCondition((None,))  # type: ignore[arg-type]
        auditor = _Auditor(('{"add_facts": [], "negate_facts": [], "reason": "No changes."}',))

        with self.assertRaisesRegex(RuntimeError, "returned no player input"):
            _runner(auditor).run(narrator, _spec(), condition)

        self.assertEqual(narrator.requests, [])
        self.assertTrue(narrator.closed)


def _runner(auditor: _Auditor) -> EpisodeRunner:
    prompts = EvaluationPrompts(
        fact_update="facts {player_role} {pre_turn_facts} {response_text}",
        conflict_check="conflict {player_role} {system_response_history} {user_input} {response_text} {candidate_fact_updates} {trajectory_progress} {current_node} {pre_turn_facts} {narrative_commitments}",
        conflict_double_check="review {player_role} {system_response_history} {user_input} {response_text} {candidate_fact_updates} {trajectory_progress} {current_node} {pre_turn_facts} {narrative_commitments} {initial_conflicts_json}",
        trajectory_check="trajectory {player_role} {facts} {history_text} {current_node}",
        status_check="status {player_role} {facts} {history_text} {trajectory_progress} {narrative_commitments}",
    )
    return EpisodeRunner(
        TurnEvaluator(auditor, prompts),
        TrajectoryChecker(auditor, prompts),
        CommitmentChecker(auditor, prompts),
        OpeningEvaluator(auditor),
    )


def _spec() -> StorySpec:
    return StorySpec(
        id="episode-1",
        title="The Door",
        genres=("mystery",),
        player_role="Player",
        synopsis="A locked door.",
        initial_facts=(Fact("f_0", "The door is closed."),),
        commitments=(
            Commitment(
                "c_0",
                "achievement",
                "Open the door.",
                "The door opens.",
                "The door stays closed.",
            ),
        ),
        trajectory=(),
    )


if __name__ == "__main__":
    unittest.main()
