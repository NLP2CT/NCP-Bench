from __future__ import annotations

import unittest
from typing import Sequence

from ncpbench.evaluator import CommitmentChecker, TrajectoryChecker, TurnEvaluator
from ncpbench.evaluator_prompts import EvaluationPrompts
from ncpbench.models import Commitment, Fact, PendingFactUpdates, TrajectoryNode
from ncpbench.narrator import EpisodeContext, Narrator, NarratorRequest, NarratorResponse, OpeningRequest
from ncpbench.opening import OpeningEvaluator
from ncpbench.runner import EpisodeRunner


class _NarratorFromAnotherProject(Narrator):
    def __init__(self, text: str) -> None:
        self._text = text
        self.started: list[EpisodeContext] = []
        self.opening_requests: list[OpeningRequest] = []
        self.requests: list[NarratorRequest] = []
        self.closed = False

    def start_episode(self, context: EpisodeContext) -> None:
        self.started.append(context)

    def open(self, request: OpeningRequest) -> NarratorResponse:
        self.opening_requests.append(request)
        return NarratorResponse("Rain rattles the shutters.")

    def respond(self, request: NarratorRequest) -> NarratorResponse:
        self.requests.append(request)
        return NarratorResponse(self._text)

    def close_episode(self) -> None:
        self.closed = True


class _ScriptedAuditor:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, messages, *, stage: str) -> str:
        self.calls.append((stage, messages[0]["content"]))
        return next(self._responses)


class EpisodeRunnerTests(unittest.TestCase):
    def test_runs_every_public_stage_and_commits_the_projected_turn(self) -> None:
        narrator = _NarratorFromAnotherProject("The key turns and the door opens.")
        auditor = _ScriptedAuditor(
            (
                '{"add_facts": ["A key is in the lock."], "negate_facts": [], "reason": "Opening state."}',
                '{"add_facts": ["The door is open."], "negate_facts": ["f_0"], "reason": "The door changed state."}',
                '{"fact_conflict_count": 0, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 0, "conflicts": []}',
                '{"trigger": {"occurred": true, "reason": "The key turns."}, "delta": {"occurred": true, "reason": "The door opens."}}',
                '{"statuses": [{"id": "c_0", "status": "SATISFIED", "reason": "The objective is complete."}]}',
            )
        )
        runner = _runner(auditor)
        context = EpisodeContext("episode-1", "Player", _trajectory())
        session = runner.start_episode(
            narrator,
            context,
            (Fact("f_0", "The door is closed."),),
            (_commitment(),),
        )

        result = runner.run_turn(narrator, session, "I turn the key.")
        runner.close_episode(narrator)

        self.assertEqual(narrator.started, [context])
        self.assertTrue(narrator.closed)
        self.assertEqual(len(narrator.opening_requests), 1)
        self.assertEqual(
            narrator.opening_requests[0].active_facts,
            (Fact("f_0", "The door is closed."),),
        )
        self.assertEqual(len(narrator.requests), 1)
        request = narrator.requests[0]
        self.assertEqual(request.turn_id, 0)
        self.assertEqual(request.history[0].narrator_response, "Rain rattles the shutters.")
        self.assertEqual(request.active_facts, (Fact("f_0", "The door is closed."),))
        self.assertEqual(request.pending_fact_updates.add_facts, ("A key is in the lock.",))
        self.assertEqual(request.current_node_id, "n_0")

        self.assertEqual(
            [stage for stage, _ in auditor.calls],
            [
                "opening_fact_extract",
                "method_response_fact_extract",
                "method_response_conflict_check",
                "trajectory_check",
                "status_check",
            ],
        )
        self.assertIn("f_1: A key is in the lock.", auditor.calls[1][1])
        self.assertIn("f_1: A key is in the lock.", auditor.calls[3][1])
        self.assertIn("f_2: The door is open.", auditor.calls[3][1])
        self.assertIn("- n_0: status=occurred", auditor.calls[4][1])

        self.assertFalse(result.has_conflict)
        self.assertEqual(
            result.session.state.facts,
            (
                Fact("f_0", "The door is closed.", active=False),
                Fact("f_1", "A key is in the lock."),
                Fact("f_2", "The door is open."),
            ),
        )
        self.assertEqual([node.occurred for node in result.session.state.trajectory], [True, False])
        self.assertEqual([turn.turn_id for turn in result.session.state.history], [-1, 0])
        self.assertEqual(result.newly_occurred_node_ids, ("n_0",))
        self.assertEqual(result.new_satisfaction_ids, ("c_0",))
        self.assertEqual(result.session.pending_fact_updates, PendingFactUpdates())

    def test_stops_before_checkers_and_state_commit_after_a_confirmed_conflict(self) -> None:
        narrator = _NarratorFromAnotherProject("The door vanishes.")
        auditor = _ScriptedAuditor(
            (
                '{"add_facts": ["A key is in the lock."], "negate_facts": [], "reason": "Opening state."}',
                '{"add_facts": ["The door vanished."], "negate_facts": ["f_0"], "reason": "The response changes the door."}',
                '{"fact_conflict_count": 1, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 1, "conflicts": [{"type": "fact", "id": "f_0", "reason": "The door cannot vanish."}]}',
                '{"confirmed": true, "fact_conflict_count": 1, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 1, "conflicts": [{"type": "fact", "id": "f_0", "reason": "The conflict stands."}], "review_reason": "Confirmed."}',
            )
        )
        runner = _runner(auditor)
        session = runner.start_episode(
            narrator,
            EpisodeContext("episode-2", "Player", _trajectory()),
            (Fact("f_0", "The door is closed."),),
            (_commitment(),),
        )

        result = runner.run_turn(narrator, session, "I erase the door.")

        self.assertTrue(result.has_conflict)
        self.assertEqual(
            [stage for stage, _ in auditor.calls],
            ["opening_fact_extract", "method_response_fact_extract", "method_response_conflict_check", "conflict_double_check"],
        )
        self.assertEqual(result.session, session)
        self.assertEqual(result.trajectory_assessments, ())
        self.assertEqual(result.commitment_assessments, ())
        self.assertEqual(result.completed_turn.narrator_response, "The door vanishes.")

def _runner(auditor: _ScriptedAuditor) -> EpisodeRunner:
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


def _trajectory() -> tuple[TrajectoryNode, ...]:
    return (
        TrajectoryNode("n_0", "The door blocks the hall.", "The key turns.", "The door opens."),
        TrajectoryNode("n_1", "The room is accessible.", "The door opens.", "The room can be entered."),
    )


def _commitment() -> Commitment:
    return Commitment(
        "c_0",
        "achievement",
        "Open the door.",
        "The door opens.",
        "The door stays closed.",
    )


if __name__ == "__main__":
    unittest.main()
