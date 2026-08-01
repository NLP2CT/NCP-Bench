from __future__ import annotations

import hashlib
import unittest
from typing import Sequence

from ncpbench.evaluator import (
    CommitmentAssessment,
    CommitmentChecker,
    CommitmentCheckRequest,
    TrajectoryChecker,
    TrajectoryCheckRequest,
)
from ncpbench.models import Commitment, Fact, StoryTurn, TrajectoryNode
from ncpbench.model_output import ModelCallError


class ScriptedAuditor:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, tuple[dict[str, str], ...]]] = []

    def complete(self, messages, *, stage: str) -> str:
        self.calls.append((stage, tuple(dict(message) for message in messages)) )
        return next(self._responses)


class StateCheckerTests(unittest.TestCase):
    def test_trajectory_checker_preserves_prompt_and_one_step_transition_rule(self) -> None:
        auditor = ScriptedAuditor(
            ['{"trigger": {"occurred": true, "reason": "A note appears."}, "delta": {"occurred": true, "reason": "The clue is secured."}}']
        )

        assessment = TrajectoryChecker(auditor).check(_trajectory_request())

        self.assertEqual(assessment[0].target_node_id, "n1")
        self.assertTrue(assessment[0].occurred)
        self.assertEqual(assessment[0].reason, "trigger: A note appears. | delta: The clue is secured.")
        self.assertEqual([stage for stage, _ in auditor.calls], ["trajectory_check"])
        self.assertEqual(
            hashlib.sha256(auditor.calls[0][1][0]["content"].encode("utf-8")).hexdigest(),
            "bdac32c8f48948b6936b43b6f4548d0959edcf8fb00f21467b4ec1341131b01e",
        )

    def test_trajectory_checker_rejects_invalid_audit_after_two_retries(self) -> None:
        invalid_auditor = ScriptedAuditor(["not json"] * 3)
        with self.assertRaisesRegex(ModelCallError, "after 3 attempts"):
            TrajectoryChecker(invalid_auditor).check(_trajectory_request())
        self.assertEqual(len(invalid_auditor.calls), 3)

    def test_trajectory_checker_skips_a_completed_trajectory(self) -> None:

        final_auditor = ScriptedAuditor([])
        final = TrajectoryChecker(final_auditor).check(
            TrajectoryCheckRequest(
                "Jason",
                _facts(),
                _history(),
                (
                    TrajectoryNode("n0", "A door blocks the hall.", "The key is revealed.", "The door opens.", True),
                    TrajectoryNode("n1", "The room is searched.", "A note appears.", "The clue is secured.", True),
                ),
            )
        )
        self.assertEqual(final, ())
        self.assertEqual(final_auditor.calls, [])

    def test_commitment_checker_retries_invalid_status(self) -> None:
        auditor = ScriptedAuditor(
            [
                '{"statuses": ['
                '{"id": "c0", "status": "SATISFIED", "reason": "The key was used."}, '
                '{"id": "c1", "status": "UNKNOWN", "reason": "Malformed label."}'
                ']}',
                '{"statuses": ['
                '{"id": "c0", "status": "SATISFIED", "reason": "The key was used."}, '
                '{"id": "c1", "status": "PENDING", "reason": "The note is not secured."}'
                ']}',
            ]
        )
        request = CommitmentCheckRequest("Jason", _facts(), _history(), _trajectory(), _commitments())

        assessments = CommitmentChecker(auditor).check(request)

        self.assertEqual(
            assessments,
            (
                CommitmentAssessment("c0", "satisfied", "The key was used."),
                CommitmentAssessment("c1", "pending", "The note is not secured."),
            ),
        )
        self.assertEqual([stage for stage, _ in auditor.calls], ["status_check"] * 2)
        self.assertEqual(
            hashlib.sha256(auditor.calls[0][1][0]["content"].encode("utf-8")).hexdigest(),
            "ae4674c7d95c01c9b06c6caaeb0e98f5e35afa8a3a8c29b32e7587334243fc90",
        )

    def test_commitment_checker_skips_empty_commitment_sets(self) -> None:
        auditor = ScriptedAuditor([])
        request = CommitmentCheckRequest("Jason", _facts(), _history(), _trajectory(), ())

        self.assertEqual(CommitmentChecker(auditor).check(request), ())
        self.assertEqual(auditor.calls, [])

    def test_commitment_checker_rejects_a_missing_id_after_two_retries(self) -> None:
        auditor = ScriptedAuditor(
            ['{"statuses": [{"id": "c0", "status": "SATISFIED", "reason": "The key was used."}]}'] * 3
        )
        commitments = (
            Commitment("c0", "invariant", "The door remains locked until the key is used.", "The key is used.", "The door opens without a key."),
            Commitment("c1", "achievement", "Recover the note.", "The note is secured.", "The note is destroyed.", "satisfied"),
        )

        with self.assertRaisesRegex(ModelCallError, "after 3 attempts"):
            CommitmentChecker(auditor).check(
                CommitmentCheckRequest("Jason", _facts(), _history(), _trajectory(), commitments)
            )


def _facts() -> tuple[Fact, ...]:
    return (Fact("f_0", "The door is locked."),)


def _history() -> tuple[StoryTurn, ...]:
    return (
        StoryTurn(-1, None, "Rain rattles the shutters."),
        StoryTurn(0, "I inspect the lock.", "The lock is cold."),
    )


def _trajectory() -> tuple[TrajectoryNode, ...]:
    return (
        TrajectoryNode("n0", "A door blocks the hall.", "The key is revealed.", "The door opens.", True),
        TrajectoryNode("n1", "The room is searched.", "A note appears.", "The clue is secured."),
    )


def _commitments() -> tuple[Commitment, ...]:
    return (
        Commitment("c0", "invariant", "The door remains locked until the key is used.", "The key is used.", "The door opens without a key."),
        Commitment("c1", "achievement", "Recover the note.", "The note is secured.", "The note is destroyed."),
    )


def _trajectory_request() -> TrajectoryCheckRequest:
    return TrajectoryCheckRequest("Jason", _facts(), _history(), _trajectory())
