from __future__ import annotations

import unittest
from dataclasses import fields
from typing import Sequence

from ncpbench.evaluator import TurnAuditContext, TurnEvaluator
from ncpbench.evaluator_prompts import EvaluationPrompts
from ncpbench.model_output import ModelCallError
from ncpbench.models import Fact, StoryTurn
from ncpbench.narrator import NarratorResponse


class RecordedAuditor:
    """A fixed transcript used to test evaluator ordering without an API call."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.stages: list[str] = []

    def complete(self, _messages, *, stage: str) -> str:
        self.stages.append(stage)
        return next(self._responses)


PROMPTS = EvaluationPrompts(
    fact_update="facts {player_role} {pre_turn_facts} {response_text}",
    conflict_check="conflict {player_role} {system_response_history} {user_input} {response_text} {candidate_fact_updates} {trajectory_progress} {current_node} {pre_turn_facts} {narrative_commitments}",
    conflict_double_check="review {player_role} {system_response_history} {user_input} {response_text} {candidate_fact_updates} {trajectory_progress} {current_node} {pre_turn_facts} {narrative_commitments} {initial_conflicts_json}",
    trajectory_check="trajectory {player_role} {facts} {history_text} {current_node}",
    status_check="status {player_role} {facts} {history_text} {trajectory_progress} {narrative_commitments}",
)


class TurnEvaluatorTests(unittest.TestCase):
    def test_narrator_response_exposes_text_only(self) -> None:
        self.assertEqual([field.name for field in fields(NarratorResponse)], ["text"])

    def test_evaluator_runs_fact_conflict_and_double_check_externally(self) -> None:
        auditor = RecordedAuditor(
            [
                '{"add_facts": ["The door is locked."], "negate_facts": [], "reason": "Durable state."}',
                '{"fact_conflict_count": 1, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 1, "conflicts": [{"type": "fact", "id": "f_0", "reason": "The response contradicts the ledger."}]}',
                '{"confirmed": true, "fact_conflict_count": 1, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 1, "conflicts": [{"type": "fact", "id": "f_0", "reason": "The contradiction stands."}], "review_reason": "Confirmed."}',
            ]
        )
        result = TurnEvaluator(auditor, PROMPTS).evaluate(_context(), "The door opens without a key.")

        self.assertEqual(auditor.stages, ["method_response_fact_extract", "method_response_conflict_check", "conflict_double_check"])
        self.assertEqual(result.fact_update.add_facts, ("The door is locked.",))
        self.assertTrue(result.conflict_decision.has_conflict)
        self.assertTrue(result.conflict_decision.double_checked)

    def test_evaluator_skips_double_check_when_no_conflict_exists(self) -> None:
        auditor = RecordedAuditor(
            [
                '{"add_facts": [], "negate_facts": [], "reason": "No durable change."}',
                '{"fact_conflict_count": 0, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 0, "conflicts": []}',
            ]
        )
        result = TurnEvaluator(auditor, PROMPTS).evaluate(_context(), "I wait by the locked door.")

        self.assertEqual(auditor.stages, ["method_response_fact_extract", "method_response_conflict_check"])
        self.assertFalse(result.conflict_decision.has_conflict)
        self.assertFalse(result.conflict_decision.double_checked)

    def test_evaluator_retries_a_malformed_conflict_response_at_the_same_stage(self) -> None:
        auditor = RecordedAuditor(
            [
                '{"add_facts": [], "negate_facts": [], "reason": "No changes."}',
                '{"fact_conflict_count": -1, "total_conflict_count": -1, "conflicts": [{"type": "unknown", "id": "x", "reason": "invalid diagnostic"}]}',
                '{"fact_conflict_count": 0, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 0, "conflicts": []}',
            ]
        )

        result = TurnEvaluator(auditor, PROMPTS).evaluate(_context(), "I wait by the locked door.")

        self.assertEqual(
            auditor.stages,
            ["method_response_fact_extract", "method_response_conflict_check", "method_response_conflict_check"],
        )
        self.assertFalse(result.conflict_decision.has_conflict)

    def test_double_check_can_overturn_the_initial_conflict(self) -> None:
        auditor = RecordedAuditor(
            [
                '{"add_facts": [], "negate_facts": [], "reason": "No changes."}',
                '{"fact_conflict_count": 1, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 1, "conflicts": [{"type": "fact", "id": "f_0", "reason": "Possible contradiction."}]}',
                '{"confirmed": false, "fact_conflict_count": 0, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 0, "conflicts": [], "review_reason": "The transition is supported."}',
            ]
        )

        result = TurnEvaluator(auditor, PROMPTS).evaluate(_context(), "The lock remains closed.")

        self.assertEqual(
            auditor.stages,
            ["method_response_fact_extract", "method_response_conflict_check", "conflict_double_check"],
        )
        self.assertFalse(result.conflict_decision.has_conflict)
        self.assertTrue(result.conflict_decision.double_checked)
        self.assertEqual(result.conflict_decision.review_reason, "The transition is supported.")

    def test_invalid_output_fails_after_initial_attempt_plus_two_retries(self) -> None:
        auditor = RecordedAuditor(["not json"] * 3)

        with self.assertRaisesRegex(ModelCallError, "after 3 attempts"):
            TurnEvaluator(auditor, PROMPTS).evaluate(_context(), "The door opens.")

        self.assertEqual(auditor.stages, ["method_response_fact_extract"] * 3)


def _context() -> TurnAuditContext:
    return TurnAuditContext(
        player_role="Player",
        player_input="I try the door.",
        history=(StoryTurn(0, "I approach.", "A locked door blocks the hall."),),
        active_facts=(Fact("f_0", "The door is locked."),),
        commitments=(),
        trajectory=(),
    )


if __name__ == "__main__":
    unittest.main()
