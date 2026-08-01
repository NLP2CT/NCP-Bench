from __future__ import annotations

import unittest
from typing import Literal

from ncpbench.evaluator import (
    CommitmentAssessment,
    ConflictDecision,
    FactUpdate,
    TrajectoryAssessment,
    TurnEvaluation,
)
from ncpbench.models import Commitment, Fact, PendingFactUpdates, StoryTurn, TrajectoryNode
from ncpbench.runner import (
    active_facts,
    all_achievement_commitments_resolved,
    apply_fact_updates,
    apply_turn_outcome,
    initialize_episode_state,
)


class EpisodeStateTests(unittest.TestCase):
    def test_initialization_leaves_every_trajectory_node_pending(self) -> None:
        state = initialize_episode_state(
            (Fact("f_0", "A sealed door."),),
            (_commitment("c_0", kind="invariant"),),
            (_node("t_0"), _node("t_1")),
        )

        self.assertEqual([node.occurred for node in state.trajectory], [False, False])
        self.assertEqual(state.facts, (Fact("f_0", "A sealed door."),))

    def test_fact_updates_keep_negated_facts_and_stable_collision_rule(self) -> None:
        facts = (Fact("f_0", "An old fact."), Fact("f_2", "A separately numbered fact."))

        result = apply_fact_updates(
            facts,
            PendingFactUpdates(
                add_facts=("  A newly learned fact.  ", "   "),
                negate_fact_ids=(" f_0 ", "unknown"),
            ),
        )

        self.assertEqual(
            result,
            (
                Fact("f_0", "An old fact.", active=False),
                Fact("f_2", "A separately numbered fact."),
                Fact("f_2_2", "A newly learned fact."),
            ),
        )
        self.assertEqual(active_facts(initialize_episode_state(result, (), ())), result[1:])

    def test_turn_transition_commits_facts_trajectory_and_commitments_in_order(self) -> None:
        state = initialize_episode_state(
            (Fact("f_0", "The door is closed."),),
            (_commitment("c_achievement", kind="achievement"), _commitment("c_invariant", kind="invariant")),
            (_node("t_0"), _node("t_1")),
        )
        turn = StoryTurn(0, "I unlock the door.", "The door opens.")

        transition = apply_turn_outcome(
            state,
            turn,
            _evaluation(add_facts=("The door is open.",), negate_fact_ids=("f_0",)),
            (TrajectoryAssessment("t_0", True, "The change occurred."),),
            (
                CommitmentAssessment("c_achievement", "satisfied", "Goal met."),
                CommitmentAssessment("c_invariant", "pending", "Still active."),
            ),
            pending_fact_updates=PendingFactUpdates(add_facts=("A key is in the lock.",)),
        )

        self.assertEqual(
            transition.state.facts,
            (
                Fact("f_0", "The door is closed.", active=False),
                Fact("f_1", "A key is in the lock."),
                Fact("f_2", "The door is open."),
            ),
        )
        self.assertEqual([node.occurred for node in transition.state.trajectory], [True, False])
        self.assertEqual(transition.state.history, (turn,))
        self.assertEqual(transition.newly_occurred_node_ids, ("t_0",))
        self.assertEqual(transition.new_satisfaction_ids, ("c_achievement",))
        self.assertTrue(all_achievement_commitments_resolved(transition.state))

    def test_all_resolved_requires_at_least_one_achievement_commitment(self) -> None:
        no_achievement = initialize_episode_state((), (_commitment("c_0", kind="invariant"),), ())
        self.assertFalse(all_achievement_commitments_resolved(no_achievement))

    def test_transition_rejects_incomplete_commitment_audits(self) -> None:
        state = initialize_episode_state((), (_commitment("c_0", kind="achievement"),), ())

        with self.assertRaisesRegex(ValueError, "Missing commitment assessments"):
            apply_turn_outcome(state, StoryTurn(0, "Wait.", "Nothing changes."), _evaluation(), (), ())


def _node(node_id: str) -> TrajectoryNode:
    return TrajectoryNode(node_id, f"Node {node_id}", "event", "delta")


def _commitment(
    commitment_id: str, *, kind: Literal["invariant", "achievement", "ordering"]
) -> Commitment:
    return Commitment(
        id=commitment_id,
        kind=kind,
        description="A benchmark commitment.",
        satisfaction_condition="It is met.",
        violation_condition="It is violated.",
    )


def _evaluation(
    *, add_facts: tuple[str, ...] = (), negate_fact_ids: tuple[str, ...] = ()
) -> TurnEvaluation:
    return TurnEvaluation(
        fact_update=FactUpdate(add_facts, negate_fact_ids, None),
        conflict_decision=ConflictDecision(0, 0, 0, 0, (), False),
    )


if __name__ == "__main__":
    unittest.main()
