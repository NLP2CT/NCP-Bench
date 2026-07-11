from __future__ import annotations

import json
import unittest

from ncpbench.evaluator import (
    CommitmentAssessment,
    Conflict,
    ConflictDecision,
    FactUpdate,
    TrajectoryAssessment,
    TurnEvaluation,
)
from ncpbench.models import Commitment, Fact, StoryTurn, TrajectoryNode
from ncpbench.narrator import EpisodeContext, NarratorResponse
from ncpbench.results import episode_trace_to_result
from ncpbench.runner import EpisodeSession, EpisodeState, EpisodeTermination, EpisodeTrace, EpisodeTurnResult


class EpisodeResultTests(unittest.TestCase):
    def test_result_contains_stable_terminal_turn_and_state_fields(self) -> None:
        result = episode_trace_to_result(_trace())

        self.assertNotIn("schema_version", result)
        self.assertEqual(result["episode"], {"id": "movie00", "player_role": "Tony Stark"})
        self.assertEqual(result["termination"], "conflict")
        self.assertEqual(
            result["turns"],
            [
                {
                    "turn_id": 0,
                    "player_input": "I break the rules.",
                    "narrator_response": "The impossible happens.",
                    "evaluation": {
                        "fact_update": {
                            "add_facts": ["An impossible fact."],
                            "negate_fact_ids": ["f_0"],
                            "reason": "The narration adds it.",
                        },
                        "conflict": {
                            "has_conflict": True,
                            "fact_count": 1,
                            "commitment_count": 0,
                            "player_input_count": 0,
                            "total_count": 1,
                            "conflicts": [
                                {
                                    "kind": "fact",
                                    "target_id": "f_0",
                                    "reason": "It contradicts the ledger.",
                                }
                            ],
                            "double_checked": True,
                            "review_reason": "Confirmed.",
                        },
                    },
                    "trajectory_assessments": [
                        {"target_node_id": "n_1", "occurred": True, "reason": "Would advance."}
                    ],
                    "commitment_assessments": [
                        {"commitment_id": "c_0", "status": "pending", "reason": "Not met."}
                    ],
                    "newly_occurred_node_ids": [],
                    "new_satisfaction_ids": [],
                }
            ],
        )
        self.assertEqual(
            result["final_state"],
            {
                "facts": [{"id": "f_0", "text": "The door is closed.", "active": True}],
                "commitments": [
                    {
                        "id": "c_0",
                        "kind": "achievement",
                        "description": "Open the door.",
                        "satisfaction_condition": "The door opens.",
                        "violation_condition": "The door remains closed.",
                        "status": "pending",
                    }
                ],
                "trajectory": [
                    {
                        "id": "n_0",
                        "description": "At the door.",
                        "trigger_event": "A key turns.",
                        "key_delta": "The door opens.",
                        "occurred": True,
                    }
                ],
                "history": [{"turn_id": -1, "player_input": None, "narrator_response": "Rain falls."}],
            },
        )

    def test_result_is_json_serializable_and_deterministic(self) -> None:
        first = episode_trace_to_result(_trace())
        second = episode_trace_to_result(_trace())

        first_json = json.dumps(first, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(first_json, json.dumps(second, ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(json.loads(first_json), first)


def _trace() -> EpisodeTrace:
    state = EpisodeState(
        facts=(Fact("f_0", "The door is closed."),),
        commitments=(
            Commitment(
                "c_0",
                "achievement",
                "Open the door.",
                "The door opens.",
                "The door remains closed.",
            ),
        ),
        trajectory=(TrajectoryNode("n_0", "At the door.", "A key turns.", "The door opens.", True),),
        history=(StoryTurn(-1, None, "Rain falls."),),
    )
    session = EpisodeSession(
        EpisodeContext("movie00", "Tony Stark", state.trajectory),
        state,
    )
    evaluation = TurnEvaluation(
        FactUpdate(("An impossible fact.",), ("f_0",), "The narration adds it."),
        ConflictDecision(
            1,
            0,
            0,
            1,
            (Conflict("fact", "f_0", "It contradicts the ledger."),),
            True,
            "Confirmed.",
        ),
    )
    turn = EpisodeTurnResult(
        session,
        StoryTurn(0, "I break the rules.", "The impossible happens."),
        NarratorResponse("The impossible happens."),
        evaluation,
        (TrajectoryAssessment("n_1", True, "Would advance."),),
        (CommitmentAssessment("c_0", "pending", "Not met."),),
        (),
        (),
    )
    return EpisodeTrace(session, (turn,), EpisodeTermination.CONFLICT)


if __name__ == "__main__":
    unittest.main()
