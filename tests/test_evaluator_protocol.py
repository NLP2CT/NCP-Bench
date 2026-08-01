from __future__ import annotations

import hashlib
import json
import unittest
from importlib.resources import files

from ncpbench.evaluator import FactUpdate, TurnAuditContext, TurnEvaluator
from ncpbench.evaluator_prompts import (
    load_evaluation_prompts,
    render_conflict_check,
    render_conflict_double_check,
    render_fact_update,
)
from ncpbench.models import Commitment, Fact, StoryTurn, TrajectoryNode


class RecordedAuditor:
    def __init__(self) -> None:
        self.stages: list[str] = []

    def complete(self, _messages, *, stage: str) -> str:
        self.stages.append(stage)
        if stage == "method_response_fact_extract":
            return '{"add_facts": [], "negate_facts": [], "reason": "No durable changes."}'
        return '{"fact_conflict_count": 0, "commitment_conflict_count": 0, "player_input_conflict_count": 0, "total_conflict_count": 0, "conflicts": []}'


class EvaluatorProtocolTests(unittest.TestCase):
    def test_frozen_prompt_files_match_the_paper_artifacts(self) -> None:
        expected_hashes = {
            "fact_update.txt": "14bf0a31e51587517a7d270d7ffa74e5083d8dc06e2414154a273e16272a72b4",
            "conflict_check.txt": "2b8761e16262a87745b28e8414fd3f47a8f8efe5c7d5e412be23787804806998",
            "conflict_double_check.txt": "353e43468819f87f52ba6516f60abe5c38198fc5f532e71ec7f83eea26b897c3",
        }
        prompt_dir = files("ncpbench.prompts")

        for filename, expected_hash in expected_hashes.items():
            self.assertEqual(hashlib.sha256(prompt_dir.joinpath(filename).read_bytes()).hexdigest(), expected_hash)

    def test_paper_prompt_rendering_is_unchanged_by_structured_state(self) -> None:
        prompts = load_evaluation_prompts()
        context = _context()
        update = FactUpdate(("The lock is rusted.",), ("f_0",), None)
        initial_conflicts = json.dumps(
            [
                {
                    "type": "fact",
                    "id": "f_0",
                    "reason": "The lock contradiction.",
                    "content": "The door is locked.",
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
        rendered = (
            render_fact_update(prompts, context, "The lock resists your hand."),
            render_conflict_check(prompts, context, "The lock resists your hand.", update),
            render_conflict_double_check(
                prompts,
                context,
                "The lock resists your hand.",
                update,
                initial_conflicts,
            ),
        )
        expected_hashes = (
            "c1935d2477b888b6a68a7447f8ad8c4afb5e6eb9dc7fa143e25f68aa935c4cfc",
            "995961d21c99c637ff3ed941e7795a0071b509776ad05d589eae4b288ed17e58",
            "040dc92801505c3d8060852b373cdde7b33b9925d76927e4a66a8ccc268553b7",
        )

        self.assertEqual(tuple(hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in rendered), expected_hashes)

    def test_default_evaluator_uses_the_frozen_prompt_protocol(self) -> None:
        auditor = RecordedAuditor()

        result = TurnEvaluator(auditor).evaluate(_context(), "The lock resists your hand.")

        self.assertFalse(result.conflict_decision.has_conflict)
        self.assertEqual(auditor.stages, ["method_response_fact_extract", "method_response_conflict_check"])


def _context() -> TurnAuditContext:
    return TurnAuditContext(
        player_role="Jason",
        player_input="I inspect the lock.",
        history=(
            StoryTurn(-1, None, "Rain rattles the shutters."),
            StoryTurn(0, "I inspect the lock.", "The lock is cold."),
        ),
        active_facts=(Fact("f_0", "The door is locked."),),
        commitments=(
            Commitment(
                "c0",
                "invariant",
                "The door remains locked until the key is used.",
                "The key is used.",
                "The door opens without a key.",
            ),
        ),
        trajectory=(
            TrajectoryNode(
                "n0",
                "A door blocks the hall.",
                "The key is revealed.",
                "The door opens.",
                True,
            ),
            TrajectoryNode(
                "n1",
                "The room is searched.",
                "A note appears.",
                "The clue is secured.",
            ),
        ),
    )
