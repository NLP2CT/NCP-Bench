from __future__ import annotations

import hashlib
import unittest
from typing import Sequence

from ncpbench.models import Commitment, Fact, TrajectoryNode
from ncpbench.narrator import NarratorResponse, OpeningRequest, render_opening_prompt
from ncpbench.opening import (
    OpeningEvaluationError,
    OpeningEvaluator,
    render_opening_fact_extract_prompt,
)


class _ScriptedAuditor:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, messages, *, stage: str) -> str:
        self.calls.append((stage, messages[0]["content"]))
        return next(self._responses)


class OpeningTests(unittest.TestCase):
    def test_narrator_opening_prompt_matches_the_frozen_protocol(self) -> None:
        prompt = render_opening_prompt(_request())

        self.assertIn("<player_role>\nCaptain\n</player_role>", prompt)
        self.assertIn("- node_id: s_0", prompt)
        self.assertIn("- f_0: The ship is at sea.", prompt)
        self.assertNotIn("f_1: A retired fact.", prompt)
        self.assertIn("type=achievement | status=pending", prompt)
        self.assertEqual(
            hashlib.sha256(prompt.encode()).hexdigest(),
            "ab8b976780cae595bebe6a879892c8204bb8680bfd7d2f5a1d3a72f65600838c",
        )

    def test_external_evaluator_extracts_deferred_opening_facts(self) -> None:
        auditor = _ScriptedAuditor(
            ['{"add_facts": ["A storm surrounds the ship."], "negate_facts": ["f_0"], "reason": "The opening establishes the storm."}']
        )

        updates = OpeningEvaluator(auditor).evaluate(
            _request(), NarratorResponse("Rain lashes the deck.")
        )

        self.assertEqual(updates.add_facts, ("A storm surrounds the ship.",))
        self.assertEqual(updates.negate_fact_ids, ("f_0",))
        self.assertEqual([stage for stage, _ in auditor.calls], ["opening_fact_extract"])
        prompt = auditor.calls[0][1]
        self.assertIn(
            "<latest_narrative_content>\nRain lashes the deck.\n</latest_narrative_content>",
            prompt,
        )
        self.assertEqual(
            hashlib.sha256(prompt.encode()).hexdigest(),
            "f1ebc38047f29b7383af282806e328d75d8553dabb3e363221bf28596b2a3866",
        )

    def test_invalid_fact_extraction_is_not_accepted(self) -> None:
        auditor = _ScriptedAuditor(["not JSON"] * 3)
        with self.assertRaisesRegex(OpeningEvaluationError, "after 3 attempts"):
            OpeningEvaluator(auditor).evaluate(
                _request(), NarratorResponse("A scene.")
            )
        self.assertEqual(len(auditor.calls), 3)

    def test_fact_extract_prompt_marks_an_empty_narrative_explicitly(self) -> None:
        prompt = render_opening_fact_extract_prompt(
            player_role="Player", opening_text="", pre_turn_facts_text="(none)"
        )
        self.assertIn(
            "<latest_narrative_content>\n(No narrative content available)\n</latest_narrative_content>",
            prompt,
        )


def _request() -> OpeningRequest:
    return OpeningRequest(
        player_role="Captain",
        active_facts=(
            Fact("f_0", "The ship is at sea."),
            Fact("f_1", "A retired fact.", active=False),
        ),
        commitments=(
            Commitment(
                "c_0",
                "achievement",
                "Reach safe harbor.",
                "The ship reaches harbor.",
                "The ship sinks.",
            ),
        ),
        trajectory=(
            TrajectoryNode("s_0", "The storm gathers.", "Dark clouds arrive.", "Rain begins."),
        ),
    )


if __name__ == "__main__":
    unittest.main()
