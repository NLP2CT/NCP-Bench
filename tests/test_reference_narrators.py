from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "reference-methods" / "baseline" / "src"),
    str(ROOT / "reference-methods" / "hiagent" / "src"),
]

from ncpbench.models import Commitment, Fact, PendingFactUpdates, StoryTurn, TrajectoryNode
from ncpbench.model_output import ModelCallError
from ncpbench.narrator import EpisodeContext, Narrator, NarratorRequest, OpeningRequest
from ncpbench_reference_baseline import BaselineNarrator
from ncpbench_reference_baseline.narrator import build_generation_prompt
from ncpbench_reference_hiagent import HiAgentNarrator
from ncpbench_reference_hiagent.hiagent_prompts import build_hiagent_turn_context


class ScriptedGenerator:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, tuple[Mapping[str, str], ...]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]], *, stage: str) -> str:
        self.calls.append((stage, tuple(messages)))
        return next(self._responses)


class ReferenceNarratorTests(unittest.TestCase):
    def test_each_reference_narrator_generates_its_own_opening(self) -> None:
        baseline_generator = ScriptedGenerator(['{"text": "Baseline opening."}'])
        baseline = BaselineNarrator(baseline_generator)
        self.assertEqual(baseline.open(_opening_request()).text, "Baseline opening.")

        hiagent_generator = ScriptedGenerator(['{"text": "HiAgent opening."}'])
        hiagent = HiAgentNarrator(hiagent_generator)
        hiagent.start_episode(EpisodeContext("movie00", "Jason", _request().trajectory))
        self.assertEqual(hiagent.open(_opening_request()).text, "HiAgent opening.")

        for generator in (baseline_generator, hiagent_generator):
            self.assertEqual([stage for stage, _ in generator.calls], ["opening"])
            self.assertEqual(
                hashlib.sha256(generator.calls[0][1][0]["content"].encode()).hexdigest(),
                "8e4ffe4ba06856b7683a4659b78262ae0b2467754186bafc2a696105ba0ed03c",
            )

    def test_baseline_is_an_ordinary_text_only_narrator(self) -> None:
        generator = ScriptedGenerator(['{"text": "The lock resists your hand."}'])
        narrator = BaselineNarrator(generator)

        response = narrator.respond(_request())

        self.assertIsInstance(narrator, Narrator)
        self.assertEqual(response.text, "The lock resists your hand.")
        self.assertEqual([stage for stage, _ in generator.calls], ["method_response_generate"])

    def test_baseline_rejects_invalid_or_textless_generation_outputs(self) -> None:
        for response in ("not json", "{}"):
            with self.subTest(response=response):
                generator = ScriptedGenerator([response] * 3)
                with self.assertRaisesRegex(ModelCallError, "after 3 attempts"):
                    BaselineNarrator(generator).respond(_request())
                self.assertEqual(len(generator.calls), 3)

    def test_hiagent_uses_the_same_narrator_boundary_and_keeps_memory_locally(self) -> None:
        generator = ScriptedGenerator(
            [
                "Subgoal: Establish the blocked doorway.\nAction: Rain hammers the locked door.",
                "Action: The lock clicks once, then holds fast.",
            ]
        )
        narrator = HiAgentNarrator(generator)
        episode = EpisodeContext("movie00", "Jason", _request().trajectory)
        narrator.start_episode(episode)

        first = narrator.respond(_request())
        second = narrator.respond(_request(player_input="I force the lock."))

        self.assertIsInstance(narrator, Narrator)
        self.assertEqual(first.text, "Rain hammers the locked door.")
        self.assertEqual(second.text, "The lock clicks once, then holds fast.")
        self.assertEqual([stage for stage, _ in generator.calls], ["method_response_generate", "method_response_generate"])
        self.assertIn("Player input observed: I force the lock.", generator.calls[1][1][1]["content"])

    def test_generation_prompts_match_the_frozen_paper_artifacts(self) -> None:
        request = _request()

        self.assertEqual(
            hashlib.sha256(build_generation_prompt(request).encode("utf-8")).hexdigest(),
            "f25d50e582da95f423edc006af1734f0bddc2356b18aebcffcff1e133cd2d446",
        )
        self.assertEqual(
            hashlib.sha256(build_hiagent_turn_context(request).encode("utf-8")).hexdigest(),
            "e1abf44c17e2a45be9e5b335366d9ecc53ce9bcaa3a8330bfd9cb24bfcb3de53",
        )

        for package in ("baseline", "hiagent"):
            artifact = (
                ROOT
                / "reference-methods"
                / package
                / "src"
                / f"ncpbench_reference_{package}"
                / "prompts"
                / "method_response_generate_user.txt"
            )
            self.assertEqual(
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "5467b46ceb9df38a31734705108301a26c6478d05138a2c63d7696169ec96076",
            )

    def test_reference_methods_do_not_depend_on_evaluator_code(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for package in ("baseline", "hiagent")
            for path in (ROOT / "reference-methods" / package / "src").rglob("*.py")
        )

        for forbidden in ("ResponseGuard", "TurnEvaluator", "AuditorClient", "ncpbench.evaluation"):
            self.assertNotIn(forbidden, source)


def _request(player_input: str = "I inspect the lock.") -> NarratorRequest:
    return NarratorRequest(
        turn_id=1,
        player_role="Jason",
        player_input=player_input,
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
            TrajectoryNode("n0", "A door blocks the hall.", "The key is revealed.", "The door opens.", True),
            TrajectoryNode("n1", "The room is searched.", "A note appears.", "The clue is secured."),
        ),
        current_node_id="n1",
        pending_fact_updates=PendingFactUpdates(("The lock is rusted.",), ("f_0",)),
    )

def _opening_request() -> OpeningRequest:
    request = _request()
    return OpeningRequest(
        player_role=request.player_role,
        active_facts=request.active_facts,
        commitments=request.commitments,
        trajectory=request.trajectory,
    )
