from __future__ import annotations

import unittest
from pathlib import Path

from ncpbench.input_conditions import (
    AdversarialInputCondition,
    NaturalInputCondition,
    create_input_condition,
    format_visible_history,
)
from ncpbench.models import StoryTurn


ROOT = Path(__file__).resolve().parents[1]


class _RecordingClient:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.result


class InputConditionTests(unittest.TestCase):
    def test_natural_condition_preserves_the_message_and_history_shape(self) -> None:
        client = _RecordingClient("I follow the footsteps.")
        history = (
            StoryTurn(-1, None, "Rain rattles the shutters."),
            StoryTurn(0, "I listen at the door.", "A distant bell answers."),
        )

        result = NaturalInputCondition(client).next_input(player_role="Detective", history=history)

        history_text = (
            "Opening:\nRain rattles the shutters.\n"
            "Turn 0:\n- Player: I listen at the door.\n- System: A distant bell answers."
        )
        prompt = (ROOT / "src/ncpbench/prompts/natural_input.txt").read_text(encoding="utf-8")
        self.assertEqual(result, "I follow the footsteps.")
        self.assertEqual(
            client.calls,
            [[{"role": "user", "content": prompt.format(player_role="Detective", history_text=history_text)}]],
        )

    def test_adversarial_condition_preserves_the_first_turn_message(self) -> None:
        client = _RecordingClient("  I demand the locked archive now.\n")

        result = AdversarialInputCondition(client).next_input(player_role="Courier", history=())

        prompt = (ROOT / "src/ncpbench/prompts/adversarial_input.txt").read_text(encoding="utf-8")
        self.assertEqual(result, "  I demand the locked archive now.\n")
        self.assertEqual(
            client.calls,
            [[
                {
                    "role": "user",
                    "content": prompt.format(
                        player_role="Courier",
                        history_text="(The story has just begun)",
                    ),
                }
            ]],
        )

    def test_history_defaults_and_missing_visible_text_match_the_protocol(self) -> None:
        self.assertEqual(format_visible_history(()), "(The story has just begun)")
        self.assertEqual(
            format_visible_history((StoryTurn(-1, None, None), StoryTurn(3, None, None))),
            "Opening:\n(No opening narrative)\nTurn 3:\n- Player: (None)\n- System: (None)",
        )

    def test_factory_uses_the_natural_default_and_rejects_unknown_conditions(self) -> None:
        client = _RecordingClient("input")
        self.assertIsInstance(create_input_condition(None, client), NaturalInputCondition)
        self.assertIsInstance(create_input_condition("adversarial", client), AdversarialInputCondition)
        with self.assertRaisesRegex(ValueError, "Unknown input condition"):
            create_input_condition("unpublished", client)

    def test_input_conditions_have_no_narrator_or_evaluator_dependency(self) -> None:
        source = (ROOT / "src/ncpbench/input_conditions.py").read_text(encoding="utf-8")
        self.assertNotIn("ncpbench.narrator", source)
        self.assertNotIn("ncpbench.evaluator", source)
        self.assertNotIn("ncpbench.runner", source)


if __name__ == "__main__":
    unittest.main()
