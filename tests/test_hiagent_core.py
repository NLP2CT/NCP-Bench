from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference-methods" / "hiagent" / "src"))

from ncpbench_reference_hiagent.config import HiAgentRuntimeConfig
from ncpbench_reference_hiagent.core import HiAgentCore
from ncpbench_reference_hiagent.summarizer import TrajectorySummarizer


class _ScriptedClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, tuple[Mapping[str, str], ...]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]], *, stage: str) -> str:
        self.calls.append((stage, tuple(messages)))
        return next(self._responses)


class HiAgentCoreTests(unittest.TestCase):
    def test_retrieve_requests_reenter_reasoning_with_the_same_memory(self) -> None:
        client = _ScriptedClient(
            (
                "Subgoal: Find the key.\nAction: retrieve(1)",
                "Action: I continue searching the locked room.",
            )
        )
        core = _core(client)
        core.reset(init_obs="Story initialized.")

        response = core.run(turn_context="Current player input: I search the room.")

        self.assertEqual(response, "I continue searching the locked room.")
        self.assertEqual(
            [stage for stage, _ in client.calls],
            ["method_response_generate", "hiagent_reasoning"],
        )
        self.assertIn("Subgoal: Find the key.", client.calls[1][1][1]["content"])

    def test_completed_subgoals_are_summarized_before_the_next_action(self) -> None:
        client = _ScriptedClient(
            (
                "Subgoal: Find the key.\nAction: I find the brass key.",
                "Subgoal: Reach the door.\nAction: I approach the locked door.",
                "The key was found and the first subgoal is complete.",
                "Action: I fit the key into the lock.",
            )
        )
        core = _core(client)
        core.reset(init_obs="Story initialized.")

        first = core.run(turn_context="Turn one.")
        core.update(first, "Player input observed: I move on.")
        second = core.run(turn_context="Turn two.")
        core.update(second, "Player input observed: I try the lock.")
        third = core.run(turn_context="Turn three.")

        self.assertEqual(third, "I fit the key into the lock.")
        self.assertEqual(
            [stage for stage, _ in client.calls],
            [
                "method_response_generate",
                "method_response_generate",
                "hiagent_summary",
                "method_response_generate",
            ],
        )
        third_prompt = client.calls[3][1][1]["content"]
        self.assertIn("1 Subgoal: Find the key.", third_prompt)
        self.assertIn("The key was found and the first subgoal is complete.", third_prompt)
        self.assertIn("2 Subgoal: Reach the door.", third_prompt)


def _core(client: _ScriptedClient) -> HiAgentCore:
    config = HiAgentRuntimeConfig()
    return HiAgentCore(
        client=client,
        config=config,
        summarizer=TrajectorySummarizer(client),
    )


if __name__ == "__main__":
    unittest.main()
