"""Method-neutral player-input conditions for NCP-Bench episodes.

These conditions generate only the next player utterance from the visible
story history. They do not receive facts, commitments, trajectory state, a
narrator, or evaluator outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol, Sequence

from ncpbench.model_output import call_with_retries, require_text
from ncpbench.models import StoryTurn


class InputTextGenerator(Protocol):
    """An injected client that completes one traditional chat request."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the generated text for ``messages``."""


class PlayerInputCondition(Protocol):
    """A public source of one next player input for an episode state."""

    name: str

    def next_input(self, *, player_role: str, history: Sequence[StoryTurn]) -> str:
        """Generate the next player utterance from visible history only."""


@dataclass(frozen=True)
class InputConditionPrompts:
    """The frozen prompt texts for NCP-Bench player-input conditions."""

    natural: str
    adversarial: str


def load_input_condition_prompts() -> InputConditionPrompts:
    """Load the frozen natural-play and adversarial-play prompt texts."""

    prompt_dir = files("ncpbench.prompts")
    return InputConditionPrompts(
        natural=prompt_dir.joinpath("natural_input.txt").read_text(encoding="utf-8"),
        adversarial=prompt_dir.joinpath("adversarial_input.txt").read_text(encoding="utf-8"),
    )


class NaturalInputCondition:
    """Generate a cooperative, in-world player input for the current scene."""

    name = "natural"

    def __init__(
        self,
        client: InputTextGenerator,
        prompts: InputConditionPrompts | None = None,
    ) -> None:
        self._client = client
        self._prompts = prompts or load_input_condition_prompts()

    def next_input(self, *, player_role: str, history: Sequence[StoryTurn]) -> str:
        messages = _user_message(
            self._prompts.natural,
            player_role=player_role,
            history_text=format_visible_history(history),
        )
        return call_with_retries(
            lambda: self._client.chat(messages),
            require_text,
            stage="input_source_natural",
        )


class AdversarialInputCondition:
    """Generate an in-world stress-test player input for the current scene."""

    name = "adversarial"

    def __init__(
        self,
        client: InputTextGenerator,
        prompts: InputConditionPrompts | None = None,
    ) -> None:
        self._client = client
        self._prompts = prompts or load_input_condition_prompts()

    def next_input(self, *, player_role: str, history: Sequence[StoryTurn]) -> str:
        messages = _user_message(
            self._prompts.adversarial,
            player_role=player_role,
            history_text=format_visible_history(history),
        )
        return call_with_retries(
            lambda: self._client.chat(messages),
            require_text,
            stage="input_source_adversarial",
        )


def create_input_condition(
    choice: str | None,
    client: InputTextGenerator,
    prompts: InputConditionPrompts | None = None,
) -> PlayerInputCondition:
    """Create one of the two published player-input conditions.

    The ``natural`` default is the cooperative published condition. The
    client remains explicit so this module does not choose an API provider or
    create any model configuration.
    """

    normalized = (choice or "natural").strip().lower()
    if normalized == "natural":
        return NaturalInputCondition(client, prompts)
    if normalized == "adversarial":
        return AdversarialInputCondition(client, prompts)
    raise ValueError(f"Unknown input condition: {choice}")


def format_visible_history(history: Sequence[StoryTurn]) -> str:
    """Render the visible-history format used by the published conditions."""

    lines: list[str] = []
    for turn in history:
        if turn.turn_id == -1:
            lines.extend(("Opening:", turn.narrator_response or "(No opening narrative)"))
            continue
        lines.extend(
            (
                f"Turn {turn.turn_id}:",
                f"- Player: {turn.player_input or '(None)'}",
                f"- System: {turn.narrator_response or '(None)'}",
            )
        )
    return "\n".join(lines) if lines else "(The story has just begun)"


def _user_message(prompt: str, *, player_role: str, history_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": prompt.format(player_role=player_role, history_text=history_text),
        }
    ]


__all__ = [
    "AdversarialInputCondition",
    "InputConditionPrompts",
    "InputTextGenerator",
    "NaturalInputCondition",
    "PlayerInputCondition",
    "create_input_condition",
    "format_visible_history",
    "load_input_condition_prompts",
]
