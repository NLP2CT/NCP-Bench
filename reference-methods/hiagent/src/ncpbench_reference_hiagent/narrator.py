"""HiAgent as an ordinary NCP-Bench narrator implementation."""

from __future__ import annotations

from collections.abc import Mapping

from ncpbench.model_output import call_with_retries, parse_text_object
from ncpbench.narrator import (
    EpisodeContext,
    Narrator,
    NarratorRequest,
    NarratorResponse,
    OpeningRequest,
    render_opening_prompt,
)

from ncpbench_reference_hiagent.client import TextGenerationClient
from ncpbench_reference_hiagent.config import HiAgentRuntimeConfig
from ncpbench_reference_hiagent.core import HiAgentCore
from ncpbench_reference_hiagent.hiagent_prompts import build_hiagent_turn_context
from ncpbench_reference_hiagent.summarizer import TrajectorySummarizer


class HiAgentNarrator(Narrator):
    """The paper HiAgent policy. It owns memory but has no evaluator access."""

    def __init__(self, client: TextGenerationClient, config: HiAgentRuntimeConfig | None = None) -> None:
        self._client = client
        self._config = config or HiAgentRuntimeConfig()
        self._core = HiAgentCore(
            client=client,
            config=self._config,
            summarizer=TrajectorySummarizer(
                client, max_chars=self._config.summary_max_chars
            ),
        )
        self._episode: EpisodeContext | None = None
        self._pending_action: str | None = None

    def start_episode(self, context: EpisodeContext) -> None:
        self._episode = context
        self._pending_action = None
        self._core.reset(init_obs="Story initialized.")

    def close_episode(self) -> None:
        self._episode = None
        self._pending_action = None

    def checkpoint_state(self) -> Mapping[str, object]:
        return {
            "core": self._core.checkpoint_state(),
            "pending_action": self._pending_action,
        }

    def restore_episode(
        self, context: EpisodeContext, state: Mapping[str, object]
    ) -> None:
        core_state = state.get("core")
        pending_action = state.get("pending_action")
        if not isinstance(core_state, Mapping):
            raise ValueError("Invalid HiAgent checkpoint core state")
        if pending_action is not None and not isinstance(pending_action, str):
            raise ValueError("Invalid HiAgent checkpoint pending action")
        self._episode = context
        self._core.restore_state(core_state)
        self._pending_action = pending_action

    def open(self, request: OpeningRequest) -> NarratorResponse:
        if self._episode is None:
            raise RuntimeError("HiAgentNarrator.start_episode() must be called before open()")
        messages = ({"role": "user", "content": render_opening_prompt(request)},)
        text = call_with_retries(
            lambda: self._client.complete(messages, stage="opening"),
            parse_text_object,
            stage="opening",
        )
        return NarratorResponse(text=text)

    def respond(self, request: NarratorRequest) -> NarratorResponse:
        if self._episode is None:
            raise RuntimeError("HiAgentNarrator.start_episode() must be called before respond()")
        if self._pending_action is not None:
            self._core.update(action=self._pending_action, state=f"Player input observed: {request.player_input}")
            self._pending_action = None

        turn_context = build_hiagent_turn_context(request)
        text = self._core.run(turn_context=turn_context)
        response = NarratorResponse(text=text)
        self._pending_action = response.text
        return response
