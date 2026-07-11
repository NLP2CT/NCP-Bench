from __future__ import annotations

import json
import unittest
from collections.abc import Mapping, Sequence

from ncpbench.checkpoint import checkpoint_from_mapping, checkpoint_to_mapping
from ncpbench.dataset import StorySpec
from ncpbench.evaluator import ConflictDecision, FactUpdate, TurnEvaluation
from ncpbench.models import Fact, PendingFactUpdates, StoryTurn
from ncpbench.narrator import (
    EpisodeContext,
    Narrator,
    NarratorRequest,
    NarratorResponse,
    OpeningRequest,
)
from ncpbench.runner import EpisodeCheckpoint, EpisodeRunner, EpisodeTermination


class CheckpointTests(unittest.TestCase):
    def test_json_round_trip_resumes_at_the_next_uncommitted_turn(self) -> None:
        runner = EpisodeRunner(_Evaluator(), _NoTrajectory(), _NoCommitments(), _Opening())
        first_narrator = _Narrator()
        checkpoints: list[EpisodeCheckpoint] = []

        first_trace = runner.run(
            first_narrator,
            _spec(),
            _Inputs(("first input",)),
            max_turns=1,
            on_checkpoint=checkpoints.append,
        )
        self.assertEqual(first_trace.termination, EpisodeTermination.MAX_TURNS)
        self.assertEqual([turn.completed_turn.turn_id for turn in first_trace.turns], [0])

        encoded = json.loads(json.dumps(checkpoint_to_mapping(checkpoints[-1])))
        restored = checkpoint_from_mapping(encoded, _spec())
        self.assertEqual(checkpoint_to_mapping(restored), encoded)

        resumed_narrator = _Narrator()
        second_trace = runner.run(
            resumed_narrator,
            _spec(),
            _Inputs(("second input",)),
            max_turns=2,
            checkpoint=restored,
        )

        self.assertEqual(second_trace.termination, EpisodeTermination.MAX_TURNS)
        self.assertEqual(
            [turn.completed_turn.turn_id for turn in second_trace.turns], [0, 1]
        )
        self.assertEqual(resumed_narrator.started, 0)
        self.assertEqual(resumed_narrator.opened, 0)
        self.assertEqual(resumed_narrator.restored, {"responses": 1})
        self.assertEqual(
            [request.turn_id for request in resumed_narrator.requests], [1]
        )


class _Narrator(Narrator):
    def __init__(self) -> None:
        self.started = 0
        self.opened = 0
        self.restored: Mapping[str, object] | None = None
        self.responses = 0
        self.requests: list[NarratorRequest] = []

    def start_episode(self, context: EpisodeContext) -> None:
        self.started += 1

    def open(self, request: OpeningRequest) -> NarratorResponse:
        self.opened += 1
        return NarratorResponse("Opening")

    def respond(self, request: NarratorRequest) -> NarratorResponse:
        self.requests.append(request)
        self.responses += 1
        return NarratorResponse(f"response {request.turn_id}")

    def checkpoint_state(self) -> Mapping[str, object]:
        return {"responses": self.responses}

    def restore_episode(
        self, context: EpisodeContext, state: Mapping[str, object]
    ) -> None:
        self.restored = state
        self.responses = int(state["responses"])


class _Inputs:
    name = "scripted"

    def __init__(self, values: Sequence[str]) -> None:
        self._values = iter(values)

    def next_input(self, *, player_role: str, history: Sequence[StoryTurn]) -> str:
        return next(self._values)


class _Opening:
    def evaluate(
        self, request: OpeningRequest, response: NarratorResponse
    ) -> PendingFactUpdates:
        return PendingFactUpdates()


class _Evaluator:
    def evaluate(self, context, narrator_text: str) -> TurnEvaluation:
        return TurnEvaluation(
            FactUpdate((), (), "No changes."),
            ConflictDecision(0, 0, 0, 0, (), False),
        )


class _NoTrajectory:
    def check(self, request) -> tuple[object, ...]:
        return ()


class _NoCommitments:
    def check(self, request) -> tuple[object, ...]:
        return ()


def _spec() -> StorySpec:
    return StorySpec(
        id="checkpoint-story",
        title="Checkpoint Story",
        genres=("Drama",),
        player_role="Player",
        synopsis="A test story.",
        initial_facts=(Fact("f_0", "The story has started."),),
        commitments=(),
        trajectory=(),
    )


if __name__ == "__main__":
    unittest.main()
