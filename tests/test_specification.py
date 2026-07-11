from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path

from ncpbench.dataset import load_spec, save_spec
from ncpbench.specification import (
    SpecificationGenerationError,
    SpecificationGenerator,
    SpecificationSource,
)


TRAJECTORY = {
    "trajectory": [
        {
            "id": "S 0",
            "description": " Initial state ",
            "trigger_event": " Pressure begins ",
            "key_delta": " Context established ",
        },
        {
            "id": "S 0",
            "description": "Second state",
            "trigger_event": "Evidence appears",
            "key_delta": "Evidence acquired",
        },
        {"id": "unused", "description": "", "trigger_event": "", "key_delta": ""},
    ]
}
COMMITMENTS = {
    "commitments": [
        {
            "id": "Gate One",
            "type": "ordering",
            "description": " First gate ",
            "satisfaction_condition": " Evidence acquired ",
            "violation_condition": " Truth appears first ",
        },
        {
            "id": "Gate One",
            "type": "unexpected",
            "description": "Second gate",
            "satisfaction_condition": "Milestone reached",
            "violation_condition": "Milestone bypassed",
        },
        {
            "id": "incomplete",
            "type": "achievement",
            "description": "Missing condition",
            "satisfaction_condition": "",
            "violation_condition": "Failure",
        },
    ]
}
FACTS = {
    "facts": [
        {"id": "Known State", "content": " The player knows the public identity. "},
        {"id": "Known State", "content": "The evidence has not been found."},
        {"id": "empty", "content": ""},
    ]
}


class ScriptedClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, tuple[Mapping[str, str], ...]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]], *, stage: str) -> str:
        self.calls.append((stage, tuple(messages)))
        return self.responses.pop(0)


class SpecificationGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SpecificationSource(
            id="movie-test",
            title="Curated title",
            genres=("Mystery", "Drama"),
            player_role="Investigator",
            synopsis="A carefully curated synopsis.",
        )

    def test_preserves_the_three_stage_data_flow_and_curated_metadata(self) -> None:
        client = ScriptedClient(
            [json.dumps(TRAJECTORY), json.dumps(COMMITMENTS), json.dumps(FACTS)]
        )

        spec = SpecificationGenerator(client).generate(self.source)

        self.assertEqual(
            [stage for stage, _ in client.calls],
            ["trajectory_extraction", "commitment_extraction", "initial_facts_extraction"],
        )
        trajectory_json = json.dumps(TRAJECTORY["trajectory"], ensure_ascii=False, indent=2)
        commitments_json = json.dumps(
            COMMITMENTS["commitments"], ensure_ascii=False, indent=2
        )
        self.assertNotIn(trajectory_json, client.calls[0][1][0]["content"])
        self.assertIn(trajectory_json, client.calls[1][1][0]["content"])
        self.assertIn(trajectory_json, client.calls[2][1][0]["content"])
        self.assertIn(commitments_json, client.calls[2][1][0]["content"])

        self.assertEqual(spec.id, "movie-test")
        self.assertEqual(spec.title, "Curated title")
        self.assertEqual(spec.genres, ("Mystery", "Drama"))
        self.assertEqual(spec.player_role, "Investigator")
        self.assertEqual(spec.synopsis, "A carefully curated synopsis.")
        self.assertEqual([node.id for node in spec.trajectory], ["s_0", "s_0_2"])
        self.assertEqual(
            [item.id for item in spec.commitments], ["c_gate_one", "c_gate_one_2"]
        )
        self.assertEqual([item.kind for item in spec.commitments], ["ordering", "invariant"])
        self.assertEqual(
            [fact.id for fact in spec.initial_facts],
            ["f_known_state", "f_known_state_2"],
        )
        self.assertTrue(all(fact.active for fact in spec.initial_facts))
        self.assertTrue(all(not node.occurred for node in spec.trajectory))
        self.assertTrue(all(item.status == "pending" for item in spec.commitments))

    def test_retries_wrapped_json_instead_of_extracting_it(self) -> None:
        client = ScriptedClient(
            [
                f"preface\n```json\n{json.dumps(TRAJECTORY)}\n```",
                json.dumps(TRAJECTORY),
                json.dumps(COMMITMENTS),
                json.dumps(FACTS),
            ]
        )

        spec = SpecificationGenerator(client).generate(self.source)

        self.assertEqual(len(spec.trajectory), 2)
        self.assertEqual(len(spec.commitments), 2)
        self.assertEqual(len(spec.initial_facts), 2)
        self.assertEqual(
            [stage for stage, _ in client.calls[:2]],
            ["trajectory_extraction"] * 2,
        )

    def test_generated_spec_round_trips_through_the_published_yaml_schema(self) -> None:
        client = ScriptedClient(
            [json.dumps(TRAJECTORY), json.dumps(COMMITMENTS), json.dumps(FACTS)]
        )
        generated = SpecificationGenerator(client).generate(self.source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_spec(generated, root / "specs" / "movie-test.yaml")
            (root / "index.yaml").write_text(
                "specs:\n- movie-test\n",
                encoding="utf-8",
            )

            restored = load_spec(root, "movie-test")

        self.assertEqual(restored, generated)

    def test_retries_json_decoding_at_the_same_stage(self) -> None:
        client = ScriptedClient(
            [
                "not json",
                json.dumps(TRAJECTORY),
                json.dumps(COMMITMENTS),
                json.dumps(FACTS),
            ]
        )

        SpecificationGenerator(client).generate(self.source)

        self.assertEqual(
            [stage for stage, _ in client.calls[:2]], ["trajectory_extraction"] * 2
        )

    def test_reports_failure_after_initial_attempt_plus_two_retries(self) -> None:
        client = ScriptedClient(["not json"] * 3)

        with self.assertRaisesRegex(
            SpecificationGenerationError,
            "trajectory_extraction failed after 3 attempts",
        ):
            SpecificationGenerator(client).generate(self.source)

        self.assertEqual(len(client.calls), 3)

    def test_rejects_a_structurally_invalid_stage_without_running_later_stages(self) -> None:
        client = ScriptedClient(
            [json.dumps({"trajectory": ["not an object"]})] * 3
        )

        with self.assertRaisesRegex(
            SpecificationGenerationError, r"trajectory\[0\] must be an object"
        ):
            SpecificationGenerator(client).generate(self.source)

        self.assertEqual(len(client.calls), 3)


if __name__ == "__main__":
    unittest.main()
