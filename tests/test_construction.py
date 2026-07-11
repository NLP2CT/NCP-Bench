from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from ncpbench.construction import (
    ConstructionFormatError,
    build_dataset,
    load_specification_sources,
)
from ncpbench.specification import SpecificationGenerator, SpecificationSource


def _responses(count: int) -> list[str]:
    responses: list[str] = []
    for _ in range(count):
        responses.extend(
            [
                json.dumps(
                    {
                        "trajectory": [
                            {
                                "id": "s_0",
                                "description": "Initial situation",
                                "trigger_event": "Pressure begins",
                                "key_delta": "The initial state is established",
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "commitments": [
                            {
                                "id": "c_0",
                                "type": "achievement",
                                "description": "Reach the ending",
                                "satisfaction_condition": "The ending occurs",
                                "violation_condition": "The ending becomes impossible",
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "facts": [
                            {"id": "f_0", "content": "The story has begun."}
                        ]
                    }
                ),
            ]
        )
    return responses


class ScriptedClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.stages: list[str] = []

    def complete(self, messages: Sequence[Mapping[str, str]], *, stage: str) -> str:
        self.stages.append(stage)
        return self.responses.pop(0)


class ConstructionTests(unittest.TestCase):
    def test_loads_the_complete_curated_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stories.yaml"
            path.write_text(
                """- id: movie02
  title: Second Story
  genres: [Mystery, Drama]
  player_role: Detective
  synopsis: A detective investigates a disappearance.
""",
                encoding="utf-8",
            )

            sources = load_specification_sources(path)

        self.assertEqual(
            sources,
            (
                SpecificationSource(
                    id="movie02",
                    title="Second Story",
                    genres=("Mystery", "Drama"),
                    player_role="Detective",
                    synopsis="A detective investigates a disappearance.",
                ),
            ),
        )

    def test_builds_a_complete_loadable_dataset_in_source_order(self) -> None:
        sources = (
            SpecificationSource(
                id="movie02",
                title="Second Story",
                genres=("Mystery",),
                player_role="Detective",
                synopsis="Second synopsis",
            ),
            SpecificationSource(
                id="movie01",
                title="First Story",
                genres=("Drama",),
                player_role="Witness",
                synopsis="First synopsis",
            ),
        )
        client = ScriptedClient(_responses(2))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            dataset = build_dataset(sources, SpecificationGenerator(client), root)
            index = yaml.safe_load((root / "index.yaml").read_text(encoding="utf-8"))

            self.assertEqual(dataset.spec_ids, ("movie02", "movie01"))
            self.assertEqual(dataset.load_spec("movie02").title, "Second Story")
            self.assertEqual(dataset.load_spec("movie01").title, "First Story")
            self.assertTrue((root / "specs" / "movie02.yaml").is_file())
            self.assertTrue((root / "specs" / "movie01.yaml").is_file())

        self.assertEqual(
            index,
            {"specs": ["movie02", "movie01"]},
        )
        self.assertEqual(
            client.stages,
            [
                "trajectory_extraction",
                "commitment_extraction",
                "initial_facts_extraction",
            ]
            * 2,
        )

    def test_rejects_invalid_sources_before_generation_or_writes(self) -> None:
        duplicate_sources = (
            SpecificationSource("movie00", "One", ("Drama",), "Player", "One"),
            SpecificationSource("movie00", "Two", ("Drama",), "Player", "Two"),
        )
        client = ScriptedClient(_responses(2))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            with self.assertRaisesRegex(
                ConstructionFormatError, "Duplicate specification id"
            ):
                build_dataset(
                    duplicate_sources,
                    SpecificationGenerator(client),
                    output,
                )
            self.assertFalse(output.exists())

        self.assertEqual(client.stages, [])

        invalid_metadata = (
            SpecificationSource("movie01", "", ("Drama",), "Player", "Synopsis"),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ConstructionFormatError, "invalid title"):
                build_dataset(
                    invalid_metadata,
                    SpecificationGenerator(client),
                    Path(directory) / "dataset",
                )
        self.assertEqual(client.stages, [])

    def test_source_manifest_rejects_the_incomplete_role_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stories.yaml"
            path.write_text(
                """- id: movie00
  title: Story
  genres: [Drama]
  character: Player
  synopsis: Synopsis
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConstructionFormatError, "player_role"):
                load_specification_sources(path)


if __name__ == "__main__":
    unittest.main()
