from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from ncpbench.dataset import DatasetFormatError, load_dataset, load_spec, save_spec


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"


class DatasetTests(unittest.TestCase):
    def test_index_preserves_the_declared_benchmark_order(self) -> None:
        dataset = load_dataset(DATASET_ROOT)

        self.assertEqual(len(dataset.spec_ids), 100)
        self.assertEqual(dataset.spec_ids[:3], ("movie48", "movie09", "movie72"))
        self.assertEqual(len(set(dataset.spec_ids)), 100)

    def test_loads_a_story_spec_ready_for_the_episode_runner(self) -> None:
        spec = load_spec(DATASET_ROOT, "movie00")

        self.assertEqual(spec.id, "movie00")
        self.assertEqual(spec.title, "The Bourne Identity")
        self.assertEqual(spec.player_role, "Jason Bourne")
        self.assertEqual(len(spec.initial_facts), 14)
        self.assertEqual(len(spec.commitments), 20)
        self.assertEqual(len(spec.trajectory), 20)
        self.assertTrue(spec.initial_facts[0].active)

    def test_every_indexed_spec_is_parseable_and_has_unique_ids(self) -> None:
        dataset = load_dataset(DATASET_ROOT)

        for spec_id in dataset.spec_ids:
            spec = dataset.load_spec(spec_id)
            self.assertEqual(spec.id, spec_id)
            self.assertEqual(len({fact.id for fact in spec.initial_facts}), len(spec.initial_facts))
            self.assertEqual(len({commitment.id for commitment in spec.commitments}), len(spec.commitments))
            self.assertEqual(len({node.id for node in spec.trajectory}), len(spec.trajectory))

    def test_every_runtime_field_matches_the_published_yaml(self) -> None:
        dataset = load_dataset(DATASET_ROOT)

        for spec_id in dataset.spec_ids:
            raw = yaml.safe_load(
                (dataset.root / "specs" / f"{spec_id}.yaml").read_text(
                    encoding="utf-8"
                )
            )
            spec = dataset.load_spec(spec_id)

            self.assertEqual(spec.id, raw["meta"]["id"])
            self.assertEqual(spec.title, raw["meta"]["title"])
            self.assertEqual(spec.genres, tuple(raw["meta"]["genres"]))
            self.assertEqual(spec.player_role, raw["meta"]["player_role"])
            self.assertEqual(spec.synopsis, raw["synopsis"])
            self.assertEqual(
                [(fact.id, fact.text, fact.active) for fact in spec.initial_facts],
                [
                    (fact["id"], fact["content"], not fact.get("negated", False))
                    for fact in raw["initial_facts"]
                ],
            )
            self.assertEqual(
                [
                    (
                        commitment.id,
                        commitment.kind,
                        commitment.description,
                        commitment.satisfaction_condition,
                        commitment.violation_condition,
                    )
                    for commitment in spec.commitments
                ],
                [
                    (
                        commitment["id"],
                        commitment["type"],
                        commitment["description"],
                        commitment["satisfaction_condition"],
                        commitment["violation_condition"],
                    )
                    for commitment in raw["commitments"]
                ],
            )
            self.assertEqual(
                [
                    (node.id, node.description, node.trigger_event, node.key_delta)
                    for node in spec.trajectory
                ],
                [
                    (node["id"], node["description"], node["trigger_event"], node["key_delta"])
                    for node in raw["trajectory"]
                ],
            )
            self.assertFalse(hasattr(spec, "token_usage"))

    def test_rejects_an_invalid_spec_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.yaml").write_text("specs:\n- ../outside\n", encoding="utf-8")

            with self.assertRaisesRegex(DatasetFormatError, "Invalid spec id"):
                load_dataset(root)

    def test_saved_spec_round_trips_every_runtime_field(self) -> None:
        original = load_spec(DATASET_ROOT, "movie93")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "specs" / "movie93.yaml"
            save_spec(original, path)
            (root / "index.yaml").write_text(
                "specs:\n- movie93\n",
                encoding="utf-8",
            )

            restored = load_spec(root, "movie93")
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(restored, original)
        self.assertNotIn("token_usage", raw)
        self.assertEqual(
            [fact["negated"] for fact in raw["initial_facts"]],
            [not fact.active for fact in original.initial_facts],
        )
        self.assertTrue(all("status" not in item for item in raw["commitments"]))
        self.assertTrue(all("occurred" not in node for node in raw["trajectory"]))

    def test_every_published_spec_survives_a_complete_yaml_round_trip(self) -> None:
        published = load_dataset(DATASET_ROOT)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_entries = []
            for spec_id in published.spec_ids:
                relative_path = Path("specs") / f"{spec_id}.yaml"
                save_spec(published.load_spec(spec_id), root / relative_path)
                index_entries.append(spec_id)
            (root / "index.yaml").write_text(
                yaml.safe_dump({"specs": index_entries}, sort_keys=False),
                encoding="utf-8",
            )
            restored = load_dataset(root)

            self.assertEqual(restored.spec_ids, published.spec_ids)
            for spec_id in published.spec_ids:
                self.assertEqual(restored.load_spec(spec_id), published.load_spec(spec_id))

    def test_save_spec_accepts_only_the_published_yaml_format(self) -> None:
        spec = load_spec(DATASET_ROOT, "movie00")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "movie00.json"
            with self.assertRaisesRegex(ValueError, "must point to YAML"):
                save_spec(spec, path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
