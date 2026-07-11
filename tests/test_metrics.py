from __future__ import annotations

import unittest

from ncpbench.metrics import (
    EpisodeMetrics,
    MetricInputError,
    aggregate_metrics,
    compute_episode_metrics,
    survival_curve,
)


class MetricsTests(unittest.TestCase):
    def test_computes_the_paper_metrics_from_one_episode_result(self) -> None:
        metrics = compute_episode_metrics(_result())

        self.assertEqual(metrics.episode_id, "movie00")
        self.assertEqual(metrics.turn_count, 2)
        self.assertEqual(metrics.termination, "conflict")
        self.assertEqual((metrics.trajectory_reached, metrics.trajectory_total), (2, 3))
        self.assertAlmostEqual(metrics.trajectory_progress, 2 / 3)
        self.assertEqual((metrics.commitments_satisfied, metrics.commitments_total), (1, 2))
        self.assertEqual(metrics.commitment_satisfaction, 0.5)
        self.assertTrue(metrics.fact_conflict)
        self.assertFalse(metrics.commitment_conflict)
        self.assertTrue(metrics.player_input_conflict)

    def test_uses_per_episode_conflict_flags_and_macro_averages(self) -> None:
        first = compute_episode_metrics(_result())
        second = EpisodeMetrics(
            episode_id="movie01",
            turn_count=4,
            termination="max_turns",
            trajectory_reached=1,
            trajectory_total=4,
            trajectory_progress=0.25,
            commitments_satisfied=3,
            commitments_total=4,
            commitment_satisfaction=0.75,
            fact_conflict=False,
            commitment_conflict=True,
            player_input_conflict=False,
        )

        aggregate = aggregate_metrics((first, second))

        self.assertEqual(aggregate.episode_count, 2)
        self.assertEqual(aggregate.average_turns, 3.0)
        self.assertAlmostEqual(aggregate.average_trajectory_progress, ((2 / 3) + 0.25) / 2)
        self.assertEqual(aggregate.average_commitment_satisfaction, 0.625)
        self.assertEqual(aggregate.fact_conflict_rate, 0.5)
        self.assertEqual(aggregate.commitment_conflict_rate, 0.5)
        self.assertEqual(aggregate.player_input_conflict_rate, 0.5)
        self.assertEqual((aggregate.conflict_count, aggregate.max_turns_count, aggregate.all_resolved_count), (1, 1, 0))

    def test_survival_curve_matches_turn_threshold_counting(self) -> None:
        first = compute_episode_metrics(_result())
        second = EpisodeMetrics(
            "movie01", 4, "max_turns", 1, 1, 1.0, 0, 0, 0.0, False, False, False
        )

        self.assertEqual(survival_curve((first, second), max_turns=4), (1.0, 1.0, 1.0, 0.5, 0.5))

    def test_rejects_missing_required_result_state(self) -> None:
        with self.assertRaisesRegex(MetricInputError, "final_state must be an object"):
            compute_episode_metrics({"episode": {"id": "movie00"}, "turns": [], "termination": "conflict"})

        with self.assertRaisesRegex(ValueError, "empty"):
            aggregate_metrics(())


def _result():
    no_conflict = {
        "fact_count": 0,
        "commitment_count": 0,
        "player_input_count": 0,
    }
    mixed_conflict = {
        "fact_count": 2,
        "commitment_count": 0,
        "player_input_count": 1,
    }
    return {
        "episode": {"id": "movie00", "player_role": "Player"},
        "termination": "conflict",
        "turns": [
            {"evaluation": {"conflict": no_conflict}},
            {"evaluation": {"conflict": mixed_conflict}},
        ],
        "final_state": {
            "trajectory": [
                {"id": "n_0", "occurred": True},
                {"id": "n_1", "occurred": True},
                {"id": "n_2", "occurred": False},
            ],
            "commitments": [
                {"id": "c_0", "status": "satisfied"},
                {"id": "c_1", "status": "pending"},
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
