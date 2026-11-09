import unittest

from code.disease_ranking import rank_disease_probabilities


class DiseaseRankingTests(unittest.TestCase):
    def test_ranks_highest_scores_and_maps_encoded_labels(self):
        ranked = rank_disease_probabilities(
            [0.12, 0.71, 0.17],
            [2, 0, 1],
            ["Condition A", "Condition B", "Condition C"],
        )

        self.assertEqual(
            ranked,
            [
                ("Condition A", 0.71),
                ("Condition B", 0.17),
                ("Condition C", 0.12),
            ],
        )

    def test_limits_results_to_requested_count(self):
        ranked = rank_disease_probabilities(
            [0.12, 0.71, 0.17],
            [0, 1, 2],
            ["Condition A", "Condition B", "Condition C"],
            top_k=2,
        )

        self.assertEqual(len(ranked), 2)

    def test_rejects_mismatched_probabilities_and_classes(self):
        with self.assertRaisesRegex(ValueError, "counts must match"):
            rank_disease_probabilities([0.5], [0, 1], ["Condition"])

    def test_rejects_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            rank_disease_probabilities([1.0], [0], ["Condition"], top_k=0)


if __name__ == "__main__":
    unittest.main()
