import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from evaluator import ranking_metrics, rmse
from hybrid_model import (
    BiasedMatrixFactorization,
    HybridRecommender,
    load_cached_training_data,
    load_matrix_factorization,
    load_training_data,
    normalize_aspect,
    normalize_cf,
    save_matrix_factorization,
    save_training_data,
    write_recommendations,
)
from text_processor import ProfileMatrices


class HybridAndEvaluatorTests(unittest.TestCase):
    def _training_file(self, root):
        path = root / "train.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\r\n")
            writer.writerow(["review_id", "user_id", "business_id", "stars", "date", "text"])
            writer.writerows([
                ["r1", "u1", "b1", 5, "2020-01-01", "x"],
                ["r2", "u1", "b1", 3, "2020-01-02", "x"],
                ["r3", "u1", "b2", 4, "2020-01-03", "x"],
                ["r4", "u2", "b2", 2, "2020-01-04", "x"],
                ["r5", "u2", "b3", 5, "2020-01-05", "x"],
                ["r6", "u3", "b4", 4, "2020-01-06", "x"],
            ])
        return path

    def _recommender(self, root):
        training = load_training_data(self._training_file(root))
        user_map = {user: index for index, user in enumerate(sorted(training.user_to_index))}
        item_map = {item: index for index, item in enumerate(sorted(training.item_to_index))}
        profiles = ProfileMatrices(
            user_aspects=csr_matrix(np.array([
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ], dtype=float)),
            item_aspects=csr_matrix(np.array([
                [0.8, 0.0, 0, 0, 0],
                [0.2, -0.2, 0, 0, 0],
                [-0.4, 0.9, 0, 0, 0],
                [0.0, 0.1, 0, 0, 0],
            ], dtype=float)),
            user_to_index=user_map,
            item_to_index=item_map,
            stats={},
        )
        mf = BiasedMatrixFactorization(factors=3, epochs=3, seed=7).fit(training.ratings)
        return HybridRecommender(training, mf, profiles)

    def test_pair_mean_sparse_matrix_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = load_training_data(self._training_file(root))
            self.assertEqual(training.ratings[training.user_to_index["u1"], training.item_to_index["b1"]], 4.0)
            first = BiasedMatrixFactorization(factors=2, epochs=2, seed=11).fit(training.ratings)
            second = BiasedMatrixFactorization(factors=2, epochs=2, seed=11).fit(training.ratings)
            np.testing.assert_allclose(first.user_factors, second.user_factors)
            np.testing.assert_allclose(first.item_factors, second.item_factors)

    def test_global_normalization_and_fallback(self):
        self.assertEqual(normalize_cf(0.0), 0.0)
        self.assertEqual(normalize_cf(6.0), 1.0)
        self.assertEqual(normalize_aspect(-1.0), 0.0)
        self.assertEqual(normalize_aspect(1.0), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recommender = self._recommender(root)
            result = recommender.recommend_user("cold", min_interactions=5, threshold=0.3)
            self.assertTrue(result.fallback)
            self.assertEqual(result.reason, "fewer_than_min_interactions")
            # b1 has two training rows, all others one; popularity ties use item ID.
            self.assertEqual([row["business_id"] for row in result.rows], ["b1", "b2", "b3", "b4"])

            normal = recommender.recommend_user(
                "u1", alpha=0.7, min_interactions=1, threshold=0.0
            )
            self.assertFalse(normal.fallback)
            self.assertTrue({"b1", "b2"}.isdisjoint({row["business_id"] for row in normal.rows}))
            output = root / "recommendations.csv"
            write_recommendations(output, normal.rows)
            text = output.read_text(encoding="utf-8")
            self.assertIn("false", text)

    def test_metric_definitions(self):
        rows = [
            {"user_id": "u1", "business_id": "x", "rank": 1},
            {"user_id": "u1", "business_id": "b1", "rank": 2},
            {"user_id": "u2", "business_id": "x", "rank": 1},
        ]
        qrels = {"u1": {"b1": 1}, "u2": {"b2": 1}}
        metrics = ranking_metrics(rows, qrels, ["u1", "u2"], ["x", "b1", "b2"], k=10)
        self.assertAlmostEqual(metrics["hr@10"], 0.5)
        self.assertAlmostEqual(metrics["ndcg@10"], (1 / np.log2(3)) / 2)
        targets = {("u1", "b1"): 5.0, ("u2", "b2"): 1.0}
        self.assertAlmostEqual(rmse(targets, lambda _u, _i: 3.0), 2.0)

    def test_checkpoint_round_trip_and_multi_configuration_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = load_training_data(self._training_file(root))
            mf = BiasedMatrixFactorization(factors=3, epochs=3, seed=7).fit(
                training.ratings
            )
            save_training_data(root / "training-cache", training)
            save_matrix_factorization(root / "mf-cache", mf)
            cached_training = load_cached_training_data(root / "training-cache")
            cached_mf = load_matrix_factorization(root / "mf-cache")
            np.testing.assert_allclose(cached_training.ratings.toarray(), training.ratings.toarray())
            np.testing.assert_allclose(cached_mf.user_factors, mf.user_factors)
            self.assertEqual(cached_training.seen_by_user, training.seen_by_user)

            recommender = self._recommender(root)
            configurations = {
                "hybrid": {
                    "alpha": 0.7,
                    "top_k": 10,
                    "min_interactions": 1,
                    "threshold": 0.0,
                    "mode": "hybrid",
                },
                "popularity": {
                    "alpha": 0.7,
                    "top_k": 10,
                    "min_interactions": 1,
                    "threshold": 0.0,
                    "mode": "popularity",
                },
            }
            rows, _ = recommender.recommend_many_configurations(["u1"], configurations)
            expected = recommender.recommend_user(
                "u1", alpha=0.7, min_interactions=1, threshold=0.0
            ).rows
            self.assertEqual(rows["hybrid"], expected)
            self.assertTrue(all(row["is_fallback"] for row in rows["popularity"]))


if __name__ == "__main__":
    unittest.main()
