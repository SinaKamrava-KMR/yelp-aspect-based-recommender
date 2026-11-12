import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from advanced_models import (
    BayesianPersonalizedRanking,
    load_bpr,
    save_bpr,
    tune_matrix_factorization,
)
from sentiment import (
    score_review_aspects_detailed,
    score_review_aspects_mandatory_detailed,
)


class Token:
    def __init__(self, text, index, *, dep="", pos="", lemma=""):
        self.text = text
        self.i = index
        self.dep_ = dep
        self.pos_ = pos
        self.lemma_ = lemma or text.lower()
        self.children = []
        self.head = self

    def add(self, child):
        self.children.append(child)
        child.head = self
        return child


class Analyzer:
    def polarity_scores(self, phrase):
        return {
            "excellent": {"compound": 0.8},
            "love": {"compound": 0.7},
        }[phrase]


class AdvancedFeatureTests(unittest.TestCase):
    def test_advanced_sentiment_does_not_change_mandatory_baseline(self):
        food = Token("food", 2, dep="obj", pos="NOUN")
        love = Token("love", 1, pos="VERB")
        food.head = love
        mandatory = score_review_aspects_mandatory_detailed(
            [("food", food)], Analyzer()
        )
        advanced = score_review_aspects_detailed([( "food", food)], Analyzer())
        self.assertEqual(mandatory.scores["food"], 0.0)
        self.assertAlmostEqual(advanced.scores["food"], 0.7)

    def test_bpr_is_reproducible_and_round_trips(self):
        interactions = csr_matrix(
            np.array(
                [
                    [5.0, 0.0, 0.0, 1.0],
                    [0.0, 4.0, 1.0, 0.0],
                    [1.0, 0.0, 5.0, 0.0],
                ]
            )
        )
        first = BayesianPersonalizedRanking(
            factors=4, epochs=2, batch_size=2, seed=9
        ).fit(interactions)
        second = BayesianPersonalizedRanking(
            factors=4, epochs=2, batch_size=2, seed=9
        ).fit(interactions)
        np.testing.assert_allclose(first.user_factors, second.user_factors)
        np.testing.assert_allclose(first.item_factors, second.item_factors)
        with tempfile.TemporaryDirectory() as directory:
            save_bpr(directory, first)
            cached = load_bpr(directory)
        np.testing.assert_allclose(first.user_factors, cached.user_factors)
        np.testing.assert_allclose(first.item_factors, cached.item_factors)

    def test_mf_grid_search_uses_validation_and_early_stopping(self):
        ratings = csr_matrix(
            np.array(
                [
                    [5.0, 3.0, 0.0],
                    [0.0, 2.0, 4.0],
                    [1.0, 0.0, 5.0],
                ]
            )
        )
        model, report = tune_matrix_factorization(
            ratings,
            [(0, 1, 3.0), (1, 2, 4.0)],
            factors=[2, 3],
            learning_rates=[0.01],
            regularizations=[0.05],
            epochs=[4],
            seed=42,
            patience=2,
        )
        self.assertEqual(len(report["trials"]), 2)
        self.assertGreaterEqual(model.best_epoch, 1)
        self.assertTrue(model.validation_history)


if __name__ == "__main__":
    unittest.main()
