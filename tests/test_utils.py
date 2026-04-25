"""Test utility functions."""

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.evaluation import evaluate_accuracy_at_4
from src.utils.popularity import top_city_ids_from_train


class TestEvaluation:
    """Test evaluation metrics."""

    def test_accuracy_at_4_perfect(self):
        """Test accuracy@4 with perfect predictions."""
        predictions = pd.DataFrame({
            'utrip_id': [1, 2],
            'city_id_1': [100, 200],
            'city_id_2': [101, 201],
            'city_id_3': [102, 202],
            'city_id_4': [103, 203],
        })

        ground_truth = pd.DataFrame({
            'utrip_id': [1, 2],
            'city_id': [100, 200]
        })

        accuracy, n = evaluate_accuracy_at_4(predictions, ground_truth)
        assert accuracy == 1.0
        assert n == 2

    def test_accuracy_at_4_partial(self):
        """Test accuracy@4 with partial matches."""
        predictions = pd.DataFrame({
            'utrip_id': [1, 2],
            'city_id_1': [100, 999],
            'city_id_2': [101, 998],
            'city_id_3': [102, 997],
            'city_id_4': [103, 200],
        })

        ground_truth = pd.DataFrame({
            'utrip_id': [1, 2],
            'city_id': [100, 200]
        })

        accuracy, n = evaluate_accuracy_at_4(predictions, ground_truth)
        assert accuracy == 1.0  # Both ground truth cities appear in top-4
        assert n == 2

    def test_accuracy_at_4_no_match(self):
        """Test accuracy@4 with no matches."""
        predictions = pd.DataFrame({
            'utrip_id': [1, 2],
            'city_id_1': [999, 998],
            'city_id_2': [997, 996],
            'city_id_3': [995, 994],
            'city_id_4': [993, 992],
        })

        ground_truth = pd.DataFrame({
            'utrip_id': [1, 2],
            'city_id': [100, 200]
        })

        accuracy, n = evaluate_accuracy_at_4(predictions, ground_truth)
        assert accuracy == 0.0
        assert n == 2

    def test_accuracy_at_4_mixed(self):
        """Test accuracy@4 with mixed results."""
        predictions = pd.DataFrame({
            'utrip_id': [1, 2, 3],
            'city_id_1': [100, 999, 300],
            'city_id_2': [101, 998, 301],
            'city_id_3': [102, 200, 302],
            'city_id_4': [103, 997, 303],
        })

        ground_truth = pd.DataFrame({
            'utrip_id': [1, 2, 3],
            'city_id': [100, 200, 999]  # 2/3 found
        })

        accuracy, n = evaluate_accuracy_at_4(predictions, ground_truth)
        expected = 2.0 / 3.0
        assert abs(accuracy - expected) < 1e-6
        assert n == 3


class TestPopularity:
    """Test popularity-based baselines."""

    def test_top_city_ids_from_train(self):
        """Test extraction of most popular cities."""
        train_data = pd.DataFrame({
            'city_id': [100, 100, 100, 200, 200, 300]
        })

        top_cities = top_city_ids_from_train(train_data, k=2)

        # Should return top 2 cities
        assert len(top_cities) == 2

        # Most frequent should be first
        assert top_cities[0] == 100  # appears 3 times
        assert top_cities[1] == 200  # appears 2 times

    def test_top_city_ids_from_train_k_larger_than_unique(self):
        """Test when k is larger than number of unique cities."""
        train_data = pd.DataFrame({
            'city_id': [100, 200]
        })

        top_cities = top_city_ids_from_train(train_data, k=10)

        # Should return all available cities
        assert len(top_cities) == 2

    def test_top_city_ids_from_train_empty(self):
        """Test with empty training data."""
        train_data = pd.DataFrame({
            'city_id': []
        })

        top_cities = top_city_ids_from_train(train_data, k=4)
        assert len(top_cities) == 0


class TestDataTokens:
    """Test tokenization and vocabulary."""

    def test_vocabulary_consistency(self):
        """Test that vocabulary mapping is consistent."""
        from src.features.city_tokens import build_city_vocab

        train_data = pd.DataFrame({
            'city_id': [100, 200, 100, 300, 200]
        })

        city_to_idx, idx_to_city = build_city_vocab(train_data)

        # All cities should be in vocab
        assert 100 in city_to_idx
        assert 200 in city_to_idx
        assert 300 in city_to_idx

        # Reverse mapping should be consistent
        for city, idx in city_to_idx.items():
            assert idx_to_city[idx] == city

    def test_special_tokens(self):
        """Test that special tokens are reserved."""
        from src.datasets import PAD_TOKEN_ID, UNK_TOKEN_ID

        # PAD and UNK should be different
        assert PAD_TOKEN_ID != UNK_TOKEN_ID

        # They should be small integers (0 or 1 typically)
        assert PAD_TOKEN_ID >= 0
        assert UNK_TOKEN_ID >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
