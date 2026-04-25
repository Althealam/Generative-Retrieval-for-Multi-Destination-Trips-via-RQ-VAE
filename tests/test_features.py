"""Test feature engineering functions."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.context import (
    build_booker_device_vocabs,
    row_to_context_indices,
)
from src.features.trips import create_multiple_sequences


class TestContextFeatures:
    """Test trip-level context feature extraction."""

    def test_build_booker_device_vocabs(self):
        """Test vocabulary building for booker country and device class."""
        # Create sample data
        data = pd.DataFrame({
            'booker_country': ['US', 'UK', 'US', 'FR'],
            'device_class': ['mobile', 'desktop', 'mobile', 'desktop']
        })

        booker_to_idx, device_to_idx, n_booker, n_device = build_booker_device_vocabs(data)

        # Should have 3 countries (FR, UK, US - sorted alphabetically)
        assert n_booker == 3
        assert 'US' in booker_to_idx
        assert 'UK' in booker_to_idx
        assert 'FR' in booker_to_idx

        # Should have 2 device classes
        assert n_device == 2
        assert 'mobile' in device_to_idx
        assert 'desktop' in device_to_idx

        # Indices should start from 1 (0 is reserved for padding/unknown)
        assert all(idx > 0 for idx in booker_to_idx.values())
        assert all(idx > 0 for idx in device_to_idx.values())

    def test_row_to_context_indices(self):
        """Test extraction of context indices from a trip row."""
        booker_to_idx = {'US': 1, 'UK': 2}
        device_to_idx = {'mobile': 1, 'desktop': 2}

        row = pd.Series({
            'booker_country': 'US',
            'device_class': 'mobile',
            'checkin_month': 3,
            'stay_duration': [2, 3, 4],
            'city_id': [100, 101, 102],
            'hotel_country': ['US', 'US', 'FR']
        })

        indices = row_to_context_indices(
            row,
            booker_to_idx,
            device_to_idx,
            prefix_len=3
        )

        # Should return 9 indices
        assert len(indices) == 9

        # Booker and device should match
        booker_idx, device_idx = indices[0], indices[1]
        assert booker_idx == 1  # US
        assert device_idx == 1  # mobile

        # Month should be 3
        assert indices[2] == 3

        # All indices should be positive integers
        assert all(isinstance(idx, (int, np.integer)) for idx in indices)
        assert all(idx >= 0 for idx in indices)

    def test_row_to_context_indices_with_prefix(self):
        """Test that prefix_len prevents data leakage."""
        booker_to_idx = {'US': 1}
        device_to_idx = {'mobile': 1}

        row = pd.Series({
            'booker_country': 'US',
            'device_class': 'mobile',
            'checkin_month': 3,
            'stay_duration': [1, 2, 3, 4, 5],
            'city_id': [100, 101, 102, 103, 104],
            'hotel_country': ['US', 'US', 'FR', 'FR', 'UK']
        })

        # Use only first 2 cities
        indices_prefix2 = row_to_context_indices(
            row,
            booker_to_idx,
            device_to_idx,
            prefix_len=2
        )

        # Use all 5 cities
        indices_full = row_to_context_indices(
            row,
            booker_to_idx,
            device_to_idx,
            prefix_len=5
        )

        # Trip length should be different
        trip_len_idx_prefix2 = indices_prefix2[4]
        trip_len_idx_full = indices_full[4]
        assert trip_len_idx_prefix2 != trip_len_idx_full

        # Unique cities count should be different
        num_unique_prefix2 = indices_prefix2[5]
        num_unique_full = indices_full[5]
        # Both should be valid
        assert num_unique_prefix2 > 0
        assert num_unique_full > 0


class TestTripCreation:
    """Test trip sequence creation."""

    def test_create_multiple_sequences(self):
        """Test trip aggregation from bookings."""
        # Create sample booking data
        data = pd.DataFrame({
            'utrip_id': [1, 1, 1, 2, 2],
            'city_id': [100, 101, 102, 200, 201],
            'checkin': pd.date_range('2024-01-01', periods=5),
            'checkout': pd.date_range('2024-01-02', periods=5),
            'booker_country': ['US', 'US', 'US', 'UK', 'UK'],
            'device_class': ['mobile', 'mobile', 'mobile', 'desktop', 'desktop'],
            'hotel_country': ['US', 'FR', 'IT', 'UK', 'FR']
        })

        trips = create_multiple_sequences(data)

        # Should have 2 trips
        assert len(trips) == 2

        # Each trip should have city_id as a list
        assert 'city_id' in trips.columns

        # First trip should have 3 cities
        trip1 = trips.iloc[0]
        assert len(trip1['city_id']) == 3
        assert trip1['city_id'] == [100, 101, 102]

        # Second trip should have 2 cities
        trip2 = trips.iloc[1]
        assert len(trip2['city_id']) == 2

    def test_create_multiple_sequences_empty(self):
        """Test handling of empty input."""
        data = pd.DataFrame({
            'utrip_id': [],
            'city_id': []
        })

        trips = create_multiple_sequences(data)
        assert len(trips) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
