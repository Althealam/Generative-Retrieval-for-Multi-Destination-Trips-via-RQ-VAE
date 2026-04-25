"""Test dataset loading and processing."""

import sys
from pathlib import Path
import torch
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import build_city_dataloaders, PAD_TOKEN_ID


class TestDataLoaders:
    """Test DataLoader creation and batching."""

    def test_build_city_dataloaders(self):
        """Test creation of train and test dataloaders."""
        # Create dummy data
        train_x = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        train_y = [10, 11, 12]
        test_x = [[13, 14], [15, 16, 17]]

        # Create dummy context (9 features)
        train_ctx = tuple([list(range(len(train_x)))] * 9)
        test_ctx = tuple([list(range(len(test_x)))] * 9)

        train_loader, test_loader = build_city_dataloaders(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            batch_size=2,
            train_ctx=train_ctx,
            test_ctx=test_ctx,
        )

        # Check train loader
        assert len(train_loader) > 0

        # Get a batch
        batch = next(iter(train_loader))
        x_batch = batch[0]
        y_batch = batch[1]
        context_batch = batch[2:]

        # x should be padded tensor
        assert isinstance(x_batch, torch.Tensor)
        assert x_batch.shape[0] <= 2  # batch size

        # y should be tensor
        assert isinstance(y_batch, torch.Tensor)

        # Should have 9 context features
        assert len(context_batch) == 9

    def test_padding_in_dataloader(self):
        """Test that sequences are properly padded."""
        train_x = [[1, 2], [3, 4, 5, 6, 7]]  # Different lengths
        train_y = [10, 11]
        train_ctx = tuple([[0, 1]] * 9)

        train_loader, _ = build_city_dataloaders(
            train_x=train_x,
            train_y=train_y,
            test_x=[[1]],
            batch_size=2,
            train_ctx=train_ctx,
            test_ctx=tuple([[0]] * 9),
        )

        batch = next(iter(train_loader))
        x_batch = batch[0]

        # All sequences should have same length (padded to max)
        assert x_batch.shape[0] == 2  # batch size
        assert x_batch.shape[1] == 5  # max length in batch

        # Check padding - due to shuffle, find which sequence is shorter
        # The shorter sequence (length 2) should be padded with PAD_TOKEN_ID
        for i in range(x_batch.shape[0]):
            # Check if this is the shorter sequence by looking at position 2
            if x_batch[i, 1].item() in [1, 2]:  # Second element is 1 or 2
                # This is likely the [1, 2] sequence, check padding
                assert x_batch[i, 2].item() == PAD_TOKEN_ID
                break


class TestSequenceProcessing:
    """Test sequence tokenization and processing."""

    def test_sequence_to_indices(self):
        """Test conversion of city sequences to indices."""
        from src.features.city_tokens import build_city_vocab

        data = pd.DataFrame({
            'city_id': [100, 200, 300, 100, 200]
        })

        city_to_idx, idx_to_city = build_city_vocab(data)

        # Map a sequence
        sequence = [100, 200, 100]
        indices = [city_to_idx[c] for c in sequence]

        # All should be valid indices
        assert all(isinstance(i, int) for i in indices)
        assert all(i > 1 for i in indices)  # Greater than PAD(0) and UNK(1)

        # Should be able to reverse
        recovered = [idx_to_city[i] for i in indices]
        assert recovered == sequence

    def test_unknown_city_handling(self):
        """Test handling of unknown cities."""
        from src.features.city_tokens import build_city_vocab
        from src.datasets import UNK_TOKEN_ID

        data = pd.DataFrame({
            'city_id': [100, 200]
        })

        city_to_idx, _ = build_city_vocab(data)

        # Unknown city should map to UNK
        unknown_city = 999
        idx = city_to_idx.get(unknown_city, UNK_TOKEN_ID)
        assert idx == UNK_TOKEN_ID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
