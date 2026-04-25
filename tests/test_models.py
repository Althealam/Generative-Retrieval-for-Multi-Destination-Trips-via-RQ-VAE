"""Test model architectures."""

import sys
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.embedding import CityTransformer, CityGRU
from src.models.rqvae import RQVAE, RQVAETransformer, RQVAEGRU
from src.models.rqkmeans import RQKMeansTransformer, RQKmeansGRU


class TestEmbeddingModels:
    """Test embedding-based models."""

    def test_city_transformer_forward(self):
        """Test Transformer forward pass."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        model = CityTransformer(
            vocab_size=vocab_size,
            pad_token_id=0,
            d_model=64,
            nhead=2,
            num_layers=1,
            n_booker_countries=5,
            n_device_classes=2,
        )

        # Create dummy input
        x = torch.randint(1, vocab_size, (batch_size, seq_len))
        booker = torch.randint(0, 5, (batch_size,))
        device = torch.randint(0, 2, (batch_size,))
        month = torch.randint(1, 12, (batch_size,))
        stay = torch.randint(1, 30, (batch_size,))
        trip_len = torch.randint(1, 30, (batch_size,))
        num_unique = torch.randint(1, 30, (batch_size,))
        repeat_ratio = torch.randint(1, 10, (batch_size,))
        last_stay = torch.randint(1, 30, (batch_size,))
        same_country = torch.randint(1, 30, (batch_size,))

        output = model(
            x, booker, device, month, stay,
            trip_len, num_unique, repeat_ratio, last_stay, same_country
        )

        # Output shape should be (batch_size, vocab_size)
        assert output.shape == (batch_size, vocab_size)

        # Should be valid logits (not NaN or Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_city_transformer_different_pooling(self):
        """Test different pooling strategies."""
        vocab_size = 100
        batch_size = 2
        seq_len = 5

        for pooling in ["last", "mean"]:
            model = CityTransformer(
                vocab_size=vocab_size,
                d_model=64,
                nhead=2,
                num_layers=1,
                n_booker_countries=5,
                n_device_classes=2,
                pooling=pooling,
            )

            x = torch.randint(1, vocab_size, (batch_size, seq_len))
            # Generate valid context values within embedding ranges
            context = [
                torch.randint(0, 5, (batch_size,)),    # booker: 0-4
                torch.randint(0, 2, (batch_size,)),    # device: 0-1
                torch.randint(1, 12, (batch_size,)),   # month: 1-11
                torch.randint(1, 30, (batch_size,)),   # stay: 1-29
                torch.randint(1, 30, (batch_size,)),   # trip_len: 1-29
                torch.randint(1, 30, (batch_size,)),   # num_unique: 1-29
                torch.randint(1, 10, (batch_size,)),   # repeat_ratio: 1-9
                torch.randint(1, 30, (batch_size,)),   # last_stay: 1-29
                torch.randint(1, 30, (batch_size,)),   # same_country: 1-29
            ]

            output = model(x, *context)
            assert output.shape == (batch_size, vocab_size)

    def test_city_gru_forward(self):
        """Test GRU forward pass."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100

        model = CityGRU(
            vocab_size=vocab_size,
            pad_token_id=0,
            embedding_dim=64,
            hidden_dim=64,
            n_booker_countries=5,
            n_device_classes=2,
        )

        x = torch.randint(1, vocab_size, (batch_size, seq_len))
        # Generate valid context values within embedding ranges
        context = [
            torch.randint(0, 5, (batch_size,)),    # booker: 0-4
            torch.randint(0, 2, (batch_size,)),    # device: 0-1
            torch.randint(1, 12, (batch_size,)),   # month: 1-11
            torch.randint(1, 30, (batch_size,)),   # stay: 1-29
            torch.randint(1, 30, (batch_size,)),   # trip_len: 1-29
            torch.randint(1, 30, (batch_size,)),   # num_unique: 1-29
            torch.randint(1, 10, (batch_size,)),   # repeat_ratio: 1-9
            torch.randint(1, 30, (batch_size,)),   # last_stay: 1-29
            torch.randint(1, 30, (batch_size,)),   # same_country: 1-29
        ]

        output = model(x, *context)

        assert output.shape == (batch_size, vocab_size)
        assert not torch.isnan(output).any()


class TestRQVAE:
    """Test RQ-VAE autoencoder."""

    def test_rqvae_forward(self):
        """Test RQ-VAE forward pass and loss computation."""
        batch_size = 8
        input_dim = 128

        model = RQVAE(
            input_dim=input_dim,
            latent_dim=64,
            hidden_dim=128,
            num_levels=2,
            codebook_size=64,
        )

        # Random input vectors (e.g., Word2Vec embeddings)
        x = torch.randn(batch_size, input_dim)

        result = model(x)

        # Check output structure
        assert 'loss' in result
        assert 'recon_loss' in result
        assert 'vq_loss' in result
        assert 'codes' in result
        assert 'x_hat' in result

        # Check shapes
        assert result['x_hat'].shape == x.shape
        assert len(result['codes']) == 2  # 2 levels

        # Check loss values are valid
        assert result['loss'].item() > 0
        assert result['recon_loss'].item() >= 0
        assert result['vq_loss'].item() >= 0

        # Codes should be integers
        for code in result['codes']:
            assert code.dtype == torch.long

    def test_rqvae_encode_codes(self):
        """Test encoding to discrete codes."""
        model = RQVAE(
            input_dim=128,
            latent_dim=64,
            num_levels=2,
            codebook_size=64,
        )

        x = torch.randn(4, 128)
        codes = model.encode_codes(x)

        # Should return (batch_size, num_levels) codes
        assert codes.shape == (4, 2)
        assert codes.dtype == torch.long

        # Codes should be in valid range [0, codebook_size)
        assert (codes >= 0).all()
        assert (codes < 64).all()


class TestRQVAEPredictionModels:
    """Test RQ-VAE prediction models."""

    def test_rqvae_transformer_forward(self):
        """Test RQ-VAE Transformer forward pass."""
        batch_size = 4
        seq_len = 10
        codebook_size = 64

        model = RQVAETransformer(
            codebook_size=codebook_size,
            d_model=64,
            nhead=2,
            num_layers=1,
            n_booker_countries=5,
            n_device_classes=2,
        )

        # Input: sequences of code pairs
        x = torch.randint(0, codebook_size, (batch_size, seq_len, 2))
        # Generate valid context values within embedding ranges
        context = [
            torch.randint(0, 5, (batch_size,)),    # booker: 0-4
            torch.randint(0, 2, (batch_size,)),    # device: 0-1
            torch.randint(1, 12, (batch_size,)),   # month: 1-11
            torch.randint(1, 30, (batch_size,)),   # stay: 1-29
            torch.randint(1, 30, (batch_size,)),   # trip_len: 1-29
            torch.randint(1, 30, (batch_size,)),   # num_unique: 1-29
            torch.randint(1, 10, (batch_size,)),   # repeat_ratio: 1-9
            torch.randint(1, 30, (batch_size,)),   # last_stay: 1-29
            torch.randint(1, 30, (batch_size,)),   # same_country: 1-29
        ]

        output = model(x, *context)

        # Output: logits for each code in the pair
        assert output.shape == (batch_size, 2, codebook_size)

    def test_rqvae_gru_forward(self):
        """Test RQ-VAE GRU forward pass."""
        model = RQVAEGRU(
            codebook_size=64,
            embedding_dim=32,
            hidden_dim=64,
            n_booker_countries=5,
            n_device_classes=2,
        )

        x = torch.randint(0, 64, (4, 10, 2))
        # Generate valid context values within embedding ranges
        context = [
            torch.randint(0, 5, (4,)),    # booker: 0-4
            torch.randint(0, 2, (4,)),    # device: 0-1
            torch.randint(1, 12, (4,)),   # month: 1-11
            torch.randint(1, 30, (4,)),   # stay: 1-29
            torch.randint(1, 30, (4,)),   # trip_len: 1-29
            torch.randint(1, 30, (4,)),   # num_unique: 1-29
            torch.randint(1, 10, (4,)),   # repeat_ratio: 1-9
            torch.randint(1, 30, (4,)),   # last_stay: 1-29
            torch.randint(1, 30, (4,)),   # same_country: 1-29
        ]

        output = model(x, *context)
        assert output.shape == (4, 2, 64)


class TestRQKMeansModels:
    """Test RQ-KMeans models."""

    def test_rqkmeans_transformer_forward(self):
        """Test RQ-KMeans Transformer forward pass."""
        model = RQKMeansTransformer(
            codebook_size=64,
            d_model=64,
            nhead=2,
            num_layers=1,
            n_booker_countries=5,
            n_device_classes=2,
        )

        x = torch.randint(0, 64, (4, 10, 2))
        # Generate valid context values within embedding ranges
        context = [
            torch.randint(0, 5, (4,)),    # booker: 0-4
            torch.randint(0, 2, (4,)),    # device: 0-1
            torch.randint(1, 12, (4,)),   # month: 1-11
            torch.randint(1, 30, (4,)),   # stay: 1-29
            torch.randint(1, 30, (4,)),   # trip_len: 1-29
            torch.randint(1, 30, (4,)),   # num_unique: 1-29
            torch.randint(1, 10, (4,)),   # repeat_ratio: 1-9
            torch.randint(1, 30, (4,)),   # last_stay: 1-29
            torch.randint(1, 30, (4,)),   # same_country: 1-29
        ]

        output = model(x, *context)
        assert output.shape == (4, 2, 64)

    def test_rqkmeans_gru_forward(self):
        """Test RQ-KMeans GRU forward pass."""
        model = RQKmeansGRU(
            codebook_size=64,
            embedding_dim=32,
            hidden_dim=64,
            n_booker_countries=5,
            n_device_classes=2,
        )

        x = torch.randint(0, 64, (4, 10, 2))
        # Generate valid context values within embedding ranges
        context = [
            torch.randint(0, 5, (4,)),    # booker: 0-4
            torch.randint(0, 2, (4,)),    # device: 0-1
            torch.randint(1, 12, (4,)),   # month: 1-11
            torch.randint(1, 30, (4,)),   # stay: 1-29
            torch.randint(1, 30, (4,)),   # trip_len: 1-29
            torch.randint(1, 30, (4,)),   # num_unique: 1-29
            torch.randint(1, 10, (4,)),   # repeat_ratio: 1-9
            torch.randint(1, 30, (4,)),   # last_stay: 1-29
            torch.randint(1, 30, (4,)),   # same_country: 1-29
        ]

        output = model(x, *context)
        assert output.shape == (4, 2, 64)


class TestModelEdgeCases:
    """Test model edge cases and error handling."""

    def test_empty_sequence(self):
        """Test model behavior with minimum sequence length."""
        model = CityGRU(
            vocab_size=100,
            embedding_dim=64,
            hidden_dim=64,
            n_booker_countries=5,
            n_device_classes=2,
        )

        # Sequence of length 1 (minimum)
        x = torch.randint(1, 100, (2, 1))
        # Generate valid context values within embedding ranges
        context = [
            torch.randint(0, 5, (2,)),    # booker: 0-4
            torch.randint(0, 2, (2,)),    # device: 0-1
            torch.randint(1, 12, (2,)),   # month: 1-11
            torch.randint(1, 30, (2,)),   # stay: 1-29
            torch.randint(1, 30, (2,)),   # trip_len: 1-29
            torch.randint(1, 30, (2,)),   # num_unique: 1-29
            torch.randint(1, 10, (2,)),   # repeat_ratio: 1-9
            torch.randint(1, 30, (2,)),   # last_stay: 1-29
            torch.randint(1, 30, (2,)),   # same_country: 1-29
        ]

        output = model(x, *context)
        assert output.shape == (2, 100)

    def test_padding_handling(self):
        """Test that padding tokens are handled correctly."""
        model = CityTransformer(
            vocab_size=100,
            pad_token_id=0,
            d_model=64,
            nhead=2,
            num_layers=1,
            n_booker_countries=5,
            n_device_classes=2,
        )

        # Create sequence with padding
        x = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 0, 0, 0],
        ])
        context = [torch.zeros(2, dtype=torch.long) for _ in range(9)]

        output = model(x, *context)
        assert output.shape == (2, 100)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
