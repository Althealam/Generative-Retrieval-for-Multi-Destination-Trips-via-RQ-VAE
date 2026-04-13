import torch
import torch.nn as nn

from src.models.rqvae.vector_quantizer import ResidualVectorQuantizer


class RQVAE(nn.Module):
    """Minimal RQ-VAE for vector inputs (e.g., city Word2Vec vectors)."""

    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_levels: int = 2,
        codebook_size: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quantizer = ResidualVectorQuantizer(
            num_levels=num_levels, codebook_size=codebook_size, embedding_dim=latent_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q_st, codes, vq_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q_st)
        recon_loss = nn.functional.mse_loss(x_hat, x)
        loss = recon_loss + 0.25 * vq_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codes": codes,
            "x_hat": x_hat,
        }

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        z_e = self.encoder(x)
        _, codes, _ = self.quantizer(z_e)
        return torch.stack(codes, dim=1)
