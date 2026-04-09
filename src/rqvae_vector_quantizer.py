import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """
    Multi-level residual vector quantizer.
    Each level quantizes the residual from previous levels.
    """

    def __init__(self, num_levels: int, codebook_size: int, embedding_dim: int):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, embedding_dim) for _ in range(num_levels)]
        )
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: [batch, embedding_dim]
        Returns:
            z_q_st: straight-through quantized vectors [batch, embedding_dim]
            codes: list[tensor] each [batch]
            vq_loss: sum over levels
        """
        residual = z_e
        quantized_sum = torch.zeros_like(z_e)
        codes: list[torch.Tensor] = []
        vq_loss = torch.tensor(0.0, device=z_e.device)

        for codebook in self.codebooks:
            embeddings = codebook.weight  # [K, D]
            # Squared L2 distance: ||x||^2 + ||e||^2 - 2x.e
            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                + embeddings.pow(2).sum(dim=1)
                - 2 * residual @ embeddings.t()
            )
            indices = torch.argmin(distances, dim=1)  # [B]
            q = F.embedding(indices, embeddings)  # [B, D]
            codes.append(indices)

            # VQ loss (codebook + commitment)
            vq_loss = vq_loss + F.mse_loss(q, residual.detach()) + F.mse_loss(q.detach(), residual)

            quantized_sum = quantized_sum + q
            residual = residual - q

        # Straight-through estimator
        z_q_st = z_e + (quantized_sum - z_e).detach()
        return z_q_st, codes, vq_loss
