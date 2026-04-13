import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """Multi-level residual vector quantizer."""

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
        residual = z_e
        quantized_sum = torch.zeros_like(z_e)
        codes: list[torch.Tensor] = []
        vq_loss = torch.tensor(0.0, device=z_e.device)

        for codebook in self.codebooks:
            embeddings = codebook.weight
            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                + embeddings.pow(2).sum(dim=1)
                - 2 * residual @ embeddings.t()
            )
            indices = torch.argmin(distances, dim=1)
            q = F.embedding(indices, embeddings)
            codes.append(indices)

            vq_loss = vq_loss + F.mse_loss(q, residual.detach()) + F.mse_loss(q.detach(), residual)

            quantized_sum = quantized_sum + q
            residual = residual - q

        z_q_st = z_e + (quantized_sum - z_e).detach()
        return z_q_st, codes, vq_loss
