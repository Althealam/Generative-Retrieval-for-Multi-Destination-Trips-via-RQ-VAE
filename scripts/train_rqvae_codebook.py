"""Train RQ-VAE on Word2Vec city vectors; export city→(code1, code2) JSON for the Transformer script."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import create_trip_sequences, train_word2vec
from src.models import RQVAE
from src.utils import data_dir, rqvae_dir


def build_city_vectors(
    train_set: pd.DataFrame, *, vector_size: int = 128, window: int = 10
) -> tuple[list[int], np.ndarray]:
    train_trips = create_trip_sequences(train_set)
    w2v = train_word2vec(train_trips, vector_size=vector_size, window=window)
    unique_cities = sorted(train_set["city_id"].unique().tolist())
    vectors = np.array([w2v.wv[str(city)] for city in unique_cities], dtype=np.float32)
    return unique_cities, vectors


def train_rqvae(
    vectors: np.ndarray, *, epochs: int = 30, batch_size: int = 512, lr: float = 1e-3
) -> tuple[RQVAE, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.from_numpy(vectors))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RQVAE(
        input_dim=vectors.shape[1],
        latent_dim=64,
        hidden_dim=256,
        num_levels=2,
        codebook_size=128,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            out["loss"].backward()
            optimizer.step()

            total_loss += out["loss"].item()
            total_recon += out["recon_loss"].item()
            total_vq += out["vq_loss"].item()

        n = len(loader)
        print(
            f"Epoch {epoch + 1:02d} | loss={total_loss / n:.4f} "
            f"| recon={total_recon / n:.4f} | vq={total_vq / n:.4f}"
        )

    return model, device


def export_city_to_codes(
    model: RQVAE, device: torch.device, unique_cities: list[int], vectors: np.ndarray
) -> dict[int, list[int]]:
    with torch.no_grad():
        x = torch.from_numpy(vectors).to(device)
        codes = model.encode_codes(x).cpu().numpy()
    return {
        int(city_id): [int(codes[i, 0]), int(codes[i, 1])] for i, city_id in enumerate(unique_cities)
    }


def main() -> None:
    output_dir = rqvae_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = pd.read_csv(data_dir() / "train_set.csv")
    unique_cities, vectors = build_city_vectors(train_set, vector_size=128, window=10)
    model, device = train_rqvae(vectors, epochs=30, batch_size=512, lr=1e-3)
    city_to_codes = export_city_to_codes(model, device, unique_cities, vectors)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapping_path = output_dir / f"city_to_codes_rqvae_{timestamp}.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(city_to_codes, f, ensure_ascii=False)

    model_path = output_dir / f"rqvae_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)

    print(f"✅ saved mapping: {mapping_path}")
    print(f"✅ saved model:   {model_path}")


if __name__ == "__main__":
    main()
