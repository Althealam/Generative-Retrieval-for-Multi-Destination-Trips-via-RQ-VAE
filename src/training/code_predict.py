"""Shared train / infer for RQ-KMeans and RQVAE Transformers (two code heads + trip context)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_code_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y, *ctx in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            ctx = tuple(t.to(device) for t in ctx)

            optimizer.zero_grad()
            pred1, pred2 = model(batch_x, *ctx)
            loss = criterion(pred1, batch_y[:, 0]) + criterion(pred2, batch_y[:, 1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    return model


def predict_top4_with_codebook(
    model: nn.Module,
    test_loader: DataLoader,
    code_to_cities: dict[tuple[int, int], list[int]],
    *,
    codebook_size: int,
    top_global: list[int] | None = None,
    topk_pairs: int = 50,
    device: torch.device | None = None,
) -> list[list[int]]:
    if top_global is None:
        raise ValueError("top_global is required (e.g. top_city_ids_from_train(train_set, k=4)).")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    out: list[list[int]] = []

    with torch.no_grad():
        for batch_x, *ctx in test_loader:
            batch_x = batch_x.to(device)
            ctx = tuple(t.to(device) for t in ctx)
            pred1, pred2 = model(batch_x, *ctx)
            log_p1 = torch.log_softmax(pred1, dim=1)
            log_p2 = torch.log_softmax(pred2, dim=1)

            for bi in range(len(batch_x)):
                joint = log_p1[bi].unsqueeze(1) + log_p2[bi].unsqueeze(0)
                flat = joint.flatten()
                k = min(topk_pairs, flat.size(0))
                _, top_indices = torch.topk(flat, k=k)
                row_indices = top_indices // codebook_size
                col_indices = top_indices % codebook_size

                recs: list[int] = []
                for r, c in zip(row_indices, col_indices):
                    pair = (r.item(), c.item())
                    if pair in code_to_cities:
                        for city in code_to_cities[pair]:
                            if city not in recs:
                                recs.append(city)
                    if len(recs) >= 4:
                        break

                for fb in top_global:
                    if len(recs) < 4 and fb not in recs:
                        recs.append(fb)
                out.append(recs[:4])

    return out
