from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_city_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    pad_token_id: int = 0,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    return model


def recommend_top4_cities(
    model: nn.Module,
    test_loader: DataLoader,
    idx_to_city: dict[int, int],
    top_popular_cities: list[int],
    reserved_token_ids: set[int] | None = None,
    topk_candidates: int = 50,
    device: torch.device | None = None,
) -> list[list[int]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if reserved_token_ids is None:
        reserved_token_ids = {0, 1}

    model.to(device)
    model.eval()
    outputs: list[list[int]] = []

    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            _, top_indices = torch.topk(probs, k=min(topk_candidates, probs.size(1)), dim=1)

            for row in top_indices:
                recs: list[int] = []
                for token_id in row.tolist():
                    if token_id in reserved_token_ids:
                        continue
                    city_id = idx_to_city.get(token_id)
                    if city_id is not None and city_id not in recs:
                        recs.append(city_id)
                    if len(recs) == 4:
                        break

                for city_id in top_popular_cities:
                    if len(recs) < 4 and city_id not in recs:
                        recs.append(city_id)
                outputs.append(recs[:4])

    return outputs


def compute_top_popular_cities(train_city_series, k: int = 4) -> list[int]:
    counts = Counter(train_city_series.tolist())
    return [city_id for city_id, _ in counts.most_common(k)]
