from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_embedding_model(
    model: nn.Module,
    train_loader: DataLoader,
    pad_token_id: int = 0,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=label_smoothing)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            (
                batch_x,
                batch_y,
                b_b,
                b_d,
                b_a,
                b_m,
                b_s,
                b_tl,
                b_nu,
                b_rr,
                b_ls,
                b_sc,
                b_lc,
                b_uc,
                b_bc,
                b_br,
            ) = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            b_b = b_b.to(device)
            b_d = b_d.to(device)
            b_a = b_a.to(device)
            b_m = b_m.to(device)
            b_s = b_s.to(device)
            b_tl = b_tl.to(device)
            b_nu = b_nu.to(device)
            b_rr = b_rr.to(device)
            b_ls = b_ls.to(device)
            b_sc = b_sc.to(device)
            b_lc = b_lc.to(device)
            b_uc = b_uc.to(device)
            b_bc = b_bc.to(device)
            b_br = b_br.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, b_b, b_d, b_a, b_m, b_s, b_tl, b_nu, b_rr, b_ls, b_sc, b_lc, b_uc, b_bc, b_br)

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
        for batch in test_loader:
            (
                batch_x,
                b_b,
                b_d,
                b_a,
                b_m,
                b_s,
                b_tl,
                b_nu,
                b_rr,
                b_ls,
                b_sc,
                b_lc,
                b_uc,
                b_bc,
                b_br,
            ) = batch
            batch_x = batch_x.to(device)
            b_b = b_b.to(device)
            b_d = b_d.to(device)
            b_a = b_a.to(device)
            b_m = b_m.to(device)
            b_s = b_s.to(device)
            b_tl = b_tl.to(device)
            b_nu = b_nu.to(device)
            b_rr = b_rr.to(device)
            b_ls = b_ls.to(device)
            b_sc = b_sc.to(device)
            b_lc = b_lc.to(device)
            b_uc = b_uc.to(device)
            b_bc = b_bc.to(device)
            b_br = b_br.to(device)
            logits = model(batch_x, b_b, b_d, b_a, b_m, b_s, b_tl, b_nu, b_rr, b_ls, b_sc, b_lc, b_uc, b_bc, b_br)

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
