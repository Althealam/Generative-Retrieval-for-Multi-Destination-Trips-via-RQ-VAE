import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_rqvae_model(
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
        for batch_x, batch_y, b_b, b_d, b_m, b_s, b_tl, b_nu, b_rr, b_ls, b_sc in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            b_b = b_b.to(device)
            b_d = b_d.to(device)
            b_m = b_m.to(device)
            b_s = b_s.to(device)
            b_tl = b_tl.to(device)
            b_nu = b_nu.to(device)
            b_rr = b_rr.to(device)
            b_ls = b_ls.to(device)
            b_sc = b_sc.to(device)
            optimizer.zero_grad()
            pred1, pred2 = model(
                batch_x, b_b, b_d, b_m, b_s, b_tl, b_nu, b_rr, b_ls, b_sc
            )
            loss1 = criterion(pred1, batch_y[:, 0])
            loss2 = criterion(pred2, batch_y[:, 1])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    return model


def predict_top4_cities_from_rqvae(
    model: nn.Module,
    test_loader: DataLoader,
    code_to_cities: dict[tuple[int, int], list[int]],
    codebook_size: int = 128,
    top_global: list[int] | None = None,
    topk_pairs: int = 100,
    device: torch.device | None = None,
) -> list[list[int]]:
    if top_global is None:
        raise ValueError(
            "top_global is required: pass top_city_ids_from_train(train_set, k=4) "
            "from src.utils.popularity (or your own fallback city ids)."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    all_predictions: list[list[int]] = []

    with torch.no_grad():
        for batch_x, b_b, b_d, b_m, b_s, b_tl, b_nu, b_rr, b_ls, b_sc in test_loader:
            batch_x = batch_x.to(device)
            b_b = b_b.to(device)
            b_d = b_d.to(device)
            b_m = b_m.to(device)
            b_s = b_s.to(device)
            b_tl = b_tl.to(device)
            b_nu = b_nu.to(device)
            b_rr = b_rr.to(device)
            b_ls = b_ls.to(device)
            b_sc = b_sc.to(device)
            pred1, pred2 = model(
                batch_x, b_b, b_d, b_m, b_s, b_tl, b_nu, b_rr, b_ls, b_sc
            )
            log_p1 = torch.log_softmax(pred1, dim=1)
            log_p2 = torch.log_softmax(pred2, dim=1)

            for b in range(len(batch_x)):
                joint = log_p1[b].unsqueeze(1) + log_p2[b].unsqueeze(0)
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

                for fallback_city in top_global:
                    if len(recs) < 4 and fallback_city not in recs:
                        recs.append(fallback_city)
                all_predictions.append(recs[:4])

    return all_predictions
