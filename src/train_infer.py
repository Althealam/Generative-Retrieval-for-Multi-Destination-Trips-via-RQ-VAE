import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 0.001,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred1, pred2 = model(batch_x)
            loss1 = criterion(pred1, batch_y[:, 0])
            loss2 = criterion(pred2, batch_y[:, 1])
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    return model


def predict_top4_cities(
    model: nn.Module,
    test_loader: DataLoader,
    code_to_cities: dict[tuple[int, int], list[int]],
    top_global: list[int] | None = None,
    topk_pairs: int = 20,
    device: torch.device | None = None,
) -> list[list[int]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if top_global is None:
        top_global = [47499, 23921, 36063, 17013]

    model.to(device)
    model.eval()
    all_predictions: list[list[int]] = []

    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            pred1, pred2 = model(batch_x)

            prob1 = torch.softmax(pred1, dim=1)
            prob2 = torch.softmax(pred2, dim=1)

            for b in range(len(batch_x)):
                combined_probs = torch.outer(prob1[b], prob2[b])
                flat_probs = combined_probs.flatten()
                _, top_indices = torch.topk(flat_probs, k=topk_pairs)

                row_indices = top_indices // 32
                col_indices = top_indices % 32

                recommended_cities: list[int] = []
                for row_code, col_code in zip(row_indices, col_indices):
                    pair = (row_code.item(), col_code.item())
                    if pair in code_to_cities:
                        for city in code_to_cities[pair]:
                            if city not in recommended_cities:
                                recommended_cities.append(city)
                    if len(recommended_cities) >= 4:
                        break

                for fallback_city in top_global:
                    if len(recommended_cities) < 4 and fallback_city not in recommended_cities:
                        recommended_cities.append(fallback_city)

                all_predictions.append(recommended_cities[:4])

    return all_predictions
