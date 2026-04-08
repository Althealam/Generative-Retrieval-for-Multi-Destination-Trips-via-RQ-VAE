import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1


class CitySequenceDataset(Dataset):
    def __init__(self, x_values: list[list[int]], y_values: list[int] | None = None):
        self.x_values = [torch.tensor(x, dtype=torch.long) for x in x_values]
        self.y_values = torch.tensor(y_values, dtype=torch.long) if y_values is not None else None

    def __len__(self) -> int:
        return len(self.x_values)

    def __getitem__(self, idx: int):
        if self.y_values is None:
            return self.x_values[idx]
        return self.x_values[idx], self.y_values[idx]


def collate_city_batch(batch):
    if isinstance(batch[0], tuple):
        xs, ys = zip(*batch)
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_TOKEN_ID)
        return xs_padded, torch.stack(ys)
    return pad_sequence(batch, batch_first=True, padding_value=PAD_TOKEN_ID)


def build_city_dataloaders(
    train_x: list[list[int]],
    train_y: list[int],
    test_x: list[list[int]],
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CitySequenceDataset(train_x, train_y)
    test_ds = CitySequenceDataset(test_x)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_city_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_city_batch)
    return train_loader, test_loader
