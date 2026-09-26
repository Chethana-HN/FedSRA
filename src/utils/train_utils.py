import torch
from torch.utils.data import DataLoader

from .metrics import batch_metrics


def make_loader(dataset, batch_size, shuffle, workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def eval_seg(model, loader, device):
    model.eval()
    values = []

    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"].to(device)

        prediction = torch.sigmoid(model(image))
        values.append(batch_metrics(prediction, target))

    if not values:
        return 0.0, 0.0, 0.0

    return tuple( sum(value[i] for value in values) / len(values) for i in range(3) )