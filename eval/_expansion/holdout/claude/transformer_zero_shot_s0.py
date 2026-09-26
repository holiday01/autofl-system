import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from lightning.pytorch.demos import Transformer, WikiText2


def build_model(config):
    dataset = WikiText2()
    vocab_size = config.get("vocab_size", dataset.vocab_size)
    model = Transformer(vocab_size=vocab_size)
    return model


def build_dataloader(config, split):
    dataset = WikiText2()
    n = len(dataset)

    train_size = config.get("train_size", n - 4000)
    val_size = config.get("val_size", 2000)
    test_size = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    batch_size = config.get("batch_size", 20)

    splits = {
        "train": (train_dataset, True),
        "val": (val_dataset, False),
        "test": (test_dataset, False),
    }

    if split not in splits:
        raise ValueError(f"split must be one of {list(splits.keys())}, got '{split}'")

    ds, shuffle = splits[split]
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_step(model, batch, optimizer, config):
    model.train()
    input, target = batch

    optimizer.zero_grad()
    output = model(input, target)
    loss = F.nll_loss(output, target.view(-1))
    loss.backward()

    grad_clip = config.get("gradient_clip_val", 0.25)
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    return loss.item()