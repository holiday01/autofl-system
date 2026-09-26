"""
Auto-generated FL client module.
Original script: autofl/examples/wsi_mlp_train.py

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── Original source (unchanged) ────────────────────────────────────────
"""
Standalone WSI feature classification training script (pre-FL version).
Based on fl_wsi MLPClassifier: 9-class cancer type classification
on top of frozen foundation model features (pre-extracted .npy files).
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class WSIFeatureDataset(Dataset):
    """Load pre-extracted foundation model features (.npy) for cancer classification."""

    def __init__(self, root: str, feature_dim: int = 1024, num_classes: int = 9):
        self.root = root
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        npy_files = []
        labels = []
        if os.path.isdir(root):
            for label_idx, subdir in enumerate(sorted(os.listdir(root))):
                subpath = os.path.join(root, subdir)
                if os.path.isdir(subpath):
                    for f in os.listdir(subpath):
                        if f.endswith(".npy"):
                            npy_files.append(os.path.join(subpath, f))
                            labels.append(label_idx)
        if npy_files:
            self.features = [np.load(p) for p in npy_files]
            self.labels = labels
        else:
            # synthetic fallback for testing
            self.features = [np.random.randn(feature_dim).astype(np.float32)
                             for _ in range(200)]
            self.labels = [i % num_classes for i in range(200)]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=9, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, loader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}  "
              f"acc={correct/total:.3f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WSIFeatureDataset(root="./wsi_features", feature_dim=1024)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = MLPClassifier(input_dim=1024, num_classes=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train(model, loader, optimizer, criterion, device, epochs=10)
    torch.save(model.state_dict(), "wsi_mlp_classifier.pt")
    print("Saved.")


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate the model. Override __init__ kwargs via config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    return MLPClassifier(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for the requested split.
    Expects config to have 'data_path' and optionally 'val_ratio'.
    Client-local batch_size and num_workers are read from config['local'].
    """
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    dataset_kwargs = config.get("dataset_kwargs", {})
    data_path = config.get("data_path", ".")
    full_dataset = WSIFeatureDataset(root=data_path, **dataset_kwargs)
    val_ratio = config.get("val_ratio", 0.1)
    n_val = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )
    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass only.  Returns the loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally.
    Do NOT call backward() or optimizer.step() here.
    Do NOT return loss.detach() — the FL runtime requires grad_fn to remain.
    """
    local  = config.get("local", {})
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    # Return raw loss WITH grad_fn — do NOT detach
    return loss
