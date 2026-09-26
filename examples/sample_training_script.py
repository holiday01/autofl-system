"""
Minimal standard PyTorch training script — used to test the AutoFL converter.
This script intentionally looks like a typical standalone training file.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SyntheticDataset(TensorDataset):
    def __init__(self, root=".", n=200, input_dim=32, num_classes=4):
        X = torch.randn(n, input_dim)
        y = torch.randint(0, num_classes, (n,))
        super().__init__(X, y)


def train(model, loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")


if __name__ == "__main__":
    dataset = SyntheticDataset(n=400)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train(model, loader, optimizer, criterion, epochs=3)
    print("Done.")
