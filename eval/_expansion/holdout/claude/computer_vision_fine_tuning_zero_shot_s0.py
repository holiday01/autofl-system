import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models import resnet50, ResNet50_Weights

log = logging.getLogger(__name__)
DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"


def build_model(config: dict) -> nn.Module:
    backbone_name = config.get("backbone", "resnet50")
    weights = ResNet50_Weights.DEFAULT if backbone_name == "resnet50" else "DEFAULT"
    backbone = resnet50(weights=weights)

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

    train_bn = config.get("train_bn", False)
    for module in feature_extractor.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(train_bn)
        else:
            module.requires_grad_(False)

    fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Linear(256, 32),
        nn.Linear(32, 1),
    )

    model = nn.ModuleDict({"feature_extractor": feature_extractor, "fc": fc})
    return model


def build_dataloader(config: dict, split: str) -> DataLoader:
    dl_path = config.get("dl_path", "data")
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 0)

    data_path = Path(dl_path) / "cats_and_dogs_filtered"

    if not data_path.exists():
        download_and_extract_archive(url=DATA_URL, download_root=dl_path, remove_finished=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = ImageFolder(root=data_path / "train", transform=transform)
        return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = ImageFolder(root=data_path / "validation", transform=transform)
        return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


def train_step(
    model: nn.ModuleDict,
    batch: tuple,
    optimizer: optim.Optimizer,
    config: dict,
) -> dict:
    x, y = batch

    features = model["feature_extractor"](x)
    features = features.squeeze(-1).squeeze(-1)
    logits = model["fc"](features)

    y_true = y.view(-1, 1).float().to(logits.device)
    loss = F.binary_cross_entropy_with_logits(input=logits, target=y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= 0.5).int()
        accuracy = (preds == y_true.int()).float().mean().item()

    return {"loss": loss.item(), "accuracy": accuracy}