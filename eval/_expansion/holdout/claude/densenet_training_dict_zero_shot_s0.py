import os
import numpy as np
import torch
import monai
from monai.data import DataLoader
from monai.transforms import (
    Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd
)


def build_model(config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = monai.networks.nets.DenseNet121(
        spatial_dims=config.get("spatial_dims", 3),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 2),
    ).to(device)
    return model


def build_dataloader(config, split):
    data_path = config["data_path"]
    images = [os.path.join(data_path, f) for f in config["images"]]
    labels = np.array(config["labels"], dtype=np.int64)

    n_train = config.get("n_train", 10)
    if split == "train":
        files = [{"img": img, "label": label} for img, label in zip(images[:n_train], labels[:n_train])]
        transforms = Compose([
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=config.get("spatial_size", (96, 96, 96))),
            RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
        ])
        shuffle = True
    else:
        files = [{"img": img, "label": label} for img, label in zip(images[n_train:], labels[n_train:])]
        transforms = Compose([
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=config.get("spatial_size", (96, 96, 96))),
        ])
        shuffle = False

    dataset = monai.data.Dataset(data=files, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=shuffle,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def train_step(model, batch, optimizer, config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    loss_fn = torch.nn.CrossEntropyLoss()

    inputs = batch["img"].to(device)
    labels = batch["label"].to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    return {"loss": loss.item()}