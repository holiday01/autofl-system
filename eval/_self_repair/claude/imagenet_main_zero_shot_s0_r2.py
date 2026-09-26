import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


def build_model(config):
    arch = config.get("arch", "resnet18")
    pretrained = config.get("pretrained", False)
    if pretrained:
        model = models.__dict__[arch](pretrained=True)
    else:
        model = models.__dict__[arch]()
    device = torch.device(config.get("device", "cpu"))
    model.to(device)
    return model


def build_dataloader(config, split):
    assert split in ("train", "val")
    batch_size = config.get("batch_size", 256)
    workers = config.get("workers", 4)
    dummy = config.get("dummy", False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_root = config.get("data", "imagenet")
    split_dir = os.path.join(data_root, split)

    if dummy or not os.path.isdir(split_dir):
        if split == "train":
            dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        else:
            dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        if split == "train":
            dataset = datasets.ImageFolder(
                split_dir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            dataset = datasets.ImageFolder(
                split_dir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=True,
    )
    return loader


def train_step(model, batch, optimizer, config):
    device = torch.device(config.get("device", "cpu"))
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()

    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.get("lr", 0.1),
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 1e-4),
        )

    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    output = model(images)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        correct = pred.t().eq(target.view(1, -1)).reshape(-1).float().sum(0)
        acc1 = correct.mul_(100.0 / images.size(0)).item()
    return {"loss": loss.item(), "acc1": acc1}