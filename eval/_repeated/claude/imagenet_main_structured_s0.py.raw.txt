import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split


# ── Helpers preserved verbatim from original script ──────────────────────────

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# ── FL interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return a torchvision model.

    Relevant config keys
    --------------------
    arch          str   torchvision model name (default: 'resnet18')
    model_kwargs  dict  extra keyword arguments forwarded to the constructor,
                        e.g. {"weights": "IMAGENET1K_V1"} or {"num_classes": 10}
    """
    arch = config.get("arch", "resnet18")
    model_kwargs = config.get("model_kwargs", {})

    _available = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )
    if arch not in models.__dict__ or not callable(models.__dict__[arch]):
        raise ValueError(
            f"Unknown torchvision architecture '{arch}'. "
            f"Available names: {_available}"
        )

    model = models.__dict__[arch](**model_kwargs)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    The dataset is expected to be an ImageFolder-structured directory at
    config['data_path'].  A reproducible 80/20 random_split (seed 42) is
    applied to produce train and val subsets; each subset receives the
    appropriate ImageNet transforms.

    Relevant config keys
    --------------------
    data_path              str    root directory of the ImageFolder dataset
    local.batch_size       int    mini-batch size (default: 16)
    val_fraction           float  fraction of data held out for val (default: 0.2)
    num_workers            int    DataLoader worker processes (default: 4)
    allow_synthetic_data   bool   if True and data_path is missing, fall back to
                                  FakeData; if False (default) raise FileNotFoundError
    num_classes            int    number of classes for synthetic data (default: 1000)
    synthetic_train_size   int    synthetic training samples (default: 1280)
    synthetic_val_size     int    synthetic validation samples (default: 256)
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    val_fraction = config.get("val_fraction", 0.2)
    num_workers = config.get("num_workers", 4)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if os.path.isdir(data_path):
        # ── real data path ────────────────────────────────────────────────────
        # Build with train_transform first to run random_split and capture indices.
        full_train_ds = datasets.ImageFolder(data_path, transform=train_transform)
        total = len(full_train_ds)
        val_size = max(1, int(total * val_fraction))
        train_size = total - val_size

        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset_ref = random_split(
            full_train_ds, [train_size, val_size], generator=generator
        )

        if split == "train":
            return DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

        # val: rebuild the dataset with val_transform and apply the same indices
        full_val_ds = datasets.ImageFolder(data_path, transform=val_transform)
        val_ds = Subset(full_val_ds, val_subset_ref.indices)
        return DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # ── real data unavailable ─────────────────────────────────────────────────
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"Data directory '{data_path}' does not exist. "
            "Organise it as an ImageFolder dataset (one sub-directory per class) "
            "or set config['allow_synthetic_data'] = True to use synthetic data "
            "for smoke-testing only."
        )

    warnings.warn(
        f"Real data not found at '{data_path}'. "
        "Falling back to synthetic FakeData — FOR TESTING ONLY.",
        UserWarning,
        stacklevel=2,
    )
    num_classes = config.get("num_classes", 1000)
    if split == "train":
        syn_size = config.get("synthetic_train_size", 1280)
        shuffle = True
    else:
        syn_size = config.get("synthetic_val_size", 256)
        shuffle = False

    fake_ds = datasets.FakeData(
        syn_size, (3, 224, 224), num_classes, transforms.ToTensor()
    )
    return DataLoader(
        fake_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """Run one forward pass and return the scalar loss tensor (grad attached).

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.
    """
    device = next(model.parameters()).device

    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    model.train()
    output = model(images)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)

    # Return loss with grad attached — do NOT call loss.backward() or
    # optimizer.step(); those are handled by the FL runtime.
    return loss