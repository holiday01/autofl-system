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
from torch.utils.data import Subset, DataLoader, random_split


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# ---------------------------------------------------------------------------
# Original helper classes preserved verbatim
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FL-specific helpers
# ---------------------------------------------------------------------------

class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a random_split Subset and applies a split-specific transform.

    ImageFolder is loaded with transform=None so that PIL Images are returned
    raw; this wrapper applies the correct augmentation pipeline per split
    without duplicating the dataset on disk.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class _SyntheticImageNetDataset(torch.utils.data.Dataset):
    """Synthetic ImageNet-shaped dataset used only when allow_synthetic_data=True."""

    def __init__(self, size=1000, num_classes=1000):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, 224, 224)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return a torchvision model.

    Recognised keys inside config["model_kwargs"]:
        arch        – torchvision model name (default: "resnet18")
        pretrained  – bool, load ImageNet weights (default: False)
        All other keys are forwarded to the model constructor verbatim
        (e.g. num_classes).
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)

    if arch not in models.__dict__ or not callable(models.__dict__[arch]):
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available: {', '.join(model_names)}"
        )

    # Strip FL-level keys before passing the rest to the constructor.
    constructor_kwargs = {
        k: v for k, v in model_kwargs.items()
        if k not in ("arch", "pretrained")
    }

    if pretrained:
        model = models.__dict__[arch](pretrained=True, **constructor_kwargs)
    else:
        model = models.__dict__[arch](**constructor_kwargs)

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Config keys consumed:
        data_path                   – root ImageFolder directory (default: ".")
        local.batch_size            – mini-batch size (default: 16)
        local.num_workers           – DataLoader worker count (default: 4)
        val_fraction                – fraction reserved for val (default: 0.2)
        model_kwargs.num_classes    – used for synthetic label range (default: 1000)
        synthetic_size              – samples in synthetic dataset (default: 1000)
        allow_synthetic_data        – MUST be True to use synthetic fallback
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    num_workers = config.get("local", {}).get("num_workers", 4)
    data_path = config.get("data_path", ".")
    val_fraction = config.get("val_fraction", 0.2)
    num_classes = config.get("model_kwargs", {}).get("num_classes", 1000)
    synthetic_size = config.get("synthetic_size", 1000)
    allow_synthetic = config.get("allow_synthetic_data", False)

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

    # ---- attempt to load real data ----------------------------------------
    real_loaded = False
    load_error = None
    dataset = None
    shuffle = False

    if os.path.isdir(data_path):
        try:
            # Load without transform so _TransformSubset can apply the correct
            # split-specific pipeline after random_split.
            full_dataset = datasets.ImageFolder(data_path, transform=None)
            total = len(full_dataset)
            val_size = max(1, int(val_fraction * total))
            train_size = total - val_size
            train_subset, val_subset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            if split == "train":
                dataset = _TransformSubset(train_subset, train_transform)
                shuffle = True
            else:
                dataset = _TransformSubset(val_subset, val_transform)
                shuffle = False
            real_loaded = True
        except Exception as exc:
            load_error = exc
    else:
        load_error = FileNotFoundError(
            f"Data path '{data_path}' does not exist or is not a directory."
        )

    # ---- synthetic fallback (only when explicitly permitted) ---------------
    if not real_loaded:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"Real dataset unavailable at '{data_path}'"
                + (f": {load_error}" if load_error else "")
                + ". Set config['allow_synthetic_data'] = True to use "
                  "synthetic data instead."
            )

        full_syn = _SyntheticImageNetDataset(
            size=synthetic_size, num_classes=num_classes
        )
        val_size = max(1, int(val_fraction * synthetic_size))
        train_size = synthetic_size - val_size
        train_subset, val_subset = random_split(
            full_syn,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        if split == "train":
            dataset = train_subset
            shuffle = True
        else:
            dataset = val_subset
            shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run a single forward pass and return the loss WITH grad attached.

    The FL runtime is responsible for loss.backward() and optimizer.step().
    This function must NOT call either.
    """
    device = next(model.parameters()).device

    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss()
    output = model(images)
    loss = criterion(output, target)

    # Return the live loss tensor; grad graph is intact for the FL runtime.
    return loss