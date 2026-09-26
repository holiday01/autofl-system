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


# ── helpers preserved verbatim from the original script ──────────────────────

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


# ── internal dataset helpers ──────────────────────────────────────────────────

class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a random_split Subset and applies a per-split transform.

    The base ImageFolder is loaded with transform=None so that each split
    can independently receive train-time augmentation or val-time resizing.
    """

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]   # img is PIL.Image when base has no transform
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class _SyntheticImageNet(torch.utils.data.Dataset):
    """In-memory synthetic dataset with ImageNet tensor shapes (3 × 224 × 224).

    Only instantiated when config['allow_synthetic_data'] is explicitly True.
    """

    def __init__(self, length: int, num_classes: int = 1000):
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        label = int(torch.randint(0, self.num_classes, (1,)).item())
        return image, label


# ── FL interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate a torchvision model.

    Recognised config['model_kwargs'] keys
    ───────────────────────────────────────
    arch        (str)  torchvision architecture name  default: "resnet18"
    pretrained  (bool) load ImageNet-pretrained weights  default: False
    num_classes (int)  output-class count               default: 1000
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)
    num_classes = model_kwargs.get("num_classes", 1000)

    valid_archs = {
        name for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    }
    if arch not in valid_archs:
        raise ValueError(
            f"Unknown torchvision architecture '{arch}'. "
            f"Available: {sorted(valid_archs)}"
        )

    if pretrained:
        model = models.__dict__[arch](pretrained=True)
    else:
        model = models.__dict__[arch](num_classes=num_classes)

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    config keys
    ───────────
    data_path                  root that contains a 'train/' ImageFolder sub-dir
                               default: "."
    local.batch_size           mini-batch size  default: 16
    local.num_workers          DataLoader worker count  default: 4
    local.synthetic_samples    sample count used when synthetic  default: 128
    val_fraction               fraction reserved for val split  default: 0.2
    model_kwargs.num_classes   class count for synthetic labels  default: 1000
    allow_synthetic_data       MUST be True to enable synthetic fallback
                               (raises FileNotFoundError when False and real
                               data is absent)
    """
    batch_size   = config.get("local", {}).get("batch_size", 16)
    num_workers  = config.get("local", {}).get("num_workers", 4)
    data_path    = config.get("data_path", ".")
    val_fraction = config.get("val_fraction", 0.2)

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

    train_dir = os.path.join(data_path, "train")

    if os.path.isdir(train_dir):
        # Load without any transform; each split wrapper applies its own.
        full_dataset = datasets.ImageFolder(train_dir, transform=None)

        n_total = len(full_dataset)
        n_val   = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val

        train_subset, val_subset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        if split == "val":
            dataset = _TransformSubset(val_subset, val_transform)
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )
        else:
            dataset = _TransformSubset(train_subset, train_transform)
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
            )

    # ── real data not found ───────────────────────────────────────────────────
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"ImageNet 'train' directory not found at '{train_dir}'. "
            "Provide a valid 'data_path' in config, or set "
            "config['allow_synthetic_data'] = True to use synthetic data "
            "for offline testing only."
        )

    # Synthetic fallback — only reachable when allow_synthetic_data is True.
    n_synthetic = config.get("local", {}).get("synthetic_samples", 128)
    num_classes = config.get("model_kwargs", {}).get("num_classes", 1000)

    full_dataset = _SyntheticImageNet(n_synthetic, num_classes=num_classes)
    n_val   = max(1, int(n_synthetic * val_fraction))
    n_train = n_synthetic - n_val

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_subset if split == "train" else val_subset
    return DataLoader(
        chosen, batch_size=batch_size, shuffle=(split == "train"),
        num_workers=0, pin_memory=False,
    )


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime owns the backward pass and the optimiser step;
    this function must NOT call loss.backward() or optimizer.step().
    """
    images, target = batch

    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss()

    model.train()
    output = model(images)
    loss   = criterion(output, target)

    # Intentionally no loss.backward() or optimizer.step() — FL runtime handles that.
    return loss