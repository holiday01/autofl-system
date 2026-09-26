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


# ---------------------------------------------------------------------------
# Original helper classes (preserved verbatim from source script)
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
# Internal FL data-loading helpers
# ---------------------------------------------------------------------------

class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a random_split Subset and applies a per-split torchvision transform
    to the raw PIL image returned by the underlying ImageFolder."""

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class _SyntheticImageNet(torch.utils.data.Dataset):
    """Minimal synthetic dataset with ImageNet tensor shapes (3 × 224 × 224).
    Only instantiated when config['allow_synthetic_data'] is explicitly True."""

    def __init__(self, size: int, num_classes: int = 1000):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, 224, 224)
        label = int(torch.randint(0, self.num_classes, (1,)).item())
        return img, label


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return a torchvision model.

    config keys
    -----------
    arch             : str   torchvision model name (default: 'resnet18')
    model_kwargs     : dict  keyword args forwarded to the model constructor
                             (e.g. {'weights': 'IMAGENET1K_V1'})
    """
    arch = config.get("arch", "resnet18")
    model_kwargs = config.get("model_kwargs", {})

    if arch not in models.__dict__ or not callable(models.__dict__[arch]):
        raise ValueError(
            f"Unknown torchvision architecture: {arch!r}. "
            f"Available: {sorted(n for n in models.__dict__ if n.islower() and not n.startswith('__') and callable(models.__dict__[n]))}"
        )

    model = models.__dict__[arch](**model_kwargs)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for *split* ('train' or 'val').

    The full dataset rooted at config['data_path'] is loaded as an
    ImageFolder (transform=None so PIL images are returned), then split
    deterministically 80 / 20 with random_split.  Split-specific transforms
    (train augmentation vs. validation centre-crop pipeline) are applied via
    _TransformSubset after the split so there is no data leakage.

    config keys
    -----------
    data_path             : str   root of an ImageFolder-compatible tree
    local.batch_size      : int   (default 16)
    num_workers           : int   DataLoader worker processes (default 4)
    val_fraction          : float fraction held out for validation (default 0.2)
    split_seed            : int   RNG seed for random_split (default 42)
    num_classes           : int   used only for synthetic data (default 1000)
    synthetic_size        : int   total synthetic samples (default 256)
    allow_synthetic_data  : bool  MUST be True to use synthetic fallback
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    num_workers = config.get("num_workers", 4)
    val_fraction = config.get("val_fraction", 0.2)
    split_seed = config.get("split_seed", 42)
    num_classes = config.get("num_classes", 1000)

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

    generator = torch.Generator().manual_seed(split_seed)

    if os.path.isdir(data_path):
        # Load without transforms so _TransformSubset receives raw PIL images.
        base_dataset = datasets.ImageFolder(data_path, transform=None)
        total = len(base_dataset)
        val_size = max(1, int(val_fraction * total))
        train_size = total - val_size

        train_sub, val_sub = random_split(
            base_dataset, [train_size, val_size], generator=generator
        )

        if split == "train":
            dataset = _TransformSubset(train_sub, train_transform)
            shuffle = True
        else:
            dataset = _TransformSubset(val_sub, val_transform)
            shuffle = False

    else:
        # ------------------------------------------------------------------
        # Real data unavailable — enforce the synthetic-data gate.
        # ------------------------------------------------------------------
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"ImageFolder root '{data_path}' does not exist or is not a "
                "directory. Provide a valid config['data_path'], or set "
                "config['allow_synthetic_data'] = True to use randomly "
                "generated tensors for smoke-testing only."
            )

        synthetic_size = config.get("synthetic_size", 256)
        val_size = max(1, int(val_fraction * synthetic_size))
        train_size = synthetic_size - val_size

        full_synthetic = _SyntheticImageNet(synthetic_size, num_classes=num_classes)
        train_sub, val_sub = random_split(
            full_synthetic, [train_size, val_size], generator=generator
        )
        # Synthetic tensors are already (3, 224, 224) floats; no transform needed.
        dataset = train_sub if split == "train" else val_sub
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run ONE forward pass and return the loss tensor with grad attached.

    The FL runtime owns the backward pass and the optimizer step;
    this function must NOT call loss.backward() or optimizer.step().

    Parameters
    ----------
    model     : the model returned by build_model (already on its target device)
    batch     : (images, target) tuple as yielded by build_dataloader
    optimizer : provided by the FL runtime (unused here, reserved for signature)
    config    : FL config dict (currently unused inside; reserved for extensions)
    """
    device = next(model.parameters()).device

    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss()
    model.train()
    output = model(images)
    loss = criterion(output, target)
    # grad is attached; backward() and optimizer.step() are the runtime's job.
    return loss