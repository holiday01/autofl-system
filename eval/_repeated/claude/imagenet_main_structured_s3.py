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
from torch.utils.data import DataLoader, Subset

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


# ---------------------------------------------------------------------------
# Helpers preserved verbatim from the original training script
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
# Internal dataset utilities
# ---------------------------------------------------------------------------

class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a random_split Subset and applies a per-split transform.

    ImageFolder is loaded once *without* transforms so that train and val
    subsets can each receive their own augmentation pipeline.
    """

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class _SyntheticImageNetDataset(torch.utils.data.Dataset):
    """Pre-generated random float tensors shaped (3, 224, 224) with integer labels.

    Only instantiated when config['allow_synthetic_data'] is explicitly True.
    """

    def __init__(self, size: int, num_classes: int = 1000):
        self.images = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return a torchvision model.

    Reads from config["model_kwargs"]:
        arch        (str)  – torchvision model name          (default: "resnet18")
        pretrained  (bool) – load ImageNet weights           (default: False)
        num_classes (int)  – number of output classes        (default: 1000)
    """
    model_kwargs = config.get("model_kwargs", {})
    arch        = model_kwargs.get("arch", "resnet18")
    pretrained  = model_kwargs.get("pretrained", False)
    num_classes = model_kwargs.get("num_classes", 1000)

    if arch not in models.__dict__ or not callable(models.__dict__[arch]):
        raise ValueError(
            f"Unknown torchvision architecture: '{arch}'. "
            f"Available: {model_names}"
        )

    if pretrained:
        try:
            # torchvision >= 0.13 weights API
            weights_enum = models.get_model_weights(arch)
            model = models.__dict__[arch](weights=weights_enum.DEFAULT)
        except Exception:
            # Legacy API fallback
            model = models.__dict__[arch](pretrained=True)
    else:
        model = models.__dict__[arch]()

    # Adjust the classification head when num_classes differs from 1000
    if num_classes != 1000:
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier'):
            clf = model.classifier
            if isinstance(clf, nn.Sequential):
                last = clf[-1]
                clf[-1] = nn.Linear(last.in_features, num_classes)
            elif isinstance(clf, nn.Linear):
                model.classifier = nn.Linear(clf.in_features, num_classes)
        elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for *split* ("train" or "val").

    Config keys
    -----------
    data_path            (str)   Root of an ImageFolder tree         (default: ".")
    local.batch_size     (int)   Mini-batch size                     (default: 16)
    local.num_workers    (int)   DataLoader worker processes         (default: 4)
    val_fraction         (float) Fraction of data reserved for val  (default: 0.2)
    allow_synthetic_data (bool)  Allow random-tensor fallback        (default: False)
    synthetic_size       (int)   Total samples when synthetic        (default: 200)
    model_kwargs.num_classes (int) Classes for synthetic labels      (default: 1000)

    When the real dataset is unavailable *and* allow_synthetic_data is False,
    a FileNotFoundError is raised — synthetic data is never silently substituted.
    """
    data_path   = config.get("data_path", ".")
    batch_size  = config.get("local", {}).get("batch_size", 16)
    num_workers = config.get("local", {}).get("num_workers", 4)
    val_frac    = config.get("val_fraction", 0.2)
    num_classes = config.get("model_kwargs", {}).get("num_classes", 1000)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
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

    # ------------------------------------------------------------------ #
    # Attempt to load the real ImageFolder dataset (no transform yet so  #
    # that train and val subsets can get different augmentation pipelines)#
    # ------------------------------------------------------------------ #
    real_dataset = None
    if os.path.isdir(data_path):
        try:
            real_dataset = datasets.ImageFolder(data_path, transform=None)
        except Exception:
            real_dataset = None

    if real_dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real ImageNet-style dataset not found at '{data_path}'. "
                "Ensure 'data_path' points to a directory whose sub-folders "
                "are class names (ImageFolder layout), or set "
                "config['allow_synthetic_data'] = True to use randomly "
                "generated tensors for debugging purposes only."
            )

        # -------- synthetic fallback (explicitly opt-in only) ---------- #
        total_synth = config.get("synthetic_size", 200)
        n_val_synth   = max(1, int(total_synth * val_frac))
        n_train_synth = total_synth - n_val_synth
        synth_size    = n_train_synth if split == "train" else n_val_synth

        synth_dataset = _SyntheticImageNetDataset(synth_size, num_classes=num_classes)
        return DataLoader(
            synth_dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,       # tensors already in memory; no worker overhead
            pin_memory=False,
        )

    # ------------------------------------------------------------------ #
    # Real data: random_split → wrap each subset with its own transform  #
    # ------------------------------------------------------------------ #
    total  = len(real_dataset)
    n_val  = max(1, int(total * val_frac))
    n_train = total - n_val

    train_subset, val_subset = torch.utils.data.random_split(
        real_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    if split == "train":
        out_dataset = _TransformSubset(train_subset, train_transform)
        shuffle     = True
        drop_last   = True
    else:
        out_dataset = _TransformSubset(val_subset, val_transform)
        shuffle     = False
        drop_last   = False

    return DataLoader(
        out_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Execute one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); neither is invoked here.
    """
    images, target = batch
    device = next(model.parameters()).device

    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    if target.dtype != torch.long:
        target = target.long()

    model.train()
    output = model(images)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)   # grad graph intact; do NOT call .backward()

    return loss