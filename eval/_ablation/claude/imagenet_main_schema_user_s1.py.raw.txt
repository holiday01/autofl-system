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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
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


class _TransformSubset(torch.utils.data.Dataset):
    # Wraps a Subset to apply a transform after random_split, since both subsets
    # share the same underlying ImageFolder (transform=None) and need different transforms.
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)
    extra_kwargs = {k: v for k, v in model_kwargs.items() if k not in ("arch", "pretrained")}

    if pretrained:
        model = models.__dict__[arch](pretrained=True, **extra_kwargs)
    else:
        model = models.__dict__[arch](**extra_kwargs)

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local_config = config.get("local", {})
    batch_size = local_config.get("batch_size", 16)
    data_path = config.get("data_path", ".")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
        # Load with transform=None so _TransformSubset can apply split-specific transforms.
        full_dataset = datasets.ImageFolder(train_dir, transform=None)

        n_total = len(full_dataset)
        n_val = max(1, int(0.1 * n_total))
        n_train = n_total - n_val

        train_subset, val_subset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        if split == "train":
            dataset = _TransformSubset(train_subset, train_transform)
        else:
            dataset = _TransformSubset(val_subset, val_transform)

    else:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Training data directory not found at '{train_dir}'. "
                "Set config['allow_synthetic_data'] = True to use synthetic data."
            )

        n_samples = 1000
        n_val = max(1, int(0.1 * n_samples))
        n_train = n_samples - n_val

        images = torch.randn(n_samples, 3, 224, 224)
        labels = torch.randint(0, 1000, (n_samples,))
        full_dataset = torch.utils.data.TensorDataset(images, labels)

        train_subset, val_subset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        dataset = train_subset if split == "train" else val_subset

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=local_config.get("num_workers", 4),
        pin_memory=True,
    )


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    output = model(images)
    loss = nn.CrossEntropyLoss()(output, target)

    return loss