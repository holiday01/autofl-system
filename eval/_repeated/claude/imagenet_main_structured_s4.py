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


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)


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
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return a torchvision model.

    Recognised keys inside config['model_kwargs']:
        arch        (str)  – torchvision architecture name  (default: 'resnet18')
        pretrained  (bool) – load ImageNet weights           (default: False)
        Any remaining keys are forwarded verbatim to the constructor
        (e.g. num_classes).
    """
    model_kwargs = dict(config.get("model_kwargs", {}))
    arch = model_kwargs.pop("arch", "resnet18")
    pretrained = model_kwargs.pop("pretrained", False)

    if arch not in models.__dict__ or not callable(models.__dict__[arch]):
        raise ValueError(
            f"Unknown torchvision architecture '{arch}'. "
            f"Available choices: {model_names}"
        )

    if pretrained:
        model = models.__dict__[arch](pretrained=True, **model_kwargs)
    else:
        model = models.__dict__[arch](**model_kwargs)

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Config keys consulted:
        data_path               (str)   – root of an ImageFolder dataset
        local.batch_size        (int)   – mini-batch size            (default: 16)
        local.num_workers       (int)   – DataLoader workers         (default: 4)
        val_split               (float) – fraction held out for val  (default: 0.2)
        seed                    (int)   – random-split seed          (default: 42)
        allow_synthetic_data    (bool)  – enable fake-data fallback  (default: False)
        synthetic_samples       (int)   – fake dataset size          (default: 1000)
    """
    local_cfg = config.get("local", {})
    batch_size = local_cfg.get("batch_size", 16)
    num_workers = local_cfg.get("num_workers", 4)
    data_path = config.get("data_path", ".")
    val_split = config.get("val_split", 0.2)
    seed = config.get("seed", 42)

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

    # ------------------------------------------------------------------
    # Real-data path
    # ------------------------------------------------------------------
    if os.path.isdir(data_path):
        # Load once without transforms so random_split operates on a single,
        # consistent dataset; transforms are applied via the wrapper below.
        base_dataset = datasets.ImageFolder(data_path, transform=None)

        total = len(base_dataset)
        val_size = int(total * val_split)
        train_size = total - val_size

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(
            base_dataset, [train_size, val_size], generator=generator
        )

        chosen_subset = train_subset if split == "train" else val_subset
        chosen_transform = train_transform if split == "train" else val_transform

        class _TransformWrapper(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                img, label = self.subset[idx]
                if self.transform is not None:
                    img = self.transform(img)
                return img, label

        dataset = _TransformWrapper(chosen_subset, chosen_transform)
        shuffle = split == "train"

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Synthetic-data fallback — must be explicitly enabled by the caller
    # ------------------------------------------------------------------
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"Data path '{data_path}' does not exist or is not a directory. "
            "Supply a valid ImageFolder root via config['data_path'], or set "
            "config['allow_synthetic_data'] = True to use synthetic data for "
            "debugging purposes only."
        )

    num_classes = config.get("model_kwargs", {}).get("num_classes", 1000)
    total_synthetic = config.get("synthetic_samples", 1000)

    class _SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, n, num_classes):
            self.n = n
            self.num_classes = num_classes

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return image, label

    full_dataset = _SyntheticDataset(total_synthetic, num_classes)
    val_size = int(total_synthetic * val_split)
    train_size = total_synthetic - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    chosen_subset = train_subset if split == "train" else val_subset
    shuffle = split == "train"

    return DataLoader(
        chosen_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is solely responsible for loss.backward() and
    optimizer.step(); neither is called here.
    """
    device = next(model.parameters()).device
    images, target = batch
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    model.train()
    criterion = nn.CrossEntropyLoss()
    output = model(images)
    loss = criterion(output, target)
    return loss