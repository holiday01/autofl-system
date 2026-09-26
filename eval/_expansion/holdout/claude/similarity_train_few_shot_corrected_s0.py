"""
Auto-generated FL client module.
Original script: references/similarity/train.py (pytorch/vision @ 7b0e250)

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.

NOTE: TripletMarginLoss returns (loss, frac_pos_triplets). train_step unpacks
the tuple and returns only the loss tensor; the fraction metric is discarded
here and may be extracted by the runtime from embeddings if needed.
"""
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.datasets import FashionMNIST


# ── Loss (inlined from loss.py) ─────────────────────────────────────────────

class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2.0, mining="batch_all"):
        super().__init__()
        self.margin = margin
        self.p = p
        self.mining = mining
        if mining == "batch_all":
            self.loss_fn = batch_all_triplet_loss
        if mining == "batch_hard":
            self.loss_fn = batch_hard_triplet_loss

    def forward(self, embeddings, labels):
        return self.loss_fn(labels, embeddings, self.margin, self.p)


def batch_hard_triplet_loss(labels, embeddings, margin, p):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=p)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)
    triplet_loss = hardest_positive_dist - hardest_negative_dist + margin
    triplet_loss[triplet_loss < 0] = 0
    return triplet_loss.mean(), -1


def batch_all_triplet_loss(labels, embeddings, margin, p):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=p)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss
    triplet_loss[triplet_loss < 0] = 0
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_positive_triplets


def _get_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    valid_labels = ~i_equal_k & i_equal_j
    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    return labels.unsqueeze(0) != labels.unsqueeze(1)


# ── Model (inlined from model.py) ───────────────────────────────────────────

class EmbeddingNet(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            backbone = models.resnet50(num_classes=128)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=1)
        return x


# ── Sampler (inlined from sampler.py) ───────────────────────────────────────

def _create_groups(groups, k):
    group_samples = defaultdict(list)
    for sample_idx, group_idx in enumerate(groups):
        group_samples[group_idx].append(sample_idx)
    keys_to_remove = [key for key in group_samples if len(group_samples[key]) < k]
    for key in keys_to_remove:
        group_samples.pop(key)
    return group_samples


class PKSampler(Sampler):
    def __init__(self, groups, p, k):
        self.p = p
        self.k = k
        self.groups = _create_groups(groups, self.k)
        if len(self.groups) < p:
            raise ValueError("There are not enough classes to sample from")

    def __iter__(self):
        for key in self.groups:
            random.shuffle(self.groups[key])
        group_samples_remaining = {key: len(self.groups[key]) for key in self.groups}
        while len(group_samples_remaining) > self.p:
            group_ids = list(group_samples_remaining.keys())
            selected_group_idxs = torch.multinomial(torch.ones(len(group_ids)), self.p).tolist()
            for i in selected_group_idxs:
                group_id = group_ids[i]
                group = self.groups[group_id]
                for _ in range(self.k):
                    sample_idx = len(group) - group_samples_remaining[group_id]
                    yield group[sample_idx]
                    group_samples_remaining[group_id] -= 1
                if group_samples_remaining[group_id] < self.k:
                    group_samples_remaining.pop(group_id)


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    embedding_dim = config.get("embedding_dim", 128)
    backbone = models.resnet50(num_classes=embedding_dim)
    return EmbeddingNet(backbone=backbone)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    dataset_dir = config.get("dataset_dir", "/tmp/fmnist/")
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory = local.get("pin_memory", True)

    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
    ])

    is_train = split == "train"
    dataset = FashionMNIST(dataset_dir, train=is_train, transform=transform, download=True)

    if is_train:
        p = config.get("labels_per_batch", 8)
        k = config.get("samples_per_label", 8)
        batch_size = p * k
        targets = dataset.targets.tolist()
        sampler = PKSampler(targets, p, k)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )
    else:
        eval_batch_size = config.get("eval_batch_size", 512)
        return DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    embeddings = model(inputs)
    criterion = TripletMarginLoss(
        margin=config.get("margin", 0.2),
        p=config.get("distance_p", 2.0),
        mining=config.get("mining", "batch_all"),
    )
    loss, _ = criterion(embeddings, targets)
    return loss