# ============================================================================
#  AutoFL benchmark expansion — multi-file script with sibling modules inlined
#  Upstream : https://github.com/pytorch/vision/blob/7b0e250acf82aac5a2389f54c6855da17bfeace9/references/similarity/train.py
#  Pinned   : repo pytorch/vision @ 7b0e250acf82aac5a2389f54c6855da17bfeace9 (default branch HEAD on 2026-09-22)
#  File     : last commit c35d3855ccbfa6a36e6ae6337a1f2c721c1f1e78 (2023-12-14); first commit at this path 2019-07-17; raw sha256 52242d4cc32a4e1b7c30ee8cf08a2dc46cd26f90e4aef62a209a3bfc56e73e5f
#  siblings loss.py, model.py, sampler.py inlined at their import sites; entry code otherwise unchanged
# ============================================================================
import os

import torch
import torchvision.transforms as transforms
# ---- BEGIN inlined sibling module loss.py (upstream references/similarity/loss.py @ d367a01a18a3ae6bee13d8be3b63fd6a581ea46f, sha256 2acb571a5a30fd198ae300ad0e808558e69a8e8ba8d1c256de7bf3f0dd1760ed) ----
"""
    Pytorch adaptation of https://omoindrot.github.io/triplet-loss
    https://github.com/omoindrot/tensorflow-triplet-loss
"""
import torch
import torch.nn as nn


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

    # hardest positive for every anchor
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # Add max value in each row to invalid negatives
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # hardest negative for every anchor
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    triplet_loss = hardest_positive_dist - hardest_negative_dist + margin
    triplet_loss[triplet_loss < 0] = 0

    triplet_loss = triplet_loss.mean()

    return triplet_loss, -1


def batch_all_triplet_loss(labels, embeddings, margin, p):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=p)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def _get_triplet_mask(labels):
    # Check that i, j and k are distinct
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
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    return labels.unsqueeze(0) != labels.unsqueeze(1)
# ---- END inlined sibling module loss.py ----
# ---- BEGIN inlined sibling module model.py (upstream references/similarity/model.py @ d367a01a18a3ae6bee13d8be3b63fd6a581ea46f, sha256 61d0c421ed3ca796b8e2041f222e4a91be9e71a6797f3a0c409e44269e30634b) ----
import torch.nn as nn
import torchvision.models as models


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
# ---- END inlined sibling module model.py ----
# ---- BEGIN inlined sibling module sampler.py (upstream references/similarity/sampler.py @ 7dc5e5bd60b55eb4e6ea5c1265d6dc7b17d2e917, sha256 151850e05036a5ccac9be51348cf8a475317f0cb31b76a0a3c90313528a95b76) ----
import random
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler


def create_groups(groups, k):
    """Bins sample indices with respect to groups, remove bins with less than k samples

    Args:
        groups (list[int]): where ith index stores ith sample's group id

    Returns:
        defaultdict[list]: Bins of sample indices, binned by group_idx
    """
    group_samples = defaultdict(list)
    for sample_idx, group_idx in enumerate(groups):
        group_samples[group_idx].append(sample_idx)

    keys_to_remove = []
    for key in group_samples:
        if len(group_samples[key]) < k:
            keys_to_remove.append(key)
            continue

    for key in keys_to_remove:
        group_samples.pop(key)

    return group_samples


class PKSampler(Sampler):
    """
    Randomly samples from a dataset  while ensuring that each batch (of size p * k)
    includes samples from exactly p labels, with k samples for each label.

    Args:
        groups (list[int]): List where the ith entry is the group_id/label of the ith sample in the dataset.
        p (int): Number of labels/groups to be sampled from in a batch
        k (int): Number of samples for each label/group in a batch
    """

    def __init__(self, groups, p, k):
        self.p = p
        self.k = k
        self.groups = create_groups(groups, self.k)

        # Ensures there are enough classes to sample from
        if len(self.groups) < p:
            raise ValueError("There are not enough classes to sample from")

    def __iter__(self):
        # Shuffle samples within groups
        for key in self.groups:
            random.shuffle(self.groups[key])

        # Keep track of the number of samples left for each group
        group_samples_remaining = {}
        for key in self.groups:
            group_samples_remaining[key] = len(self.groups[key])

        while len(group_samples_remaining) > self.p:
            # Select p groups at random from valid/remaining groups
            group_ids = list(group_samples_remaining.keys())
            selected_group_idxs = torch.multinomial(torch.ones(len(group_ids)), self.p).tolist()
            for i in selected_group_idxs:
                group_id = group_ids[i]
                group = self.groups[group_id]
                for _ in range(self.k):
                    # No need to pick samples at random since group samples are shuffled
                    sample_idx = len(group) - group_samples_remaining[group_id]
                    yield group[sample_idx]
                    group_samples_remaining[group_id] -= 1

                # Don't sample from group if it has less than k samples remaining
                if group_samples_remaining[group_id] < self.k:
                    group_samples_remaining.pop(group_id)
# ---- END inlined sibling module sampler.py ----
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


def train_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq):
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        samples, targets = data[0].to(device), data[1].to(device)

        embeddings = model(samples)

        loss, frac_pos_triplets = criterion(embeddings, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_frac_pos_triplets += float(frac_pos_triplets)

        if i % print_freq == print_freq - 1:
            i += 1
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            print(f"[{epoch:d}, {i:d}] | loss: {avg_loss:.4f} | % avg hard triplets: {avg_trip:.2f}%")
            running_loss = 0
            running_frac_pos_triplets = 0


def find_best_threshold(dists, targets, device):
    best_thresh = 0.01
    best_correct = 0
    for thresh in torch.arange(0.0, 1.51, 0.01):
        predictions = dists <= thresh.to(device)
        correct = torch.sum(predictions == targets.to(device)).item()
        if correct > best_correct:
            best_thresh = thresh
            best_correct = correct

    accuracy = 100.0 * best_correct / dists.size(0)

    return best_thresh, accuracy


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    embeds, labels = [], []
    dists, targets = None, None

    for data in loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    dists = dists[mask == 1]
    targets = targets[mask == 1]

    threshold, accuracy = find_best_threshold(dists, targets, device)

    print(f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}")


def save(model, epoch, save_dir, file_name):
    file_name = "epoch_" + str(epoch) + "__" + file_name
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    p = args.labels_per_batch
    k = args.samples_per_label
    batch_size = p * k

    model = EmbeddingNet()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))

    model.to(device)

    criterion = TripletMarginLoss(margin=args.margin)
    optimizer = Adam(model.parameters(), lr=args.lr)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    # Using FMNIST to demonstrate embedding learning using triplet loss. This dataset can
    # be replaced with any classification dataset.
    train_dataset = FashionMNIST(args.dataset_dir, train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(args.dataset_dir, train=False, transform=transform, download=True)

    # targets is a list where the i_th element corresponds to the label of i_th dataset element.
    # This is required for PKSampler to randomly sample from exactly p classes. You will need to
    # construct targets while building your dataset. Some datasets (such as ImageFolder) have a
    # targets attribute with the same format.
    targets = train_dataset.targets.tolist()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=PKSampler(targets, p, k), num_workers=args.workers
    )
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers)

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, test_loader, device)
        return

    for epoch in range(1, args.epochs + 1):
        print("Training...")
        train_epoch(model, optimizer, criterion, train_loader, device, epoch, args.print_freq)

        print("Evaluating...")
        evaluate(model, test_loader, device)

        print("Saving...")
        save(model, epoch, args.save_dir, "ckpt.pth")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Embedding Learning")

    parser.add_argument("--dataset-dir", default="/tmp/fmnist/", type=str, help="FashionMNIST dataset directory path")
    parser.add_argument(
        "-p", "--labels-per-batch", default=8, type=int, help="Number of unique labels/classes per batch"
    )
    parser.add_argument("-k", "--samples-per-label", default=8, type=int, help="Number of samples per label in a batch")
    parser.add_argument("--eval-batch-size", default=512, type=int, help="batch size for evaluation")
    parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--margin", default=0.2, type=float, help="Triplet loss margin")
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--save-dir", default=".", type=str, help="Model save directory")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
