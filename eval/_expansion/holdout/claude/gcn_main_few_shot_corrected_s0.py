"""
Auto-generated FL client module.
Original script: gcn_cora_train.py

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.

NOTE: GCN on Cora is a transductive task — the entire graph is processed
each step; the DataLoader yields one item (full graph + node mask).
Sparse tensors are not stackable, so a passthrough collate_fn is used and
num_workers defaults to 0.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False):
        super().__init__()
        self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_normal_(self.kernel)
        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)

    def forward(self, input_tensor, adj_mat):
        support = torch.mm(input_tensor, self.kernel)
        output = torch.spmm(adj_mat, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True, dropout_p=0.1):
        super().__init__()
        self.gc1 = GraphConv(input_dim, hidden_dim, use_bias=use_bias)
        self.gc2 = GraphConv(hidden_dim, output_dim, use_bias=use_bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor, adj_mat):
        x = self.gc1(input_tensor, adj_mat)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj_mat)
        return F.log_softmax(x, dim=1)


def _load_cora(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    content_path = os.path.join(path, "cora.content")
    cites_path = os.path.join(path, "cora.cites")

    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32))
    scale_vector = torch.sum(features, dim=1)
    scale_vector = 1 / scale_vector
    scale_vector[scale_vector == float("inf")] = 0
    scale_vector = torch.diag(scale_vector).to_sparse()
    features = scale_vector @ features

    _, labels = np.unique(content_tensor[:, -1], return_inverse=True)
    labels = torch.LongTensor(labels)

    idx = content_tensor[:, 0].astype(np.int32)
    idx_map = {id: pos for pos, id in enumerate(idx)}
    edges = np.array(
        [[ idx_map[e[0]], idx_map[e[1]] ] for e in cites_tensor],
        dtype=np.int32,
    )

    V = len(idx)
    E = edges.shape[0]
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64)
    adj_mat = torch.eye(V) + adj_mat

    degree_mat = torch.sum(adj_mat, dim=1)
    degree_mat = torch.sqrt(1 / degree_mat)
    degree_mat[degree_mat == float("inf")] = 0
    degree_mat = torch.diag(degree_mat).to_sparse()
    adj_mat = degree_mat @ adj_mat @ degree_mat

    return features.to_sparse(), labels, adj_mat.to_sparse()


class CoraGraphDataset(Dataset):
    """Wraps the full Cora graph for transductive node classification.

    __len__ == 1: the single item is (features, adj_mat, labels, mask).
    The mask selects which nodes belong to the requested split.
    """

    def __init__(
        self,
        data_path: str = "./cora",
        split: str = "train",
        seed: int = 42,
        val_size: int = 500,
        test_size: int = 1000,
    ):
        features, labels, adj_mat = _load_cora(data_path)
        self.features = features
        self.labels = labels
        self.adj_mat = adj_mat

        rng = torch.Generator()
        rng.manual_seed(seed)
        idx = torch.randperm(len(labels), generator=rng)
        idx_test = idx[:test_size]
        idx_val = idx[test_size: test_size + val_size]
        idx_train = idx[test_size + val_size:]

        self.mask = {"train": idx_train, "val": idx_val, "test": idx_test}[split]

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.features, self.adj_mat, self.labels, self.mask


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return GCN(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    # sparse tensors cannot be forked safely across workers
    num_workers = local.get("num_workers", 0)

    data_path = config.get("data_path", "./cora")
    seed = config.get("seed", 42)
    dataset_kwargs = config.get("dataset_kwargs", {})

    dataset = CoraGraphDataset(
        data_path=data_path,
        split=split,
        seed=seed,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        # sparse tensors cannot be stacked by the default collate; unwrap the
        # single-element list instead
        collate_fn=lambda batch: batch[0],
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    features, adj_mat, labels, mask = batch

    features = features.to(device)
    adj_mat = adj_mat.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    output = model(features, adj_mat)
    loss = nn.NLLLoss()(output[mask], labels[mask])
    return loss