import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConv, self).__init__()
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
        super(GCN, self).__init__()
        self.gc1 = GraphConv(input_dim, hidden_dim, use_bias=use_bias)
        self.gc2 = GraphConv(hidden_dim, output_dim, use_bias=use_bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor, adj_mat):
        x = self.gc1(input_tensor, adj_mat)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj_mat)
        return F.log_softmax(x, dim=1)


def _load_cora(path='./cora', device='cpu'):
    content_path = os.path.join(path, 'cora.content')
    cites_path = os.path.join(path, 'cora.cites')

    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32))
    scale_vector = torch.sum(features, dim=1)
    scale_vector = 1 / scale_vector
    scale_vector[scale_vector == float('inf')] = 0
    scale_vector = torch.diag(scale_vector).to_sparse()
    features = scale_vector @ features

    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True)
    labels = torch.LongTensor(labels)

    idx = content_tensor[:, 0].astype(np.int32)
    idx_map = {id: pos for pos, id in enumerate(idx)}
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], cites_tensor)),
        dtype=np.int32,
    )

    V = len(idx)
    E = edges.shape[0]
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64)
    adj_mat = torch.eye(V) + adj_mat

    degree_mat = torch.sum(adj_mat, dim=1)
    degree_mat = torch.sqrt(1 / degree_mat)
    degree_mat[degree_mat == float('inf')] = 0
    degree_mat = torch.diag(degree_mat).to_sparse()
    adj_mat = degree_mat @ adj_mat @ degree_mat

    return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)


def build_model(config):
    """
    config keys:
        input_dim (int)
        hidden_dim (int, default 16)
        output_dim (int)
        use_bias (bool, default True)
        dropout_p (float, default 0.5)
        device (str, default 'cpu')
    """
    device = config.get('device', 'cpu')
    model = GCN(
        input_dim=config['input_dim'],
        hidden_dim=config.get('hidden_dim', 16),
        output_dim=config['output_dim'],
        use_bias=config.get('use_bias', True),
        dropout_p=config.get('dropout_p', 0.5),
    ).to(device)
    return model


def build_dataloader(config, split):
    """
    Loads Cora and returns a single-element list containing a dict with the full graph
    masked to the requested split. The GCN operates on the full graph, so masking is
    encoded as an index tensor rather than sub-sampling.

    config keys:
        cora_path (str, default './cora')
        device (str, default 'cpu')
        seed (int, default 42)
        train_size (int, default nodes after first 1500; val: 500; test: 1000)

    split: one of 'train', 'val', 'test'

    Returns:
        list of one dict: {'features', 'adj_mat', 'labels', 'mask'}
    """
    device = config.get('device', 'cpu')
    path = config.get('cora_path', './cora')
    seed = config.get('seed', 42)

    features, labels, adj_mat = _load_cora(path=path, device=device)

    torch.manual_seed(seed)
    idx = torch.randperm(len(labels)).to(device)
    idx_test = idx[:1000]
    idx_val = idx[1000:1500]
    idx_train = idx[1500:]

    mask_map = {'train': idx_train, 'val': idx_val, 'test': idx_test}
    if split not in mask_map:
        raise ValueError(f"split must be one of 'train', 'val', 'test', got '{split}'")

    batch = {
        'features': features,
        'adj_mat': adj_mat,
        'labels': labels,
        'mask': mask_map[split],
    }
    return [batch]


def train_step(model, batch, optimizer, config):
    """
    Performs one federated training step on the provided batch.

    Args:
        model: GCN instance
        batch (dict): output element from build_dataloader with keys
                      'features', 'adj_mat', 'labels', 'mask'
        optimizer: torch optimizer bound to model parameters
        config (dict): may contain 'device'

    Returns:
        dict with 'loss' (float) and 'acc' (float) on the training mask
    """
    features = batch['features']
    adj_mat = batch['adj_mat']
    labels = batch['labels']
    mask = batch['mask']

    criterion = nn.NLLLoss()

    model.train()
    optimizer.zero_grad()

    output = model(features, adj_mat)
    loss = criterion(output[mask], labels[mask])
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = output[mask].argmax(dim=1)
        acc = (preds == labels[mask]).float().mean().item()

    return {'loss': loss.item(), 'acc': acc}