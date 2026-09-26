from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer.
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


class APP_MATCHER(Dataset):
    def __init__(self, root, train, download=False):
        super(APP_MATCHER, self).__init__()
        self.dataset = datasets.MNIST(root, train=train, download=download)
        self.data = self.dataset.data.unsqueeze(1).clone()
        self.group_examples()

    def group_examples(self):
        np_arr = np.array(self.dataset.targets.clone(), dtype=None, copy=None)
        self.grouped_examples = {}
        for i in range(0, 10):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        selected_class = random.randint(0, 9)
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
        index_1 = self.grouped_examples[selected_class][random_index_1]
        image_1 = self.data[index_1].clone().float()

        if index % 2 == 0:
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
            index_2 = self.grouped_examples[selected_class][random_index_2]
            image_2 = self.data[index_2].clone().float()
            target = torch.tensor(1, dtype=torch.float)
        else:
            other_selected_class = random.randint(0, 9)
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, 9)
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0] - 1)
            index_2 = self.grouped_examples[other_selected_class][random_index_2]
            image_2 = self.data[index_2].clone().float()
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target


class _SyntheticSiameseDataset(Dataset):
    """Synthetic fallback dataset producing random image pairs and binary similarity labels."""

    def __init__(self, size: int = 1000):
        self.size = size
        self.image_1 = torch.randn(size, 1, 28, 28)
        self.image_2 = torch.randn(size, 1, 28, 28)
        self.targets = torch.randint(0, 2, (size,)).float()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.image_1[index], self.image_2[index], self.targets[index]


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the SiameseNetwork.

    SiameseNetwork takes no constructor arguments, but model_kwargs is
    forwarded for forward-compatibility with subclass configurations.
    """
    model_kwargs = config.get("model_kwargs", {})
    model = SiameseNetwork(**model_kwargs)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Attempts to load the real MNIST-backed APP_MATCHER dataset from
    config['data_path'].  If loading fails and
    config['allow_synthetic_data'] is True, falls back to a synthetic
    dataset.  If loading fails and the flag is False (the default), raises
    FileNotFoundError so the FL runtime can surface the problem clearly.

    The full dataset is split 80 / 20 into train and val subsets using
    random_split.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    full_dataset = None
    load_error: Exception | None = None

    try:
        full_dataset = APP_MATCHER(data_path, train=True, download=True)
    except Exception as exc:
        load_error = exc

    if full_dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real MNIST dataset could not be loaded from '{data_path}' "
                f"(original error: {load_error}). "
                "Provide a valid data_path with the MNIST dataset, or set "
                "config['allow_synthetic_data'] = True to use synthetic data."
            )
        full_dataset = _SyntheticSiameseDataset(size=1000)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    if split == "train":
        return DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    elif split == "val":
        return DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Unknown split '{split}': expected 'train' or 'val'.")


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do so.
    """
    device = next(model.parameters()).device

    images_1, images_2, targets = batch
    images_1 = images_1.to(device)
    images_2 = images_2.to(device)
    targets = targets.to(device)

    criterion = nn.BCELoss()
    outputs = model(images_1, images_2).squeeze()
    loss = criterion(outputs, targets)

    return loss