from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.resnet.apply(self._init_weights)
        self.fc.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        return output.view(output.size()[0], -1)

    def forward(self, input1, input2):
        output = torch.cat((self.forward_once(input1), self.forward_once(input2)), 1)
        return self.sigmoid(self.fc(output))


class APP_MATCHER(Dataset):
    def __init__(self, root, train, download=False):
        super().__init__()
        self.dataset = datasets.MNIST(root, train=train, download=download)
        self.data = self.dataset.data.unsqueeze(1).clone()
        self._group_examples()

    def _group_examples(self):
        np_arr = np.array(self.dataset.targets.clone())
        self.grouped_examples = {i: np.where(np_arr == i)[0] for i in range(10)}

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
            other_class = random.randint(0, 9)
            while other_class == selected_class:
                other_class = random.randint(0, 9)
            random_index_2 = random.randint(0, self.grouped_examples[other_class].shape[0] - 1)
            index_2 = self.grouped_examples[other_class][random_index_2]
            image_2 = self.data[index_2].clone().float()
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target


def build_model(config: dict) -> nn.Module:
    return SiameseNetwork()


def build_dataloader(config: dict, split: str) -> DataLoader:
    is_train = split == "train"
    dataset = APP_MATCHER(
        root=config.get("data_root", "../data"),
        train=is_train,
        download=config.get("download", True),
    )
    batch_size = config.get("batch_size", 64) if is_train else config.get("test_batch_size", 1000)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )


def train_step(
    model: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    device = config.get("device", next(model.parameters()).device)
    images_1, images_2, targets = (t.to(device) for t in batch)
    criterion = nn.BCELoss()
    model.train()
    optimizer.zero_grad()
    outputs = model(images_1, images_2).squeeze()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss