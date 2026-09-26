# ============================================================================
#  AutoFL benchmark expansion — notebook converted to script
#  Upstream : https://github.com/Project-MONAI/tutorials/blob/2ce51396477367791eb53b49cd62617b630d58dd/3d_regression/densenet_training_array.ipynb
#  Pinned   : repo Project-MONAI/tutorials @ 2ce51396477367791eb53b49cd62617b630d58dd (default branch HEAD on 2026-09-22)
#  File     : last commit 3c891ec76d26e437b7b26387589fb9dc40f5e2f8 (2024-09-07); first commit at this path 2024-06-27; raw sha256 49d29dbcc919c65f6773e273bdac2c12972466472772ef8231a4e68d99f81d4b
#  Conversion: nbformat 5.10.4 (jupytext NOT installed); 8 code cells concatenated in notebook order (percent-style cell markers); 1 IPython magic/shell line(s) commented out; markdown cells dropped; nothing else changed.
# ============================================================================
# %% [code cell 2]
# [nb magic stripped] !python -c "import monai" || pip install -q "monai-weekly[nibabel, tqdm]"

# %% [code cell 4]
import logging
import os
import sys
import shutil
import tempfile

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from monai.networks.nets import Regressor

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

# %% [code cell 6]
# Set data directory
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# %% [code cell 7]
# IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
images = [
    os.sep.join([root_dir, "ixi", "IXI314-IOP-0889-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI249-Guys-1072-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI609-HH-2600-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI173-HH-1590-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI020-Guys-0700-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI342-Guys-0909-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI134-Guys-0780-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI577-HH-2661-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI066-Guys-0731-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI130-HH-1528-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI607-Guys-1097-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI175-HH-1570-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI385-HH-2078-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI344-Guys-0905-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI409-Guys-0960-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI584-Guys-1129-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI253-HH-1694-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI092-HH-1436-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI574-IOP-1156-T1.nii.gz"]),
    os.sep.join([root_dir, "ixi", "IXI585-Guys-1130-T1.nii.gz"]),
]

# ages of subjects
ages = np.array(
    [
        45.86,
        68.27,
        29.0,
        29.57,
        39.47,
        48.68,
        47.35,
        64.19,
        46.17,
        38.77,
        83.81,
        72.27,
        64.65,
        62.09,
        70.95,
        41.33,
        24.0,
        33.24,
        50.57,
        28.12,
    ]
)

# %% [code cell 8]
if not os.path.isfile(images[0]):
    resource = "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/IXI-T1.tar"
    md5 = "34901a0593b41dd19c1a1f746eac2d58"

    dataset_dir = os.path.join(root_dir, "ixi")
    tarfile_name = f"{dataset_dir}.tar"

    download_and_extract(resource, tarfile_name, dataset_dir, md5)

# %% [code cell 10]
# Define transforms
train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])

val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

# Define nifti dataset, data loader
check_ds = ImageDataset(image_files=images, labels=ages, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=3, num_workers=2, pin_memory=pin_memory)

im, label = monai.utils.misc.first(check_loader)
print(type(im), im.shape, label, label.shape)

# create a training data loader
train_ds = ImageDataset(image_files=images[:10], labels=ages[:10], transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=pin_memory)

# create a validation data loader
val_ds = ImageDataset(image_files=images[-10:], labels=ages[-10:], transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)

# %% [code cell 12]
model = Regressor(in_shape=[1, 96, 96, 96], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
if torch.cuda.is_available():
    model.cuda()
# It is important that we use nn.MSELoss for regression.
loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()
max_epochs = 5

lowest_rmse = sys.float_info.max
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        all_labels = []
        all_val_outputs = []
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            all_labels.extend(val_labels.cpu().detach().numpy())
            with torch.no_grad():
                val_outputs = model(val_images)
                flattened_val_outputs = [val for sublist in val_outputs.cpu().detach().numpy() for val in sublist]
                all_val_outputs.extend(flattened_val_outputs)

        mse = np.square(np.subtract(all_labels, all_val_outputs)).mean()
        rmse = np.sqrt(mse)

        if rmse < lowest_rmse:
            lowest_rmse = rmse
            lowest_rmse_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current RMSE: {rmse:.4f} ")
        print(f"Best RMSE: {lowest_rmse:.4f} at epoch {lowest_rmse_epoch}")
        writer.add_scalar("val_rmse", rmse, epoch + 1)

print(f"Training completed, lowest_rmse: {lowest_rmse:.4f} at epoch: {lowest_rmse_epoch}")
writer.close()

# %% [code cell 14]
if directory is None:
    shutil.rmtree(root_dir)
