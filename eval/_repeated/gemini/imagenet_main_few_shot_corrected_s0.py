"""
Auto-generated FL client module.
Original script: PyTorch ImageNet Training Example

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
"""
import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the model based on the configuration.
    The FL runtime handles device placement and distribution.
    """
    model_kwargs = config.get("model_kwargs", {})
    
    # Get model architecture, pretrained status, and number of classes from config
    # Defaults are set to match ImageNet training typical values.
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)
    num_classes = model_kwargs.get("num_classes", 1000) # Default to ImageNet classes

    if arch not in models.__dict__:
        raise ValueError(f"Model architecture '{arch}' not found in torchvision.models.")

    # Load the base model
    if pretrained:
        print(f"=> using pre-trained model '{arch}'")
        # Note: pretrained=True is deprecated in newer torchvision versions,
        # but used here to match the original script's style.
        # For new code, consider `weights=models.ResNet18_Weights.IMAGENET1K_V1` etc.
        model = models.__dict__[arch](pretrained=True)
    else:
        print(f"=> creating model '{arch}'")
        # Many torchvision models accept num_classes directly in their constructor
        # when not loading pretrained weights.
        model = models.__dict__[arch](num_classes=num_classes)

    # If a pretrained model was loaded and the task has a different number of classes
    # than the pretraining task (typically 1000 for ImageNet), the final
    # classification layer needs to be replaced.
    if pretrained and num_classes != 1000:
        if hasattr(model, 'fc'):  # Common for ResNets
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential): # Common for VGG, AlexNet
            # Check if the last layer of the classifier is a Linear layer
            if isinstance(model.classifier[-1], nn.Linear):
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            else:
                warnings.warn(f"Classifier's last layer for '{arch}' is not nn.Linear. "
                              "Cannot automatically adjust for `num_classes`.")
        elif hasattr(model, 'head') and hasattr(model.head, 'fc'): # Common for some Vision Transformers
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, num_classes)
        else:
            warnings.warn(f"Cannot automatically adjust final layer for model '{arch}' to {num_classes} classes. "
                          "Model might not be configured correctly for the specified num_classes.")
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a DataLoader for the specified split (e.g., "train", "val").
    """
    local_config = config.get("local", {})
    batch_size = local_config.get("batch_size", config.get("batch_size", 256))
    num_workers = local_config.get("num_workers", config.get("workers", 4))
    pin_memory = local_config.get("pin_memory", True)
    
    data_path = config.get("data_path", "imagenet")
    dummy_data = config.get("dummy_data", False)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if dummy_data:
        print("=> Dummy data is used!")
        # The number of classes for FakeData also defaults to 1000.
        if split == "train":
            dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        else: # "val" or "test"
            dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        # Determine data directory and transforms based on split
        if split == "train":
            dir_path = os.path.join(data_path, 'train')
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif split == "val": # Use "val" split for validation
            dir_path = os.path.join(data_path, 'val')
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise ValueError(f"Unsupported split: {split}. Expected 'train' or 'val'.")

        # Check if the data directory exists for non-dummy data
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Data directory not found: {dir_path}. "
                                    "Please ensure your 'data_path' config is correct "
                                    "or set 'dummy_data' to True for testing.")

        dataset = datasets.ImageFolder(dir_path, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"), # Only shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # Kept for signature consistency as per CONTRACT, not used directly
    config: dict,
) -> torch.Tensor:
    """
    Performs ONE forward pass and returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    # Determine the device the model is currently on
    device = next(model.parameters()).device
    
    # Original script's batch unpacking
    # Assuming batch is (images, target) as yielded by ImageFolder DataLoader
    images, target = batch
    
    # Move data to the same device as the model, if not already there
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Compute model output
    output = model(images)
    
    # Define loss function (CrossEntropyLoss from original script)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    return loss