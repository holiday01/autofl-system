"""
Auto-generated FL client module.
Original script: keras_image_classification_from_scratch.py

Exposes:
  build_model(config)               -> keras.Model
  build_dataloader(config, split)   -> tf.data.Dataset
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach() or .numpy()).
    For Keras, this means returning the tf.Tensor directly from the loss function.
  - Do NOT call loss.backward() (PyTorch) or tape.gradient()/optimizer.apply_gradients() (Keras) inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() (PyTorch) or optimizer.apply_gradients() (Keras) inside train_step.
  - Do NOT call .item() (PyTorch) or .numpy() (Keras) on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data

# Keras data augmentation layers definition
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Model definition based on the original script's make_model function
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> keras.Model:
    """
    Builds and returns a Keras model based on the provided configuration.
    """
    model_kwargs = config.get("model_kwargs", {})
    # Default values from the original script
    input_shape = model_kwargs.get("input_shape", (180, 180, 3))
    num_classes = model_kwargs.get("num_classes", 2)
    return make_model(input_shape=input_shape, num_classes=num_classes)


def build_dataloader(config: dict, split: str = "train") -> tf_data.Dataset:
    """
    Builds and returns a tf.data.Dataset for the specified split.
    Handles data loading, splitting, augmentation (for train split), and prefetching.
    """
    local_config = config.get("local", {})
    image_size = config.get("image_size", (180, 180))
    batch_size = local_config.get("batch_size", config.get("batch_size", 128))
    data_path = config.get("data_path", "PetImages")
    seed = config.get("seed", 1337)
    validation_split = config.get("validation_split", 0.2)

    if not os.path.exists(data_path):
        # In a real FL scenario, the data_path should be correctly set and exist
        # on the client. For this example, if it doesn't exist,
        # `image_dataset_from_directory` will raise an error.
        raise FileNotFoundError(
            f"Data path '{data_path}' not found. Please ensure the 'PetImages' "
            "directory is available and contains 'Cat' and 'Dog' subdirectories."
        )

    # Note: The original script includes logic to filter out corrupted images.
    # This is a data preparation step that should typically happen once
    # before FL training begins, as it modifies the filesystem.
    # We assume the data at data_path is already clean or handled externally.

    # keras.utils.image_dataset_from_directory directly creates batched datasets
    # and handles the validation split internally when subset="both".
    all_datasets = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        interpolation="bilinear",
        label_mode="int", # "int" for integer labels, used by BinaryCrossentropy
    )

    train_ds_raw, val_ds_raw = all_datasets
    ds = train_ds_raw if split == "train" else val_ds_raw

    # Apply data augmentation only to the training set as per original script
    if split == "train":
        ds = ds.map(
            lambda img, label: (data_augmentation(img), label),
            num_parallel_calls=tf_data.AUTOTUNE,
        )

    # Apply prefetching for performance
    ds = ds.prefetch(tf_data.AUTOTUNE)

    return ds


def train_step(
    model: keras.Model,
    batch: tuple | list,
    optimizer, # Keras optimizer instance; provided by FL runtime, but typically compiled into model
    config: dict,
) -> keras.Tensor:
    """
    Performs one forward pass and returns the raw loss tensor.
    Follows the FL contract: no backward pass, no optimizer step, no tensor detachment/conversion.
    """
    inputs, targets = batch

    # Ensure inputs are appropriate for the model
    # The image_dataset_from_directory already yields float32 images in [0, 255]
    # The model's first layer (Rescaling) handles normalization to [0, 1].

    # Perform a forward pass
    outputs = model(inputs, training=True) # training=True enables dropout, batch norm updates

    # Get loss function from config, or use default from script
    loss_fn_name = config.get("loss_fn", "BinaryCrossentropy")
    loss_fn_kwargs = config.get("loss_fn_kwargs", {"from_logits": True})

    # Instantiate the loss criterion
    if loss_fn_name == "BinaryCrossentropy":
        criterion = keras.losses.BinaryCrossentropy(**loss_fn_kwargs)
    elif hasattr(keras.losses, loss_fn_name):
        criterion = getattr(keras.losses, loss_fn_name)(**loss_fn_kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")

    # Calculate the loss
    loss = criterion(targets, outputs)

    return loss