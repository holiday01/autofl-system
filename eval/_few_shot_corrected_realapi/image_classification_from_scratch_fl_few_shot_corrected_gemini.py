"""
Auto-generated FL client module.
Original script: keras_cats_vs_dogs_from_scratch.py

Exposes:
  build_model(config)               -> keras.Model
  build_dataloader(config, split)   -> tf.data.Dataset
  train_step(model, batch, opt, config) -> tf.Tensor (loss)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST be part of the computation graph (do NOT call .detach() or equivalent).
  - Do NOT call loss.backward() (or tf.GradientTape.gradient) inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .numpy() or .item() on the returned loss.
  The FL runtime owns gradient computation, application, and metric extraction.
"""
import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data


# --- Data Augmentation Layers ---
# These layers are stateless and can be defined once.
_data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def _apply_data_augmentation(images):
    """Applies data augmentation layers to a batch of images."""
    for layer in _data_augmentation_layers:
        images = layer(images)
    return images


# --- Model Definition ---
def _make_model(input_shape, num_classes):
    """
    Builds a Keras model (small Xception-like network) for image classification.
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs) # Standardize pixel values to [0, 1]
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
        units = 1 # For binary classification with BinaryCrossentropy(from_logits=True)
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
    # Default values match the original script's usage
    input_shape = config.get("input_shape", (180, 180, 3))
    num_classes = config.get("num_classes", 2)
    return _make_model(input_shape=input_shape, num_classes=num_classes)


def build_dataloader(config: dict, split: str = "train") -> tf_data.Dataset:
    """
    Builds and returns a tf.data.Dataset for the specified split.

    Assumes the 'PetImages' directory (containing 'Cat' and 'Dog' subdirectories)
    is already present at the specified `data_path` and has been cleaned
    (corrupted images removed).
    """
    local = config.get("local", {})
    # data_path should point to the directory containing 'Cat' and 'Dog' folders
    data_path = config.get("data_path", "PetImages")
    image_size = config.get("image_size", (180, 180))
    batch_size = local.get("batch_size", config.get("batch_size", 128))
    seed = config.get("seed", 1337)
    validation_split = config.get("val_ratio", 0.2)

    # Create the base dataset(s) using keras.utils.image_dataset_from_directory
    # This utility returns a tuple (train_ds, val_ds) when subset="both"
    train_ds_raw, val_ds_raw = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    ds = train_ds_raw if split == "train" else val_ds_raw

    # Apply data augmentation only to the training dataset
    if split == "train":
        ds = ds.map(
            lambda img, label: (_apply_data_augmentation(img), label),
            num_parallel_calls=tf_data.AUTOTUNE,
        )

    # Prefetching samples for performance
    ds = ds.prefetch(tf_data.AUTOTUNE)
    return ds


def train_step(
    model: keras.Model,
    batch: tuple | list,
    optimizer, # Keras optimizer, typically not used for applying gradients in this contract
    config: dict,
) -> tf.Tensor:
    """
    Performs ONE forward pass and returns the raw loss tensor.

    The FL runtime is responsible for wrapping this call in tf.GradientTape,
    computing gradients, and applying them to the model's trainable variables.
    Do NOT call model.train_on_batch, optimizer.apply_gradients, or compute gradients here.
    """
    inputs, targets = batch[0], batch[1]

    # The original script uses BinaryCrossentropy(from_logits=True)
    # Keras loss functions expect (y_true, y_pred)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    # Perform forward pass
    # `training=True` ensures dropout layers are active and batch normalization
    # updates its moving averages.
    outputs = model(inputs, training=True)

    # Calculate loss
    loss = loss_fn(targets, outputs)

    return loss