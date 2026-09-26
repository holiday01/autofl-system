"""
Auto-generated FL client module.
Original script: keras-io/examples/vision/image_classification_from_scratch.py

Exposes:
  build_model(config)               -> keras.Model
  build_dataloader(config, split)   -> tf.data.Dataset
  train_step(model, batch, opt, config) -> loss tensor (Keras/TF)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST allow for gradient computation (do NOT call .stop_gradient()).
  - Do NOT call loss.backward() (PyTorch) or tape.gradient()/optimizer.apply_gradients() (Keras/TF) inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss (PyTorch) or .numpy() (Keras/TF).
  The FL runtime owns backward()/tape.gradient(), step()/apply_gradients(), and metric extraction.
"""

import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf # Required for tf.data operations and tf.Tensor type hints
from tensorflow import data as tf_data # Explicitly used for tf_data.AUTOTUNE


# --- Model Definition (from original script) ---

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


# --- FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> keras.Model:
    model_kwargs = config.get("model_kwargs", {})
    # Default parameters based on the original script
    image_size = model_kwargs.get("image_size", (180, 180))
    num_classes = model_kwargs.get("num_classes", 2) # Cats vs Dogs is binary classification

    # Input shape for Conv2D layers is (height, width, channels)
    input_shape = image_size + (3,)
    return make_model(input_shape=input_shape, num_classes=num_classes)


def build_dataloader(config: dict, split: str = "train") -> tf_data.Dataset:
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 128))
    seed = config.get("seed", 1337)
    data_path = config.get("data_path", "PetImages")
    image_size = config.get("image_size", (180, 180))

    # --- Data Cleanup (from original script) ---
    # This part modifies the dataset on disk by deleting corrupted images.
    # In a real FL scenario, clients might be expected to have clean data,
    # or this cleanup might be a one-time pre-federation step.
    # However, since it's in the original script before dataset loading,
    # we include it here. It's designed to be safe if the data_path
    # does not exist or has already been cleaned.
    num_skipped = 0
    full_data_path = data_path
    if os.path.isdir(full_data_path):
        for folder_name in ("Cat", "Dog"):
            folder_path = os.path.join(full_data_path, folder_name)
            if os.path.isdir(folder_path):
                # Using list() to create a copy of the list of files,
                # so modifications (deletions) don't affect iteration.
                for fname in list(os.listdir(folder_path)):
                    fpath = os.path.join(folder_path, fname)
                    # Check if it's a file before attempting to open
                    if not os.path.isfile(fpath):
                        continue
                    is_jfif = False
                    fobj = None
                    try:
                        fobj = open(fpath, "rb")
                        # .peek() reads without advancing the file pointer
                        is_jfif = b"JFIF" in fobj.peek(10)
                    except Exception:
                        # Catch various exceptions during file reading; assume not JFIF if error.
                        pass
                    finally:
                        if fobj:
                            fobj.close()

                    if not is_jfif:
                        num_skipped += 1
                        try:
                            os.remove(fpath)
                        except OSError as e:
                            print(f"Error deleting corrupted image {fpath}: {e}")
            else:
                print(f"Warning: Data subfolder '{folder_path}' not found during cleanup.")
        if num_skipped > 0:
            print(f"Deleted {num_skipped} images from {full_data_path}.")
    else:
        print(f"Warning: Data path '{full_data_path}' not found. Skipping image cleanup.")
        # `keras.utils.image_dataset_from_directory` will raise an error if path is invalid.

    # Dataset loading
    try:
        # validation_split and subset are used to get both train and validation splits
        # from a single call to `image_dataset_from_directory`.
        train_ds_raw, val_ds_raw = keras.utils.image_dataset_from_directory(
            full_data_path,
            validation_split=0.2, # As per original script
            subset="both", # Request both train and validation splits
            seed=seed,
            image_size=image_size,
            batch_size=batch_size, # This sets the batch size for the raw datasets
            labels="inferred", # Default behavior
            label_mode="int", # Default for 0/1 integer labels
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from '{full_data_path}': {e}")

    ds = train_ds_raw if split == "train" else val_ds_raw

    # Data Augmentation (applied only to the training split)
    if split == "train":
        data_augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
        def data_augmentation(images):
            for layer in data_augmentation_layers:
                images = layer(images)
            return images

        # Apply augmentation as a map operation on the dataset, parallelized
        ds = ds.map(
            lambda img, label: (data_augmentation(img), label),
            num_parallel_calls=tf_data.AUTOTUNE,
        )

    # Configure dataset for performance (prefetching)
    ds = ds.prefetch(tf_data.AUTOTUNE)

    return ds


def train_step(
    model: keras.Model,
    batch: tuple | list,
    optimizer, # Keras optimizer; NOT used directly in this function per contract
    config: dict,
) -> tf.Tensor:
    """
    ONE forward pass in Keras/TensorFlow. Returns the raw loss tensor.
    The FL runtime handles gradient computation and optimizer application.
    """
    inputs, targets = batch[0], batch[1]

    # Instantiate the loss function. The original script uses BinaryCrossentropy with from_logits=True.
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    # Perform the forward pass. `training=True` is crucial for layers like Dropout and BatchNorm
    # to behave correctly (e.g., apply dropout, use batch stats during training).
    outputs = model(inputs, training=True)

    # Calculate the loss. The targets from `image_dataset_from_directory` are typically `int32`.
    # `BinaryCrossentropy` with `from_logits=True` can work with `int32` 0/1 targets directly.
    loss = loss_fn(targets, outputs)

    # Return the raw loss tensor. The FL runtime will be responsible for creating a
    # `tf.GradientTape` (or equivalent), computing gradients from this loss, and applying
    # them with an optimizer.
    return loss