"""
Auto-generated FL client module.
Original script: simple_mnist_convnet.py

Exposes:
  build_model(config)               -> keras.Model
  build_dataloader(config, split)   -> tf.data.Dataset
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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> keras.Model:
    """
    Builds and returns the Keras model.
    Model compilation (optimizer and loss) is handled by the FL runtime.
    """
    num_classes = config.get("num_classes", 10)
    input_shape = config.get("input_shape", (28, 28, 1))
    dropout_rate = config.get("dropout_rate", 0.5)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> tf.data.Dataset:
    """
    Builds and returns a tf.data.Dataset for the specified split.
    This function assumes a client's full dataset is the MNIST training set,
    which is then split locally into train/validation. The MNIST test set
    is used if split='test'.
    """
    num_classes = config.get("num_classes", 10)
    batch_size = config.get("batch_size", 128)
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.1)
    shuffle_buffer_size = config.get("shuffle_buffer_size", 10000)

    # Load the full data for the current client's "pool"
    if split == "test":
        # For the 'test' split, load the global MNIST test set
        (_, _), (x_data, y_data) = keras.datasets.mnist.load_data()
    else:
        # For 'train' or 'val', load the global MNIST training set and split it
        (x_data, y_data), (_, _) = keras.datasets.mnist.load_data()

    # Preprocessing
    x_data = x_data.astype("float32") / 255
    x_data = np.expand_dims(x_data, -1)
    y_data = keras.utils.to_categorical(y_data, num_classes)

    full_tf_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    if split in ("train", "val"):
        n_total = len(x_data)
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val

        # Ensure deterministic shuffling for splitting
        shuffled_dataset = full_tf_dataset.shuffle(
            buffer_size=min(n_total, shuffle_buffer_size), seed=seed
        )

        if split == "train":
            ds = shuffled_dataset.skip(n_val)
            # Re-shuffle the training split for better epoch-to-epoch randomness
            ds = ds.shuffle(
                buffer_size=min(n_train, shuffle_buffer_size), seed=seed + 1
            )
        else:  # split == "val"
            ds = shuffled_dataset.take(n_val)
    elif split == "test":
        ds = full_tf_dataset  # The full test set is used directly
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train_step(
    model: keras.Model,
    batch: tuple | list,
    optimizer: keras.optimizers.Optimizer,  # Optimizer is passed but not used internally per contract
    config: dict,
) -> tf.Tensor:
    """
    Performs ONE forward pass and returns the raw loss tensor with gradient history.
    The FL runtime is responsible for `tf.GradientTape` context, gradient computation,
    and applying updates to the model.
    """
    # Unpack the batch (inputs, targets)
    inputs, targets = batch

    # Perform the forward pass with `training=True` for dropout/batch normalization
    predictions = model(inputs, training=True)

    # Compute the loss
    # The original script used 'categorical_crossentropy' with one-hot encoded labels.
    loss_fn = keras.losses.CategoricalCrossentropy()
    loss = loss_fn(targets, predictions)

    # Return the loss tensor.
    # The FL runtime is expected to manage the tf.GradientTape context
    # around this call to retrieve gradients from this loss.
    return loss