"""
Auto-generated FL client module.
Original script: keras.io/examples/vision/mnist_convnet/

Exposes:
  build_model(config)               -> tf.keras.Model
  build_dataloader(config, split)   -> tf.data.Dataset
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()/.numpy()).
  - Do NOT call loss.backward() (or tf.GradientTape.gradient) inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.data import Dataset as TFDataset

# --- Data loading and preprocessing (internal helper) ---
_MNIST_DATA = None

def _load_and_preprocess_mnist(num_classes: int, input_shape: tuple):
    """Loads and preprocesses the MNIST dataset globally, once."""
    global _MNIST_DATA
    if _MNIST_DATA is not None:
        return _MNIST_DATA

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    _MNIST_DATA = {
        "x_train_full": x_train,
        "y_train_full": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "num_classes": num_classes,
        "input_shape": input_shape,
    }
    return _MNIST_DATA


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> Model:
    """
    Builds and returns the Keras model.
    """
    num_classes = config.get("num_classes", 10)
    input_shape = config.get("input_shape", (28, 28, 1))

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> TFDataset:
    """
    Builds and returns a tf.data.Dataset for the specified split.
    """
    local_config = config.get("local", {})
    batch_size = local_config.get("batch_size", config.get("batch_size", 128))
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.1)

    num_classes = config.get("num_classes", 10)
    input_shape = config.get("input_shape", (28, 28, 1))

    # Load and preprocess data (done once)
    data = _load_and_preprocess_mnist(num_classes, input_shape)
    x_train_full, y_train_full = data["x_train_full"], data["y_train_full"]
    x_test, y_test = data["x_test"], data["y_test"]

    if split in ("train", "val"):
        n_train_full = len(x_train_full)
        n_val = int(n_train_full * val_ratio)
        n_train = n_train_full - n_val

        # Create a TensorFlow dataset from the full training data
        full_train_ds = TFDataset.from_tensor_slices((x_train_full, y_train_full))
        
        # Shuffle the full dataset once to ensure a consistent train/val split
        # `reshuffle_each_iteration=False` makes the split deterministic given the seed
        full_train_ds = full_train_ds.shuffle(
            buffer_size=n_train_full, seed=seed, reshuffle_each_iteration=False
        ) 

        if split == "train":
            ds = full_train_ds.skip(n_val).take(n_train)
            # For training, reshuffle data for each epoch
            ds = ds.shuffle(buffer_size=n_train, seed=seed) 
        else: # split == "val"
            ds = full_train_ds.take(n_val)
            # Validation set typically not shuffled
    elif split == "test":
        ds = TFDataset.from_tensor_slices((x_test, y_test))
        # Test set typically not shuffled
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_step(
    model: Model,
    batch: tuple | list,
    optimizer: keras.optimizers.Optimizer,
    config: dict,
) -> tf.Tensor:
    """
    Performs one forward pass and returns the raw loss tensor.
    The FL runtime is responsible for managing the tf.GradientTape context,
    computing gradients, and applying them.
    """
    inputs, targets = batch

    # Call the model in training mode (important for layers like Dropout, BatchNorm)
    outputs = model(inputs, training=True)
    
    # The original script uses "categorical_crossentropy" with "softmax" activation,
    # so we use CategoricalCrossentropy with from_logits=False.
    criterion = keras.losses.CategoricalCrossentropy(from_logits=False)
    
    loss = criterion(targets, outputs)
    
    # Return the raw loss tensor. The FL runtime will handle gradients.
    return loss