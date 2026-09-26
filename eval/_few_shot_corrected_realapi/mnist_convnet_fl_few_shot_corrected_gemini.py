"""
Auto-generated FL client module.
Original script: keras_mnist_convnet.py

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
import numpy as np
import tensorflow as tf
import keras
from keras import layers


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> keras.Model:
    """
    Builds and returns a Keras model based on the provided configuration.
    """
    input_shape = config.get("input_shape", (28, 28, 1))
    num_classes = config.get("num_classes", 10)

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


def build_dataloader(config: dict, split: str = "train") -> tf.data.Dataset:
    """
    Builds and returns a tf.data.Dataset for the specified split.
    """
    num_classes = config.get("num_classes", 10)
    input_shape = config.get("input_shape", (28, 28, 1))
    batch_size = config.get("batch_size", 128)
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.1)

    # Load the data
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess data
    def preprocess(images, labels):
        images = images.astype("float32") / 255
        images = np.expand_dims(images, -1)  # Make sure images have shape (28, 28, 1)
        labels = keras.utils.to_categorical(labels, num_classes)
        return images, labels

    x_train_full, y_train_full = preprocess(x_train_full, y_train_full)
    x_test, y_test = preprocess(x_test, y_test)

    # Split training data into train and validation
    n_train_full = x_train_full.shape[0]
    n_val = max(1, int(n_train_full * val_ratio))
    n_train = n_train_full - n_val

    # Use a fixed seed for reproducibility during splitting
    rng = np.random.default_rng(seed)
    indices = np.arange(n_train_full)
    rng.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    x_train, y_train = x_train_full[train_indices], y_train_full[train_indices]
    x_val, y_val = x_train_full[val_indices], y_train_full[val_indices]

    if split == "train":
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(buffer_size=10000, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    elif split == "val":
        dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    elif split == "test":
        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        raise ValueError(f"Unknown split: {split}")

    return dataset


def train_step(
    model: keras.Model,
    batch: tuple | list,
    optimizer: keras.optimizers.Optimizer,
    config: dict,
) -> tf.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    inputs, targets = batch

    # Ensure inputs and targets are TensorFlow tensors
    if not isinstance(inputs, tf.Tensor):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    if not isinstance(targets, tf.Tensor):
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Forward pass: model(inputs, training=True) is important for layers like Dropout/BatchNorm
        outputs = model(inputs, training=True)
        
        # Calculate loss using CategoricalCrossentropy, as specified in the original script
        loss_fn = keras.losses.CategoricalCrossentropy()
        loss = loss_fn(targets, outputs)

    # Return the raw loss tensor. The FL runtime will use tf.GradientTape to compute
    # gradients from this loss and apply them with the optimizer.
    return loss