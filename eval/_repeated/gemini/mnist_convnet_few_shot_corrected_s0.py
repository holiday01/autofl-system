"""
Auto-generated FL client module (Keras/TensorFlow adaptation).
Original script: [path-to-your-script]/mnist_convnet.py

Exposes:
  build_model(config)               -> keras.Model
  build_dataloader(config, split)   -> tf.data.Dataset
  train_step(model, batch, opt, config) -> loss tf.Tensor (with gradient tape context)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (implicit via tf.GradientTape
    if the FL runtime wraps the call). Do NOT call .numpy() on it.
  - Do NOT call `tape.gradient()` or `optimizer.apply_gradients()` inside train_step.
  - Do NOT call `optimizer.zero_grad()` (not applicable for TF) inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns gradient computation (`tape.gradient`), optimization (`optimizer.apply_gradients`),
  and metric extraction.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> keras.Model:
    kwargs = config.get("model_kwargs", {})
    input_shape = kwargs.get("input_shape", (28, 28, 1))
    num_classes = kwargs.get("num_classes", 10)
    dropout_rate = kwargs.get("dropout_rate", 0.5)

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
    local_config = config.get("local", {})
    batch_size = local_config.get("batch_size", config.get("batch_size", 128))
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.1)
    num_classes = config.get("model_kwargs", {}).get("num_classes", 10)

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    def preprocess_data(images, labels):
        images = images.astype("float32") / 255
        images = np.expand_dims(images, -1)
        labels = keras.utils.to_categorical(labels, num_classes)
        return images, labels

    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    if split == "test":
        x_selected, y_selected = x_test, y_test
    else:  # "train" or "val"
        # Split training data into train and validation sets
        num_total_train_samples = x_train.shape[0]
        num_val_samples = max(1, int(num_total_train_samples * val_ratio))
        num_actual_train_samples = num_total_train_samples - num_val_samples

        if split == "train":
            x_selected, y_selected = x_train[:num_actual_train_samples], y_train[:num_actual_train_samples]
        elif split == "val":
            x_selected, y_selected = x_train[num_actual_train_samples:], y_train[num_actual_train_samples:]
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_selected, y_selected))

    if split == "train":
        # Shuffle only the training dataset
        dataset = dataset.shuffle(buffer_size=1024, seed=seed)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_step(
    model: keras.Model,
    batch: tuple | list,
    optimizer,  # Keras optimizer instance, provided by FL runtime but not used directly here.
    config: dict,
) -> tf.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH gradient tape context (implicit).
    The FL runtime is responsible for creating a `tf.GradientTape` context around this
    call to track gradients. It will then compute gradients via `tape.gradient()`
    and apply them via `optimizer.apply_gradients()`.

    Do NOT perform gradient computation or weight updates here.
    """
    inputs, targets = batch

    # Perform a forward pass, ensuring dropout/batchnorm are in training mode.
    outputs = model(inputs, training=True)

    # Compute the loss. `from_logits=False` because the model's last layer has 'softmax' activation.
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    loss = loss_fn(targets, outputs)

    return loss