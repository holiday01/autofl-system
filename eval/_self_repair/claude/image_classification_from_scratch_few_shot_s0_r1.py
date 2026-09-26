"""
Auto-generated FL client module.
Original script: cats_vs_dogs_train.py

Exposes:
  build_model(config)                   -> keras.Model
  build_dataloader(config, split)       -> tf.data.Dataset
  train_step(model, batch, opt, config) -> loss tensor
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import data as tf_data

# ── Original source (unchanged) ────────────────────────────────────────
"""
Title: Image classification from scratch
Author: fchollet
Date created: 2020/04/27
Last modified: 2023/11/09
Description: Training an image classifier from scratch on the Kaggle Cats vs Dogs dataset.
"""

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    units = 1 if num_classes == 2 else num_classes
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
    print(f"Deleted {num_skipped} images.")

    image_size = (180, 180)
    batch_size = 128

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "PetImages",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    ).prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    model.fit(train_ds, epochs=25, callbacks=callbacks, validation_data=val_ds)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> keras.Model:
    """Instantiate the model. Override kwargs via config['model_kwargs']."""
    image_size = tuple(config.get("image_size", [180, 180]))
    num_classes = config.get("model_kwargs", {}).get("num_classes", config.get("num_classes", 2))
    return make_model(input_shape=image_size + (3,), num_classes=num_classes)


def build_dataloader(config: dict, split: str = "train") -> tf_data.Dataset:
    """
    Build a tf.data.Dataset for the requested split.
    Expects config to have 'data_path' and optionally 'val_split'.
    Client-local batch_size is read from config['local'].
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    image_size  = tuple(config.get("image_size", [180, 180]))
    val_split   = config.get("val_split", 0.2)
    seed        = config.get("seed", 1337)
    data_path   = config.get("data_path", "PetImages")

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=val_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    if split == "train":
        return train_ds.map(
            lambda img, label: (data_augmentation(img), label),
            num_parallel_calls=tf_data.AUTOTUNE,
        ).prefetch(tf_data.AUTOTUNE)
    else:
        return val_ds.prefetch(tf_data.AUTOTUNE)


def train_step(
    model: keras.Model,
    batch,
    optimizer,
    config: dict,
) -> tf.Tensor:
    """
    Perform one forward (+ optionally backward) step via GradientTape.
    If optimizer is None (preflight forward-only check), skip backward.
    """
    num_classes = config.get("num_classes", 2)

    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if num_classes == 2:
        criterion = keras.losses.BinaryCrossentropy(from_logits=True)
        targets = tf.cast(targets, tf.float32)
    else:
        criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    with tf.GradientTape() as tape:
        outputs = model(inputs, training=(optimizer is not None))
        if num_classes == 2:
            outputs = tf.squeeze(outputs, axis=-1)
        loss = criterion(targets, outputs)

    if optimizer is not None:
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss