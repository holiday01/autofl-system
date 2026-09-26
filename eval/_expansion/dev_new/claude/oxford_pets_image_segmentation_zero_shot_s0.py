import os
import random

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io


def build_model(config):
    img_size = tuple(config.get("img_size", (160, 160)))
    num_classes = config.get("num_classes", 3)

    inputs = keras.Input(shape=img_size + (3,))

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    return keras.Model(inputs, outputs)


def build_dataloader(config, split):
    input_dir = config.get("input_dir", "images/")
    target_dir = config.get("target_dir", "annotations/trimaps/")
    img_size = tuple(config.get("img_size", (160, 160)))
    batch_size = config.get("batch_size", 32)
    val_samples = config.get("val_samples", 1000)
    max_dataset_len = config.get("max_dataset_len", None)
    seed = config.get("seed", 1337)

    if split not in ("train", "val"):
        raise ValueError(f"Unknown split {split!r}. Expected 'train' or 'val'.")

    input_img_paths = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".jpg")
    )
    target_img_paths = sorted(
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".png") and not f.startswith(".")
    )

    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)

    if split == "train":
        input_paths = input_img_paths[:-val_samples]
        target_paths = target_img_paths[:-val_samples]
        if max_dataset_len:
            input_paths = input_paths[:max_dataset_len]
            target_paths = target_paths[:max_dataset_len]
    else:
        input_paths = input_img_paths[-val_samples:]
        target_paths = target_img_paths[-val_samples:]

    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")
        target_img -= 1
        return input_img, target_img

    dataset = tf_data.Dataset.from_tensor_slices((input_paths, target_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)


def train_step(model, batch, optimizer, config):
    inputs, targets = batch
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {"loss": float(loss)}