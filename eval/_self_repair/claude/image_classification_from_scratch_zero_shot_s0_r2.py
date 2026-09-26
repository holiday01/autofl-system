import os
import keras
from keras import layers
import keras.ops as ops

try:
    from tensorflow import data as tf_data
    _AUTOTUNE = tf_data.AUTOTUNE
except ImportError:
    _AUTOTUNE = -1


def build_model(config):
    image_size = config.get("image_size", (180, 180))
    num_classes = config.get("num_classes", 2)
    input_shape = image_size + (3,)

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


def build_dataloader(config, split):
    data_dir = config["data_dir"]
    image_size = config.get("image_size", (180, 180))
    batch_size = config.get("batch_size", 128)
    seed = config.get("seed", 1337)
    validation_split = config.get("validation_split", 0.2)

    assert split in ("train", "val"), f"split must be 'train' or 'val', got {split}"
    subset = "training" if split == "train" else "validation"

    ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset=subset,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    if split == "train":
        augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]

        def augment(img, label):
            for layer in augmentation_layers:
                img = layer(img)
            return img, label

        ds = ds.map(augment, num_parallel_calls=_AUTOTUNE)

    return ds.prefetch(_AUTOTUNE)


def train_step(model, batch, optimizer, config):
    num_classes = config.get("num_classes", 2)
    images, labels = batch

    if num_classes == 2:
        labels = ops.cast(labels, "float32")
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    with keras.GradientTape() as tape:
        logits = model(images, training=True)
        if num_classes == 2:
            logits = ops.squeeze(logits, axis=-1)
        loss = loss_fn(labels, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if num_classes == 2:
        preds = ops.cast(ops.sigmoid(logits) >= 0.5, "float32")
        accuracy = ops.mean(ops.cast(ops.equal(preds, labels), "float32"))
    else:
        preds = ops.argmax(logits, axis=-1)
        accuracy = ops.mean(ops.cast(ops.equal(preds, ops.cast(labels, "int64")), "float32"))

    return {"loss": float(loss), "accuracy": float(accuracy)}