"""
FL client module – Cats vs Dogs image classifier.

Exposes:
  build_model(config)                   -> keras.Model
  build_dataloader(config, split)       -> iterable dataset
  train_step(model, batch, opt, config) -> loss scalar
"""
import os
import numpy as np

try:
    import keras
    from keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers

try:
    import tensorflow as tf
    from tensorflow import data as tf_data
    _TF_AVAILABLE = True
except ImportError:
    tf = None
    tf_data = None
    _TF_AVAILABLE = False


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
    if _TF_AVAILABLE:
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


# ── FL Interface ───────────────────────────────────────────────────────

def build_model(config: dict):
    image_size = tuple(config.get("image_size", [180, 180]))
    num_classes = config.get("model_kwargs", {}).get(
        "num_classes", config.get("num_classes", 2)
    )
    return make_model(input_shape=image_size + (3,), num_classes=num_classes)


def build_dataloader(config: dict, split: str = "train"):
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 32))
    image_size = tuple(config.get("image_size", [180, 180]))
    val_split = config.get("val_split", 0.2)
    seed = config.get("seed", 1337)
    data_path = config.get("data_path", "PetImages")

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=val_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    if split == "train":
        map_kwargs = {"num_parallel_calls": tf_data.AUTOTUNE} if _TF_AVAILABLE else {}
        ds = train_ds.map(
            lambda img, label: (data_augmentation(img), label), **map_kwargs
        )
        return ds.prefetch(tf_data.AUTOTUNE) if _TF_AVAILABLE else ds
    else:
        return val_ds.prefetch(tf_data.AUTOTUNE) if _TF_AVAILABLE else val_ds


def train_step(model, batch, optimizer, config: dict):
    num_classes = config.get("num_classes", 2)

    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        inputs = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if num_classes == 2:
        criterion = keras.losses.BinaryCrossentropy(from_logits=True)
        targets = keras.ops.cast(targets, "float32")
    else:
        criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    training = optimizer is not None

    if _TF_AVAILABLE:
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=training)
            if num_classes == 2:
                outputs = keras.ops.squeeze(outputs, axis=-1)
            loss = criterion(targets, outputs)
        if optimizer is not None:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    backend = keras.backend.backend()

    if backend == "torch":
        import torch
        outputs = model(inputs, training=training)
        if num_classes == 2:
            outputs = keras.ops.squeeze(outputs, axis=-1)
        loss = criterion(targets, outputs)
        if optimizer is not None:
            if hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad()
            loss.backward()
            if hasattr(optimizer, "step"):
                optimizer.step()
            else:
                optimizer.apply(model.trainable_variables)
        return loss

    if backend == "jax":
        import jax

        def _loss_fn(trainable_vars):
            _orig = [v.numpy() for v in model.trainable_variables]
            for var, val in zip(model.trainable_variables, trainable_vars):
                var.assign(val)
            out = model(inputs, training=training)
            if num_classes == 2:
                out = keras.ops.squeeze(out, axis=-1)
            l = criterion(targets, out)
            for var, val in zip(model.trainable_variables, _orig):
                var.assign(val)
            return l

        if optimizer is not None:
            grads_and_loss = jax.value_and_grad(_loss_fn)(
                [v.numpy() for v in model.trainable_variables]
            )
            loss, grads = grads_and_loss
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            outputs = model(inputs, training=False)
            if num_classes == 2:
                outputs = keras.ops.squeeze(outputs, axis=-1)
            loss = criterion(targets, outputs)
        return loss

    outputs = model(inputs, training=False)
    if num_classes == 2:
        outputs = keras.ops.squeeze(outputs, axis=-1)
    return criterion(targets, outputs)