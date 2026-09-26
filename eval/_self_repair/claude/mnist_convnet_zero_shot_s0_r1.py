import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(config):
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


def build_dataloader(config, split):
    num_classes = config.get("num_classes", 10)
    batch_size = config.get("batch_size", 128)
    partition = config.get("partition", None)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if split == "train":
        x, y = x_train, y_train
    elif split in ("test", "val"):
        x, y = x_test, y_test
    else:
        raise ValueError(f"Unknown split: {split!r}. Expected 'train', 'val', or 'test'.")

    x = x.astype("float32") / 255
    x = np.expand_dims(x, -1)
    y = keras.utils.to_categorical(y, num_classes)

    if partition is not None:
        start, end = partition
        x, y = x[start:end], y[start:end]

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if split == "train":
        ds = ds.shuffle(buffer_size=len(x), seed=config.get("seed", 42))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train_step(model, batch, optimizer, config):
    x, y = batch
    loss_fn = keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    preds = tf.argmax(logits, axis=1)
    labels = tf.argmax(y, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    return {"loss": float(loss), "accuracy": float(accuracy)}