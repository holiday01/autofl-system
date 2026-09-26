import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import pandas as pd
import keras
from keras import layers
import tensorflow as tf


COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
TARGET_FEATURE_NAME = "target"
NUMERIC_FEATURE_NAMES = ["age", "trestbps", "thalach", "oldpeak", "slope", "chol"]

# Fixed global vocabulary so all FL clients share the same input schema and model architecture.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "sex":     [0, 1],
    "cp":      [0, 1, 2, 3, 4],
    "fbs":     [0, 1],
    "restecg": [0, 1, 2],
    "exang":   [0, 1],
    "ca":      [0, 1, 2, 3],
    "thal":    ["fixed", "normal", "reversable"],
}

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())


def _encode_categorical(features, target):
    for feature_name in list(features.keys()):
        if feature_name not in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            continue
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        is_string = features[feature_name].dtype == tf.string
        lookup_cls = layers.StringLookup if is_string else layers.IntegerLookup
        index = lookup_cls(
            vocabulary=vocabulary,
            mask_token=None,
            num_oov_indices=0,
            output_mode="binary",
        )
        features[feature_name] = index(features[feature_name])
    return dict(features), target


def _split_dataframe(config):
    data_path = config.get(
        "data_path",
        "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv",
    )
    val_frac = config.get("val_frac", 0.2)
    seed = config.get("seed", 1337)
    df = pd.read_csv(data_path)
    val_df = df.sample(frac=val_frac, random_state=seed)
    train_df = df.drop(val_df.index)
    return train_df, val_df


def build_dataloader(config, split):
    train_df, val_df = _split_dataframe(config)
    df = train_df if split == "train" else val_df
    batch_size = config.get("batch_size", 32)
    seed = config.get("seed", 1337)

    working = df.copy()
    labels = working.pop(TARGET_FEATURE_NAME)
    ds = tf.data.Dataset.from_tensor_slices((dict(working), labels.values))
    ds = ds.map(_encode_categorical)
    if split == "train":
        ds = ds.shuffle(buffer_size=len(df), seed=seed)
    return ds.batch(batch_size)


def build_model(config):
    train_df, _ = _split_dataframe(config)
    hidden_units = config.get("hidden_units", 32)
    dropout_rate = config.get("dropout_rate", 0.5)

    inputs = {}
    processed = {}

    for feature_name in FEATURE_NAMES:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            num_cats = len(CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name])
            inp = layers.Input(name=feature_name, shape=(num_cats,), dtype="int64")
            inputs[feature_name] = inp
            processed[feature_name] = inp
        else:
            inp = layers.Input(name=feature_name, shape=(1,), dtype="float32")
            normalizer = layers.Normalization()
            normalizer.adapt(
                train_df[feature_name].values.reshape(-1, 1).astype("float32")
            )
            inputs[feature_name] = inp
            processed[feature_name] = normalizer(inp)

    all_features = layers.concatenate(list(processed.values()))
    x = layers.Dense(hidden_units, activation="relu")(all_features)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=output)


def train_step(model, batch, optimizer, config):
    # tf.GradientTape requires KERAS_BACKEND=tensorflow; swap for backend-native
    # autograd (torch.autograd / jax.grad) when using other backends.
    x, y = batch
    loss_fn = keras.losses.BinaryCrossentropy()
    y_true = tf.cast(y, tf.float32)

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y_true, y_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply(grads, model.trainable_variables)

    accuracy = tf.reduce_mean(keras.metrics.binary_accuracy(y_true, y_pred))
    return {"loss": float(loss), "accuracy": float(accuracy)}