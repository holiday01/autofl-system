import numpy as np
import keras
from keras import layers


def build_model(config):
    max_features = config.get("max_features", 20000)
    inputs = keras.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(max_features, 128)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model


def build_dataloader(config, split):
    max_features = config.get("max_features", 20000)
    maxlen = config.get("maxlen", 200)
    batch_size = config.get("batch_size", 32)

    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=max_features
    )

    if split == "train":
        x, y = x_train, y_train
    elif split in ("val", "validation"):
        x, y = x_val, y_val
    else:
        raise ValueError(f"Unknown split: {split!r}. Expected 'train' or 'val'.")

    x = keras.utils.pad_sequences(x, maxlen=maxlen)
    y = np.array(y, dtype=np.float32)

    dataset = keras.utils.PyDataset if hasattr(keras.utils, "PyDataset") else None

    indices = np.arange(len(x))
    batches = [
        (x[indices[i:i + batch_size]], y[indices[i:i + batch_size]])
        for i in range(0, len(x), batch_size)
    ]
    return batches


def train_step(model, batch, optimizer, config):
    x_batch, y_batch = batch
    x_batch = keras.ops.convert_to_tensor(x_batch)
    y_batch = keras.ops.convert_to_tensor(y_batch)

    with keras.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        predictions = keras.ops.squeeze(predictions, axis=-1)
        loss = keras.losses.binary_crossentropy(y_batch, predictions)
        loss = keras.ops.mean(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    predictions_binary = keras.ops.cast(predictions >= 0.5, dtype="float32")
    accuracy = keras.ops.mean(
        keras.ops.cast(predictions_binary == y_batch, dtype="float32")
    )

    return {"loss": float(loss), "accuracy": float(accuracy)}