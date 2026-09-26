import os
import numpy as np
import keras
from keras import backend as K, layers
from scipy.signal import butter, filtfilt
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class ComplexSpectrum(keras.layers.Layer):
    def __init__(self, nfft=512, fft_start=8, fft_end=64):
        super().__init__()
        self.nfft = nfft
        self.fft_start = fft_start
        self.fft_end = fft_end

    def call(self, x):
        samples = x.shape[-1]
        x = keras.ops.rfft(x, fft_length=self.nfft)
        real = x[0] / samples
        imag = x[1] / samples
        real = real[:, :, self.fft_start:self.fft_end]
        imag = imag[:, :, self.fft_start:self.fft_end]
        return keras.ops.concatenate((real, imag), axis=-1)


class ChannelComb(keras.layers.Layer):
    def __init__(self, n_channels, drop_rate=0.5):
        super().__init__()
        self.conv = layers.Conv1D(
            2 * n_channels, 1, padding="same",
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        )
        self.normalization = layers.LayerNormalization()
        self.activation = layers.Activation("gelu")
        self.drop = layers.Dropout(drop_rate)

    def call(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return self.drop(x)


class ConvAttention(keras.layers.Layer):
    def __init__(self, n_channels, drop_rate=0.5):
        super().__init__()
        self.norm = layers.LayerNormalization()
        self.conv = layers.Conv1D(
            2 * n_channels, 31, padding="same",
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        )
        self.activation = layers.Activation("gelu")
        self.drop = layers.Dropout(drop_rate)

    def call(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.drop(x)
        return x + residual


class ChannelMLP(keras.layers.Layer):
    def __init__(self, n_features, drop_rate=0.5):
        super().__init__()
        self.norm = layers.LayerNormalization()
        self.mlp = layers.Dense(
            2 * n_features,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        )
        self.activation = layers.Activation("gelu")
        self.drop = layers.Dropout(drop_rate)
        self.cat = layers.Concatenate(axis=1)

    def call(self, x):
        residual = x
        channels = x.shape[1]
        x = self.norm(x)
        output_channels = []
        for i in range(channels):
            c = self.mlp(x[:, :, i])
            c = layers.Reshape([1, -1])(c)
            output_channels.append(c)
        x = self.cat(output_channels)
        x = self.activation(x)
        x = self.drop(x)
        return x + residual


class Encoder(keras.layers.Layer):
    def __init__(self, n_channels, n_features, drop_rate=0.5):
        super().__init__()
        self.attention1 = ConvAttention(n_channels, drop_rate=drop_rate)
        self.mlp1 = ChannelMLP(n_features, drop_rate=drop_rate)
        self.attention2 = ConvAttention(n_channels, drop_rate=drop_rate)
        self.mlp2 = ChannelMLP(n_features, drop_rate=drop_rate)

    def call(self, x):
        x = self.attention1(x)
        x = self.mlp1(x)
        x = self.attention2(x)
        return self.mlp2(x)


class MlpHead(keras.layers.Layer):
    def __init__(self, n_classes, drop_rate=0.5):
        super().__init__()
        self.flatten = layers.Flatten()
        self.drop = layers.Dropout(drop_rate)
        self.linear1 = layers.Dense(
            6 * n_classes,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        )
        self.norm = layers.LayerNormalization()
        self.activation = layers.Activation("gelu")
        self.drop2 = layers.Dropout(drop_rate)
        self.linear2 = layers.Dense(
            n_classes,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        )

    def call(self, x):
        x = self.flatten(x)
        x = self.drop(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.drop2(x)
        return self.linear2(x)


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config):
    """
    config keys:
        fs          : sampling frequency (default 256)
        resolution  : FFT frequency resolution (default 0.25)
        band        : [low, high] Hz (default [8, 64])
        n_channels  : number of EEG electrodes (default 8)
        n_classes   : number of target classes (default 12)
        drop_rate   : dropout probability (default 0.5)
        duration    : epoch length in seconds (default 1.0)
        keras_backend: Keras backend string (default "jax")
    """
    backend = config.get("keras_backend", "jax")
    os.environ["KERAS_BACKEND"] = backend
    K.set_image_data_format("channels_first")

    fs = config.get("fs", 256)
    resolution = config.get("resolution", 0.25)
    band = config.get("band", [8, 64])
    n_channels = config.get("n_channels", 8)
    n_classes = config.get("n_classes", 12)
    drop_rate = config.get("drop_rate", 0.5)
    duration = config.get("duration", 1.0)

    samples = int(duration * fs)
    input_shape = (n_channels, samples)

    nfft = round(fs / resolution)
    fft_start = int(band[0] / resolution)
    fft_end = int(band[1] / resolution) + 1
    n_features = fft_end - fft_start

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        ComplexSpectrum(nfft, fft_start, fft_end),
        ChannelComb(n_channels=n_channels, drop_rate=drop_rate),
        Encoder(n_channels=n_channels, n_features=n_features, drop_rate=drop_rate),
        Encoder(n_channels=n_channels, n_features=n_features, drop_rate=drop_rate),
        MlpHead(n_classes=n_classes, drop_rate=drop_rate),
        layers.Activation("softmax"),
    ])
    return model


def build_dataloader(config, split):
    """
    config keys (beyond build_model keys):
        data_folder : path to directory with s1.mat … s10.mat
        subjects    : list of subject indices (0-based) assigned to this client
        batch_size  : mini-batch size (default 128)
        band        : bandpass filter [low, high] Hz (default [8, 64])
        order       : filter order (default 4)
        fs          : sampling frequency (default 256)
        duration    : epoch length in seconds (default 1.0)
        onset       : visual-latency offset in seconds (default 0.135)
        val_ratio   : fraction of local data held out for validation (default 0.2)

    split : "train" | "val" | "test"
        "test" uses the first subject NOT in config["subjects"] if subjects < 10,
        otherwise falls back to the val split.

    Returns a tf.data.Dataset yielding (x, y) batches.
    """
    import tensorflow as tf

    data_folder = config["data_folder"]
    subjects = config.get("subjects", list(range(10)))
    batch_size = config.get("batch_size", 128)
    band = config.get("band", [8, 64])
    order = config.get("order", 4)
    fs = config.get("fs", 256)
    duration = config.get("duration", 1.0)
    onset_sec = config.get("onset", 0.135)
    val_ratio = config.get("val_ratio", 0.2)

    onset = 38 + int(onset_sec * fs)
    end = int(duration * fs)

    def _load_subject(subj_0based):
        subj = subj_0based + 1
        data = loadmat(f"{data_folder}/s{subj}.mat")
        eeg = data["eeg"].transpose((2, 1, 3, 0))  # samples, ch, blocks, targets
        B, A = butter(order, np.array(band) / (fs / 2), btype="bandpass")
        eeg = filtfilt(B, A, eeg, axis=0)
        eeg = eeg[onset:onset + end, :, :, :]
        _, channels, blocks, targets = eeg.shape
        y = np.tile(np.arange(1, targets + 1), (blocks, 1))
        y = y.reshape((1, blocks * targets), order="F").squeeze()
        x = eeg.reshape((end, channels, blocks * targets), order="F")
        x = x.transpose((2, 1, 0)).astype(np.float32)  # trials x channels x samples
        y = (y - 1).astype(np.int64)
        return x, y

    all_subjects = set(range(10))

    if split in ("train", "val"):
        xs, ys = [], []
        for s in subjects:
            x, y = _load_subject(s)
            xs.append(x)
            ys.append(y)
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)

        n = len(X)
        n_val = max(1, int(n * val_ratio))
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        if split == "train":
            X, Y = X[train_idx], Y[train_idx]
        else:
            X, Y = X[val_idx], Y[val_idx]

    else:  # "test"
        held_out = sorted(all_subjects - set(subjects))
        if held_out:
            X, Y = _load_subject(held_out[0])
        else:
            xs, ys = [], []
            for s in subjects:
                x, y = _load_subject(s)
                xs.append(x)
                ys.append(y)
            X = np.concatenate(xs, axis=0)
            Y = np.concatenate(ys, axis=0)
            n = len(X)
            n_val = max(1, int(n * val_ratio))
            rng = np.random.default_rng(42)
            idx = rng.permutation(n)
            X, Y = X[idx[:n_val]], Y[idx[:n_val]]

    dataset = (
        tf.data.Dataset.from_tensor_slices((X, Y))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def train_step(model, batch, optimizer, config):
    """
    Performs a single gradient-update step.

    Parameters
    ----------
    model     : Keras model returned by build_model
    batch     : (x, y) tuple from the dataloader
    optimizer : Keras optimizer (e.g. keras.optimizers.SGD)
    config    : same config dict (unused here but kept for interface parity)

    Returns
    -------
    dict with keys "loss" and "accuracy" (Python floats)
    """
    import tensorflow as tf

    x, y = batch
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = keras.losses.sparse_categorical_crossentropy(y, logits)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    preds = tf.argmax(logits, axis=-1, output_type=y.dtype)
    acc = tf.reduce_mean(tf.cast(preds == y, tf.float32))

    return {"loss": float(loss.numpy()), "accuracy": float(acc.numpy())}