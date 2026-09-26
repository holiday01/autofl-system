"""
Auto-generated FL client module.
Original script: keras_char_seq2seq.py

Exposes:
  build_model(config)                       -> keras.Model
  build_dataloader(config, split)           -> Seq2SeqDataLoader
  train_step(model, batch, opt, config)     -> tf.Tensor (scalar loss)
"""

import os
import numpy as np
import keras
import tensorflow as tf
from pathlib import Path

# ── Original source (unchanged) ────────────────────────────────────────
"""
Title: Character-level recurrent sequence-to-sequence model
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2017/09/29
Last modified: 2023/11/22
Description: Character-level recurrent sequence-to-sequence model.
Accelerator: GPU
"""

"""
## Introduction

This example demonstrates how to implement a basic character-level
recurrent sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

**Summary of the algorithm**

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
"""

import numpy as np
import keras
import os
from pathlib import Path

fpath = keras.utils.get_file(origin="http://www.manythings.org/anki/fra-eng.zip")
dirpath = Path(fpath).parent.absolute()
os.system(f"unzip -q {fpath} -d {dirpath}")

batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000
data_path = os.path.join(dirpath, "fra.txt")

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
model.save("s2s_model.keras")

model = keras.models.load_model("s2s_model.keras")

encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]
    return decoded_sentence


for seq_index in range(20):
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)


# ── Data helpers ────────────────────────────────────────────────────────

class Seq2SeqCharDataset:
    """Vectorized character-level seq2seq dataset loaded from a tab-separated text file."""

    def __init__(self, data_path: str, num_samples: int = 10000):
        input_texts, target_texts = [], []
        input_characters, target_characters = set(), set()

        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        for line in lines[: min(num_samples, len(lines) - 1)]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            input_text, target_text = parts[0], parts[1]
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                input_characters.add(char)
            for char in target_text:
                target_characters.add(char)

        input_characters = sorted(input_characters)
        target_characters = sorted(target_characters)
        self.input_token_index = {c: i for i, c in enumerate(input_characters)}
        self.target_token_index = {c: i for i, c in enumerate(target_characters)}
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max(len(t) for t in input_texts)
        self.max_decoder_seq_length = max(len(t) for t in target_texts)

        n = len(input_texts)
        enc = np.zeros((n, self.max_encoder_seq_length, self.num_encoder_tokens), dtype="float32")
        dec_in = np.zeros((n, self.max_decoder_seq_length, self.num_decoder_tokens), dtype="float32")
        dec_tgt = np.zeros((n, self.max_decoder_seq_length, self.num_decoder_tokens), dtype="float32")

        for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(inp):
                enc[i, t, self.input_token_index[char]] = 1.0
            enc[i, t + 1:, self.input_token_index[" "]] = 1.0
            for t, char in enumerate(tgt):
                dec_in[i, t, self.target_token_index[char]] = 1.0
                if t > 0:
                    dec_tgt[i, t - 1, self.target_token_index[char]] = 1.0
            dec_in[i, t + 1:, self.target_token_index[" "]] = 1.0
            dec_tgt[i, t:, self.target_token_index[" "]] = 1.0

        self.encoder_input_data = enc
        self.decoder_input_data = dec_in
        self.decoder_target_data = dec_tgt

    def __len__(self):
        return len(self.encoder_input_data)


class Seq2SeqDataLoader:
    """Iterable yielding (encoder_input, decoder_input, decoder_target) numpy batches."""

    def __init__(
        self,
        dataset: Seq2SeqCharDataset,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ):
        self.enc = dataset.encoder_input_data[indices]
        self.dec_in = dataset.decoder_input_data[indices]
        self.dec_tgt = dataset.decoder_target_data[indices]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(indices)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = np.random.permutation(self.n) if self.shuffle else np.arange(self.n)
        for start in range(0, self.n, self.batch_size):
            idx = order[start : start + self.batch_size]
            yield self.enc[idx], self.dec_in[idx], self.dec_tgt[idx]


def _maybe_download(config: dict) -> str:
    """Return a local path to fra.txt, downloading if needed."""
    data_path = config.get("data_path", None)
    if data_path and os.path.isfile(data_path):
        return data_path
    fpath = keras.utils.get_file(origin="http://www.manythings.org/anki/fra-eng.zip")
    dirpath = Path(fpath).parent.absolute()
    os.system(f"unzip -q {fpath} -d {dirpath}")
    return str(os.path.join(dirpath, "fra.txt"))


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> keras.Model:
    """Instantiate the seq2seq training model. Reads vocab sizes from config['num_encoder_tokens']
    and config['num_decoder_tokens'] (populated as a side effect of build_dataloader)."""
    latent_dim = config.get("latent_dim", 256)
    num_encoder_tokens = config["num_encoder_tokens"]
    num_decoder_tokens = config["num_decoder_tokens"]

    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    return keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)


def build_dataloader(config: dict, split: str = "train") -> Seq2SeqDataLoader:
    """
    Build a Seq2SeqDataLoader for the requested split.
    Expects config to have 'data_path' (optional — will auto-download if absent).
    Reads 'num_samples', 'val_ratio', 'seed', and client-local 'batch_size' from config.
    Populates config['num_encoder_tokens'] and config['num_decoder_tokens'] as a side effect
    so that build_model can be called after this without extra arguments.
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 64))
    num_samples = config.get("num_samples", 10000)
    val_ratio = config.get("val_ratio", 0.2)
    seed = config.get("seed", 42)

    data_path = _maybe_download(config)
    dataset = Seq2SeqCharDataset(data_path=data_path, num_samples=num_samples)

    config["num_encoder_tokens"] = dataset.num_encoder_tokens
    config["num_decoder_tokens"] = dataset.num_decoder_tokens

    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    indices = train_idx if split == "train" else val_idx
    return Seq2SeqDataLoader(
        dataset, indices, batch_size=batch_size, shuffle=(split == "train")
    )


def train_step(
    model: keras.Model,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> tf.Tensor:
    """
    Perform one forward (+ optionally backward) step via GradientTape.
    batch: (encoder_input, decoder_input, decoder_target) numpy arrays or tensors.
    If optimizer is None (preflight forward-only check), skip backward.
    """
    if isinstance(batch, (list, tuple)):
        enc_in, dec_in, dec_tgt = batch[0], batch[1], batch[2]
    elif isinstance(batch, dict):
        enc_in = batch.get("encoder_input", batch.get("x_enc"))
        dec_in = batch.get("decoder_input", batch.get("x_dec"))
        dec_tgt = batch.get("decoder_target", batch.get("y"))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    with tf.GradientTape() as tape:
        outputs = model([enc_in, dec_in], training=(optimizer is not None))
        loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(dec_tgt, outputs)
        )

    if optimizer is not None:
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss