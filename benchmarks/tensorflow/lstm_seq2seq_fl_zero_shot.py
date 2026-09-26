import numpy as np
import keras
import tensorflow as tf
import os
from pathlib import Path
from typing import Any, Dict, Tuple


def build_model(config: Dict[str, Any]) -> keras.Model:
    latent_dim = config["latent_dim"]
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


def _vectorize(config: Dict[str, Any]):
    data_path = config["data_path"]
    num_samples = config.get("num_samples", 10000)

    input_texts, target_texts = [], []
    input_characters, target_characters = set(), set()

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    for line in lines[: min(num_samples, len(lines) - 1)]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        input_text, target_text = parts[0], "\t" + parts[1] + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        input_characters.update(input_text)
        target_characters.update(target_text)

    input_characters = sorted(input_characters)
    target_characters = sorted(target_characters)
    input_token_index = {c: i for i, c in enumerate(input_characters)}
    target_token_index = {c: i for i, c in enumerate(target_characters)}

    num_enc = len(input_characters)
    num_dec = len(target_characters)
    max_enc_len = max(len(t) for t in input_texts)
    max_dec_len = max(len(t) for t in target_texts)
    n = len(input_texts)

    enc_data = np.zeros((n, max_enc_len, num_enc), dtype="float32")
    dec_in = np.zeros((n, max_dec_len, num_dec), dtype="float32")
    dec_tgt = np.zeros((n, max_dec_len, num_dec), dtype="float32")

    space_enc = input_token_index.get(" ", 0)
    space_dec = target_token_index.get(" ", 0)

    for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
        t = 0
        for t, char in enumerate(inp):
            enc_data[i, t, input_token_index[char]] = 1.0
        enc_data[i, t + 1:, space_enc] = 1.0
        for t, char in enumerate(tgt):
            dec_in[i, t, target_token_index[char]] = 1.0
            if t > 0:
                dec_tgt[i, t - 1, target_token_index[char]] = 1.0
        dec_in[i, t + 1:, space_dec] = 1.0
        dec_tgt[i, t:, space_dec] = 1.0

    return enc_data, dec_in, dec_tgt, num_enc, num_dec, input_token_index, target_token_index


def build_dataloader(config: Dict[str, Any], split: str) -> tf.data.Dataset:
    enc_data, dec_in, dec_tgt, num_enc, num_dec, input_token_index, target_token_index = (
        _vectorize(config)
    )

    config["num_encoder_tokens"] = num_enc
    config["num_decoder_tokens"] = num_dec
    config["input_token_index"] = input_token_index
    config["target_token_index"] = target_token_index

    val_split = config.get("validation_split", 0.2)
    split_idx = int(len(enc_data) * (1 - val_split))

    if split == "train":
        e, d, t = enc_data[:split_idx], dec_in[:split_idx], dec_tgt[:split_idx]
    else:
        e, d, t = enc_data[split_idx:], dec_in[split_idx:], dec_tgt[split_idx:]

    batch_size = config.get("batch_size", 64)
    dataset = tf.data.Dataset.from_tensor_slices(((e, d), t))
    if split == "train":
        dataset = dataset.shuffle(buffer_size=len(e), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_step(
    model: keras.Model,
    batch: Tuple,
    optimizer: keras.optimizers.Optimizer,
    config: Dict[str, Any],
) -> Dict[str, float]:
    (encoder_input, decoder_input), decoder_target = batch

    with tf.GradientTape() as tape:
        predictions = model([encoder_input, decoder_input], training=True)
        loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(decoder_target, predictions)
        )

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    accuracy = tf.reduce_mean(
        keras.metrics.categorical_accuracy(decoder_target, predictions)
    )

    return {"loss": float(loss.numpy()), "accuracy": float(accuracy.numpy())}