"""
Auto-generated FL client module.
Original script: keras-seq2seq-char-level.py
"""
import os
import numpy as np
import keras
import tensorflow as tf
from pathlib import Path

# Module-level variables to store data metadata once processed.
# This avoids re-processing the data multiple times across calls to build_model/build_dataloader.
_data_metadata = {}

def _prepare_data_metadata(data_path_arg: str, num_samples: int):
    """
    Helper function to load data, build vocabularies and calculate sequence lengths.
    Caches results in _data_metadata.
    """
    if _data_metadata:
        return _data_metadata # Already processed

    # Normalize data_path_arg to be a directory where 'fra.txt' is expected or will be downloaded.
    data_dir = Path(data_path_arg)
    
    # First, try to find fra.txt directly within data_dir
    actual_data_file = data_dir / "fra.txt"

    if not actual_data_file.exists():
        # If not found directly, attempt to download and extract using keras.utils.get_file.
        # This utility often extracts into a specific sub-directory within the cache.
        print(f"'{actual_data_file}' not found. Attempting to download fra-eng.zip...")
        
        # Define a consistent cache location for the dataset
        target_cache_dir = Path.home() / ".keras" / "datasets"
        target_cache_subdir = "fra-eng" # This will be the direct parent of fra.txt after extraction
        
        extracted_zip_path = keras.utils.get_file(
            origin="http://www.manythings.org/anki/fra-eng.zip",
            extract=True,
            cache_dir=str(target_cache_dir),
            cache_subdir=target_cache_subdir
        )
        # `extracted_zip_path` here is the *directory* where `fra.txt` is now located,
        # e.g., `~/.keras/datasets/fra-eng`.
        actual_data_file = Path(extracted_zip_path) / "fra.txt"
        
        if not actual_data_file.exists():
            raise FileNotFoundError(f"Failed to find 'fra.txt' after downloading and extracting to {extracted_zip_path}. Expected at {actual_data_file}")
        print(f"Data found at: {actual_data_file}")

    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    with open(actual_data_file, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    for line in lines[: min(num_samples, len(lines) - 1)]:
        parts = line.split("\t")
        if len(parts) < 2: # Skip incomplete lines
            continue
        input_text, target_text = parts[0], parts[1]
        target_text = "\t" + target_text + "\n" # Add start-of-sequence and end-of-sequence tokens
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            input_characters.add(char)
        for char in target_text:
            target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    metadata = {
        "input_texts": input_texts,
        "target_texts": target_texts,
        "input_characters": input_characters,
        "target_characters": target_characters,
        "num_encoder_tokens": num_encoder_tokens,
        "num_decoder_tokens": num_decoder_tokens,
        "max_encoder_seq_length": max_encoder_seq_length,
        "max_decoder_seq_length": max_decoder_seq_length,
        "input_token_index": input_token_index,
        "target_token_index": target_token_index,
        "reverse_input_char_index": dict((i, char) for char, i in input_token_index.items()),
        "reverse_target_char_index": dict((i, char) for char, i in target_token_index.items()),
    }
    _data_metadata.update(metadata) # Cache for future calls
    return _data_metadata

def _vectorize_data(metadata: dict):
    """Vectorize input/target texts into one-hot encoded NumPy arrays."""
    num_samples_actual = len(metadata["input_texts"])
    max_encoder_seq_length = metadata["max_encoder_seq_length"]
    num_encoder_tokens = metadata["num_encoder_tokens"]
    max_decoder_seq_length = metadata["max_decoder_seq_length"]
    num_decoder_tokens = metadata["num_decoder_tokens"]
    input_token_index = metadata["input_token_index"]
    target_token_index = metadata["target_token_index"]

    encoder_input_data = np.zeros(
        (num_samples_actual, max_encoder_seq_length, num_encoder_tokens),
        dtype="float32",
    )
    decoder_input_data = np.zeros(
        (num_samples_actual, max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )
    decoder_target_data = np.zeros(
        (num_samples_actual, max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )

    for i, (input_text, target_text) in enumerate(zip(metadata["input_texts"], metadata["target_texts"])):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        # Pad remaining part of encoder input with spaces if space char exists
        if t + 1 < max_encoder_seq_length and " " in input_token_index:
            encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0

        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        # Pad remaining parts of decoder input and target with spaces if space char exists
        if t + 1 < max_decoder_seq_length and " " in target_token_index:
            decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        if t < max_decoder_seq_length and " " in target_token_index: # Corrected from t: to t+1: to match target padding style
             decoder_target_data[i, t:, target_token_index[" "]] = 1.0 # This matches the original script's padding

    return encoder_input_data, decoder_input_data, decoder_target_data

# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> keras.Model:
    """
    Builds the Keras sequence-to-sequence model.
    Config should contain 'latent_dim', and data dimensions derived from data processing.
    """
    # Ensure data metadata is available. If not, attempt to prepare it with defaults.
    if not _data_metadata:
        _prepare_data_metadata(
            config.get("data_path", str(Path.home() / ".keras" / "datasets" / "fra-eng")),
            config.get("num_samples", 10000)
        )

    num_encoder_tokens = _data_metadata["num_encoder_tokens"]
    num_decoder_tokens = _data_metadata["num_decoder_tokens"]

    model_kwargs = config.get("model_kwargs", {})
    latent_dim = model_kwargs.get("latent_dim", 256)

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens), name="encoder_input")
    encoder = keras.layers.LSTM(latent_dim, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens), name="decoder_input")

    # We set up our decoder to return full output sequences,
    # and to return internal states as well.
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model")

    return model

def build_dataloader(config: dict, split: str = "train") -> tf.data.Dataset:
    """
    Prepares and vectorizes the data, then returns a tf.data.Dataset for batches.
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 64))
    num_samples = config.get("num_samples", 10000)
    data_path = config.get("data_path", str(Path.home() / ".keras" / "datasets" / "fra-eng"))

    # Ensure metadata is prepared
    metadata = _prepare_data_metadata(data_path, num_samples)
    encoder_input_data, decoder_input_data, decoder_target_data = _vectorize_data(metadata)

    # Split data based on `validation_split` from config
    val_ratio = config.get("validation_split", 0.2)
    n_total = len(encoder_input_data)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    if split == "train":
        start_idx, end_idx = 0, n_train
        shuffle = True # Usually shuffle training data
    elif split == "val":
        start_idx, end_idx = n_train, n_total
        shuffle = False # Usually don't shuffle validation data
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    # Select the split data
    encoder_split = encoder_input_data[start_idx:end_idx]
    decoder_input_split = decoder_input_data[start_idx:end_idx]
    decoder_target_split = decoder_target_data[start_idx:end_idx]

    # Create a tf.data.Dataset
    # The 'inputs' for the Keras model are [encoder_inputs, decoder_inputs]
    # The 'targets' for the Keras model is decoder_targets
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"encoder_input": encoder_split, "decoder_input": decoder_input_split},
            decoder_target_split
        )
    )

    if shuffle:
        # Buffer size typically set to num_samples for full shuffling or a large number
        dataset = dataset.shuffle(buffer_size=len(encoder_split), seed=config.get("seed", 42))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # Optimize performance

    return dataset


def train_step(
    model: keras.Model,
    batch: tuple | list | dict,
    optimizer, # Keras optimizers are typically subclasses of tf.keras.optimizers.Optimizer, but not used directly here.
    config: dict,
) -> tf.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH computation graph attached.
    The FL runtime calls tape.gradient() and optimizer.apply_gradients() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    # The tf.data.Dataset yields a tuple: (inputs_dict, targets_tensor)
    inputs_dict, decoder_targets = batch

    # Unpack model inputs from the dictionary
    encoder_inputs = inputs_dict["encoder_input"]
    decoder_inputs = inputs_dict["decoder_input"]

    # Perform a forward pass.
    # training=True is important for Keras models with layers like Dropout or BatchNorm.
    outputs = model({"encoder_input": encoder_inputs, "decoder_input": decoder_inputs}, training=True)

    # Calculate loss. The original script uses 'categorical_crossentropy'.
    # keras.losses.CategoricalCrossentropy expects one-hot encoded targets.
    # `from_logits=False` because the decoder_dense layer has 'softmax' activation.
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    loss = loss_fn(decoder_targets, outputs)

    return loss