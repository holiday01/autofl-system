import tensorflow as tf
import keras
import numpy as np
import os
from pathlib import Path
import subprocess

# Cache for data_info and processed data arrays to avoid re-processing
_cached_data_info = None
_cached_data_arrays = None


def _download_and_prepare_data(data_path_base: str, num_samples: int):
    """
    Helper function to download, unzip, and preprocess the data.
    Caches results to avoid redundant processing if called multiple times.
    """
    global _cached_data_info, _cached_data_arrays

    if _cached_data_info is not None and _cached_data_arrays is not None:
        return _cached_data_arrays + (_cached_data_info,)

    # `keras.utils.get_file` downloads to a global cache (e.g., ~/.keras/datasets).
    # We use `data_path_base` as the cache_dir.
    fpath = keras.utils.get_file(
        origin="http://www.manythings.org/anki/fra-eng.zip", cache_dir=data_path_base
    )
    dirpath = Path(fpath).parent.absolute()

    # Check if already unzipped to avoid re-unzipping on subsequent calls
    actual_data_file = os.path.join(dirpath, "fra.txt")
    if not os.path.exists(actual_data_file):
        # Ensure the directory exists for unzipping
        os.makedirs(dirpath, exist_ok=True)
        try:
            subprocess.run(["unzip", "-q", fpath, "-d", dirpath], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error unzipping file: {e}")
            raise

    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(actual_data_file, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines[: min(num_samples, len(lines) - 1)]:
        # Handle potential empty lines or malformed lines
        parts = line.split("\t")
        if len(parts) < 2:
            continue  # Skip malformed lines
        input_text, target_text = parts[0], parts[1]

        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
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

    # Add space token if not present, for padding
    if " " not in input_token_index:
        input_token_index[" "] = num_encoder_tokens
        num_encoder_tokens += 1
    if " " not in target_token_index:
        target_token_index[" "] = num_decoder_tokens
        num_decoder_tokens += 1

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype="float32",
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        # Pad remaining sequence length with space token
        if t + 1 < max_encoder_seq_length:
            encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0

        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data is ahead of decoder_input_data by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        # Pad remaining sequence length with space token
        if t + 1 < max_decoder_seq_length:
            decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        # Pad remaining target sequence length with space token
        # The target sequence is one step ahead, so padding starts from the current 't'
        if t < max_decoder_seq_length:
            decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    _cached_data_arrays = (encoder_input_data, decoder_input_data, decoder_target_data)
    _cached_data_info = {
        "input_characters": input_characters,
        "target_characters": target_characters,
        "num_encoder_tokens": num_encoder_tokens,
        "num_decoder_tokens": num_decoder_tokens,
        "max_encoder_seq_length": max_encoder_seq_length,
        "max_decoder_seq_length": max_decoder_seq_length,
        "input_token_index": input_token_index,
        "target_token_index": target_token_index,
    }
    return _cached_data_arrays + (_cached_data_info,)


# ── FL Interface ────────────────────────────────────────────────────────


def build_model(config: dict) -> keras.Model:
    """
    Builds and returns the Keras Seq2Seq model.

    Args:
        config: A dictionary containing model configuration parameters,
                including 'latent_dim', 'num_encoder_tokens', 'num_decoder_tokens'.

    Returns:
        A tf.keras.Model instance.
    """
    latent_dim = config.get("latent_dim", 256)
    # These should be populated by build_dataloader first
    num_encoder_tokens = config["num_encoder_tokens"]
    num_decoder_tokens = config["num_decoder_tokens"]

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(
        shape=(None, num_encoder_tokens), name="encoder_inputs"
    )
    encoder = keras.layers.LSTM(latent_dim, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(
        shape=(None, num_decoder_tokens), name="decoder_inputs"
    )
    decoder_lstm = keras.layers.LSTM(
        latent_dim, return_sequences=True, return_state=True, name="decoder_lstm"
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(
        num_decoder_tokens, activation="softmax", name="decoder_dense"
    )
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model"
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> tf.data.Dataset:
    """
    Builds and returns a tf.data.Dataset for the specified split.

    Args:
        config: A dictionary containing data configuration parameters,
                including 'batch_size', 'num_samples', 'data_path', 'seed', 'val_ratio'.
                This config object will be updated with data_info (e.g., token counts).
        split: A string indicating the dataset split ('train' or 'val').

    Returns:
        A tf.data.Dataset instance.
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 64))
    num_samples = config.get("num_samples", 10000)
    # Use a temporary directory for cache_dir if data_path is not specified
    data_path_base = config.get(
        "data_path", os.path.join(Path.home(), ".keras", "datasets")
    )
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.2)

    # Download and prepare data (this will populate the config with token info)
    # This function also caches its results to avoid re-processing on subsequent calls
    (
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        data_info,
    ) = _download_and_prepare_data(data_path_base, num_samples)

    # Update config with data_info for build_model.
    # This assumes config is a mutable dictionary passed by reference.
    config.update(data_info)

    # Split data
    num_samples_total = len(encoder_input_data)
    num_val_samples = int(num_samples_total * val_ratio)
    num_train_samples = num_samples_total - num_val_samples

    if split == "train":
        x_encoder = encoder_input_data[:num_train_samples]
        x_decoder = decoder_input_data[:num_train_samples]
        y_target = decoder_target_data[:num_train_samples]
    elif split == "val":  # Use "val" for validation split
        x_encoder = encoder_input_data[num_train_samples:]
        x_decoder = decoder_input_data[num_train_samples:]
        y_target = decoder_target_data[num_train_samples:]
    else:
        raise ValueError(f"Unsupported split: {split}. Must be 'train' or 'val'.")

    # Create a tf.data.Dataset
    # The model expects inputs as a list or dict, so we use a dict for clarity
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"encoder_inputs": x_encoder, "decoder_inputs": x_decoder}, y_target)
    )

    if split == "train":
        dataset = dataset.shuffle(buffer_size=len(x_encoder), seed=seed)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_step(
    model: keras.Model,
    batch: tuple | list,
    optimizer: keras.optimizers.Optimizer,
    config: dict,
) -> tf.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.

    Args:
        model: The Keras model to train.
        batch: A tuple/list containing the inputs and targets for the current batch.
               Expected format: ({"encoder_inputs": ..., "decoder_inputs": ...}, targets).
        optimizer: The Keras optimizer instance. (Note: Not used directly in this function
                   as per the contract, but included in signature for FL runtime compatibility).
        config: A dictionary containing configuration parameters.

    Returns:
        A tf.Tensor representing the raw loss for the batch.
    """
    # The batch from tf.data.Dataset will be (inputs_dict, targets_tensor)
    inputs_dict, targets = batch
    encoder_inputs = inputs_dict["encoder_inputs"]
    decoder_inputs = inputs_dict["decoder_inputs"]

    # Use tf.GradientTape to compute the loss
    with tf.GradientTape() as tape:
        # Pass training=True to handle layers like Dropout and BatchNorm correctly
        outputs = model([encoder_inputs, decoder_inputs], training=True)

        # Keras loss functions are typically called with (y_true, y_pred)
        # The original script uses 'categorical_crossentropy'
        # from_logits=False because the decoder_dense has 'softmax' activation
        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
        loss = loss_fn(targets, outputs)

    # The contract states "returns the raw loss tensor WITH grad_fn attached".
    # In TensorFlow, the `loss` tensor itself is part of the computation graph
    # and `tape.gradient` would be used by the FL runtime to get gradients.
    # We return the loss tensor directly.
    return loss