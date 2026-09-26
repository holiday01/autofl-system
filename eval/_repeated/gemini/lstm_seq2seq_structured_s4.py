import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import numpy as np
import os
from pathlib import Path
import urllib.request
import zipfile
import io

# Dummy Keras imports just to preserve original script's feel,
# but they are not used for the PyTorch implementation logic.
try:
    import keras
    from keras.utils import get_file # Only used for path resolution, not execution
except ImportError:
    # Provide dummy for local execution if Keras not installed
    class KerasDummy:
        class utils:
            @staticmethod
            def get_file(origin, **kwargs):
                # Simulate get_file behavior for local path creation
                filename = origin.split('/')[-1]
                path = Path(os.getcwd()) / filename
                return str(path)
    keras = KerasDummy()


# --- PyTorch Model Equivalent to Keras Seq2Seq ---
class Seq2SeqKerasLike(nn.Module):
    """
    A PyTorch equivalent of the Keras character-level sequence-to-sequence model.
    """
    def __init__(self, num_encoder_tokens: int, num_decoder_tokens: int, latent_dim: int):
        super().__init__()
        self.encoder_lstm = nn.LSTM(num_encoder_tokens, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(num_decoder_tokens, latent_dim, batch_first=True)
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)
        self.num_decoder_tokens = num_decoder_tokens

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the sequence-to-sequence model.

        Args:
            encoder_input: Input sequence for the encoder.
                           Shape: (batch_size, max_encoder_seq_length, num_encoder_tokens)
            decoder_input: Input sequence for the decoder (teacher forcing).
                           Shape: (batch_size, max_decoder_seq_length, num_decoder_tokens)

        Returns:
            Logits for the decoder's output sequence.
            Shape: (batch_size, max_decoder_seq_length, num_decoder_tokens)
        """
        # Encoder:
        # Keras encoder discards outputs and only keeps the final states.
        # PyTorch LSTM returns output, (h_n, c_n)
        _, (state_h, state_c) = self.encoder_lstm(encoder_input)

        # Decoder:
        # Keras decoder uses encoder states as initial state.
        # PyTorch LSTM expects (h_0, c_0) as initial state.
        decoder_outputs, _, _ = self.decoder_lstm(decoder_input, (state_h, state_c))

        # Apply dense layer to each timestep of the decoder outputs
        # decoder_outputs shape: (batch_size, max_decoder_seq_length, latent_dim)
        decoder_outputs = self.decoder_dense(decoder_outputs)
        # final decoder_outputs shape: (batch_size, max_decoder_seq_length, num_decoder_tokens)

        return decoder_outputs # These are logits, not probabilities, suitable for F.cross_entropy


# --- FL Client Module Functions ---

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiates and returns the PyTorch model.

    Args:
        config: Configuration dictionary containing model_kwargs.

    Returns:
        An instance of torch.nn.Module.
    """
    model_kwargs = config.get("model_kwargs", {})
    num_encoder_tokens = model_kwargs.get("num_encoder_tokens")
    num_decoder_tokens = model_kwargs.get("num_decoder_tokens")
    latent_dim = model_kwargs.get("latent_dim")

    if num_encoder_tokens is None or num_decoder_tokens is None or latent_dim is None:
        raise ValueError(
            "model_kwargs in config must contain 'num_encoder_tokens', 'num_decoder_tokens', "
            "and 'latent_dim'. These values are typically determined during a pre-processing "
            "step (e.g., initial data analysis on a sample client or globally) "
            "and passed through the FL configuration."
        )

    model = Seq2SeqKerasLike(num_encoder_tokens, num_decoder_tokens, latent_dim)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Returns a DataLoader for the requested split ("train" or "val").

    Args:
        config: Configuration dictionary containing local, data_path,
                allow_synthetic_data, and data_config settings.
        split: The data split to return ("train" or "val").

    Returns:
        A torch.utils.data.DataLoader instance.

    Raises:
        FileNotFoundError: If real data is unavailable and synthetic data is not allowed.
        ValueError: If an invalid split is requested or configuration is inconsistent.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path_base = Path(config.get("data_path", ".")) # Base directory for data files
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    validation_split = config.get("data_config", {}).get("validation_split", 0.2)
    num_samples_limit = config.get("data_config", {}).get("num_samples", 10000) # Limit for real data

    # Expected model parameters, these should be in config.model_kwargs for consistency
    model_kwargs = config.get("model_kwargs", {})
    expected_num_encoder_tokens = model_kwargs.get("num_encoder_tokens")
    expected_num_decoder_tokens = model_kwargs.get("num_decoder_tokens")
    expected_max_encoder_seq_length = model_kwargs.get("max_encoder_seq_length")
    expected_max_decoder_seq_length = model_kwargs.get("max_decoder_seq_length")

    # Fallback defaults for synthetic data if not provided in config
    SYNTHETIC_NUM_SAMPLES = config.get("data_config", {}).get("synthetic_num_samples", 1000)
    SYNTHETIC_ENCODER_TOKENS = expected_num_encoder_tokens if expected_num_encoder_tokens is not None else 50
    SYNTHETIC_DECODER_TOKENS = expected_num_decoder_tokens if expected_num_decoder_tokens is not None else 50
    SYNTHETIC_MAX_ENC_SEQ_LEN = expected_max_encoder_seq_length if expected_max_encoder_seq_length is not None else 20
    SYNTHETIC_MAX_DEC_SEQ_LEN = expected_max_decoder_seq_length if expected_max_decoder_seq_length is not None else 20

    encoder_input_data = None
    decoder_input_data = None
    decoder_target_data = None # Will store class indices

    try:
        # Replicate data download and preparation logic
        fpath_zip = data_path_base / "fra-eng.zip"
        fpath_txt = data_path_base / "fra.txt"

        if not fpath_txt.exists():
            print(f"Data file '{fpath_txt}' not found. Attempting to download and unzip...")
            os.makedirs(data_path_base, exist_ok=True)
            url = "http://www.manythings.org/anki/fra-eng.zip"
            try:
                with urllib.request.urlopen(url) as response:
                    zip_content = io.BytesIO(response.read())
                with zipfile.ZipFile(zip_content, 'r') as zip_ref:
                    zip_ref.extractall(data_path_base)
                print(f"Successfully downloaded and unzipped data to '{data_path_base}'")
            except Exception as e:
                if allow_synthetic_data:
                    print(f"Could not download real data ({e}). Falling back to synthetic data.")
                    raise FileNotFoundError # Trigger synthetic data path
                else:
                    raise FileNotFoundError(
                        f"Real data file 'fra.txt' not found at '{fpath_txt}' "
                        f"and download failed from '{url}'. "
                        f"To proceed with synthetic data, set 'allow_synthetic_data' to True in the config."
                    ) from e

        if fpath_txt.exists():
            input_texts = []
            target_texts = []
            input_characters = set()
            target_characters = set()
            with open(fpath_txt, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
            for line in lines[: min(num_samples_limit, len(lines) - 1)]:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                input_text, target_text = parts[0], parts[1]
                target_text = "\t" + target_text + "\n" # Add start/end tokens
                input_texts.append(input_text)
                target_texts.append(target_text)
                for char in input_text:
                    input_characters.add(char)
                for char in target_text:
                    target_characters.add(char)

            if not input_texts:
                if allow_synthetic_data:
                    print(f"No valid data lines found in '{fpath_txt}'. Falling back to synthetic data.")
                    raise FileNotFoundError # Trigger synthetic data path
                else:
                    raise FileNotFoundError(
                        f"No valid data lines found in '{fpath_txt}'. "
                        f"To proceed with synthetic data, set 'allow_synthetic_data' to True in the config."
                    )

            input_characters = sorted(list(input_characters))
            target_characters = sorted(list(target_characters))
            derived_num_encoder_tokens = len(input_characters)
            derived_num_decoder_tokens = len(target_characters)
            derived_max_encoder_seq_length = max([len(txt) for txt in input_texts])
            derived_max_decoder_seq_length = max([len(txt) for txt in target_texts])

            # Validate derived parameters against config, if provided
            if expected_num_encoder_tokens is not None and expected_num_encoder_tokens != derived_num_encoder_tokens:
                raise ValueError(f"Config 'num_encoder_tokens' ({expected_num_encoder_tokens}) does not match derived ({derived_num_encoder_tokens}) from data.")
            if expected_num_decoder_tokens is not None and expected_num_decoder_tokens != derived_num_decoder_tokens:
                raise ValueError(f"Config 'num_decoder_tokens' ({expected_num_decoder_tokens}) does not match derived ({derived_num_decoder_tokens}) from data.")
            # Max sequence lengths are more flexible, can use derived if config is different
            if expected_max_encoder_seq_length is not None and expected_max_encoder_seq_length < derived_max_encoder_seq_length:
                print(f"Warning: Config 'max_encoder_seq_length' ({expected_max_encoder_seq_length}) is less than derived ({derived_max_encoder_seq_length}). Using derived for data processing. This might truncate data.")
            if expected_max_decoder_seq_length is not None and expected_max_decoder_seq_length < derived_max_decoder_seq_length:
                print(f"Warning: Config 'max_decoder_seq_length' ({expected_max_decoder_seq_length}) is less than derived ({derived_max_decoder_seq_length}). Using derived for data processing. This might truncate data.")

            # Use derived values for data preparation
            num_encoder_tokens = derived_num_encoder_tokens
            num_decoder_tokens = derived_num_decoder_tokens
            max_encoder_seq_length = derived_max_encoder_seq_length
            max_decoder_seq_length = derived_max_decoder_seq_length

            input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
            target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

            encoder_input_data = np.zeros(
                (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
                dtype="float32",
            )
            decoder_input_data = np.zeros(
                (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
                dtype="float32",
            )
            # decoder_target_data will be class indices (Long type) for PyTorch's CrossEntropyLoss
            decoder_target_data_indices = np.zeros(
                (len(input_texts), max_decoder_seq_length),
                dtype="int64", # Long type for class indices
            )

            # Find space character index for padding, if it exists
            space_char_input_idx = input_token_index.get(" ", 0) # Default to 0 if not found
            space_char_target_idx = target_token_index.get(" ", 0) # Default to 0 if not found

            for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
                for t, char in enumerate(input_text):
                    if t < max_encoder_seq_length: # Ensure not exceeding max length
                        encoder_input_data[i, t, input_token_index[char]] = 1.0
                # Pad remaining part of encoder input with space character (one-hot)
                if ' ' in input_token_index:
                    encoder_input_data[i, t + 1 : max_encoder_seq_length, space_char_input_idx] = 1.0

                for t, char in enumerate(target_text):
                    if t < max_decoder_seq_length: # Ensure not exceeding max length
                        decoder_input_data[i, t, target_token_index[char]] = 1.0
                        if t > 0: # Target is ahead by one timestep
                            decoder_target_data_indices[i, t - 1] = target_token_index[char]
                # Pad remaining parts of decoder input and target with space character
                if ' ' in target_token_index:
                    decoder_input_data[i, t + 1 : max_decoder_seq_length, space_char_target_idx] = 1.0
                    decoder_target_data_indices[i, t : max_decoder_seq_length] = space_char_target_idx
                # If space char not in vocab, numpy's zeros will implicitly pad with 0s

            decoder_target_data = decoder_target_data_indices # Assign the prepared indices

            print(f"Loaded {len(input_texts)} real samples.")

        else: # fpath_txt does not exist after potential download attempt
            raise FileNotFoundError("fra.txt not found after download attempt.")

    except FileNotFoundError as e:
        if allow_synthetic_data:
            print(f"Real data not available or failed to load: {e}. Generating synthetic data...")
            num_samples = SYNTHETIC_NUM_SAMPLES
            num_encoder_tokens = SYNTHETIC_ENCODER_TOKENS
            num_decoder_tokens = SYNTHETIC_DECODER_TOKENS
            max_encoder_seq_length = SYNTHETIC_MAX_ENC_SEQ_LEN
            max_decoder_seq_length = SYNTHETIC_MAX_DEC_SEQ_LEN

            # Generate random one-hot like input data
            encoder_input_data = np.zeros((num_samples, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
            # For each sample and timestep, randomly pick one token to be 1.0
            idx = np.random.randint(0, num_encoder_tokens, (num_samples, max_encoder_seq_length))
            encoder_input_data[np.arange(num_samples)[:, None], np.arange(max_encoder_seq_length), idx] = 1.0

            decoder_input_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens), dtype="float32")
            idx = np.random.randint(0, num_decoder_tokens, (num_samples, max_decoder_seq_length))
            decoder_input_data[np.arange(num_samples)[:, None], np.arange(max_decoder_seq_length), idx] = 1.0

            # Target data as class indices
            decoder_target_data = np.random.randint(0, num_decoder_tokens, (num_samples, max_decoder_seq_length)).astype("int64")

            print(f"Generated {num_samples} synthetic samples.")
        else:
            raise

    # Convert to PyTorch tensors
    encoder_input_data_tensor = torch.tensor(encoder_input_data)
    decoder_input_data_tensor = torch.tensor(decoder_input_data)
    decoder_target_data_tensor = torch.tensor(decoder_target_data)

    dataset = TensorDataset(encoder_input_data_tensor, decoder_input_data_tensor, decoder_target_data_tensor)

    # Split dataset
    total_samples = len(dataset)
    val_samples = int(total_samples * validation_split)
    train_samples = total_samples - val_samples

    if split == "train":
        # Use a fixed seed for random_split for reproducibility, important in FL
        subset, _ = random_split(dataset, [train_samples, val_samples], generator=torch.Generator().manual_seed(42))
    elif split == "val":
        _, subset = random_split(dataset, [train_samples, val_samples], generator=torch.Generator().manual_seed(42))
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    return DataLoader(subset, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: torch.nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Performs one forward pass and returns the loss tensor with gradients attached.

    Args:
        model: The PyTorch model to train.
        batch: A tuple containing (encoder_input, decoder_input, decoder_target).
        optimizer: The optimizer (not used for its step in this function, but for device info).
        config: Configuration dictionary (can be used for additional settings, e.g., loss weights).

    Returns:
        The computed loss tensor.
    """
    device = next(model.parameters()).device # Get model's current device
    encoder_input, decoder_input, decoder_target = [t.to(device) for t in batch]

    # Forward pass
    output_logits = model(encoder_input, decoder_input)

    # Calculate loss
    # output_logits shape: (batch_size, max_decoder_seq_length, num_decoder_tokens) - contains logits
    # decoder_target shape: (batch_size, max_decoder_seq_length) - contains class indices
    
    # Reshape for CrossEntropyLoss: (N, C) and (N)
    # N = batch_size * max_decoder_seq_length
    # C = num_decoder_tokens
    loss = F.cross_entropy(
        output_logits.view(-1, output_logits.size(-1)), # Logits: (N, C)
        decoder_target.view(-1)                          # Targets: (N)
    )

    return loss