import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from pathlib import Path
import collections

# --- Helper for data processing and vocabulary management ---
class Seq2SeqDataProcessor:
    def __init__(self, data_path, num_samples=None, allow_synthetic_data=False):
        self.data_path = data_path
        self.num_samples_config = num_samples # Store original num_samples from config
        self.allow_synthetic_data = allow_synthetic_data

        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()
        self.input_token_index = {}
        self.target_token_index = {}
        self.reverse_input_char_index = {}
        self.reverse_target_char_index = {}
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.max_encoder_seq_length = 0
        self.max_decoder_seq_length = 0

        self._process_data()

    def _process_data(self):
        real_data_available = False
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")

            # Filter out malformed lines and apply num_samples limit
            valid_lines = []
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 2:
                    valid_lines.append(parts)

            num_lines_to_process = self.num_samples_config if self.num_samples_config is not None else len(valid_lines)
            
            for line_parts in valid_lines[: min(num_lines_to_process, len(valid_lines))]:
                input_text, target_text = line_parts[0], line_parts[1]
                target_text = "\t" + target_text + "\n" # Add start/end tokens
                self.input_texts.append(input_text)
                self.target_texts.append(target_text)
                for char in input_text:
                    self.input_characters.add(char)
                for char in target_text:
                    self.target_characters.add(char)
            
            if self.input_texts: # If any real data was processed
                real_data_available = True
                self.input_characters = sorted(list(self.input_characters))
                self.target_characters = sorted(list(self.target_characters))
                self.num_encoder_tokens = len(self.input_characters)
                self.num_decoder_tokens = len(self.target_characters)
                self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
                self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

                self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
                self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
                self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
                self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())
        
        # If no real data or data_path doesn't exist, and synthetic data is allowed
        if not real_data_available and self.allow_synthetic_data:
            print("Generating synthetic data...")
            synthetic_num_samples = self.num_samples_config if self.num_samples_config is not None else 1000
            self.max_encoder_seq_length = 20
            self.max_decoder_seq_length = 20
            self.num_encoder_tokens = 50 # Example size
            self.num_decoder_tokens = 60 # Example size
            # Ensure essential characters are present if used in logic
            self.input_characters = [chr(i) for i in range(ord('a'), ord('a') + self.num_encoder_tokens - 2)] + [' ', '\t'] 
            self.target_characters = [chr(i) for i in range(ord('A'), ord('A') + self.num_decoder_tokens - 3)] + [' ', '\t', '\n']
            
            self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
            self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])
            self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
            self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())
            
            # Populate dummy input/target texts for the dataset creation later
            self.input_texts = ["synthetic input " * (self.max_encoder_seq_length // 15) for _ in range(synthetic_num_samples)]
            self.target_texts = ["\tsynthetic target \n" * (self.max_decoder_seq_length // 15) for _ in range(synthetic_num_samples)]
        
        # If no data was loaded (real or synthetic) and synthetic is not allowed
        if not self.input_texts and not self.allow_synthetic_data:
             raise FileNotFoundError(
                f"No data found or processed from '{self.data_path}' and 'allow_synthetic_data' is False."
            )


# --- PyTorch Dataset for Seq2Seq ---
class Seq2SeqDataset(Dataset):
    def __init__(self, data_processor: Seq2SeqDataProcessor):
        self.data_processor = data_processor
        self.num_samples = len(self.data_processor.input_texts)

        if self.num_samples == 0:
            raise ValueError("No samples found to create the dataset.")

        self.encoder_input_data = np.zeros(
            (self.num_samples, self.data_processor.max_encoder_seq_length, self.data_processor.num_encoder_tokens),
            dtype="float32",
        )
        self.decoder_input_data = np.zeros(
            (self.num_samples, self.data_processor.max_decoder_seq_length, self.data_processor.num_decoder_tokens),
            dtype="float32",
        )
        self.decoder_target_data = np.zeros(
            (self.num_samples, self.data_processor.max_decoder_seq_length, self.data_processor.num_decoder_tokens),
            dtype="float32",
        )

        self._vectorize_data()

    def _vectorize_data(self):
        for i, (input_text, target_text) in enumerate(zip(self.data_processor.input_texts, self.data_processor.target_texts)):
            for t, char in enumerate(input_text):
                if char in self.data_processor.input_token_index:
                    self.encoder_input_data[i, t, self.data_processor.input_token_index[char]] = 1.0
            # Pad remaining input sequence with space character, if ' ' exists in input_token_index
            if ' ' in self.data_processor.input_token_index:
                self.encoder_input_data[i, len(input_text):, self.data_processor.input_token_index[" "]] = 1.0


            for t, char in enumerate(target_text):
                if char in self.data_processor.target_token_index:
                    # decoder_input_data is fed to the decoder at training time
                    self.decoder_input_data[i, t, self.data_processor.target_token_index[char]] = 1.0
                    if t > 0:
                        # decoder_target_data is one timestep ahead of decoder_input_data,
                        # and does not include the start character ('\t').
                        # Note: decoder_target_data is effectively "shifted target"
                        self.decoder_target_data[i, t - 1, self.data_processor.target_token_index[char]] = 1.0
            
            # Pad remaining decoder sequences with space character, if ' ' exists in target_token_index
            if ' ' in self.data_processor.target_token_index:
                self.decoder_input_data[i, len(target_text):, self.data_processor.target_token_index[" "]] = 1.0
                # Pad decoder_target_data from where the target text ends.
                # The length of target_text in data_processor already includes '\t' and '\n'.
                # The effective sequence length for decoder_target_data is len(target_text) - 1
                # because it doesn't include the initial '\t'.
                # So padding starts from max(0, len(target_text) - 1).
                target_data_padding_start_idx = max(0, len(target_text) - 1)
                self.decoder_target_data[i, target_data_padding_start_idx:, self.data_processor.target_token_index[" "]] = 1.0


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoder_input_data[idx], dtype=torch.float32),
            torch.tensor(self.decoder_input_data[idx], dtype=torch.float32),
            torch.tensor(self.decoder_target_data[idx], dtype=torch.float32),
        )

# --- PyTorch Models ---
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, latent_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x) # We only care about the final states
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, latent_dim, batch_first=True)
        self.dense = nn.Linear(latent_dim, output_dim)

    def forward(self, x, hidden_state, cell_state):
        # x: (batch_size, sequence_length, input_dim)
        # hidden_state: (num_layers, batch_size, latent_dim)
        # cell_state: (num_layers, batch_size, latent_dim)
        output, (_, _) = self.lstm(x, (hidden_state, cell_state))
        # output: (batch_size, sequence_length, latent_dim)
        prediction = self.dense(output)
        # prediction: (batch_size, sequence_length, output_dim)
        return prediction

class Seq2Seq(nn.Module):
    def __init__(self, num_encoder_tokens, num_decoder_tokens, latent_dim):
        super().__init__()
        self.encoder = Encoder(num_encoder_tokens, latent_dim)
        self.decoder = Decoder(num_decoder_tokens, latent_dim, num_decoder_tokens)

    def forward(self, encoder_input, decoder_input):
        # encoder_input: (batch_size, max_encoder_seq_length, num_encoder_tokens)
        # decoder_input: (batch_size, max_decoder_seq_length, num_decoder_tokens)
        encoder_hidden, encoder_cell = self.encoder(encoder_input)
        # encoder_hidden, encoder_cell: (1, batch_size, latent_dim)

        decoder_output = self.decoder(decoder_input, encoder_hidden, encoder_cell)
        # decoder_output: (batch_size, max_decoder_seq_length, num_decoder_tokens)
        return decoder_output


# --- FL Client Module Functions ---

# Global data processor to avoid re-processing data for each call to build_dataloader
# and to pass configuration parameters derived from data to build_model.
_data_processor = None
_global_config_cache = {}

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    - Use config.get("model_kwargs", {}) for constructor args.
    """
    global _data_processor, _global_config_cache

    # Ensure data processor is initialized to get model dimensions
    # Reinitialize if config changes or not initialized
    if _data_processor is None or _global_config_cache != config:
        _global_config_cache = config.copy()
        data_path = config.get("data_path", ".")
        num_samples = config.get("num_samples", 10000) # num_samples from original script config
        allow_synthetic_data = config.get("allow_synthetic_data", False)

        _data_processor = Seq2SeqDataProcessor(
            data_path=data_path,
            num_samples=num_samples,
            allow_synthetic_data=allow_synthetic_data
        )

    model_kwargs = config.get("model_kwargs", {})
    latent_dim = model_kwargs.get("latent_dim", 256)

    model = Seq2Seq(
        num_encoder_tokens=_data_processor.num_encoder_tokens,
        num_decoder_tokens=_data_processor.num_decoder_tokens,
        latent_dim=latent_dim,
    )
    return model

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    - Read batch_size from config.get("local", {}).get("batch_size", 16).
    - Read data_path from config.get("data_path", ".").
    - Use random_split to produce train/val subsets from a single dataset.
    - Include a synthetic data fallback.
    """
    global _data_processor, _global_config_cache

    # Ensure data processor is initialized
    # Reinitialize if config changes or not initialized
    if _data_processor is None or _global_config_cache != config:
        _global_config_cache = config.copy()
        data_path = config.get("data_path", ".")
        num_samples = config.get("num_samples", 10000)
        allow_synthetic_data = config.get("allow_synthetic_data", False)
        
        _data_processor = Seq2SeqDataProcessor(
            data_path=data_path,
            num_samples=num_samples,
            allow_synthetic_data=allow_synthetic_data
        )
    
    # Check if any data was processed by the data processor.
    # If not, and synthetic data is disallowed, raise error.
    if not _data_processor.input_texts and not _data_processor.allow_synthetic_data:
        raise FileNotFoundError(
            f"No data found or processed from '{_data_processor.data_path}' and 'allow_synthetic_data' is False."
        )
    
    full_dataset = Seq2SeqDataset(_data_processor)
    
    # Use validation_split from original script (0.2) or config
    val_split_ratio = config.get("validation_split", 0.2)
    
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        raise ValueError("The dataset is empty after processing.")

    val_size = int(dataset_size * val_split_ratio)
    train_size = dataset_size - val_size

    # Ensure reproducibility of split if needed with a fixed generator state
    g = torch.Generator().manual_seed(config.get("seed", 42))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

    batch_size = config.get("local", {}).get("batch_size", 16)

    if split == "train":
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elif split == "val":
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Invalid split name: {split}. Expected 'train' or 'val'.")

def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    - Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    - Move tensors to the device of the model parameters.
    """
    model.train()
    
    encoder_input, decoder_input, decoder_target_one_hot = batch

    # Move tensors to the device of the model
    device = next(model.parameters()).device
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    decoder_target_one_hot = decoder_target_one_hot.to(device)

    # Forward pass
    decoder_outputs = model(encoder_input, decoder_input)

    # Calculate loss
    # Original Keras uses categorical_crossentropy with one-hot targets.
    # In PyTorch, nn.CrossEntropyLoss expects class indices for target and (N, C, ...) for input.
    # Reshape decoder_outputs from (B, S, V) to (B*S, V)
    # Convert decoder_target_one_hot from (B, S, V) to (B*S) class indices
    
    output_dim = decoder_outputs.shape[-1]
    reshaped_outputs = decoder_outputs.view(-1, output_dim) # (Batch*SeqLength, VocabSize)

    # Convert one-hot targets to class indices
    reshaped_targets = torch.argmax(decoder_target_one_hot, dim=-1).view(-1) # (Batch*SeqLength)

    # The original script pads with space character and it's included in target_data.
    # We should ignore the loss contribution from these padding tokens.
    # Get the index of the space character in target vocabulary.
    # If space is not in the vocabulary, or if _data_processor is not initialized,
    # then ignore_index will be -1 (which means no index is ignored by CrossEntropyLoss).
    ignore_idx = _data_processor.target_token_index.get(' ', -1) if _data_processor else -1
    
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    
    loss = criterion(reshaped_outputs, reshaped_targets)

    # Do NOT call loss.backward() or optimizer.step()
    return loss