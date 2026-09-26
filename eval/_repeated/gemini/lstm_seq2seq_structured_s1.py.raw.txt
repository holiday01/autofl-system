import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import collections


class CharacterSeq2SeqDataset(Dataset):
    def __init__(self, data_path, num_samples, allow_synthetic_data, config_for_synthetic=None):
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()

        # Determine file_path from data_path
        file_path = None
        if data_path:
            if os.path.isdir(data_path):
                file_path = os.path.join(data_path, "fra.txt")
            else:  # Assume data_path is directly the file fra.txt
                file_path = data_path

        if file_path and os.path.exists(file_path):
            # Original data loading logic
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
            
            for line in lines[: min(num_samples, len(lines) - 1)]:
                parts = line.split("\t")
                if len(parts) < 2:  # Skip malformed lines
                    continue
                input_text, target_text = parts[0], parts[1]  # Discard potential third part
                
                target_text = "\t" + target_text + "\n"  # Add start/end tokens
                self.input_texts.append(input_text)
                self.target_texts.append(target_text)

                for char in input_text:
                    self.input_characters.add(char)
                for char in target_text:
                    self.target_characters.add(char)
        else: # Data file not found or data_path not provided
            if allow_synthetic_data:
                print(f"Warning: Data file not found at {file_path}. Generating synthetic data.")
                self._generate_synthetic_data(num_samples, config_for_synthetic)
                # After synthetic data generation, return to avoid further real data processing
                return 
            else:
                if file_path:
                    raise FileNotFoundError(f"Data file not found at {file_path} and 'allow_synthetic_data' is False.")
                else:
                    raise ValueError("No data_path provided and 'allow_synthetic_data' is False.")

        # Ensure essential characters are present if not in data (e.g., for padding, start/end tokens)
        self.input_characters.add(' ')  # Space for padding
        self.target_characters.add(' ')  # Space for padding
        self.target_characters.add('\t') # Start token
        self.target_characters.add('\n') # End token

        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self._num_encoder_tokens = len(self.input_characters)
        self._num_decoder_tokens = len(self.target_characters)
        self._max_encoder_seq_length = max([len(txt) for txt in self.input_texts]) if self.input_texts else 1
        self._max_decoder_seq_length = max([len(txt) for txt in self.target_texts]) if self.target_texts else 1
        
        # Ensure minimum sequence length of 1 if no texts or single char texts.
        self._max_encoder_seq_length = max(self._max_encoder_seq_length, 1)
        self._max_decoder_seq_length = max(self._max_decoder_seq_length, 1)

        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])

        # Pre-allocate and vectorize data
        self.encoder_input_data = np.zeros(
            (len(self.input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype="float32",
        )
        self.decoder_input_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype="float32",
        )
        self.decoder_target_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype="float32",
        )

        # One-hot encoding
        space_input_idx = self.input_token_index.get(" ", 0)
        space_target_idx = self.target_token_index.get(" ", 0)
        
        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            # Pad remaining encoder sequence with space
            if t + 1 < self.max_encoder_seq_length:
                self.encoder_input_data[i, t + 1 :, space_input_idx] = 1.0

            for t, char in enumerate(target_text):
                # decoder_input_data is the target sequence with start token
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.0
                # decoder_target_data is ahead of decoder_input_data by one timestep
                # and does not include the start character.
                if t > 0:  # Target starts from the second character of the original target_text
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.0
            # Pad remaining decoder input sequence with space
            if t + 1 < self.max_decoder_seq_length:
                self.decoder_input_data[i, t + 1 :, space_target_idx] = 1.0
            # Pad remaining decoder target sequence with space
            # Note: target has one less effective time step due to shift; if target_text had length L,
            # decoder_target_data has data up to index L-2, and then padding from L-1 onwards.
            # So, if t is the last valid index in target_text, decoder_target_data is filled up to t-1.
            # Padding starts from index t.
            if t < self.max_decoder_seq_length: 
                self.decoder_target_data[i, t :, space_target_idx] = 1.0
        
        self.encoder_input_data = torch.from_numpy(self.encoder_input_data)
        self.decoder_input_data = torch.from_numpy(self.decoder_input_data)
        self.decoder_target_data = torch.from_numpy(self.decoder_target_data)

    def _generate_synthetic_data(self, num_samples, config_for_synthetic):
        # Default values for synthetic data if not in config
        self._num_encoder_tokens = config_for_synthetic.get("num_encoder_tokens", 50)
        self._num_decoder_tokens = config_for_synthetic.get("num_decoder_tokens", 60)
        self._max_encoder_seq_length = config_for_synthetic.get("max_encoder_seq_length", 30)
        self._max_decoder_seq_length = config_for_synthetic.get("max_decoder_seq_length", 40)

        self.input_texts = ["synthetic_input"] * num_samples  # Dummy values for len calculation
        self.target_texts = ["synthetic_target"] * num_samples # Dummy values for len calculation

        self.encoder_input_data = torch.randn(
            (num_samples, self.max_encoder_seq_length, self.num_encoder_tokens), dtype=torch.float32
        )
        self.decoder_input_data = torch.randn(
            (num_samples, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=torch.float32
        )
        
        # For decoder_target_data, generate one-hot like tensors for CrossEntropyLoss
        self.decoder_target_data = torch.zeros(
            (num_samples, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=torch.float32
        )
        # Populate with random one-hot vectors
        for i in range(num_samples):
            for t in range(self.max_decoder_seq_length):
                rand_idx = torch.randint(0, self.num_decoder_tokens, (1,)).item()
                self.decoder_target_data[i, t, rand_idx] = 1.0

        # Create dummy token indices for metadata, even if not directly used by synthetic data
        self.input_characters = [chr(i + 97) for i in range(self.num_encoder_tokens)] # Example chars 'a', 'b', ...
        self.target_characters = [chr(i + 97) for i in range(self.num_decoder_tokens)]
        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        return (
            self.encoder_input_data[idx],
            self.decoder_input_data[idx],
            self.decoder_target_data[idx],
        )

    # Properties to expose character/token metadata for model construction
    @property
    def num_encoder_tokens(self):
        return self._num_encoder_tokens

    @property
    def num_decoder_tokens(self):
        return self._num_decoder_tokens
    
    @property
    def max_encoder_seq_length(self):
        return self._max_encoder_seq_length
    
    @property
    def max_decoder_seq_length(self):
        return self._max_decoder_seq_length


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size, latent_dim, batch_first=True)

    def forward(self, input_seq):
        _, (h_n, c_n) = self.lstm(input_seq)
        # h_n, c_n shape: (num_layers * num_directions, batch, hidden_size)
        # Here, (1, batch, latent_dim).
        return h_n, c_n


class Decoder(nn.Module):
    def __init__(self, input_size, latent_dim, output_size):
        super().__init__()
        # PyTorch LSTM expects (input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, latent_dim, batch_first=True)
        # Dense layer for output prediction
        self.dense = nn.Linear(latent_dim, output_size)

    def forward(self, decoder_input, encoder_states):
        # initial_state (h_0, c_0) must be (num_layers * num_directions, batch, hidden_size)
        # The encoder output (h_n, c_n) fits this format directly for 1-layer LSTM.
        output, _ = self.lstm(decoder_input, encoder_states)
        output = self.dense(output)
        return output  # These are logits


class Seq2Seq(nn.Module):
    def __init__(self, num_encoder_tokens, num_decoder_tokens, latent_dim):
        super().__init__()
        self.encoder = Encoder(num_encoder_tokens, latent_dim)
        self.decoder = Decoder(num_decoder_tokens, latent_dim, num_decoder_tokens)

    def forward(self, encoder_input, decoder_input):
        h_n, c_n = self.encoder(encoder_input)
        encoder_states = (h_n, c_n)  # Prepare for decoder initial state
        decoder_outputs = self.decoder(decoder_input, encoder_states)
        return decoder_outputs  # These are logits


def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    latent_dim = model_kwargs.get("latent_dim", 256)
    
    # These dimensions must be available in model_kwargs, typically populated by build_dataloader
    num_encoder_tokens = model_kwargs.get("num_encoder_tokens")
    num_decoder_tokens = model_kwargs.get("num_decoder_tokens")

    if num_encoder_tokens is None or num_decoder_tokens is None:
        raise ValueError(
            "Model dimensions (num_encoder_tokens, num_decoder_tokens) "
            "not found in config.model_kwargs. "
            "Ensure build_dataloader runs first to populate these, or provide them directly."
        )

    model = Seq2Seq(num_encoder_tokens, num_decoder_tokens, latent_dim)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    num_samples = config.get("num_samples", 10000)  # From original script config
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    
    # config_for_synthetic needs default values, which can come from config itself if not explicitly set
    # e.g., if model_kwargs are passed for synthetic data parameters
    config_for_synthetic = config.get("model_kwargs", {}).copy() # Use model_kwargs for synthetic defaults
    config_for_synthetic["num_encoder_tokens"] = config_for_synthetic.get("num_encoder_tokens", 50)
    config_for_synthetic["num_decoder_tokens"] = config_for_synthetic.get("num_decoder_tokens", 60)
    config_for_synthetic["max_encoder_seq_length"] = config_for_synthetic.get("max_encoder_seq_length", 30)
    config_for_synthetic["max_decoder_seq_length"] = config_for_synthetic.get("max_decoder_seq_length", 40)

    dataset = CharacterSeq2SeqDataset(data_path, num_samples, allow_synthetic_data, config_for_synthetic)

    # Update config with actual dataset dimensions (important for build_model later)
    # This modifies the config dict passed by reference, which is common in FL contexts.
    if "model_kwargs" not in config:
        config["model_kwargs"] = {}
    config["model_kwargs"]["num_encoder_tokens"] = dataset.num_encoder_tokens
    config["model_kwargs"]["num_decoder_tokens"] = dataset.num_decoder_tokens
    config["model_kwargs"]["max_encoder_seq_length"] = dataset.max_encoder_seq_length
    config["model_kwargs"]["max_decoder_seq_length"] = dataset.max_decoder_seq_length
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Ensure sizes are non-negative
    train_size = max(0, train_size)
    val_size = max(0, val_size)

    # If dataset is empty, create empty splits.
    if len(dataset) == 0:
        train_dataset = []
        val_dataset = []
    else:
        # random_split requires sum of lengths to be len(dataset)
        # Adjust if there's a discrepancy due to integer truncation
        if train_size + val_size != len(dataset):
            train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if split == "train":
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elif split == "val":
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    # Ensure model is on the correct device
    device = next(model.parameters()).device

    encoder_input, decoder_input, decoder_target = batch
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    decoder_target = decoder_target.to(device)  # One-hot target

    # Forward pass
    logits = model(encoder_input, decoder_input)  # (batch_size, max_decoder_seq_length, num_decoder_tokens)

    # Reshape for CrossEntropyLoss:
    # logits: (batch_size * seq_len, num_decoder_tokens)
    # target_indices: (batch_size * seq_len)
    logits_reshaped = logits.view(-1, logits.size(-1))
    
    # Convert one-hot decoder_target to class indices
    # decoder_target: (batch_size, max_decoder_seq_length, num_decoder_tokens)
    target_indices = torch.argmax(decoder_target, dim=-1)  # (batch_size, max_decoder_seq_length)
    target_indices_reshaped = target_indices.view(-1)

    # Calculate loss using CrossEntropyLoss, which expects logits and class indices
    loss = torch.nn.functional.cross_entropy(logits_reshaped, target_indices_reshaped)

    return loss