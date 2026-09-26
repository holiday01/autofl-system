import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from pathlib import Path
import requests
import zipfile

# --- Model Definition ---
class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # Keras LSTM default activation is tanh for hidden, sigmoid for recurrent.
        # PyTorch LSTM defaults to tanh. This should be consistent enough.
        # Keras LSTM `input_shape=(None, num_encoder_tokens)` means variable length sequence,
        # where `num_encoder_tokens` is input_size. `batch_first=True` handles (batch, seq, feature).
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, input_seq: torch.Tensor):
        # input_seq: (batch_size, seq_len, input_size)
        # We only need the final hidden and cell states for the decoder's initial state.
        # output, (h_n, c_n) where h_n, c_n are (num_layers * num_directions, batch, hidden_size)
        # For a single layer, single direction LSTM, it's (1, batch, hidden_size)
        _, (h_n, c_n) = self.lstm(input_seq)
        return h_n, c_n

class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # Keras decoder_lstm `return_sequences=True, return_state=True`
        # PyTorch LSTM returns outputs for all timesteps by default if `batch_first=True` is set.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Keras decoder_dense `keras.layers.Dense(num_decoder_tokens, activation="softmax")`
        # CrossEntropyLoss expects logits, so we don't apply softmax here.
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq: torch.Tensor, initial_states: tuple):
        # input_seq: (batch_size, seq_len, input_size)
        # initial_states: (h_0, c_0), each (1, batch_size, hidden_size)
        outputs, _ = self.lstm(input_seq, initial_states)
        # outputs: (batch_size, seq_len, hidden_size)
        # Apply dense layer to each timestep output
        decoder_outputs = self.dense(outputs)
        # decoder_outputs: (batch_size, seq_len, output_size) (logits)
        return decoder_outputs

class Seq2Seq(nn.Module):
    def __init__(self, num_encoder_tokens: int, num_decoder_tokens: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(num_encoder_tokens, latent_dim)
        self.decoder = Decoder(num_decoder_tokens, latent_dim, num_decoder_tokens)

    def forward(self, encoder_input_data: torch.Tensor, decoder_input_data: torch.Tensor):
        # encoder_input_data: (batch_size, max_encoder_seq_length, num_encoder_tokens)
        # decoder_input_data: (batch_size, max_decoder_seq_length, num_decoder_tokens)
        
        encoder_h, encoder_c = self.encoder(encoder_input_data)
        # encoder_h, encoder_c shape: (1, batch_size, latent_dim)
        
        decoder_outputs = self.decoder(decoder_input_data, (encoder_h, encoder_c))
        # decoder_outputs shape: (batch_size, max_decoder_seq_length, num_decoder_tokens)
        return decoder_outputs

def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    
    # These parameters are crucial for model building and are derived from the dataset.
    # build_dataloader is expected to populate them into the config dictionary.
    num_encoder_tokens = config.get("num_encoder_tokens")
    num_decoder_tokens = config.get("num_decoder_tokens")
    
    # Check if necessary parameters are available (populated by build_dataloader)
    if num_encoder_tokens is None or num_decoder_tokens is None:
        raise ValueError(
            "Missing 'num_encoder_tokens' or 'num_decoder_tokens' in config. "
            "These should be computed by build_dataloader and added to config."
        )

    latent_dim = model_kwargs.get("latent_dim", 256) # Default from original script

    model = Seq2Seq(num_encoder_tokens, num_decoder_tokens, latent_dim)
    return model

# --- Dataset and DataLoader Definition ---
class TranslationDataset(Dataset):
    def __init__(self, data_path: str, num_samples: int, allow_synthetic_data: bool = False):
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()
        self.num_samples = num_samples

        if not os.path.exists(data_path):
            if allow_synthetic_data:
                print(f"Data file not found at {data_path}. Generating synthetic data.")
                self._generate_synthetic_data()
            else:
                raise FileNotFoundError(
                    f"Data file not found at {data_path}. Set 'allow_synthetic_data' to True in config to use synthetic data."
                )
        else:
            print(f"Loading data from {data_path}")
            with open(data_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")

            for line in lines[: min(num_samples, len(lines) - 1)]:
                parts = line.split("\t")
                if len(parts) >= 2: # Ensure at least two parts (input and target) exist
                    input_text, target_text = parts[0], parts[1]
                    # We use "tab" as the "start sequence" character
                    # for the targets, and "\n" as "end sequence" character.
                    target_text = "\t" + target_text + "\n"
                    self.input_texts.append(input_text)
                    self.target_texts.append(target_text)
                    for char in input_text:
                        self.input_characters.add(char)
                    for char in target_text:
                        self.target_characters.add(char)
                # else: skip malformed lines (lines with fewer than 2 tabs)

        if not self.input_texts and not allow_synthetic_data:
            # If no real data was loaded and synthetic data is not allowed, raise an error
            raise ValueError("No data loaded. Check data_path or set 'allow_synthetic_data' to True.")
        elif not self.input_texts and allow_synthetic_data:
            # If synthetic data was allowed but no samples were generated (e.g., if _generate_synthetic_data
            # was called but did nothing due to an edge case or previous real data parsing failed),
            # ensure some synthetic data is generated.
            print("No real data loaded, ensuring synthetic data generation.")
            self._generate_synthetic_data()

        # Ensure minimal characters for processing even if input_texts/target_texts is empty (e.g., from synthetic fallback)
        if not self.input_characters: self.input_characters.add('a'); self.input_characters.add(' ')
        if not self.target_characters: self.target_characters.add('b'); self.target_characters.add('\t'); self.target_characters.add('\n'); self.target_characters.add(' ')

        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        
        # Calculate max sequence lengths. If no texts, use a default value.
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts]) if self.input_texts else 10
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts]) if self.target_texts else 12 # +2 for \t and \n

        self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])

        # Pre-process and store data as numpy arrays for efficiency
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

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            # Process encoder input
            for t, char in enumerate(input_text):
                if char in self.input_token_index:
                    self.encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            # Pad remaining sequence length with space character if available
            if ' ' in self.input_token_index:
                for t_pad in range(len(input_text), self.max_encoder_seq_length):
                    self.encoder_input_data[i, t_pad, self.input_token_index[" "]] = 1.0

            # Process decoder input and target
            for t, char in enumerate(target_text):
                if char in self.target_token_index:
                    self.decoder_input_data[i, t, self.target_token_index[char]] = 1.0
                    if t > 0: # decoder_target_data is ahead of decoder_input_data by one timestep
                        self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.0
            # Pad remaining sequence length with space character if available
            if ' ' in self.target_token_index:
                # Pad decoder input data
                for t_pad in range(len(target_text), self.max_decoder_seq_length):
                    self.decoder_input_data[i, t_pad, self.target_token_index[" "]] = 1.0
                # Pad decoder target data (shifted by one, so padding starts earlier)
                for t_pad in range(len(target_text) -1, self.max_decoder_seq_length): # -1 because target_data is shifted
                    if t_pad >= 0: # Ensure valid index
                        self.decoder_target_data[i, t_pad, self.target_token_index[" "]] = 1.0


    def _generate_synthetic_data(self):
        # Simplified synthetic data generation to ensure functional dataset
        self.input_characters = set('abcde ')
        self.target_characters = set('vwxyz\t\n ')
        
        # Define max lengths for synthetic data
        self.max_encoder_seq_length = 15
        self.max_decoder_seq_length = 17 # +2 for \t and \n (start/end tokens)
        
        input_chars_list = sorted(list(self.input_characters))
        target_chars_list = sorted(list(self.target_characters))

        self.input_texts = []
        self.target_texts = []

        for _ in range(self.num_samples):
            input_len = np.random.randint(5, self.max_encoder_seq_length - 1)
            target_len = np.random.randint(7, self.max_decoder_seq_length - 1) # target_len includes '\t' and '\n'

            input_text = ''.join(np.random.choice(input_chars_list, input_len))
            # Ensure target has start and end tokens for consistency with real data
            target_text = '\t' + ''.join(np.random.choice(target_chars_list, target_len - 2)) + '\n' # -2 for \t \n
            
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)

        print(f"Generated {len(self.input_texts)} synthetic samples.")

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx: int):
        # Convert numpy arrays to torch tensors
        encoder_input = torch.tensor(self.encoder_input_data[idx], dtype=torch.float32)
        decoder_input = torch.tensor(self.decoder_input_data[idx], dtype=torch.float32)
        decoder_target = torch.tensor(self.decoder_target_data[idx], dtype=torch.float32)
        return encoder_input, decoder_input, decoder_target


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    num_samples = config.get("num_samples", 10000)
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    
    # Determine the directory for data. Prioritize config's data_dir, then a default FL location.
    data_dir_from_config = config.get("data_dir")
    if data_dir_from_config:
        data_dir = Path(data_dir_from_config)
    else:
        # Default to a common location like ~/.fl_data/s2s_char_translation
        data_dir = Path.home() / ".fl_data" / "s2s_char_translation"
    data_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    
    data_file_path = data_dir / "fra.txt"

    # Attempt to download and extract data if not found and synthetic data is not allowed
    if not data_file_path.exists() and not allow_synthetic_data:
        print(f"Data file not found at {data_file_path}. Attempting to download fra-eng.zip...")
        zip_fpath = data_dir / "fra-eng.zip"
        try:
            download_url = "http://www.manythings.org/anki/fra-eng.zip"
            print(f"Downloading {download_url} to {zip_fpath}...")
            response = requests.get(download_url, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(zip_fpath, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=8192):
                    fd.write(chunk)
            print(f"Unzipping {zip_fpath} to {data_dir}...")
            with zipfile.ZipFile(zip_fpath, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Download and extraction complete.")
        except Exception as e:
            raise FileNotFoundError(
                f"Data file not found at {data_file_path} and download failed: {e}. "
                "Set 'allow_synthetic_data' to True in config to use synthetic data."
            )
    
    dataset = TranslationDataset(
        data_path=str(data_file_path),
        num_samples=num_samples,
        allow_synthetic_data=allow_synthetic_data
    )

    # Store derived parameters in config. This is a common pattern in FL,
    # where build_dataloader determines dataset characteristics and passes them
    # to build_model via the config.
    config["num_encoder_tokens"] = dataset.num_encoder_tokens
    config["num_decoder_tokens"] = dataset.num_decoder_tokens
    config["max_encoder_seq_length"] = dataset.max_encoder_seq_length
    config["max_decoder_seq_length"] = dataset.max_decoder_seq_length
    # Optional: store token indices if needed for inference outside of dataset
    config["input_token_index"] = dataset.input_token_index
    config["target_token_index"] = dataset.target_token_index
    config["reverse_input_char_index"] = {i: char for char, i in dataset.input_token_index.items()}
    config["reverse_target_char_index"] = {i: char for char, i in dataset.target_token_index.items()}


    # Split dataset based on validation_split ratio from original script (0.2)
    val_split_ratio = config.get("validation_split", 0.2)
    train_size = int((1.0 - val_split_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    
    # Ensure reproducibility for random_split
    g = torch.Generator().manual_seed(config.get("seed", 42))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    if split == "train":
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elif split == "val":
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

# --- Training Step Definition ---
def train_step(model: torch.nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    # Move tensors to the device of the model parameters
    device = next(model.parameters()).device
    encoder_input_data, decoder_input_data, decoder_target_data = [t.to(device) for t in batch]

    # Forward pass
    outputs = model(encoder_input_data, decoder_input_data)

    # Calculate loss
    # decoder_target_data is one-hot encoded (batch_size, seq_len, num_decoder_tokens)
    # torch.nn.CrossEntropyLoss expects class indices for targets (batch_size, seq_len)
    # and logits for predictions in the format (N, C, L) where N=batch, C=classes, L=seq_len.

    # Convert one-hot target to class indices by taking argmax along the last dimension
    target_indices = torch.argmax(decoder_target_data, dim=-1) # shape: (batch_size, max_decoder_seq_length)

    # Permute outputs to match CrossEntropyLoss expected format: (batch_size, num_decoder_tokens, max_decoder_seq_length)
    outputs_permuted = outputs.permute(0, 2, 1) # From (N, L, C) to (N, C, L)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs_permuted, target_indices)

    # Do NOT call loss.backward() or optimizer.step()
    return loss