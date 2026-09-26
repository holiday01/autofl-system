# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from io import open
import types as _types
from torch.utils.data import DataLoader, Dataset, random_split

# ---- BEGIN inlined sibling module data.py ----

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids

# ---- END inlined sibling module data.py ----

data = _types.SimpleNamespace(Dictionary=Dictionary, Corpus=Corpus)

# ---- BEGIN inlined sibling module model.py ----
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

# ---- END inlined sibling module model.py ----


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class LanguageModelDataset(Dataset):
    """Sliding-window dataset yielding non-overlapping (input_seq, target_seq) pairs.

    Each sample is a pair of 1-D int64 tensors of length *bptt*.
    The target is the input shifted right by one token position.
    """

    def __init__(self, token_data: torch.Tensor, bptt: int = 35) -> None:
        self.token_data = token_data
        self.bptt = bptt
        # Need at least bptt+1 tokens for one complete (input, target) pair.
        self.n_samples = max(0, (len(token_data) - 1) // bptt)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.bptt
        x = self.token_data[start: start + self.bptt]
        y = self.token_data[start + 1: start + self.bptt + 1]
        return x, y


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the language model.

    config["model_kwargs"] keys (all optional, shown with defaults):
      model   : str   = "LSTM"   one of LSTM | GRU | RNN_TANH | RNN_RELU | Transformer
      ntoken  : int   = 33278    vocabulary size (wikitext-2 default)
      emsize  : int   = 200      embedding / d_model dimension
      nhid    : int   = 200      hidden units per layer (or Transformer ff dim)
      nlayers : int   = 2        number of recurrent / encoder layers
      dropout : float = 0.2
      tied    : bool  = False    tie input/output embeddings (RNN variants only)
      nhead   : int   = 2        attention heads (Transformer only)
    """
    kwargs = config.get("model_kwargs", {})
    model_type = kwargs.get("model", "LSTM")
    ntoken = kwargs.get("ntoken", 33278)
    emsize = kwargs.get("emsize", 200)
    nhid = kwargs.get("nhid", 200)
    nlayers = kwargs.get("nlayers", 2)
    dropout = kwargs.get("dropout", 0.2)

    if model_type == "Transformer":
        nhead = kwargs.get("nhead", 2)
        model = TransformerModel(ntoken, emsize, nhead, nhid, nlayers, dropout)
    else:
        tied = kwargs.get("tied", False)
        model = RNNModel(model_type, ntoken, emsize, nhid, nlayers, dropout, tied)

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Config keys consumed:
      config["data_path"]                 root dir with train.txt / valid.txt / test.txt
      config["local"]["batch_size"]       mini-batch size (default 16)
      config["model_kwargs"]["bptt"]      BPTT window length (default 35)
      config["seed"]                      RNG seed for random_split (default 42)
      config["allow_synthetic_data"]      if True, fall back to random tokens when the
                                          real corpus is absent (default False)

    The concatenated train+valid token sequence is divided 80/20 via random_split.
    """
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)
    bptt = config.get("model_kwargs", {}).get("bptt", 35)
    seed = config.get("seed", 42)

    try:
        corpus = Corpus(data_path)
        all_tokens = torch.cat([corpus.train, corpus.valid])
        full_dataset = LanguageModelDataset(all_tokens, bptt=bptt)
    except (AssertionError, FileNotFoundError, OSError) as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Language model corpus not found at '{data_path}' "
                "(expected train.txt, valid.txt, and test.txt inside that directory). "
                "Set config['allow_synthetic_data']=True to train on synthetic data instead."
            ) from exc
        # Synthetic fallback: random token IDs matching real data dtype/shape contract.
        ntoken = config.get("model_kwargs", {}).get("ntoken", 1000)
        n_tokens = 5000 * bptt + 1
        synthetic_tokens = torch.randint(0, ntoken, (n_tokens,))
        full_dataset = LanguageModelDataset(synthetic_tokens, bptt=bptt)

    n_total = len(full_dataset)
    if n_total < 2:
        raise ValueError(
            f"Dataset at '{data_path}' yields only {n_total} BPTT sample(s) "
            f"(bptt={bptt}). At least 2 samples are required to produce a "
            "train/val split."
        )

    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    chosen = train_set if split == "train" else val_set
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=True,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step().  This function must NOT do either.

    batch : tuple (data_seq, target_seq) from LanguageModelDataset / DataLoader
            data_seq   shape (batch_size, bptt)  int64
            target_seq shape (batch_size, bptt)  int64
    """
    device = next(model.parameters()).device
    criterion = nn.NLLLoss()

    data_seq, target_seq = batch
    data_seq = data_seq.to(device)       # (B, T)
    target_seq = target_seq.to(device)   # (B, T)

    # Both RNNModel and TransformerModel expect layout (seq_len, batch) = (T, B)
    data_seq = data_seq.t().contiguous()   # (T, B)
    flat_target = target_seq.reshape(-1)   # (T*B,)

    model.train()

    if isinstance(model, TransformerModel):
        output = model(data_seq)                        # (T, B, ntoken)
        output = output.view(-1, output.size(-1))       # (T*B, ntoken)
    else:
        bsz = data_seq.size(1)
        hidden = model.init_hidden(bsz)
        hidden = repackage_hidden(hidden)
        output, _hidden = model(data_seq, hidden)       # output: (T*B, ntoken)

    loss = criterion(output, flat_target)
    return loss