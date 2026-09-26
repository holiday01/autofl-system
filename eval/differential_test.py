"""Source-to-converted differential tests for the AutoFL structured-prompt conversions.

Motivation (reviewer): passing the five-stage runtime harness certifies that a
converted module is *executable* under the FL runtime, not that it is a
*faithful migration* of the source script.  This script drives the original
script's own code (model classes, loss computation, optimizer loop) and the
converted module (build_model / build_dataloader / train_step, executed through
the real ``fl_runtime.client.FLClient.local_train`` procedure) on the same real
data and reports, per pair:

  (a) preprocessing equivalence      -- dataset class, length, shapes, dtype,
                                        sample-aligned value statistics, labels
  (b) architecture equivalence       -- parameter count, leaf-module sequence,
                                        state_dict keys/shapes
  (c) loss at initialisation         -- identical parameters, same real batch
  (d) one-step update equivalence    -- gradient cosine, parameter-delta cosine
                                        and relative L2 (source loop vs runtime)
  (e) task outcome after a fixed step budget

Pairs
  P1 pytorch/mnist_main.py                 -> mnist_main_fl_structured.py
  P2 lightning/mnist_lite.py (GAN)         -> mnist_lite_fl_structured.py
  P3 lightning/backbone_image_classifier   -> backbone_image_classifier_fl_structured.py
  P4 pytorch/dcgan_main.py (DCGAN)         -> dcgan_main_fl_structured.py
  P5 sklearn/mlp_digits.py                 -> mlp_digits_fl_structured.py
  P6 sklearn/svm_iris.py    (surrogate)    -> svm_iris_fl_structured.py
  P7 xgboost/xgb_breast_cancer.py (surr.)  -> xgb_breast_cancer_fl_structured.py

No LLM calls, no network downloads: MNIST and CIFAR-10 are read from the local
copies under the repository root; sklearn datasets are bundled.

Usage
  python eval/differential_test.py [--pairs P1,P2,...] [--steps 300]
                                   [--batch-size 64] [--device cpu|cuda]

Outputs
  results/differential_test.csv
  <scratch>/differential_test_tables.md   (markdown tables for the notes)
"""
from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------
# Device selection must happen before torch is imported so that the runtime's
# FLClient (which picks CUDA whenever it is visible) lands on the same device.
# --------------------------------------------------------------------------
_DEVICE_ARG = "cpu"
for _i, _a in enumerate(sys.argv):
    if _a == "--device" and _i + 1 < len(sys.argv):
        _DEVICE_ARG = sys.argv[_i + 1]
    elif _a.startswith("--device="):
        _DEVICE_ARG = _a.split("=", 1)[1]
if _DEVICE_ARG == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import ast
import copy
import csv
import importlib.util
import math
import random
import time
import types
import warnings
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parent.parent            # repository root
DATA_ROOT = str(REPO)                                     # <REPO>/MNIST/raw, <REPO>/cifar-10-batches-py
RESULTS = REPO / "results"
CSV_PATH = RESULTS / "differential_test.csv"
SCRATCH = Path(os.environ.get(
    "AUTOFL_SCRATCH",
    str(REPO / "results" / "scratch")))
TABLES_PATH = SCRATCH / "differential_test_tables.md"
SEED = 42

if str(REPO.parent) not in sys.path:
    sys.path.insert(0, str(REPO.parent))
from autofl.fl_runtime.client import FLClient  # noqa: E402  (the real runtime)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.set_num_threads(int(os.environ.get("AUTOFL_THREADS", min(16, os.cpu_count() or 4))))

PAIRS = {
    "P1": ("benchmarks/pytorch/mnist_main.py", "benchmarks/pytorch/mnist_main_fl_structured.py"),
    "P2": ("benchmarks/lightning/mnist_lite.py", "benchmarks/lightning/mnist_lite_fl_structured.py"),
    "P3": ("benchmarks/lightning/backbone_image_classifier.py",
           "benchmarks/lightning/backbone_image_classifier_fl_structured.py"),
    "P4": ("benchmarks/pytorch/dcgan_main.py", "benchmarks/pytorch/dcgan_main_fl_structured.py"),
    "P5": ("benchmarks/sklearn/mlp_digits.py", "benchmarks/sklearn/mlp_digits_fl_structured.py"),
    "P6": ("benchmarks/sklearn/svm_iris.py", "benchmarks/sklearn/svm_iris_fl_structured.py"),
    "P7": ("benchmarks/xgboost/xgb_breast_cancer.py",
           "benchmarks/xgboost/xgb_breast_cancer_fl_structured.py"),
}

# ==========================================================================
# Result recording
# ==========================================================================
ROWS: list[dict] = []
CSV_FIELDS = ["pair_id", "source_script", "converted_script", "stage", "metric",
              "source_value", "converted_value", "comparison_value", "verdict",
              "notes", "device"]


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "nan"
        return f"{v:.6g}"
    if isinstance(v, torch.Tensor):
        return _fmt(v.item() if v.numel() == 1 else v.tolist())
    return str(v)


def record(pair: str, stage: str, metric: str, source=None, converted=None,
           comparison=None, verdict: str = "", notes: str = "") -> None:
    src_path, conv_path = PAIRS[pair]
    row = {
        "pair_id": pair, "source_script": src_path, "converted_script": conv_path,
        "stage": stage, "metric": metric,
        "source_value": _fmt(source), "converted_value": _fmt(converted),
        "comparison_value": _fmt(comparison), "verdict": verdict, "notes": notes,
        "device": str(DEVICE),
    }
    ROWS.append(row)
    print(f"  [{pair}/{stage}] {metric}: src={row['source_value']} conv={row['converted_value']} "
          f"cmp={row['comparison_value']} -> {verdict}")


# ==========================================================================
# Generic helpers
# ==========================================================================
def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install_lightning_shim() -> None:
    """Map ``lightning.pytorch`` (2.x namespace used by the sources) onto the
    installed ``pytorch_lightning`` 1.9.5 so the source LightningModules can be
    imported and their own ``training_step`` executed."""
    if "lightning" in sys.modules:
        return
    import pytorch_lightning as pl
    import pytorch_lightning.core, pytorch_lightning.trainer, pytorch_lightning.utilities  # noqa: E401
    import pytorch_lightning.utilities.imports, pytorch_lightning.cli  # noqa: E401
    import pytorch_lightning.demos, pytorch_lightning.demos.mnist_datamodule  # noqa: E401
    root = types.ModuleType("lightning"); root.__path__ = []
    lp = types.ModuleType("lightning.pytorch"); lp.__path__ = []
    for name in ("LightningModule", "LightningDataModule", "Trainer", "cli_lightning_logo",
                 "seed_everything"):
        setattr(lp, name, getattr(pl, name))
    root.pytorch = lp
    sys.modules["lightning"] = root
    sys.modules["lightning.pytorch"] = lp
    sys.modules["lightning.pytorch.core"] = pytorch_lightning.core
    sys.modules["lightning.pytorch.trainer"] = pytorch_lightning.trainer
    sys.modules["lightning.pytorch.utilities"] = pytorch_lightning.utilities
    sys.modules["lightning.pytorch.utilities.imports"] = pytorch_lightning.utilities.imports
    sys.modules["lightning.pytorch.cli"] = pytorch_lightning.cli
    sys.modules["lightning.pytorch.demos"] = pytorch_lightning.demos
    sys.modules["lightning.pytorch.demos.mnist_datamodule"] = pytorch_lightning.demos.mnist_datamodule


def make_config(lr: float, batch_size: int, **extra) -> dict:
    """Same key set as eval/evaluator.py::_MINIMAL_CONFIG, with the real data
    root and the batch size / learning rate used for this test."""
    cfg = {
        "num_rounds": 1, "local_epochs": 1, "aggregation_algorithm": "FedAvg",
        "learning_rate": lr, "seed": SEED, "data_path": DATA_ROOT,
        "allow_synthetic_data": False,
        "local": {"batch_size": batch_size, "use_amp": False, "num_workers": 0},
    }
    cfg.update(extra)
    return cfg


def make_client(pair: str, config: dict) -> FLClient:
    seed_all()
    return FLClient(f"{pair}-client", REPO / PAIRS[pair][1], config)


def underlying(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def dataset_targets(ds) -> list:
    ds = underlying(ds)
    if isinstance(ds, TensorDataset):
        return ds.tensors[1].tolist()
    t = getattr(ds, "targets", None)
    if t is not None:
        return t.tolist() if isinstance(t, torch.Tensor) else list(t)
    return [int(ds[i][1]) for i in range(len(ds))]


def stack_first(ds, n: int = 256):
    xs, ys = [], []
    for i in range(min(n, len(ds))):
        x, y = ds[i]
        xs.append(torch.as_tensor(x))
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys)


def tstats(x: torch.Tensor) -> dict:
    x = x.float()
    return {"min": x.min().item(), "max": x.max().item(), "mean": x.mean().item(),
            "std": x.std().item()}


def compare_preprocessing(pair: str, src_ds, conv_train_loader, conv_val_loader,
                          src_train_len, src_val_len, src_split_note: str, n: int = 256) -> None:
    """Sample-aligned comparison of the source dataset object and the dataset
    underlying the converted build_dataloader (same index -> same example)."""
    conv_u = underlying(conv_train_loader.dataset)
    xs, ys = stack_first(src_ds, n)
    xc, yc = stack_first(conv_u, n)
    record(pair, "a_preprocessing", "dataset_class", type(src_ds).__name__, type(conv_u).__name__,
           None, "match" if type(src_ds).__name__ == type(conv_u).__name__ else "differs",
           "class of the dataset actually instantiated (synthetic fallback would be TensorDataset)")
    real = type(conv_u).__name__ in ("MNIST", "CIFAR10")
    record(pair, "a_preprocessing", "converted_used_real_data", None, real, None,
           "match" if real else "differs", "True when the converted loader wraps torchvision's real dataset")
    record(pair, "a_preprocessing", "dataset_len_underlying", len(src_ds), len(conv_u), None,
           "match" if len(src_ds) == len(conv_u) else "differs", "length of the underlying dataset")
    record(pair, "a_preprocessing", "train_split_len", src_train_len, len(conv_train_loader.dataset), None, "info",
           f"source: {src_split_note}; converted: 90/10 random_split(seed 42)")
    record(pair, "a_preprocessing", "val_split_len", src_val_len, len(conv_val_loader.dataset), None, "info", "")
    record(pair, "a_preprocessing", "example_shape", tuple(xs.shape[1:]), tuple(xc.shape[1:]), None,
           "match" if xs.shape == xc.shape else "differs", "")
    record(pair, "a_preprocessing", "example_dtype", str(xs.dtype), str(xc.dtype), None,
           "match" if xs.dtype == xc.dtype else "differs", "")
    ss, sc = tstats(xs), tstats(xc)
    for k in ("min", "max", "mean", "std"):
        record(pair, "a_preprocessing", f"value_{k}_first{n}", ss[k], sc[k], sc[k] - ss[k],
               "match" if abs(sc[k] - ss[k]) < 1e-6 else "differs", f"over the first {n} examples")
    if xs.shape == xc.shape:
        mad = (xs - xc).abs().max().item()
        record(pair, "a_preprocessing", f"max_abs_pixel_diff_first{n}", None, None, mad,
               "match" if mad == 0 else "differs", "sample-aligned: same index in both datasets")
        lab = int((ys == yc).all().item())
        record(pair, "a_preprocessing", f"labels_identical_first{n}", None, None, lab,
               "match" if lab else "differs", "")
    ls, lc = sorted(set(dataset_targets(src_ds))), sorted(set(dataset_targets(conv_u)))
    record(pair, "a_preprocessing", "label_set", ls, lc, None, "match" if ls == lc else "differs", "")
    bs = conv_train_loader.batch_size
    record(pair, "a_preprocessing", "loader_batch_size_used", None, bs, None, "info",
           "converted loader batch size from config['local']['batch_size']")
    record(pair, "a_preprocessing", "converted_train_shuffle",
           None, isinstance(conv_train_loader.sampler, torch.utils.data.RandomSampler), None, "info", "")


def leaf_types(model: nn.Module) -> list[str]:
    return [type(m).__name__ for m in model.modules() if len(list(m.children())) == 0]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compare_architecture(pair: str, src_model: nn.Module, conv_model: nn.Module, tag: str = "",
                         note: str = "") -> bool:
    sfx = f"_{tag}" if tag else ""
    ps, pc = n_params(src_model), n_params(conv_model)
    record(pair, "b_architecture", f"param_count{sfx}", ps, pc, pc - ps,
           "match" if ps == pc else "differs", note)
    ls, lc = leaf_types(src_model), leaf_types(conv_model)
    record(pair, "b_architecture", f"leaf_modules{sfx}", ">".join(ls), ">".join(lc), None,
           "match" if ls == lc else "differs", "leaf nn.Module types in registration order")
    ss, sc = src_model.state_dict(), conv_model.state_dict()
    keys_eq = list(ss.keys()) == list(sc.keys())
    shapes_eq = keys_eq and all(tuple(ss[k].shape) == tuple(sc[k].shape) for k in ss)
    record(pair, "b_architecture", f"state_dict_keys_match{sfx}", len(ss), len(sc), None,
           "match" if keys_eq else "differs", "" if keys_eq else f"src={list(ss)[:3]}... conv={list(sc)[:3]}...")
    record(pair, "b_architecture", f"state_dict_shapes_match{sfx}", None, None, shapes_eq,
           "match" if shapes_eq else "differs", "")
    return shapes_eq


def snapshot(model: nn.Module, prefix: str = "") -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(prefix)}


def flat(d: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).double().cpu() for v in d.values()])


def delta_vec(before: dict, after: dict) -> torch.Tensor:
    return torch.cat([(after[k] - before[k]).reshape(-1).double().cpu() for k in before])


def grads_of(model: nn.Module, prefix: str = "") -> torch.Tensor:
    vs = []
    for n, p in model.named_parameters():
        if n.startswith(prefix):
            vs.append((p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double().cpu())
    return torch.cat(vs)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.norm() == 0 or b.norm() == 0:
        return float("nan")
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def rel_l2(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a - ref).norm() / ref.norm()) if ref.norm() > 0 else float("nan")


def record_delta(pair: str, metric: str, d_rt: torch.Tensor, d_src: torch.Tensor, notes: str,
                 threshold: float = 0.99) -> None:
    c = cosine(d_rt, d_src)
    record(pair, "d_one_step", f"{metric}_cosine", None, None, c,
           "match" if c >= threshold else ("opposite" if c < 0 else "differs"), notes)
    record(pair, "d_one_step", f"{metric}_rel_l2", float(d_src.norm()), float(d_rt.norm()),
           rel_l2(d_rt, d_src), "info", "||delta_runtime - delta_source|| / ||delta_source||")


class CappedLoader:
    """Iterates over at most `max_steps` batches of `loader` (or a list of
    batches).  Exposes `.dataset` and `__len__` so that the source scripts'
    loops (which log len(loader.dataset)) and FLClient.local_train both accept it."""

    def __init__(self, loader, max_steps: int | None = None):
        self.loader = loader
        self.max_steps = max_steps
        self.dataset = getattr(loader, "dataset", None)
        if self.dataset is None:
            self.dataset = list(range(sum(int(b[0].shape[0]) for b in loader)))

    def __iter__(self):
        n = 0
        while True:
            for b in self.loader:
                if self.max_steps is not None and n >= self.max_steps:
                    return
                yield b
                n += 1
            if self.max_steps is None:
                return
            if n >= self.max_steps:
                return

    def __len__(self):
        base = len(self.loader) if hasattr(self.loader, "__len__") else 0
        return min(base, self.max_steps) if self.max_steps is not None else base


class _ProxyMod:
    """Presents a fixed DataLoader through the converted module's interface so
    that the *real* FLClient.local_train can be run on a chosen batch list."""

    def __init__(self, mod, loader):
        self._mod = mod
        self._loader = loader
        self.train_step = mod.train_step
        self.build_model = mod.build_model

    def build_dataloader(self, config, split="train"):
        return self._loader


def runtime_train(client: FLClient, loader, max_steps: int | None = None, local_epochs: int = 1) -> dict:
    """Execute the unmodified FLClient.local_train (AdamW at config lr, grad-norm
    clip 1.0, zero_grad/backward/step pattern) over the given loader."""
    real_mod = client.mod
    old_epochs = client.config.get("local_epochs", 1)
    client.mod = _ProxyMod(real_mod, CappedLoader(loader, max_steps))
    client.config["local_epochs"] = local_epochs
    try:
        _, metrics = client.local_train()
    finally:
        client.mod = real_mod
        client.config["local_epochs"] = old_epochs
    return metrics


@contextmanager
def loss_recorder():
    """Records every loss value produced by the criteria used in the tested
    modules (BCE-with-logits, BCELoss, nll_loss, cross_entropy) in call order,
    without modifying the converted files."""
    calls: list[tuple[str, float]] = []
    orig = {"bcel": F.binary_cross_entropy_with_logits, "bce_fwd": nn.BCELoss.forward,
            "nll": F.nll_loss, "ce": F.cross_entropy}

    def wrap(name, fn):
        def inner(*a, **k):
            out = fn(*a, **k)
            calls.append((name, float(out.detach())))
            return out
        return inner

    F.binary_cross_entropy_with_logits = wrap("bce_logits", orig["bcel"])
    nn.BCELoss.forward = wrap("bce", orig["bce_fwd"])
    F.nll_loss = wrap("nll", orig["nll"])
    F.cross_entropy = wrap("ce", orig["ce"])
    try:
        yield calls
    finally:
        F.binary_cross_entropy_with_logits = orig["bcel"]
        nn.BCELoss.forward = orig["bce_fwd"]
        F.nll_loss = orig["nll"]
        F.cross_entropy = orig["ce"]


def first_batch(loader) -> tuple:
    for b in loader:
        return tuple(x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in b)
    raise RuntimeError("empty loader")


@torch.no_grad()
def classifier_accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    model.train()
    return correct / max(total, 1)


def timed(fn: Callable, *a, **k):
    t0 = time.time()
    out = fn(*a, **k)
    return out, time.time() - t0


# ==========================================================================
# Shared pieces for the classifier pairs
# ==========================================================================
def vdiff(a: float, b: float, tol: float = 1e-5) -> str:
    return "match" if abs(a - b) <= tol * max(1.0, abs(a)) else "differs"


def state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def to_dev(batch) -> tuple:
    return tuple(x.to(DEVICE) if isinstance(x, torch.Tensor) else x for x in batch)


# ==========================================================================
# P1  pytorch/mnist_main.py  ->  mnist_main_fl_structured.py
# ==========================================================================
def run_p1(steps: int, bs: int) -> None:
    pair = "P1"
    print(f"\n=== {pair}: {PAIRS[pair][0]} -> {PAIRS[pair][1]} ===")
    src = load_module(REPO / PAIRS[pair][0], "src_mnist_main")
    lr = 1e-3            # harness default (the source's Adadelta lr=1.0 is not transferable to AdamW)
    config = make_config(lr, bs)
    client = make_client(pair, config)
    conv = client.mod

    # (a) --- mnist_main.py L118-125; root '../data' (cwd-relative) redirected to DATA_ROOT
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    src_train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    src_test = datasets.MNIST(DATA_ROOT, train=False, transform=transform)
    seed_all()
    conv_train = conv.build_dataloader(config, "train")
    conv_val = conv.build_dataloader(config, "val")
    compare_preprocessing(pair, src_train, conv_train, conv_val, 60000, "-", "no split (60000 train; separate 10k test set)")

    # (b)
    seed_all()
    src_model = src.Net().to(DEVICE)
    conv_model = client.model
    compare_architecture(pair, src_model, conv_model)
    conv_model.load_state_dict(src_model.state_dict())

    seed_all()
    src_loader = DataLoader(src_train, batch_size=bs, shuffle=True)
    batch = first_batch(src_loader)
    data, target = batch

    # (c) --- source loss: train() L40-41  (output = model(data); loss = F.nll_loss(output, target))
    seed_all(); src_model.train(); loss_s = F.nll_loss(src_model(data), target)
    seed_all(); conv_model.train(); loss_c = conv.train_step(conv_model, batch, None, config)
    record(pair, "c_loss_init", "loss_train_mode", loss_s.item(), loss_c.item(), loss_c.item() - loss_s.item(),
           vdiff(loss_s.item(), loss_c.item()), "identical params, same real batch, dropout RNG re-seeded")
    with torch.no_grad():
        src_model.eval(); conv_model.eval()
        ls_e = F.nll_loss(src_model(data), target).item()
        lc_e = conv.train_step(conv_model, batch, None, config).item()
    record(pair, "c_loss_init", "loss_eval_mode", ls_e, lc_e, lc_e - ls_e, vdiff(ls_e, lc_e), "dropout disabled")
    src_model.train(); conv_model.train()

    # (d) gradients
    seed_all(); src_model.zero_grad(); F.nll_loss(src_model(data), target).backward(); g_s = grads_of(src_model)
    seed_all(); conv_model.zero_grad(); conv.train_step(conv_model, batch, None, config).backward(); g_c = grads_of(conv_model)
    record(pair, "d_one_step", "grad_cosine", None, None, cosine(g_c, g_s),
           "match" if cosine(g_c, g_s) > 0.9999 else "differs", "gradient of source loss vs gradient of converted train_step, same params/batch")
    state0 = state_cpu(src_model)
    args = Namespace(log_interval=10 ** 9, dry_run=False)

    def source_step(opt_factory):
        m = src.Net().to(DEVICE); m.load_state_dict(state0); m.train()
        opt = opt_factory(m)
        before = snapshot(m); seed_all()
        src.train(args, m, DEVICE, CappedLoader([batch]), opt, 1)     # the source's own train() loop
        return delta_vec(before, snapshot(m))

    d_native = source_step(lambda m: optim.Adadelta(m.parameters(), lr=1.0))       # main() L128
    d_matched = source_step(lambda m: optim.AdamW(m.parameters(), lr=lr))
    client.set_weights(state0); before = snapshot(client.model); seed_all()
    runtime_train(client, [batch])
    d_rt = delta_vec(before, snapshot(client.model))
    record_delta(pair, "param_delta_runtime_vs_source_native", d_rt, d_native,
                 "source: Adadelta(lr=1.0) via mnist_main.train(); runtime: AdamW(lr=1e-3)+clip_grad_norm(1.0)", 0.99)
    record_delta(pair, "param_delta_runtime_vs_source_adamw", d_rt, d_matched,
                 "source loop with AdamW(lr=1e-3) substituted (optimizer-matched control)", 0.99)

    # (e) fixed budget
    test_loader = DataLoader(src_test, batch_size=1000)
    seed_all(); init = state_cpu(src.Net())

    def source_budget(opt_factory):
        m = src.Net().to(DEVICE); m.load_state_dict(init); m.train()
        opt = opt_factory(m); seed_all()
        src.train(args, m, DEVICE, CappedLoader(src_loader, steps), opt, 1)
        return classifier_accuracy(m, test_loader)

    acc_native, t1 = timed(source_budget, lambda m: optim.Adadelta(m.parameters(), lr=1.0))
    acc_matched, t2 = timed(source_budget, lambda m: optim.AdamW(m.parameters(), lr=lr))
    client.set_weights(init); seed_all()
    _, t3 = timed(runtime_train, client, conv_train, steps)
    acc_rt = classifier_accuracy(client.model, test_loader)
    record(pair, "e_budget", f"test_acc_{steps}steps_bs{bs}_source_native", acc_native, None, None, "info",
           f"Adadelta(lr=1.0), source train(); MNIST test 10k; {t1:.0f}s")
    record(pair, "e_budget", f"test_acc_{steps}steps_bs{bs}_source_adamw", acc_matched, None, None, "info",
           f"source train() with AdamW(lr={lr}); {t2:.0f}s")
    record(pair, "e_budget", f"test_acc_{steps}steps_bs{bs}_converted_runtime", None, acc_rt, acc_rt - acc_matched,
           "match" if abs(acc_rt - acc_matched) < 0.02 else "differs",
           f"FLClient.local_train, AdamW(lr={lr})+clip; delta vs optimizer-matched source; {t3:.0f}s")


# ==========================================================================
# P3  lightning/backbone_image_classifier.py -> backbone_image_classifier_fl_structured.py
# ==========================================================================
def run_p3(steps: int, bs: int) -> None:
    pair = "P3"
    print(f"\n=== {pair}: {PAIRS[pair][0]} -> {PAIRS[pair][1]} ===")
    install_lightning_shim()
    bb = load_module(REPO / PAIRS[pair][0], "src_backbone")
    lr = 1e-4            # LitClassifier(learning_rate=0.0001) default, L67
    config = make_config(lr, bs)
    client = make_client(pair, config)
    conv = client.mod

    # (a) --- MyDataModule L104-121; DATASETS_PATH (<repo>/Datasets, absent) redirected to DATA_ROOT
    bb.DATASETS_PATH = DATA_ROOT
    seed_all()
    dm = bb.MyDataModule(batch_size=bs)
    src_loader = dm.train_dataloader()           # NOTE: source loader has no shuffle
    seed_all()
    conv_train = conv.build_dataloader(config, "train")
    conv_val = conv.build_dataloader(config, "val")
    compare_preprocessing(pair, underlying(dm.mnist_train), conv_train, conv_val, len(dm.mnist_train),
                          len(dm.mnist_val), "random_split([55000, 5000], seed 42)")
    record(pair, "a_preprocessing", "source_train_shuffle", False, None, None, "info",
           "MyDataModule.train_dataloader() passes no shuffle=True; converted shuffles")

    # (b)
    seed_all()
    lit = bb.LitClassifier().to(DEVICE)
    compare_architecture(pair, lit.backbone, client.model,
                         note="source Backbone is LitClassifier.backbone (state_dict keys carry 'backbone.' prefix at LightningModule level)")
    client.model.load_state_dict(lit.backbone.state_dict())
    lit.log = lambda *a, **k: None               # no Trainer attached

    batch = first_batch(src_loader)

    # (c)
    loss_s = lit.training_step(batch, 0)         # the source's own training_step
    loss_c = conv.train_step(client.model, batch, None, config)
    record(pair, "c_loss_init", "loss_train_mode", loss_s.item(), loss_c.item(), loss_c.item() - loss_s.item(),
           vdiff(loss_s.item(), loss_c.item()), "LitClassifier.training_step vs converted train_step, identical params")

    # (d)
    lit.zero_grad(); loss_s.backward(); g_s = grads_of(lit)
    client.model.zero_grad(); loss_c.backward(); g_c = grads_of(client.model)
    record(pair, "d_one_step", "grad_cosine", None, None, cosine(g_c, g_s),
           "match" if cosine(g_c, g_s) > 0.9999 else "differs", "")
    state0 = state_cpu(lit)

    def auto_loop(model, loader, opt, n):
        # Lightning automatic optimisation: zero_grad -> training_step -> backward -> step
        for b in CappedLoader(loader, n):
            opt.zero_grad()
            loss = model.training_step(to_dev(b), 0)
            loss.backward()
            opt.step()

    def source_step(opt_factory):
        m = bb.LitClassifier().to(DEVICE); m.load_state_dict(state0); m.log = lambda *a, **k: None
        opt = opt_factory(m); before = snapshot(m); seed_all()
        auto_loop(m, [batch], opt, 1)
        return delta_vec(before, snapshot(m))

    d_native = source_step(lambda m: m.configure_optimizers())          # Adam(lr=1e-4), L93
    d_matched = source_step(lambda m: optim.AdamW(m.parameters(), lr=lr))
    client.set_weights({k[len("backbone."):]: v for k, v in state0.items()})
    before = snapshot(client.model); seed_all()
    runtime_train(client, [batch])
    d_rt = delta_vec(before, snapshot(client.model))
    record_delta(pair, "param_delta_runtime_vs_source_native", d_rt, d_native,
                 "source: configure_optimizers()=Adam(lr=1e-4); runtime: AdamW(lr=1e-4)+clip(1.0)")
    record_delta(pair, "param_delta_runtime_vs_source_adamw", d_rt, d_matched,
                 "source loop with AdamW(lr=1e-4) substituted")

    # (e)
    test_loader = DataLoader(dm.mnist_test, batch_size=1000)
    seed_all(); init = state_cpu(bb.LitClassifier())

    def source_budget(opt_factory):
        m = bb.LitClassifier().to(DEVICE); m.load_state_dict(init); m.log = lambda *a, **k: None
        opt = opt_factory(m); seed_all()
        auto_loop(m, src_loader, opt, steps)
        return classifier_accuracy(m, test_loader)

    acc_native, t1 = timed(source_budget, lambda m: m.configure_optimizers())
    acc_matched, t2 = timed(source_budget, lambda m: optim.AdamW(m.parameters(), lr=lr))
    client.set_weights({k[len("backbone."):]: v for k, v in init.items()}); seed_all()
    _, t3 = timed(runtime_train, client, conv_train, steps)
    acc_rt = classifier_accuracy(client.model, test_loader)
    # control: the source's training_step + AdamW consuming the converted module's own loader
    # (same seed => the same batch sequence as the runtime arm); isolates data order from code semantics
    m_ctl = bb.LitClassifier().to(DEVICE); m_ctl.load_state_dict(init); m_ctl.log = lambda *a, **k: None
    opt_ctl = optim.AdamW(m_ctl.parameters(), lr=lr); seed_all()
    auto_loop(m_ctl, conv_train, opt_ctl, steps)
    acc_ctl = classifier_accuracy(m_ctl, test_loader)
    record(pair, "e_budget", f"test_acc_{steps}steps_bs{bs}_source_native", acc_native, None, None, "info",
           f"Adam(lr={lr}); source loader (unshuffled 55000-subset); MNIST test 10k; {t1:.0f}s")
    record(pair, "e_budget", f"test_acc_{steps}steps_bs{bs}_source_adamw", acc_matched, None, None, "info",
           f"source loop with AdamW(lr={lr}); source loader; {t2:.0f}s")
    record(pair, "e_budget", f"test_acc_{steps}steps_bs{bs}_source_adamw_on_converted_loader", acc_ctl, None, None, "info",
           "source training_step + AdamW fed by the converted build_dataloader (same seed, same batches as the runtime arm)")
    record(pair, "e_budget", f"test_acc_{steps}steps_bs{bs}_converted_runtime", None, acc_rt, acc_rt - acc_ctl,
           "match" if abs(acc_rt - acc_ctl) < 0.02 else "differs",
           f"FLClient.local_train, AdamW(lr={lr})+clip; delta vs the same-batches source control "
           f"(delta vs source_adamw on its own loader: {acc_rt - acc_matched:+.4f}); {t3:.0f}s")


# ==========================================================================
# GAN helpers
# ==========================================================================
def patch_lightning_manual(gan, opt_g, opt_d) -> dict:
    """Provide, on the instance, the Trainer-backed services that the source
    GAN.training_step uses under manual optimisation: optimizers(),
    toggle/untoggle_optimizer (freeze parameters not owned by the optimizer,
    exactly as Lightning does), manual_backward and log_dict."""
    logged: dict = {}
    all_params = list(gan.parameters())
    saved: dict = {}

    def toggle(optimizer, *a, **k):
        owned = {id(p) for g in optimizer.param_groups for p in g["params"]}
        for p in all_params:
            saved[id(p)] = p.requires_grad
            p.requires_grad = id(p) in owned

    def untoggle(optimizer, *a, **k):
        for p in all_params:
            p.requires_grad = saved.get(id(p), True)
        saved.clear()

    gan.optimizers = lambda *a, **k: (opt_g, opt_d)
    gan.toggle_optimizer = toggle
    gan.untoggle_optimizer = untoggle
    gan.manual_backward = lambda loss, *a, **k: loss.backward()
    gan.log_dict = lambda d, *a, **k: logged.update({k_: float(v) for k_, v in d.items()})
    return logged


@torch.no_grad()
def gan_metrics(G: nn.Module, D: nn.Module, real: torch.Tensor, z: torch.Tensor,
                d_outputs_logits: bool) -> dict:
    """Discriminator real/fake accuracy and generator output statistics.
    Two variants: modules in eval() mode (BatchNorm running statistics) and,
    on deep copies, in train() mode (batch statistics, i.e. exactly how D saw
    real and fake batches during training)."""
    out = {}
    for suffix, mode in (("", "eval"), ("_bn_batchstats", "train")):
        Gc, Dc = copy.deepcopy(G), copy.deepcopy(D)
        (Gc.eval(), Dc.eval()) if mode == "eval" else (Gc.train(), Dc.train())
        fake = Gc(z)
        d_real, d_fake = Dc(real).reshape(-1), Dc(fake).reshape(-1)
        if d_outputs_logits:
            d_real, d_fake = torch.sigmoid(d_real), torch.sigmoid(d_fake)
        out[f"d_acc_real{suffix}"] = float((d_real > 0.5).float().mean())
        out[f"d_acc_fake{suffix}"] = float((d_fake < 0.5).float().mean())
        out[f"d_acc_mean{suffix}"] = 0.5 * (out[f"d_acc_real{suffix}"] + out[f"d_acc_fake{suffix}"])
        out[f"d_prob_real_mean{suffix}"] = float(d_real.mean())
        out[f"d_prob_fake_mean{suffix}"] = float(d_fake.mean())
        out[f"g_out_mean{suffix}"] = float(fake.mean())
        out[f"g_out_std{suffix}"] = float(fake.std())
    out["real_mean"], out["real_std"] = float(real.mean()), float(real.std())
    return out


GAN_METRIC_KEYS = ("d_acc_real", "d_acc_fake", "d_acc_mean", "d_prob_real_mean", "d_prob_fake_mean",
                   "g_out_mean", "g_out_std",
                   "d_acc_real_bn_batchstats", "d_acc_fake_bn_batchstats", "d_acc_mean_bn_batchstats",
                   "d_prob_real_mean_bn_batchstats", "d_prob_fake_mean_bn_batchstats",
                   "g_out_mean_bn_batchstats", "g_out_std_bn_batchstats")


def record_gan_metrics(pair: str, arm: str, m: dict, note: str, steps: int, bs: int) -> None:
    for k in GAN_METRIC_KEYS:
        if arm.startswith("source"):
            record(pair, "e_budget", f"{k}_{steps}steps_bs{bs}_{arm}", m[k], None, None, "info", note)
        else:
            record(pair, "e_budget", f"{k}_{steps}steps_bs{bs}_{arm}", None, m[k], None, "info", note)


# ==========================================================================
# P2  lightning/mnist_lite.py (GAN, manual optimisation) -> mnist_lite_fl_structured.py
# ==========================================================================
def gan_source_components(gan, imgs: torch.Tensor) -> dict:
    """The loss quantities of GAN.training_step (mnist_lite.py L140-176) computed
    with the source's own adversarial_loss / discriminator / forward, at the
    current parameters and without the optimizer steps.  Seeded so that z is
    the first randn draw, as in both the source and the converted train_step."""
    seed_all()
    z = torch.randn(imgs.shape[0], gan.hparams.latent_dim).type_as(imgs)
    valid = torch.ones(imgs.size(0), 1).type_as(imgs)
    fake = torch.zeros(imgs.size(0), 1).type_as(imgs)
    g_loss = gan.adversarial_loss(gan.discriminator(gan(z)), valid)
    real_loss = gan.adversarial_loss(gan.discriminator(imgs), valid)
    fake_loss = gan.adversarial_loss(gan.discriminator(gan(z).detach()), fake)
    d_loss = (real_loss + fake_loss) / 2
    return {"z": z, "g_loss": g_loss, "real_loss": real_loss, "fake_loss": fake_loss, "d_loss": d_loss}


def gan_gradient_analysis(pair: str, comp: dict, model_src: nn.Module, g_pref: str, d_pref: str,
                          conv_total: torch.Tensor, conv_model: nn.Module, cg_pref: str, cd_pref: str,
                          d_loss_name: str) -> None:
    """Gradient-level comparison at identical parameters.  Source: the two
    per-optimizer gradients (D from d_loss, G from g_loss).  Converted: one
    backward of the summed loss.  Also isolates the generator-loss gradient
    that leaks into D through the non-detached fake images."""
    model_src.zero_grad(); comp["d_loss"].backward(retain_graph=True); gD_src = grads_of(model_src, d_pref)
    model_src.zero_grad(); comp["fake_loss"].backward(retain_graph=True); gD_fake = grads_of(model_src, d_pref)
    model_src.zero_grad(); comp["g_loss"].backward(retain_graph=True)
    gG_src = grads_of(model_src, g_pref); gD_from_g = grads_of(model_src, d_pref)
    model_src.zero_grad()
    conv_model.zero_grad(); conv_total.backward()
    gG_c, gD_c = grads_of(conv_model, cg_pref), grads_of(conv_model, cd_pref)
    c = cosine(gG_c, gG_src)
    record(pair, "d_one_step", "grad_G_cosine", None, None, c, "match" if c > 0.9999 else "differs",
           "generator gradient: converted summed loss vs source g_loss (same z)")
    c = cosine(gD_c, gD_src)
    record(pair, "d_one_step", "grad_D_cosine", None, None, c, "match" if c > 0.99 else ("opposite" if c < 0 else "differs"),
           f"discriminator gradient: converted summed loss vs source {d_loss_name}")
    c = cosine(gD_from_g, gD_fake)
    record(pair, "d_one_step", "grad_D_leak_cosine(gloss_vs_dfake)", None, None, c,
           "opposite" if c < 0 else "info",
           "cos(dL_g/dD, dL_fake/dD): the g_loss gradient that leaks into D vs the source's fake-branch D gradient")
    ratio = float((gD_fake + gD_from_g).norm() / gD_fake.norm())
    record(pair, "d_one_step", "grad_D_fake_branch_norm_ratio", float(gD_fake.norm()),
           float((gD_fake + gD_from_g).norm()), ratio, "info",
           "||dL_fake/dD + dL_g/dD|| / ||dL_fake/dD||: <1 means the fake-branch signal to D is cancelled in the summed loss")
    ratio2 = float(gD_from_g.norm() / gD_src.norm())
    record(pair, "d_one_step", "grad_D_leak_norm_over_source_norm", float(gD_src.norm()),
           float(gD_from_g.norm()), ratio2, "info", "||dL_g/dD|| / ||d(source D loss)/dD||")


def run_p2(steps: int, bs: int) -> None:
    pair = "P2"
    print(f"\n=== {pair}: {PAIRS[pair][0]} -> {PAIRS[pair][1]} ===")
    install_lightning_shim()
    gm = load_module(REPO / PAIRS[pair][0], "src_mnist_lite")
    from pytorch_lightning.demos.mnist_datamodule import MNISTDataModule
    lr = 2e-4            # GAN(lr=0.0002, b1=0.5, b2=0.999) defaults, L115-118
    config = make_config(lr, bs)
    client = make_client(pair, config)
    conv = client.mod

    # (a) --- main() L224: MNISTDataModule() (data_dir './data' -> DATA_ROOT; num_workers 16 -> 0; batch 32 -> bs)
    dm = MNISTDataModule(data_dir=DATA_ROOT, num_workers=0, batch_size=bs)
    seed_all(); dm.setup("fit")                  # random_split([55000, 5000]) uses the global RNG
    src_loader = dm.train_dataloader()           # shuffle=True, drop_last=True
    seed_all()
    conv_train = conv.build_dataloader(config, "train")
    conv_val = conv.build_dataloader(config, "val")
    compare_preprocessing(pair, underlying(dm.dataset_train), conv_train, conv_val, len(dm.dataset_train),
                          len(dm.dataset_val), "random_split([55000, 5000], global RNG, MNISTDataModule.setup)")
    record(pair, "a_preprocessing", "source_default_batch_size", 32, bs, None, "info",
           "MNISTDataModule default batch_size=32; both sides use bs for this test")

    # (b)
    seed_all()
    gan = gm.GAN().to(DEVICE)
    compare_architecture(pair, gan.generator, client.model.generator, "generator")
    compare_architecture(pair, gan.discriminator, client.model.discriminator, "discriminator")
    full_keys = list(gan.state_dict().keys()) == list(client.model.state_dict().keys())
    record(pair, "b_architecture", "state_dict_keys_match_full_model", None, None, full_keys,
           "match" if full_keys else "differs", "GAN(LightningModule) vs GANModel(nn.Module)")
    client.model.load_state_dict(gan.state_dict())

    batch = first_batch(src_loader)
    imgs = batch[0]

    # (c) --- loss components at identical parameters and identical z
    comp = gan_source_components(gan, imgs)
    seed_all()
    with loss_recorder() as calls:
        total_c = conv.train_step(client.model, batch, None, config)
    vals = [v for _, v in calls]
    d_real_c, d_fake_c, g_c = vals[0], vals[1], vals[2]
    record(pair, "c_loss_init", "g_loss", comp["g_loss"].item(), g_c, g_c - comp["g_loss"].item(),
           vdiff(comp["g_loss"].item(), g_c), "BCE-with-logits(D(G(z)), 1)")
    record(pair, "c_loss_init", "d_real_loss", comp["real_loss"].item(), d_real_c, d_real_c - comp["real_loss"].item(),
           vdiff(comp["real_loss"].item(), d_real_c), "")
    record(pair, "c_loss_init", "d_fake_loss", comp["fake_loss"].item(), d_fake_c, d_fake_c - comp["fake_loss"].item(),
           vdiff(comp["fake_loss"].item(), d_fake_c), "")
    d_loss_s = comp["d_loss"].item()
    record(pair, "c_loss_init", "d_loss_as_optimised", d_loss_s, d_real_c + d_fake_c,
           (d_real_c + d_fake_c) - d_loss_s, "differs",
           "source optimises (real+fake)/2 with opt_d; converted sums real+fake (no /2) into the joint loss")
    src_sum = comp["real_loss"].item() + comp["fake_loss"].item() + comp["g_loss"].item()
    record(pair, "c_loss_init", "returned_total_loss", src_sum, total_c.item(), total_c.item() - src_sum,
           vdiff(src_sum, total_c.item()),
           "converted returns d_real+d_fake+g_loss; source never forms this scalar (compared to the same sum of source components)")
    record(pair, "c_loss_init", "n_loss_calls_in_train_step", 2, len(vals), None, "info",
           "source: two backward targets (g_loss, d_loss); converted: three BCE terms in one scalar")

    # (d)
    gan_gradient_analysis(pair, comp, gan, "generator.", "discriminator.", total_c, client.model,
                          "generator.", "discriminator.", "d_loss=(real+fake)/2")
    state0 = state_cpu(gan)

    def source_step(opt_factory):
        m = gm.GAN().to(DEVICE); m.load_state_dict(state0); m.train()
        opt_g, opt_d = opt_factory(m)
        logged = patch_lightning_manual(m, opt_g, opt_d)
        bG, bD = snapshot(m, "generator."), snapshot(m, "discriminator."); seed_all()
        m.training_step(batch)                               # the source's own training_step
        return delta_vec(bG, snapshot(m, "generator.")), delta_vec(bD, snapshot(m, "discriminator.")), logged

    dG_nat, dD_nat, logged = source_step(lambda m: m.configure_optimizers())   # Adam(2e-4, (0.5, 0.999)) x2
    dG_mat, dD_mat, _ = source_step(lambda m: (optim.AdamW(m.generator.parameters(), lr=lr),
                                               optim.AdamW(m.discriminator.parameters(), lr=lr)))
    record(pair, "d_one_step", "source_training_step_logged_d_loss", logged.get("d_loss"), None, None, "info",
           "d_loss logged by the source training_step (computed after the G update)")
    record(pair, "d_one_step", "source_training_step_logged_g_loss", logged.get("g_loss"), None, None, "info", "")
    client.set_weights(state0)
    bG, bD = snapshot(client.model, "generator."), snapshot(client.model, "discriminator."); seed_all()
    runtime_train(client, [batch])
    dG_rt, dD_rt = delta_vec(bG, snapshot(client.model, "generator.")), delta_vec(bD, snapshot(client.model, "discriminator."))
    record_delta(pair, "G_param_delta_runtime_vs_source_native", dG_rt, dG_nat,
                 "source: opt_g=Adam(2e-4,(0.5,0.999)); runtime: single AdamW(2e-4)+clip over G and D", 0.9)
    record_delta(pair, "D_param_delta_runtime_vs_source_native", dD_rt, dD_nat,
                 "source: opt_d=Adam step on (real+fake)/2 only; runtime: joint step on real+fake+g_loss", 0.9)
    record_delta(pair, "G_param_delta_runtime_vs_source_adamw", dG_rt, dG_mat,
                 "source training_step with two AdamW(2e-4) optimizers substituted", 0.9)
    record_delta(pair, "D_param_delta_runtime_vs_source_adamw", dD_rt, dD_mat, "", 0.9)

    # (e)
    real_test = stack_first(datasets.MNIST(DATA_ROOT, train=False, transform=transforms.ToTensor()), 1000)[0].to(DEVICE)
    seed_all(7); z_eval = torch.randn(1000, gan.hparams.latent_dim, device=DEVICE)
    seed_all(); init = state_cpu(gm.GAN())

    def source_budget(opt_factory):
        m = gm.GAN().to(DEVICE); m.load_state_dict(init); m.train()
        opt_g, opt_d = opt_factory(m)
        logged = patch_lightning_manual(m, opt_g, opt_d); seed_all()
        for b in CappedLoader(src_loader, steps):
            m.training_step(to_dev(b))
        met = gan_metrics(m.generator, m.discriminator, real_test, z_eval, d_outputs_logits=True)
        met["final_d_loss"], met["final_g_loss"] = logged.get("d_loss"), logged.get("g_loss")
        return met

    m_nat, t1 = timed(source_budget, lambda m: m.configure_optimizers())
    m_mat, t2 = timed(source_budget, lambda m: (optim.AdamW(m.generator.parameters(), lr=lr),
                                                optim.AdamW(m.discriminator.parameters(), lr=lr)))
    client.set_weights(init); seed_all()
    rt_metrics, t3 = timed(runtime_train, client, conv_train, steps)
    m_rt = gan_metrics(client.model.generator, client.model.discriminator, real_test, z_eval, d_outputs_logits=True)
    record_gan_metrics(pair, "source_native", m_nat, f"two Adam(2e-4,(0.5,0.999)); {t1:.0f}s", steps, bs)
    record_gan_metrics(pair, "source_adamw", m_mat, f"two AdamW(2e-4); {t2:.0f}s", steps, bs)
    record_gan_metrics(pair, "converted_runtime", m_rt, f"single AdamW(2e-4)+clip on summed loss; mean runtime loss {rt_metrics['loss']}; {t3:.0f}s", steps, bs)
    record(pair, "e_budget", f"final_losses_{steps}steps_source_native", f"d={m_nat['final_d_loss']:.4f},g={m_nat['final_g_loss']:.4f}",
           None, None, "info", "last logged d_loss/g_loss of the source training_step")
    for k in ("d_acc_mean", "d_acc_mean_bn_batchstats", "d_acc_fake_bn_batchstats"):
        record(pair, "e_budget", f"{k}_{steps}steps_delta", m_mat[k], m_rt[k], m_rt[k] - m_mat[k],
               "differs" if abs(m_rt[k] - m_mat[k]) > 0.1 else "match",
               "discriminator real/fake accuracy on 1000 MNIST test images + 1000 samples: converted-runtime minus optimizer-matched source")


# ==========================================================================
# P4  pytorch/dcgan_main.py (two-optimizer DCGAN) -> dcgan_main_fl_structured.py
# ==========================================================================
def extract_dcgan_classes() -> dict:
    """dcgan_main.py is a script (module-level argparse with a required
    --dataset).  Compile only its weights_init / Generator / Discriminator
    definitions, verbatim, with the module-level globals they read
    (nz=100, ngf=64, ndf=64; nc=3 for --dataset cifar10)."""
    path = REPO / PAIRS["P4"][0]
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in ("weights_init", "Generator", "Discriminator")]
    ns = {"torch": torch, "nn": nn, "nz": 100, "ngf": 64, "ndf": 64, "nc": 3}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


def dcgan_source_step(netD, netG, optimizerD, optimizerG, criterion, data, device, nz) -> dict:
    """Verbatim body of the training loop in dcgan_main.py (L213-239)."""
    real_label, fake_label = 1, 0
    netD.zero_grad()
    real_cpu = data[0].to(device)
    batch_size = real_cpu.size(0)
    label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)
    output = netD(real_cpu)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(fake_label)
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()
    netG.zero_grad()
    label.fill_(real_label)
    output = netD(fake)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()
    return {"errD": errD.item(), "errG": errG.item(), "D_x": D_x, "D_G_z1": D_G_z1, "D_G_z2": D_G_z2}


def dcgan_source_components(netD, netG, real_cpu, nz) -> dict:
    """The three loss terms of one dcgan_main.py iteration at the current
    parameters (no updates), with the source's criterion (nn.BCELoss) and
    label conventions; noise is the first randn draw after seeding."""
    criterion = nn.BCELoss()
    seed_all()
    batch_size = real_cpu.size(0)
    ones = torch.full((batch_size,), 1, dtype=real_cpu.dtype, device=DEVICE)
    zeros = torch.full((batch_size,), 0, dtype=real_cpu.dtype, device=DEVICE)
    errD_real = criterion(netD(real_cpu), ones)
    noise = torch.randn(batch_size, nz, 1, 1, device=DEVICE)
    fake = netG(noise)
    errD_fake = criterion(netD(fake.detach()), zeros)
    errG = criterion(netD(fake), ones)
    return {"g_loss": errG, "real_loss": errD_real, "fake_loss": errD_fake, "d_loss": errD_real + errD_fake}


def run_p4(steps: int, bs: int) -> None:
    pair = "P4"
    print(f"\n=== {pair}: {PAIRS[pair][0]} -> {PAIRS[pair][1]} ===")
    ns = extract_dcgan_classes()
    Generator, Discriminator, weights_init = ns["Generator"], ns["Discriminator"], ns["weights_init"]
    nz, lr = 100, 2e-4                                   # --nz 100, --lr 0.0002, --beta1 0.5 defaults
    config = make_config(lr, bs, model_kwargs={"workers": 0})   # in-process loading only; model defaults unchanged
    client = make_client(pair, config)
    conv = client.mod

    # (a) --- dcgan_main.py L83-89 (--dataset cifar10 --dataroot DATA_ROOT --imageSize 64)
    src_tf = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    src_ds = datasets.CIFAR10(root=DATA_ROOT, download=True, transform=src_tf)
    seed_all()
    src_loader = DataLoader(src_ds, batch_size=bs, shuffle=True, num_workers=0)   # source: num_workers=2
    seed_all()
    conv_train = conv.build_dataloader(config, "train")
    conv_val = conv.build_dataloader(config, "val")
    compare_preprocessing(pair, src_ds, conv_train, conv_val, 50000, "-", "no split (50000 train)")
    record(pair, "a_preprocessing", "dataset_choice", "cifar10 (source supports it natively)", "cifar10 (converted default)",
           None, "info", "no ImageFolder needed; converted adds drop_last=True")

    # (b) --- L155-156 / L196-197
    seed_all()
    netG = Generator(1).to(DEVICE); netG.apply(weights_init)
    netD = Discriminator(1).to(DEVICE); netD.apply(weights_init)
    compare_architecture(pair, netG, client.model.netG, "generator")
    compare_architecture(pair, netD, client.model.netD, "discriminator")
    client.model.netG.load_state_dict(netG.state_dict())
    client.model.netD.load_state_dict(netD.state_dict())
    src_wrap = nn.ModuleDict({"netG": netG, "netD": netD})   # prefixes for gradient bookkeeping

    batch = first_batch(src_loader)
    real = batch[0]

    # (c)
    comp = dcgan_source_components(netD, netG, real, nz)
    seed_all()
    with loss_recorder() as calls:
        total_c = conv.train_step(client.model, batch, None, config)
    vals = [v for _, v in calls]
    d_real_c, d_fake_c, g_c = vals[0], vals[1], vals[2]
    record(pair, "c_loss_init", "g_loss", comp["g_loss"].item(), g_c, g_c - comp["g_loss"].item(),
           vdiff(comp["g_loss"].item(), g_c), "BCELoss(D(G(z)), 1) on sigmoid outputs")
    record(pair, "c_loss_init", "d_real_loss", comp["real_loss"].item(), d_real_c, d_real_c - comp["real_loss"].item(),
           vdiff(comp["real_loss"].item(), d_real_c), "")
    record(pair, "c_loss_init", "d_fake_loss", comp["fake_loss"].item(), d_fake_c, d_fake_c - comp["fake_loss"].item(),
           vdiff(comp["fake_loss"].item(), d_fake_c), "")
    record(pair, "c_loss_init", "d_loss_as_optimised", comp["d_loss"].item(), d_real_c + d_fake_c,
           (d_real_c + d_fake_c) - comp["d_loss"].item(), vdiff(comp["d_loss"].item(), d_real_c + d_fake_c),
           "source errD = errD_real + errD_fake (optimizerD only); converted folds it into the joint loss")
    src_sum = comp["d_loss"].item() + comp["g_loss"].item()
    record(pair, "c_loss_init", "returned_total_loss", src_sum, total_c.item(), total_c.item() - src_sum,
           vdiff(src_sum, total_c.item()), "converted returns errD + errG; source never forms this scalar")
    record(pair, "c_loss_init", "n_loss_calls_in_train_step", 3, len(vals), None, "info",
           "source: three criterion calls but two separate optimizer steps")

    # (d)
    gan_gradient_analysis(pair, comp, src_wrap, "netG.", "netD.", total_c, client.model, "netG.", "netD.",
                          "errD=errD_real+errD_fake")
    state_G, state_D = state_cpu(netG), state_cpu(netD)

    def source_step(opt_factory):
        G2, D2 = Generator(1).to(DEVICE), Discriminator(1).to(DEVICE)
        G2.load_state_dict(state_G); D2.load_state_dict(state_D); G2.train(); D2.train()
        optD, optG = opt_factory(D2, G2)
        bG, bD = snapshot(G2), snapshot(D2); seed_all()
        info = dcgan_source_step(D2, G2, optD, optG, nn.BCELoss(), batch, DEVICE, nz)
        return delta_vec(bG, snapshot(G2)), delta_vec(bD, snapshot(D2)), info

    dG_nat, dD_nat, info = source_step(lambda D, G: (optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999)),
                                                     optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))))
    dG_mat, dD_mat, _ = source_step(lambda D, G: (optim.AdamW(D.parameters(), lr=lr), optim.AdamW(G.parameters(), lr=lr)))
    record(pair, "d_one_step", "source_loop_losses", f"errD={info['errD']:.4f},errG={info['errG']:.4f}", None, None,
           "info", "errG is evaluated with the already-updated D (source ordering: D step, then G step)")
    client.set_weights(state_cpu(client.model))
    bG, bD = snapshot(client.model, "netG."), snapshot(client.model, "netD."); seed_all()
    runtime_train(client, [batch])
    dG_rt, dD_rt = delta_vec(bG, snapshot(client.model, "netG.")), delta_vec(bD, snapshot(client.model, "netD."))
    record_delta(pair, "G_param_delta_runtime_vs_source_native", dG_rt, dG_nat,
                 "source: optimizerG=Adam(2e-4,(0.5,0.999)); runtime: single AdamW(2e-4)+clip over G and D", 0.9)
    record_delta(pair, "D_param_delta_runtime_vs_source_native", dD_rt, dD_nat,
                 "source: optimizerD step on errD only; runtime: joint step on errD+errG", 0.9)
    record_delta(pair, "G_param_delta_runtime_vs_source_adamw", dG_rt, dG_mat, "two AdamW(2e-4) substituted in the source loop", 0.9)
    record_delta(pair, "D_param_delta_runtime_vs_source_adamw", dD_rt, dD_mat, "", 0.9)

    # (e)
    test_ds = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=src_tf)
    real_test = stack_first(test_ds, 512)[0].to(DEVICE)
    seed_all(7); z_eval = torch.randn(512, nz, 1, 1, device=DEVICE)
    seed_all()
    G0, D0 = Generator(1), Discriminator(1); G0.apply(weights_init); D0.apply(weights_init)
    init_G, init_D = state_cpu(G0), state_cpu(D0)

    def source_budget(opt_factory):
        G2, D2 = Generator(1).to(DEVICE), Discriminator(1).to(DEVICE)
        G2.load_state_dict(init_G); D2.load_state_dict(init_D); G2.train(); D2.train()
        optD, optG = opt_factory(D2, G2); crit = nn.BCELoss(); seed_all(); info = {}
        for b in CappedLoader(src_loader, steps):
            info = dcgan_source_step(D2, G2, optD, optG, crit, to_dev(b), DEVICE, nz)
        met = gan_metrics(G2, D2, real_test, z_eval, d_outputs_logits=False)
        met["final_errD"], met["final_errG"] = info.get("errD"), info.get("errG")
        return met

    m_nat, t1 = timed(source_budget, lambda D, G: (optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999)),
                                                   optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))))
    m_mat, t2 = timed(source_budget, lambda D, G: (optim.AdamW(D.parameters(), lr=lr), optim.AdamW(G.parameters(), lr=lr)))
    client.model.netG.load_state_dict(init_G); client.model.netD.load_state_dict(init_D); seed_all()
    rt_metrics, t3 = timed(runtime_train, client, conv_train, steps)
    m_rt = gan_metrics(client.model.netG, client.model.netD, real_test, z_eval, d_outputs_logits=False)
    record_gan_metrics(pair, "source_native", m_nat, f"two Adam(2e-4,(0.5,0.999)); {t1:.0f}s", steps, bs)
    record_gan_metrics(pair, "source_adamw", m_mat, f"two AdamW(2e-4); {t2:.0f}s", steps, bs)
    record_gan_metrics(pair, "converted_runtime", m_rt, f"single AdamW(2e-4)+clip on summed loss; mean runtime loss {rt_metrics['loss']}; {t3:.0f}s", steps, bs)
    record(pair, "e_budget", f"final_losses_{steps}steps_source_native", f"errD={m_nat['final_errD']:.4f},errG={m_nat['final_errG']:.4f}",
           None, None, "info", "")
    for k in ("d_acc_mean", "d_acc_mean_bn_batchstats", "d_acc_fake_bn_batchstats"):
        record(pair, "e_budget", f"{k}_{steps}steps_delta", m_mat[k], m_rt[k], m_rt[k] - m_mat[k],
               "differs" if abs(m_rt[k] - m_mat[k]) > 0.1 else "match",
               "discriminator real/fake accuracy on 512 CIFAR-10 test images + 512 samples: converted-runtime minus optimizer-matched source")


# ==========================================================================
# Shared pieces for the sklearn / xgboost pairs
# ==========================================================================
def tabular_preprocessing(pair: str, X, y, src_scaler, src_test_idx, conv_train, conv_val, conv_recompute,
                          scaler_note: str) -> None:
    """(a) for the in-memory sklearn datasets: the converted module scales the
    whole dataset before splitting, the source fits its scaler on the train
    split only; splits are random vs stratified."""
    conv_u = underlying(conv_train.dataset)
    Xc, yc = conv_u.tensors[0].numpy(), conv_u.tensors[1].numpy()
    used_real = bool(np.allclose(Xc, conv_recompute, atol=1e-4)) and bool(np.allclose(yc, y))
    record(pair, "a_preprocessing", "dataset_class", "ndarray (sklearn loader)", type(conv_u).__name__, None, "info",
           "converted wraps the sklearn arrays in a TensorDataset")
    record(pair, "a_preprocessing", "converted_used_real_data", None, used_real, None,
           "match" if used_real else "differs", "converted tensors equal the re-scaled sklearn arrays (not the randn fallback)")
    record(pair, "a_preprocessing", "dataset_len_underlying", len(X), len(conv_u), None,
           "match" if len(X) == len(conv_u) else "differs", "")
    src_all = src_scaler.transform(X).astype(np.float32)
    mad, mean_ad = float(np.abs(src_all - Xc).max()), float(np.abs(src_all - Xc).mean())
    record(pair, "a_preprocessing", "scaler_fit_population", f"train split ({len(X) - len(src_test_idx)})", f"all rows ({len(X)})",
           None, "differs", scaler_note)
    record(pair, "a_preprocessing", "max_abs_feature_diff_sample_aligned", None, None, mad,
           "match" if mad < 1e-5 else "differs", "source scaler applied to every row vs the converted tensor, same row index")
    record(pair, "a_preprocessing", "mean_abs_feature_diff_sample_aligned", None, None, mean_ad, "info", "")
    n_src_train = len(X) - len(src_test_idx)
    record(pair, "a_preprocessing", "train_split_len", n_src_train, len(conv_train.dataset), None,
           "match" if n_src_train == len(conv_train.dataset) else "differs", "source: stratified train_test_split(0.2, seed 42); converted: random_split")
    record(pair, "a_preprocessing", "val_split_len", len(src_test_idx), len(conv_val.dataset), None,
           "match" if len(src_test_idx) == len(conv_val.dataset) else "differs", "")
    conv_val_idx = set(int(i) for i in conv_val.dataset.indices)
    jac = len(conv_val_idx & set(src_test_idx)) / len(conv_val_idx | set(src_test_idx))
    record(pair, "a_preprocessing", "heldout_set_overlap_jaccard", None, None, jac, "differs" if jac < 0.99 else "match",
           "source test split vs converted val split (row indices)")
    record(pair, "a_preprocessing", "split_stratified", True, False, None, "differs", "")
    xs = torch.tensor(src_all[[i for i in range(len(X)) if i not in set(src_test_idx)][:256]])
    xc = stack_first(conv_train.dataset, 256)[0]
    ss, sc = tstats(xs), tstats(xc)
    for k in ("min", "max", "mean", "std"):
        record(pair, "a_preprocessing", f"value_{k}_first256", ss[k], sc[k], sc[k] - ss[k], "info",
               "first 256 rows of each pipeline's own train split (not sample-aligned)")
    record(pair, "a_preprocessing", "example_shape", tuple(xs.shape[1:]), tuple(xc.shape[1:]), None,
           "match" if xs.shape[1:] == xc.shape[1:] else "differs", "")
    record(pair, "a_preprocessing", "example_dtype", "float64 (numpy)", str(xc.dtype), None, "info", "")
    ls, lc = sorted(set(y.tolist())), sorted(set(int(v) for v in yc.tolist()))
    record(pair, "a_preprocessing", "label_set", ls, lc, None, "match" if ls == lc else "differs", "")


def make_tensor_loader(X, y, bs: int, y_dtype=torch.long, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.tensor(np.asarray(X, dtype=np.float32)), torch.tensor(np.asarray(y), dtype=y_dtype))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


# ==========================================================================
# P5  sklearn/mlp_digits.py -> mlp_digits_fl_structured.py
# ==========================================================================
def run_p5(steps: int, bs: int) -> None:
    pair = "P5"
    print(f"\n=== {pair}: {PAIRS[pair][0]} -> {PAIRS[pair][1]} ===")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
    from sklearn.preprocessing import StandardScaler
    lr = 1e-3                                            # --lr default
    config = make_config(lr, bs)
    client = make_client(pair, config)
    conv = client.mod

    # (a) --- mlp_digits.py L29-41
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    seed_all()
    conv_train = conv.build_dataloader(config, "train")
    conv_val = conv.build_dataloader(config, "val")
    tabular_preprocessing(pair, X, y, scaler, idx_test.tolist(), conv_train, conv_val,
                          StandardScaler().fit_transform(X).astype(np.float32),
                          "converted fits StandardScaler on all 1797 rows before splitting (val rows leak into the scaler)")

    # (b) --- MLPClassifier(hidden_layer_sizes=(128, 64)) L45-54; sklearn allocates weights inside fit()
    clf_ref = MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=lr, max_iter=1,
                            random_state=SEED, early_stopping=False)
    clf_ref.fit(X_train_s, y_train)
    conv_model = client.model
    linears = [m for m in conv_model.modules() if isinstance(m, nn.Linear)]
    src_shapes = [tuple(c.shape) for c in clf_ref.coefs_]
    conv_shapes = [tuple(m.weight.shape) for m in linears]
    n_src = sum(c.size for c in clf_ref.coefs_) + sum(b.size for b in clf_ref.intercepts_)
    record(pair, "b_architecture", "param_count", n_src, n_params(conv_model), n_params(conv_model) - n_src,
           "match" if n_src == n_params(conv_model) else "differs", "sklearn coefs_+intercepts_ vs torch parameters")
    transposed_ok = [(o, i) for (i, o) in src_shapes] == conv_shapes
    record(pair, "b_architecture", "layer_weight_shapes", src_shapes, conv_shapes, None,
           "match" if transposed_ok else "differs", "sklearn stores (in,out); torch stores (out,in): layer-wise transposes")
    record(pair, "b_architecture", "leaf_modules", "Linear>relu>Linear>relu>Linear>softmax/log_loss",
           ">".join(leaf_types(conv_model)) + " + cross_entropy", None, "match",
           "same topology; softmax+log-loss == logits+cross-entropy")
    record(pair, "b_architecture", "state_dict_keys_match", "coefs_[0..2], intercepts_[0..2]",
           ",".join(conv_model.state_dict().keys()), None, "n/a", "no state_dict on the sklearn side; correspondence established layer-wise")
    record(pair, "b_architecture", "l2_regularisation", "alpha=1e-4 (in objective)", "none (AdamW weight_decay=0.01 in runtime)", None,
           "differs", "sklearn adds 0.5*alpha*||W||^2/n to the loss; converted train_step has no penalty")
    # inject the converted model's initial parameters into the sklearn estimator (identical parameters)
    for i, m in enumerate(linears):
        clf_ref.coefs_[i][:] = m.weight.detach().cpu().numpy().T.astype(np.float64)
        clf_ref.intercepts_[i][:] = m.bias.detach().cpu().numpy().astype(np.float64)

    rng = np.random.RandomState(SEED)
    bi = rng.permutation(len(X_train_s))[:bs]
    Xb64, yb = X_train_s[bi], y_train[bi]
    batch = (torch.tensor(Xb64, dtype=torch.float32).to(DEVICE), torch.tensor(yb, dtype=torch.long).to(DEVICE))

    # (c)
    layer_units = [64, 128, 64, 10]
    activations = [Xb64] + [None] * (len(layer_units) - 1)
    deltas = [None] * (len(activations) - 1)
    coef_grads = [np.empty((a, b)) for a, b in zip(layer_units[:-1], layer_units[1:])]
    intercept_grads = [np.empty(b) for b in layer_units[1:]]
    Yb = clf_ref._label_binarizer.transform(yb).astype(np.float64)
    loss_reg, coef_grads, intercept_grads = clf_ref._backprop(Xb64, Yb, None, activations, deltas, coef_grads, intercept_grads)
    proba = clf_ref.predict_proba(Xb64)
    logloss = float(-np.mean(np.log(np.clip(proba[np.arange(len(yb)), yb], 1e-12, None))))
    loss_c = conv.train_step(conv_model, batch, None, config)
    record(pair, "c_loss_init", "log_loss", logloss, loss_c.item(), loss_c.item() - logloss, vdiff(logloss, loss_c.item()),
           "sklearn softmax log-loss vs torch cross_entropy, identical parameters, same 64-row batch")
    record(pair, "c_loss_init", "objective_with_l2", float(loss_reg), loss_c.item(), loss_c.item() - float(loss_reg),
           vdiff(float(loss_reg), loss_c.item(), 1e-3), "sklearn _backprop objective includes 0.5*alpha*||W||^2/n")

    # (d)
    g_s = torch.cat([torch.tensor(np.concatenate([coef_grads[i].T.ravel(), intercept_grads[i].ravel()]))
                     for i in range(3)]).double()
    conv_model.zero_grad(); loss_c.backward(); g_c = grads_of(conv_model)
    record(pair, "d_one_step", "grad_cosine", None, None, cosine(g_c, g_s),
           "match" if cosine(g_c, g_s) > 0.999 else "differs", "sklearn _backprop gradient (incl. L2 term) vs torch autograd of train_step")

    def sk_flat(clf):
        return torch.cat([torch.tensor(np.concatenate([clf.coefs_[i].T.ravel(), clf.intercepts_[i].ravel()]))
                          for i in range(3)]).double()

    clf_ref._optimizer = AdamOptimizer(clf_ref.coefs_ + clf_ref.intercepts_, lr, 0.9, 0.999, 1e-8)  # fresh Adam state
    before_s = sk_flat(clf_ref)
    clf_ref.partial_fit(Xb64, yb)                        # one sklearn Adam update on this batch (batch_size auto = 64)
    d_native = sk_flat(clf_ref) - before_s
    state0 = state_cpu(conv_model)

    def torch_step(opt_factory, clip: bool):
        m = conv.build_model(config).to(DEVICE); m.load_state_dict(state0); m.train()
        opt = opt_factory(m); before = snapshot(m)
        opt.zero_grad(); conv.train_step(m, batch, None, config).backward()
        if clip:
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        return delta_vec(before, snapshot(m))

    d_adam = torch_step(lambda m: optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8), clip=False)
    client.set_weights(state0); before = snapshot(client.model); seed_all()
    runtime_train(client, [batch])
    d_rt = delta_vec(before, snapshot(client.model))
    record_delta(pair, "param_delta_runtime_vs_source_native", d_rt, d_native,
                 "source: sklearn AdamOptimizer(lr=1e-3) via partial_fit; runtime: AdamW(lr=1e-3)+clip(1.0)", 0.99)
    record_delta(pair, "param_delta_runtime_vs_torch_adam", d_rt, d_adam,
                 "torch.optim.Adam(lr=1e-3) on the converted train_step without clipping (optimizer-type control)", 0.99)
    record_delta(pair, "param_delta_torch_adam_vs_source_native", d_adam, d_native,
                 "torch Adam vs sklearn Adam from identical parameters/batch", 0.99)

    # (e) --- source: the script's estimator (L45-58)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=lr, max_iter=200, random_state=SEED,
                        verbose=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
    _, t1 = timed(clf.fit, X_train_s, y_train)
    acc_src = float((clf.predict(X_test_s) == y_test).mean())
    n_iter = int(clf.n_iter_)
    record(pair, "e_budget", "test_acc_source_native", acc_src, None, None, "info",
           f"sklearn MLPClassifier as in the script (early stopping); n_iter_={n_iter} epochs, batch 200; {t1:.0f}s")
    seed_all(); init = state_cpu(conv.build_model(config))
    test_loader = make_tensor_loader(X_test_s, y_test, 1000, shuffle=False)
    client.set_weights(init); seed_all()
    src_train_loader = make_tensor_loader(X_train_s, y_train, bs)
    _, t2 = timed(runtime_train, client, src_train_loader, None, n_iter)
    acc_rt_same = classifier_accuracy(client.model, test_loader)
    record(pair, "e_budget", f"test_acc_converted_runtime_{n_iter}epochs_source_split", None, acc_rt_same, acc_rt_same - acc_src,
           "match" if abs(acc_rt_same - acc_src) < 0.02 else "differs",
           f"torch port trained by FLClient.local_train on the source's scaled train split for the same {n_iter} epochs (bs {bs}); {t2:.0f}s")
    client.set_weights(init); seed_all()
    _, t3 = timed(runtime_train, client, src_train_loader, steps)
    acc_rt_steps = classifier_accuracy(client.model, test_loader)
    record(pair, "e_budget", f"test_acc_converted_runtime_{steps}steps_source_split", None, acc_rt_steps, acc_rt_steps - acc_src,
           "match" if abs(acc_rt_steps - acc_src) < 0.02 else "differs", f"fixed {steps}-step budget; {t3:.0f}s")
    client.set_weights(init); seed_all()
    _, t4 = timed(runtime_train, client, conv_train, None, n_iter)
    acc_rt_own = classifier_accuracy(client.model, DataLoader(conv_val.dataset, batch_size=1000))
    record(pair, "e_budget", f"val_acc_converted_runtime_{n_iter}epochs_own_split", None, acc_rt_own, acc_rt_own - acc_src,
           "info", "converted pipeline end-to-end (its own random 80/20 split, whole-data scaler); {t4:.0f}s".format(t4=t4))


# ==========================================================================
# P6  sklearn/svm_iris.py -> svm_iris_fl_structured.py   (surrogate model family)
# ==========================================================================
def run_p6(steps: int, bs: int) -> None:
    pair = "P6"
    print(f"\n=== {pair}: {PAIRS[pair][0]} -> {PAIRS[pair][1]} ===")
    from sklearn import svm
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    lr = 1e-3
    config = make_config(lr, bs)
    client = make_client(pair, config)
    conv = client.mod

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    _, idx_test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    seed_all()
    conv_train = conv.build_dataloader(config, "train")
    conv_val = conv.build_dataloader(config, "val")
    recompute = ((X - X.mean(0)) / np.clip(X.std(0, ddof=1), 1e-8, None)).astype(np.float32)   # converted: torch std (ddof=1)
    tabular_preprocessing(pair, X, y, scaler, idx_test.tolist(), conv_train, conv_val, recompute,
                          "converted standardises all 150 rows with torch .std() (ddof=1); source uses sklearn StandardScaler (ddof=0) on the train split")

    clf = svm.SVC(C=1.0, kernel="rbf", random_state=SEED, probability=True)
    clf.fit(X_train_s, y_train)
    n_sv = int(clf.support_vectors_.shape[0])
    record(pair, "b_architecture", "model_family", "sklearn.svm.SVC(rbf)", type(client.model).__name__ + " (MLP 4-64-64-3 with BatchNorm)",
           None, "surrogate", "no parameter correspondence by design")
    record(pair, "b_architecture", "param_count", int(clf.dual_coef_.size + clf.intercept_.size + clf.support_vectors_.size),
           n_params(client.model), None, "surrogate", f"SVC: {n_sv} support vectors x4 features + dual_coef_ {clf.dual_coef_.shape} + intercepts; MLP: trainable tensors")
    record(pair, "b_architecture", "leaf_modules", "kernel QP (no layers)", ">".join(leaf_types(client.model)), None, "surrogate", "")
    for m in ("loss_at_init", "grad_cosine", "param_delta_cosine"):
        record(pair, "c_loss_init" if m == "loss_at_init" else "d_one_step", m, None, None, None, "n/a",
               "SVC has no per-batch differentiable loss or gradient step (dual QP solved by libsvm)")
    acc_src = float((clf.predict(X_test_s) == y_test).mean())
    record(pair, "e_budget", "test_acc_source_native", acc_src, None, None, "info", "SVC as in the script, source test split (30 rows)")
    seed_all(); init = state_cpu(conv.build_model(config))
    client.set_weights(init); seed_all()
    _, t = timed(runtime_train, client, make_tensor_loader(X_train_s, y_train, bs), steps)
    acc_rt = classifier_accuracy(client.model, make_tensor_loader(X_test_s, y_test, 1000, shuffle=False))
    record(pair, "e_budget", f"test_acc_converted_runtime_{steps}steps_source_split", None, acc_rt, acc_rt - acc_src,
           "match" if abs(acc_rt - acc_src) <= 1 / len(y_test) + 1e-9 else "differs",
           f"surrogate MLP trained by FLClient.local_train on the source train split, AdamW(lr={lr}); {t:.0f}s")


# ==========================================================================
# P7  xgboost/xgb_breast_cancer.py -> xgb_breast_cancer_fl_structured.py   (surrogate)
# ==========================================================================
def run_p7(steps: int, bs: int) -> None:
    pair = "P7"
    print(f"\n=== {pair}: {PAIRS[pair][0]} -> {PAIRS[pair][1]} ===")
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    lr = 1e-3
    config = make_config(lr, bs)
    client = make_client(pair, config)
    conv = client.mod

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    _, idx_test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    seed_all()
    conv_train = conv.build_dataloader(config, "train")
    conv_val = conv.build_dataloader(config, "val")
    tabular_preprocessing(pair, X, y.astype(np.float32), scaler, idx_test.tolist(), conv_train, conv_val,
                          StandardScaler().fit_transform(X).astype(np.float32),
                          "converted fits StandardScaler on all 569 rows before splitting")

    kw = dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, eval_metric="logloss",
              random_state=SEED, verbosity=0)
    try:
        clf = xgb.XGBClassifier(use_label_encoder=False, **kw)
        clf.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
    except TypeError:
        clf = xgb.XGBClassifier(**kw)
        clf.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
    n_trees = len(clf.get_booster().get_dump())
    n_leaves = sum(d.count("leaf=") for d in clf.get_booster().get_dump())
    record(pair, "b_architecture", "model_family", "xgboost.XGBClassifier (200 trees, depth 4)",
           type(client.model).__name__ + " (MLP 30-128-64-32-1, BatchNorm+Dropout)", None, "surrogate", "no parameter correspondence by design")
    record(pair, "b_architecture", "param_count", n_leaves, n_params(client.model), None, "surrogate",
           f"XGB: {n_trees} trees with {n_leaves} leaf values; MLP: trainable tensors")
    record(pair, "b_architecture", "leaf_modules", "boosted decision trees (no layers)", ">".join(leaf_types(client.model)), None, "surrogate", "")
    for m in ("loss_at_init", "grad_cosine", "param_delta_cosine"):
        record(pair, "c_loss_init" if m == "loss_at_init" else "d_one_step", m, None, None, None, "n/a",
               "gradient boosting has no parameter-space gradient step comparable to an nn.Module update")
    proba_src = clf.predict_proba(X_test_s)[:, 1]
    acc_src = float(((proba_src > 0.5).astype(int) == y_test).mean())
    auc_src = float(roc_auc_score(y_test, proba_src))
    record(pair, "e_budget", "test_acc_source_native", acc_src, None, None, "info", "XGBClassifier as in the script, source test split (114 rows)")
    record(pair, "e_budget", "test_auc_source_native", auc_src, None, None, "info", "")
    seed_all(); init = state_cpu(conv.build_model(config))
    client.set_weights(init); seed_all()
    _, t = timed(runtime_train, client, make_tensor_loader(X_train_s, y_train, bs, y_dtype=torch.float32), steps)
    client.model.eval()
    with torch.no_grad():
        logits = client.model(torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)).reshape(-1).cpu()
    prob_rt = torch.sigmoid(logits).numpy()
    acc_rt = float(((prob_rt > 0.5).astype(int) == y_test).mean())
    auc_rt = float(roc_auc_score(y_test, prob_rt))
    record(pair, "e_budget", f"test_acc_converted_runtime_{steps}steps_source_split", None, acc_rt, acc_rt - acc_src,
           "match" if abs(acc_rt - acc_src) < 0.02 else "differs",
           f"surrogate MLP trained by FLClient.local_train on the source train split, AdamW(lr={lr}); {t:.0f}s")
    record(pair, "e_budget", f"test_auc_converted_runtime_{steps}steps_source_split", None, auc_rt, auc_rt - auc_src,
           "match" if abs(auc_rt - auc_src) < 0.02 else "differs", "")


# ==========================================================================
# Output
# ==========================================================================
def write_csv(path: Path = CSV_PATH, rows: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in (ROWS if rows is None else rows):
            w.writerow(r)


def write_tables(path: Path = TABLES_PATH, rows: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ROWS if rows is None else rows
    lines = []
    for pid in PAIRS:
        rows_p = [r for r in rows if r["pair_id"] == pid]
        if not rows_p:
            continue
        lines.append(f"\n### {pid}: `{PAIRS[pid][0]}` -> `{PAIRS[pid][1]}`\n")
        lines.append("| stage | metric | source | converted | comparison | verdict | notes |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows_p:
            cells = [r["stage"], r["metric"], r["source_value"], r["converted_value"],
                     r["comparison_value"], r["verdict"], r["notes"]]
            lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def merge_csvs(paths: list[Path], out: Path, tables: Path) -> None:
    """Concatenate per-run CSVs (e.g. a CPU run and a CUDA run) in pair order."""
    rows: list[dict] = []
    for p in paths:
        with open(p, newline="") as f:
            new_rows = list(csv.DictReader(f))
        replaced = {r["pair_id"] for r in new_rows}
        rows = [r for r in rows if r["pair_id"] not in replaced] + new_rows   # later file wins per pair
    order = {pid: i for i, pid in enumerate(PAIRS)}
    rows.sort(key=lambda r: order.get(r["pair_id"], 99))      # stable: keeps within-pair order
    write_csv(out, rows)
    write_tables(tables, rows)
    print(f"Merged {len(rows)} rows from {len(paths)} files into {out} and {tables}")


RUNNERS = {"P1": run_p1, "P2": run_p2, "P3": run_p3, "P4": run_p4, "P5": run_p5, "P6": run_p6, "P7": run_p7}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=",".join(PAIRS), help="comma-separated subset of P1..P7")
    ap.add_argument("--steps", type=int, default=300, help="fixed step budget for stage (e)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", type=Path, default=CSV_PATH, help="CSV output path")
    ap.add_argument("--tables", type=Path, default=TABLES_PATH, help="markdown tables output path")
    ap.add_argument("--merge", default="", help="comma-separated CSVs to merge into --out (no tests are run)")
    args = ap.parse_args()
    if args.merge:
        merge_csvs([Path(p) for p in args.merge.split(",")], args.out, args.tables)
        return
    os.chdir(REPO)      # the Lightning demo MNIST factory probes './data' relative to cwd
    print(f"torch {torch.__version__} | device {DEVICE} | cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r} "
          f"| steps={args.steps} batch={args.batch_size} seed={SEED} | data_root={DATA_ROOT}")
    t0 = time.time()
    for pid in [p.strip() for p in args.pairs.split(",") if p.strip()]:
        t = time.time()
        try:
            RUNNERS[pid](args.steps, args.batch_size)
            record(pid, "run", "elapsed_sec", None, None, round(time.time() - t, 1), "info", "")
        except Exception as e:                       # keep going; the failure is part of the record
            import traceback
            traceback.print_exc()
            record(pid, "run", "error", None, None, None, "failed", f"{type(e).__name__}: {e}")
        write_csv(args.out); write_tables(args.tables)
    print(f"\nWrote {args.out} ({len(ROWS)} rows) and {args.tables}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
