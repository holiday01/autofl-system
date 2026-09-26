"""
Template (rule-based) converter — a competent, LLM-free baseline.

Rewrites a training script into the AutoFL client contract using AST pattern
rules only (no model calls of any kind).  Generated modules expose:

    build_model(config)                      -> torch.nn.Module
    build_dataloader(config, split="train")  -> DataLoader (tuple or dict batches)
    train_step(model, batch, optimizer, cfg) -> scalar loss WITH grad attached
                                                (no backward/step/zero_grad inside)

Rule families (all general — no benchmark script is special-cased by name):

  R1  model discovery
      R1.1 nn.Module / LightningModule subclasses; the training model is the
           callable applied to batch data inside the training loop, resolved
           through parameters -> call sites -> assignment sites
      R1.2 library model factories kept verbatim (torchvision.models.*,
           monai.networks.nets.*, timm.create_model, models.__dict__[...])
      R1.3 several models called in one loop (GANs) -> nn.Module bundle
      R1.4 `<model>.apply(<init_fn>)` post-construction calls are kept
  R2  data discovery
      R2.1 dataset constructor that feeds the training loader (DataLoader
           data-flow), otherwise every dataset constructor found
      R2.2 data-root substitution: string roots, argparse attributes and
           variables named like a data root -> config["data_path"]
      R2.3 each candidate is built AND probed (ds[0]) inside try/except;
           on failure -> synthetic data gated on allow_synthetic_data
      R2.4 framework aliases: keras.datasets.<x>.load_data,
           image_dataset_from_directory, lightning demo MNIST -> torchvision;
           sklearn.datasets.load_* -> TensorDataset (StandardScaler mirrored)
  R3  train-step extraction
      R3.1 innermost loop containing .backward()/manual_backward(), or
           LightningModule.training_step
      R3.2 AMP autocast `with` blocks inlined (runtime owns AMP)
      R3.3 call-site scrubbing: zero_grad/backward/step/manual_backward/
           toggle_optimizer/log/log_dict/clip_grad_*/scaler.* deleted
      R3.4 cut after the assignment of the last back-propagated loss;
           backward slice keeps only statements the loss depends on
      R3.5 free names hoisted from enclosing scopes (criterion, transforms,
           constants) with argparse-default constant folding
      R3.6 in-place tensor mutation (fill_/zero_) -> out-of-place (single
           backward safety); .cuda() -> .to(device); self -> model
      R3.7 multi-loss loops return the sum of all back-propagated losses
      R3.8 fallback: generic `logits = model(x); loss = <detected loss>(logits, y)`
  R4  synthetic-shape inference
      explicit shapes (keras Input, FakeData, MONAI in_channels/spatial_dims,
      Resize/spatial_size) > torchvision transforms (Normalize/Resize/Crop)
      > dataset catalogue (MNIST, CIFAR, sklearn toy sets, ...) > model
      introspection (first Conv in_channels / Linear in_features, img_shape
      defaults) > defaults; labels from final Linear/out_channels/Dense/
      num_classes; target kind from the loss (class/binary/regression/seg)
  R5  framework translation
      R5.1 Keras Sequential / linear functional chains -> torch.nn.Sequential
           with static shape tracking (NHWC -> NCHW); anything else -> generic
           MLP marked TEMPLATE_FALLBACK = "generic"
      R5.2 Lightning: LightningModule -> nn.Module, save_hyperparameters ->
           SimpleNamespace, hooks stripped, training_step body -> train_step
      R5.3 MONAI: network constructor kept; synthetic N-D volumes from
           spatial_dims/in_channels; dict batches preserved
      R5.4 sklearn / XGBoost: MLP surrogate on the tabular input
           (marked TEMPLATE_FALLBACK = "surrogate")
  R6  import tree-shaking, conditional imports -> try/except, __future__ hoist

Every generated file starts with a header listing the rules that fired and
the fallbacks that were used.
"""
from __future__ import annotations

import ast
import builtins
import copy
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}

# ---------------------------------------------------------------------------
# Catalogues / tables
# ---------------------------------------------------------------------------

# name -> (input_shape (C,H,W) or (F,), num_classes or None for regression)
_DATASET_CATALOGUE: dict[str, tuple[tuple, Optional[int]]] = {
    "MNIST": ((1, 28, 28), 10), "FashionMNIST": ((1, 28, 28), 10),
    "KMNIST": ((1, 28, 28), 10), "EMNIST": ((1, 28, 28), 47), "QMNIST": ((1, 28, 28), 10),
    "USPS": ((1, 16, 16), 10),
    "CIFAR10": ((3, 32, 32), 10), "CIFAR100": ((3, 32, 32), 100),
    "SVHN": ((3, 32, 32), 10), "STL10": ((3, 96, 96), 10),
    "ImageNet": ((3, 224, 224), 1000), "Caltech101": ((3, 224, 224), 101),
    "Flowers102": ((3, 224, 224), 102), "Food101": ((3, 224, 224), 101),
    "CelebA": ((3, 218, 178), 40), "OxfordIIITPet": ((3, 224, 224), 37),
    # Lightning demo data modules
    "MNISTDataModule": ((1, 28, 28), 10), "CIFAR10DataModule": ((3, 32, 32), 10),
    # MONAI apps
    "MedNISTDataset": ((1, 64, 64), 6),
    # keras.datasets.<name>
    "mnist": ((1, 28, 28), 10), "fashion_mnist": ((1, 28, 28), 10),
    "cifar10": ((3, 32, 32), 10), "cifar100": ((3, 32, 32), 100),
    # sklearn toy loaders (n_features,), n_classes
    "load_iris": ((4,), 3), "load_digits": ((64,), 10), "load_wine": ((13,), 3),
    "load_breast_cancer": ((30,), 2), "load_diabetes": ((10,), None),
    "fetch_california_housing": ((8,), None), "fetch_covtype": ((54,), 7),
    "fetch_olivetti_faces": ((4096,), 40), "make_classification": ((20,), 2),
    "make_moons": ((2,), 2), "make_circles": ((2,), 2), "make_blobs": ((2,), 3),
    "make_regression": ((100,), None),
}

_TORCHVISION_DATASETS = {
    "MNIST", "FashionMNIST", "KMNIST", "EMNIST", "QMNIST", "USPS", "CIFAR10", "CIFAR100",
    "SVHN", "STL10", "ImageNet", "Caltech101", "Caltech256", "Flowers102", "Food101",
    "CelebA", "OxfordIIITPet", "ImageFolder", "DatasetFolder", "FakeData", "LSUN",
    "VOCSegmentation", "VOCDetection", "CocoDetection", "Places365", "DTD", "EuroSAT",
    "PCAM", "Omniglot", "SEMEION", "Country211", "FGVCAircraft", "StanfordCars", "SUN397",
    "Cityscapes", "SBDataset", "Kinetics", "HMDB51", "UCF101",
}
_MONAI_DATASETS = {
    "ImageDataset", "Dataset", "CacheDataset", "PersistentDataset", "SmartCacheDataset",
    "ArrayDataset", "DecathlonDataset", "MedNISTDataset", "ZipDataset", "CSVDataset",
}
_TORCH_DATASETS = {"TensorDataset"}
_DOWNLOADABLE = {"MNIST", "FashionMNIST", "KMNIST", "EMNIST", "QMNIST", "USPS", "CIFAR10",
                 "CIFAR100", "SVHN", "STL10", "Caltech101", "Flowers102", "Food101",
                 "OxfordIIITPet", "DTD", "EuroSAT", "PCAM", "Omniglot", "Country211"}
_FOLDER_DATASETS = {"ImageFolder", "DatasetFolder"}
_SKLEARN_LOADER_RE = re.compile(r"^(load|fetch|make)_[a-z0-9_]+$")

# Data-root names (compared lower-cased) that are bound to config["data_path"].
_ROOT_NAMES = {
    "data", "dataroot", "data_root", "data_path", "data_dir", "datadir", "root",
    "root_dir", "rootdir", "dataset_dir", "dataset_path", "datasets_path", "datasets_dir",
    "input_dir", "data_folder", "datapath", "image_dir", "img_dir",
}
_PATH_FUNCS = {"os.path.join", "path.join", "glob", "glob.glob", "os.listdir", "Path",
               "os.sep.join", "open", "os.path.exists", "os.path.isdir", "pathlib.Path"}

# Loss name -> target kind
_LOSS_KIND: dict[str, str] = {
    "cross_entropy": "class", "nll_loss": "class", "CrossEntropyLoss": "class",
    "NLLLoss": "class", "categorical_crossentropy": "class",
    "sparse_categorical_crossentropy": "class", "CategoricalCrossentropy": "class",
    "SparseCategoricalCrossentropy": "class", "multi_margin_loss": "class",
    "label_smoothing_cross_entropy": "class",
    "BCELoss": "binary", "BCEWithLogitsLoss": "binary", "binary_cross_entropy": "binary",
    "binary_cross_entropy_with_logits": "binary", "binary_crossentropy": "binary",
    "BinaryCrossentropy": "binary", "soft_margin_loss": "binary",
    "MSELoss": "regression", "mse_loss": "regression", "L1Loss": "regression",
    "l1_loss": "regression", "SmoothL1Loss": "regression", "smooth_l1_loss": "regression",
    "HuberLoss": "regression", "huber_loss": "regression", "mse": "regression",
    "mae": "regression", "mean_squared_error": "regression",
    "mean_absolute_error": "regression", "MeanSquaredError": "regression",
    "MeanAbsoluteError": "regression", "Huber": "regression",
    "GaussianNLLLoss": "regression", "poisson_nll_loss": "regression",
    "DiceLoss": "segmentation", "DiceCELoss": "segmentation", "DiceFocalLoss": "segmentation",
    "GeneralizedDiceLoss": "segmentation", "GeneralizedDiceFocalLoss": "segmentation",
    "TverskyLoss": "segmentation", "FocalLoss": "segmentation", "MaskedDiceLoss": "segmentation",
}
_TORCH_LOSS_CLASSES = {"CrossEntropyLoss", "NLLLoss", "BCELoss", "BCEWithLogitsLoss",
                       "MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss", "GaussianNLLLoss"}
_TORCH_LOSS_FUNCS = {"cross_entropy", "nll_loss", "binary_cross_entropy",
                     "binary_cross_entropy_with_logits", "mse_loss", "l1_loss",
                     "smooth_l1_loss", "huber_loss", "multi_margin_loss",
                     "soft_margin_loss", "poisson_nll_loss"}
_KERAS_LOSS_TO_TORCH = {
    "categorical_crossentropy": "cross_entropy",
    "sparse_categorical_crossentropy": "cross_entropy",
    "CategoricalCrossentropy": "cross_entropy",
    "SparseCategoricalCrossentropy": "cross_entropy",
    "binary_crossentropy": "binary_cross_entropy_with_logits",
    "BinaryCrossentropy": "binary_cross_entropy_with_logits",
    "mse": "mse_loss", "mean_squared_error": "mse_loss", "MeanSquaredError": "mse_loss",
    "mae": "l1_loss", "mean_absolute_error": "l1_loss", "MeanAbsoluteError": "l1_loss",
    "Huber": "huber_loss", "huber": "huber_loss",
}

_OPTIMIZER_CLASSES = {"Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax",
                      "NAdam", "RAdam", "LBFGS", "SparseAdam"}

_MODEL_BASES = {"Module", "LightningModule", "LightningLite"}
_LIGHTNING_HOOK_PREFIXES = ("on_", "configure_", "setup", "teardown", "optimizer_",
                            "lr_scheduler_", "backward", "manual_backward")
_LIGHTNING_HOOK_SUFFIXES = ("_step", "_dataloader", "_epoch_end", "_step_end",
                            "_batch_transfer", "_end")
_LIGHTNING_STRIP_CALLS = {"save_hyperparameters", "log", "log_dict", "manual_backward",
                          "toggle_optimizer", "untoggle_optimizer", "clip_gradients",
                          "print", "freeze", "unfreeze"}
_LIGHTNING_PKGS = {"lightning", "pytorch_lightning", "lightning_fabric", "lightning_utilities"}
_KERAS_PKGS = {"keras", "tensorflow", "tf_keras", "tf"}

_SCRUB_EXPR_ATTRS = {
    "zero_grad", "backward", "manual_backward", "step", "toggle_optimizer",
    "untoggle_optimizer", "log", "log_dict", "unscale_", "add_scalar", "add_scalars",
    "add_image", "add_images", "add_histogram", "add_figure", "display",
    "set_description", "set_postfix", "flush", "save_image", "set_epoch",
}
_SCRUB_FUNC_NAMES = {"print", "clip_grad_norm_", "clip_grad_value_"}
_SCRUB_ASSIGN_FROM = {"optimizers", "lr_schedulers"}
_AMP_CONTEXTS = {"autocast", "torch.autocast", "torch.cuda.amp.autocast", "amp.autocast",
                 "torch.enable_grad", "enable_grad", "torch.autograd.set_detect_anomaly"}
_INPLACE_REWRITE = {"fill_": "torch.full_like({x}, {v})", "zero_": "torch.zeros_like({x})"}

_SYNTHETIC_ELEMENT_BUDGET = 1 << 24   # 16M floats (64 MB) for the synthetic set
_SYNTHETIC_MAX_SAMPLES = 256
_SYNTHETIC_MIN_SAMPLES = 8


class _Unresolvable(Exception):
    """Raised when an expression/name cannot be rewritten into the target module."""


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------

def _dotted(node: ast.AST) -> str:
    """'a.b.c' for Name/Attribute chains, '' otherwise."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _last_attr(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _kw(call: ast.Call, name: str) -> Optional[ast.AST]:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _walk_no_defs(node: ast.AST):
    """ast.walk that does not descend into nested function/class definitions."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.append(child)


def _bound_inside(node: ast.AST) -> set[str]:
    """Names bound by inner constructs (comprehensions, lambdas, nested defs,
    for/with/except targets inside compound statements)."""
    bound: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.comprehension):
            bound |= _store_names(n.target)
        elif isinstance(n, ast.Lambda):
            bound |= {a.arg for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs}
            if n.args.vararg:
                bound.add(n.args.vararg.arg)
            if n.args.kwarg:
                bound.add(n.args.kwarg.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(n.name)
            bound |= {a.arg for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs}
            if n.args.vararg:
                bound.add(n.args.vararg.arg)
            if n.args.kwarg:
                bound.add(n.args.kwarg.arg)
        elif isinstance(n, ast.ClassDef):
            bound.add(n.name)
        elif isinstance(n, (ast.For, ast.AsyncFor)) and n is not node:
            bound |= _store_names(n.target)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            bound |= _store_names(n.optional_vars)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Global):
            pass
    return bound


def _store_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _load_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def free_names(node: ast.AST, compound_optimistic: bool = True) -> set[str]:
    """Names loaded by `node` that are not bound inside it."""
    loads = _load_names(node)
    bound = _bound_inside(node)
    if compound_optimistic and isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try,
                                                 ast.FunctionDef, ast.ClassDef)):
        bound |= _store_names(node)
    if isinstance(node, (ast.For, ast.AsyncFor)):
        bound |= _store_names(node.target)
    return loads - bound - _BUILTINS


def _def_free_names(node: ast.AST) -> set[str]:
    """Free names of a class/function definition (optimistic)."""
    loads = _load_names(node)
    bound = _bound_inside(node) | _store_names(node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        bound |= {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
    return loads - bound - _BUILTINS


def _src(lines: list[str], node: ast.AST) -> str:
    """Verbatim source of a statement (including decorators)."""
    start = node.lineno
    for d in getattr(node, "decorator_list", []) or []:
        start = min(start, d.lineno)
    return "\n".join(lines[start - 1: node.end_lineno])


def _indent(code: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join((pad + l) if l.strip() else l for l in code.splitlines())


# ---------------------------------------------------------------------------
# Safe constant evaluation
# ---------------------------------------------------------------------------

_SAFE_FUNCS: dict[str, Any] = {
    "int": int, "float": float, "str": str, "len": len, "tuple": tuple, "list": list,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "bool": bool,
    "sorted": sorted, "range": lambda *a: list(range(*a)), "set": set, "dict": dict,
    "math.prod": math.prod, "math.sqrt": math.sqrt, "math.ceil": math.ceil,
    "math.floor": math.floor, "np.prod": lambda x: int(math.prod(x)),
    "numpy.prod": lambda x: int(math.prod(x)), "os.path.join": os.path.join,
    "path.join": os.path.join, "os.path.dirname": os.path.dirname,
    "os.path.basename": os.path.basename, "os.path.abspath": os.path.abspath,
    "os.path.expanduser": os.path.expanduser, "os.getcwd": os.getcwd,
}
_SAFE_ATTRS: dict[str, Any] = {"os.sep": os.sep, "math.pi": math.pi, "os.pathsep": os.pathsep}
_SAFE_STR_METHODS = {"join", "format", "lower", "upper", "strip", "replace", "split", "rstrip", "lstrip"}
_SAFE_SEQ_METHODS = {"index", "count"}


def const_eval(node: ast.AST, lookup) -> Any:
    """Evaluate a constant expression. `lookup(name)` returns a value or raises _Unresolvable."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(const_eval(e, lookup) for e in node.elts)
    if isinstance(node, ast.List):
        return [const_eval(e, lookup) for e in node.elts]
    if isinstance(node, ast.Set):
        return {const_eval(e, lookup) for e in node.elts}
    if isinstance(node, ast.Dict):
        return {const_eval(k, lookup): const_eval(v, lookup) for k, v in zip(node.keys, node.values) if k is not None}
    if isinstance(node, ast.Name):
        if node.id == "__name__":
            return "__main__"
        return lookup(node.id)
    if isinstance(node, ast.Attribute):
        d = _dotted(node)
        if d in _SAFE_ATTRS:
            return _SAFE_ATTRS[d]
        if isinstance(node.value, ast.Name) and hasattr(lookup, "attr_lookup"):
            try:
                return lookup.attr_lookup(node.value.id, node.attr)
            except _Unresolvable:
                pass
        try:
            base = const_eval(node.value, lookup)
        except _Unresolvable:
            raise
        if isinstance(base, (str, tuple, list)) and node.attr in (_SAFE_STR_METHODS | _SAFE_SEQ_METHODS):
            return getattr(base, node.attr)
        raise _Unresolvable(d or "attribute")
    if isinstance(node, ast.BinOp):
        a, b = const_eval(node.left, lookup), const_eval(node.right, lookup)
        op = type(node.op)
        table = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                 ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b,
                 ast.Pow: lambda: a ** b}
        if op in table:
            try:
                return table[op]()
            except Exception as e:
                raise _Unresolvable(str(e))
        raise _Unresolvable("binop")
    if isinstance(node, ast.UnaryOp):
        v = const_eval(node.operand, lookup)
        if isinstance(node.op, ast.Not):
            return not v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        raise _Unresolvable("unaryop")
    if isinstance(node, ast.BoolOp):
        vals = []
        for v in node.values:
            vals.append(const_eval(v, lookup))
        if isinstance(node.op, ast.And):
            r = True
            for v in vals:
                r = v
                if not v:
                    return v
            return r
        for v in vals:
            if v:
                return v
        return vals[-1]
    if isinstance(node, ast.Compare):
        left = const_eval(node.left, lookup)
        for op, comp in zip(node.ops, node.comparators):
            right = const_eval(comp, lookup)
            ok = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                  ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                  ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right,
                  ast.In: lambda: left in right, ast.NotIn: lambda: left not in right,
                  ast.Is: lambda: left is right, ast.IsNot: lambda: left is not right}
            if type(op) not in ok:
                raise _Unresolvable("compare")
            try:
                if not ok[type(op)]():
                    return False
            except Exception as e:
                raise _Unresolvable(str(e))
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return const_eval(node.body, lookup) if const_eval(node.test, lookup) else const_eval(node.orelse, lookup)
    if isinstance(node, ast.Subscript):
        base = const_eval(node.value, lookup)
        if isinstance(node.slice, ast.Slice):
            lo = const_eval(node.slice.lower, lookup) if node.slice.lower else None
            hi = const_eval(node.slice.upper, lookup) if node.slice.upper else None
            st = const_eval(node.slice.step, lookup) if node.slice.step else None
            try:
                return base[lo:hi:st]
            except Exception as e:
                raise _Unresolvable(str(e))
        idx = const_eval(node.slice, lookup)
        try:
            return base[idx]
        except Exception as e:
            raise _Unresolvable(str(e))
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                out.append(str(const_eval(v.value, lookup)))
        return "".join(out)
    if isinstance(node, ast.Call):
        d = _dotted(node.func)
        args = [const_eval(a, lookup) for a in node.args]
        kwargs = {k.arg: const_eval(k.value, lookup) for k in node.keywords if k.arg}
        if d in _SAFE_FUNCS:
            try:
                return _SAFE_FUNCS[d](*args, **kwargs)
            except Exception as e:
                raise _Unresolvable(str(e))
        if isinstance(node.func, ast.Attribute):
            base = const_eval(node.func.value, lookup)
            if isinstance(base, str) and node.func.attr in _SAFE_STR_METHODS:
                try:
                    return getattr(base, node.func.attr)(*args, **kwargs)
                except Exception as e:
                    raise _Unresolvable(str(e))
            if isinstance(base, (tuple, list)) and node.func.attr in _SAFE_SEQ_METHODS:
                return getattr(base, node.func.attr)(*args)
        raise _Unresolvable(f"call {d}")
    raise _Unresolvable(type(node).__name__)


def _literal(value: Any) -> str:
    """Python literal source for a constant value."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return f"float('{value}')"
    return repr(value)


# ---------------------------------------------------------------------------
# Source index and scopes
# ---------------------------------------------------------------------------

@dataclass
class Scope:
    func: Optional[ast.AST]                 # FunctionDef or None for module scope
    cls: Optional[ast.ClassDef] = None
    before_lineno: Optional[int] = None     # only consider assignments before this line

    @property
    def key(self):
        return (id(self.func), id(self.cls), self.before_lineno)


@dataclass
class _Entry:
    kind: str                       # "assign" | "param" | "blocked"
    stmt: Optional[ast.AST]
    conds: list                     # list of (test_expr, truth)
    scope: Scope
    param_default: Optional[ast.AST] = None
    func: Optional[ast.AST] = None  # for params


class SourceIndex:
    def __init__(self, source: str, path: str):
        self.source = source
        self.path = path
        self.lines = source.splitlines()
        self.tree = ast.parse(source)
        self.parents: dict[int, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[id(child)] = node
        self.functions = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}
        self.classes = {n.name: n for n in self.tree.body if isinstance(n, ast.ClassDef)}
        self.imports: list[tuple[ast.stmt, bool]] = []
        self.import_names: dict[str, str] = {}      # bound name -> top-level package
        self.import_full: dict[str, str] = {}       # bound name -> dotted origin
        self.future_imports: list[str] = []
        self._collect_imports()
        self.argparse_defaults: dict[str, Any] = {}
        self.argparse_required: set[str] = set()
        self.ns_names: set[str] = set()
        self._collect_argparse()
        self.mutated: set[str] = self._collect_mutated()
        self.module_scope = Scope(None)

    _MUTATORS = {"add", "append", "extend", "update", "insert", "pop", "remove", "clear",
                 "discard", "setdefault", "sort", "reverse", "fill_", "zero_", "copy_"}

    def _collect_mutated(self) -> set[str]:
        """Names whose value is mutated in place somewhere (never constant-folded)."""
        out: set[str] = set()
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and isinstance(n.func.value, ast.Name) and n.func.attr in self._MUTATORS:
                out.add(n.func.value.id)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Subscript):
                        base = t.value
                        while isinstance(base, (ast.Subscript, ast.Attribute)):
                            base = base.value
                        if isinstance(base, ast.Name):
                            out.add(base.id)
        return out

    # -- imports ------------------------------------------------------------
    def _collect_imports(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                self.future_imports.append(ast.unparse(node))
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                parent = self.parents.get(id(node))
                conditional = not isinstance(parent, ast.Module)
                self.imports.append((node, conditional))
                if isinstance(node, ast.Import):
                    for a in node.names:
                        bound = a.asname or a.name.split(".")[0]
                        self.import_names[bound] = a.name.split(".")[0]
                        self.import_full[bound] = a.name
                else:
                    mod = node.module or ""
                    for a in node.names:
                        bound = a.asname or a.name
                        self.import_names[bound] = mod.split(".")[0] if mod else ""
                        self.import_full[bound] = f"{mod}.{a.name}" if mod else a.name

    # -- argparse -----------------------------------------------------------
    def _collect_argparse(self):
        has_parser = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and _last_attr(node) == "add_argument":
                has_parser = True
                names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                if not names:
                    continue
                dest_node = _kw(node, "dest")
                if isinstance(dest_node, ast.Constant):
                    dest = dest_node.value
                else:
                    longs = [n for n in names if n.startswith("--")]
                    dest = (longs[0] if longs else names[0]).lstrip("-").replace("-", "_")
                action = _kw(node, "action")
                action_v = action.value if isinstance(action, ast.Constant) else None
                default = _kw(node, "default")
                required = _kw(node, "required")
                if action_v == "store_true":
                    self.argparse_defaults[dest] = False
                elif action_v == "store_false":
                    self.argparse_defaults[dest] = True
                elif default is not None:
                    try:
                        self.argparse_defaults[dest] = const_eval(default, self._no_lookup)
                    except _Unresolvable:
                        pass
                elif isinstance(required, ast.Constant) and required.value:
                    self.argparse_required.add(dest)
                elif not names[0].startswith("-"):
                    self.argparse_required.add(dest)     # positional without default
                else:
                    self.argparse_defaults[dest] = None
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and _last_attr(node.value) in {"parse_args", "parse_known_args"}:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.ns_names.add(t.id)
                    elif isinstance(t, ast.Tuple) and t.elts and isinstance(t.elts[0], ast.Name):
                        self.ns_names.add(t.elts[0].id)
        if has_parser:
            self.ns_names |= {"args", "opt", "opts", "flags", "FLAGS", "arguments", "params"}

    @staticmethod
    def _no_lookup(name):
        raise _Unresolvable(name)

    # -- navigation ---------------------------------------------------------
    def parent(self, node):
        return self.parents.get(id(node))

    def enclosing(self, node, types):
        n = self.parent(node)
        while n is not None and not isinstance(n, types):
            n = self.parent(n)
        return n

    def scope_of(self, node) -> Scope:
        func = self.enclosing(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        cls = self.enclosing(node, (ast.ClassDef,)) if func is not None else None
        return Scope(func, cls)

    def call_sites(self, fname: str) -> list[ast.Call]:
        out = []
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fname:
                out.append(n)
        return out

    def is_lightning_import(self, bound: str) -> bool:
        return self.import_names.get(bound, "") in _LIGHTNING_PKGS

    def is_keras_import(self, bound: str) -> bool:
        return self.import_names.get(bound, "") in _KERAS_PKGS


# ---------------------------------------------------------------------------
# Emission context + resolver (hoisting with constant folding)
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    kind: str                                  # model | data | train | module
    defined: set = field(default_factory=set)  # names available in this context
    stmts: list = field(default_factory=list)  # hoisted statement sources
    in_progress: set = field(default_factory=set)
    hoisted: list = field(default_factory=list)   # names hoisted (provenance)
    notes: list = field(default_factory=list)


class _Lookup:
    """Name/attribute lookup handed to const_eval (argparse defaults folded)."""

    def __init__(self, resolver, scope, depth=0):
        self.resolver, self.scope, self.depth = resolver, scope, depth

    def __call__(self, name):
        return self.resolver.constant_of(name, self.scope, self.depth)

    def attr_lookup(self, base, attr):
        idx = self.resolver.index
        if base in idx.ns_names and attr in idx.argparse_defaults:
            return idx.argparse_defaults[attr]
        raise _Unresolvable(f"{base}.{attr}")


class Resolver:
    """Rewrites expressions so that they only reference names available in
    the generated module, hoisting supporting assignments as needed."""

    def ceval(self, node, scope, depth: int = 0):
        return const_eval(node, _Lookup(self, scope, depth))

    def __init__(self, index: SourceIndex, module_names: set[str]):
        self.index = index
        self.module_names = set(module_names)   # names defined at module level of the output
        self.module_ctx = Ctx("module")
        self._const_cache: dict = {}
        self.model_var_names: set[str] = set()

    # -- availability -------------------------------------------------------
    def available(self, name: str, ctx: Ctx) -> bool:
        return (name in ctx.defined or name in self.module_names or name in _BUILTINS
                or name in self.module_ctx.defined)

    # -- assignment lookup --------------------------------------------------
    def _walk_assigns(self, body: list, name: str, scope: Scope, before: Optional[int] = None,
                      conds: Optional[list] = None) -> list[_Entry]:
        conds = conds or []
        out: list[_Entry] = []
        for stmt in body:
            if before is not None and stmt.lineno >= before:
                break
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        out.append(_Entry("assign", stmt, list(conds), scope))
                    elif isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == name for e in t.elts):
                        out.append(_Entry("assign", stmt, list(conds), scope))
            elif isinstance(stmt, ast.If):
                out += self._walk_assigns(stmt.body, name, scope, before, conds + [(stmt.test, True)])
                out += self._walk_assigns(stmt.orelse, name, scope, before, conds + [(stmt.test, False)])
            elif isinstance(stmt, (ast.For, ast.While)):
                if isinstance(stmt, ast.For) and name in _store_names(stmt.target):
                    out.append(_Entry("blocked", stmt, list(conds), scope))
                out += self._walk_assigns(stmt.body, name, scope, before, conds)
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    if item.optional_vars is not None and name in _store_names(item.optional_vars):
                        out.append(_Entry("blocked", stmt, list(conds), scope))
                out += self._walk_assigns(stmt.body, name, scope, before, conds)
            elif isinstance(stmt, ast.Try):
                out += self._walk_assigns(stmt.body, name, scope, before, conds)
                for h in stmt.handlers:
                    out += self._walk_assigns(h.body, name, scope, before, conds)
                out += self._walk_assigns(stmt.orelse, name, scope, before, conds)
                out += self._walk_assigns(stmt.finalbody, name, scope, before, conds)
        return out

    def entries(self, name: str, scope: Scope) -> list[_Entry]:
        out: list[_Entry] = []
        if scope.func is not None:
            out += self._walk_assigns(scope.func.body, name, scope, scope.before_lineno)
            params = scope.func.args
            all_params = params.posonlyargs + params.args + params.kwonlyargs
            for i, a in enumerate(params.posonlyargs + params.args):
                if a.arg == name:
                    n_default_start = len(params.posonlyargs + params.args) - len(params.defaults)
                    default = params.defaults[i - n_default_start] if i >= n_default_start else None
                    out.append(_Entry("param", None, [], scope, param_default=default, func=scope.func))
            for a, d in zip(params.kwonlyargs, params.kw_defaults):
                if a.arg == name:
                    out.append(_Entry("param", None, [], scope, param_default=d, func=scope.func))
        mod_scope = self.index.module_scope
        before = scope.before_lineno if scope.func is None else None
        out += self._walk_assigns(self.index.tree.body, name, mod_scope, before)
        return out

    # -- conditions ---------------------------------------------------------
    def cond_value(self, test: ast.AST, scope: Scope) -> Optional[bool]:
        try:
            return bool(self.ceval(test, scope))
        except _Unresolvable:
            return None

    def _filter_conds(self, entries: list[_Entry]) -> list[_Entry]:
        keep = []
        for e in entries:
            ok = True
            for test, truth in e.conds:
                v = self.cond_value(test, e.scope)
                if v is not None and v != truth:
                    ok = False
                    break
            if ok:
                keep.append(e)
        return keep

    # -- constants ----------------------------------------------------------
    def constant_of(self, name: str, scope: Scope, depth: int = 0) -> Any:
        """Constant value of `name` in `scope` (raises _Unresolvable)."""
        key = (name, scope.key)
        if key in self._const_cache:
            v = self._const_cache[key]
            if isinstance(v, _Unresolvable):
                raise v
            return v
        if depth > 12:
            raise _Unresolvable(name)
        self._const_cache[key] = _Unresolvable(name)   # cycle guard
        try:
            value = self._constant_of(name, scope, depth)
        except _Unresolvable as e:
            self._const_cache[key] = e
            raise
        self._const_cache[key] = value
        return value

    def _constant_of(self, name: str, scope: Scope, depth: int) -> Any:
        if name in self.index.mutated:
            raise _Unresolvable(f"{name} (mutated in place)")
        entries = self._filter_conds(self.entries(name, scope))
        if not entries:
            raise _Unresolvable(name)
        values = []
        for e in entries:
            if e.kind == "blocked":
                raise _Unresolvable(name)
            if e.kind == "param":
                for call in self.index.call_sites(e.func.name):
                    arg = self._call_arg(call, e.func, name)
                    if arg is None:
                        continue
                    cscope = self.index.scope_of(call)
                    try:
                        values.append(self.ceval(arg, cscope, depth + 1))
                        break
                    except _Unresolvable:
                        continue
                if not values and e.param_default is not None:
                    try:
                        values.append(self.ceval(e.param_default, self.index.module_scope, depth + 1))
                    except _Unresolvable:
                        pass
                continue
            stmt = e.stmt
            value_node = stmt.value
            if value_node is None:
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            t = targets[0]
            try:
                v = self.ceval(value_node, e.scope, depth + 1)
            except _Unresolvable:
                continue
            if isinstance(t, ast.Tuple):
                for i, el in enumerate(t.elts):
                    if isinstance(el, ast.Name) and el.id == name:
                        try:
                            v = v[i]
                        except Exception:
                            raise _Unresolvable(name)
            values.append(v)
        if not values:
            raise _Unresolvable(name)
        # majority vote among hashable constants, else the last assignment
        try:
            c = Counter(values)
            best, n = c.most_common(1)[0]
            if n > 1:
                return best
        except TypeError:
            pass
        return values[-1]

    def _call_arg(self, call: ast.Call, func: ast.AST, pname: str) -> Optional[ast.AST]:
        params = func.args.posonlyargs + func.args.args
        names = [a.arg for a in params]
        if func.args.args and self.index.enclosing(func, (ast.ClassDef,)) is not None and names and names[0] == "self":
            names = names[1:]
        for k in call.keywords:
            if k.arg == pname:
                return k.value
        if pname in names:
            i = names.index(pname)
            if i < len(call.args) and not isinstance(call.args[i], ast.Starred):
                return call.args[i]
        return None

    # -- materialisation ----------------------------------------------------
    def can_resolve(self, name: str, scope: Scope, ctx: Ctx) -> bool:
        scratch = Ctx(ctx.kind, set(ctx.defined), [], set(ctx.in_progress), [], [])
        return self.materialize(name, scope, scratch)

    def materialize(self, name: str, scope: Scope, ctx: Ctx, depth: int = 0) -> bool:
        if self.available(name, ctx):
            return True
        guard = (name, scope.key)
        if guard in ctx.in_progress or depth > 10:
            return False
        if ctx.kind == "data" and name.lower() in _ROOT_NAMES:
            ctx.defined.add(name)
            if name != "data_path":
                ctx.stmts.append(f"{name} = data_path  # R2.2 data-root substitution")
            ctx.notes.append(f"root:{name}")
            return True
        ctx.in_progress.add(guard)
        try:
            entries = self._filter_conds(self.entries(name, scope))
            if not entries or any(e.kind == "blocked" for e in entries):
                return False
            # chain: several assignments in the same scope where a later one
            # reads the name itself (e.g. images = [...]; images = [f(i) for i in images])
            same = [e for e in entries if e.kind == "assign" and e.scope.key == entries[0].scope.key]
            if len(same) > 1 and any(name in free_names(e.stmt) for e in same[1:]):
                snapshot = (list(ctx.stmts), set(ctx.defined))
                ok = True
                for e in same:
                    if not self._emit_assign(e, name, ctx, depth, allow_self=True):
                        ok = False
                        break
                if ok:
                    return True
                ctx.stmts, ctx.defined = snapshot
            # constant majority (in the data context the rewrite path runs first so
            # that data-root names/attributes are substituted rather than folded)
            if ctx.kind != "data":
                try:
                    v = self.constant_of(name, scope)
                    if self._emit_literal(name, v, ctx):
                        return True
                except _Unresolvable:
                    pass
            for e in entries:
                snapshot = (list(ctx.stmts), set(ctx.defined))
                if e.kind == "assign":
                    if self._emit_assign(e, name, ctx, depth):
                        return True
                elif e.kind == "param":
                    if self._emit_param(e, name, ctx, depth):
                        return True
                ctx.stmts, ctx.defined = snapshot
            if ctx.kind == "data":
                try:
                    v = self.constant_of(name, scope)
                    if self._emit_literal(name, v, ctx):
                        return True
                except _Unresolvable:
                    pass
            return False
        finally:
            ctx.in_progress.discard(guard)

    def _emit_literal(self, name: str, v: Any, ctx: Ctx) -> bool:
        if isinstance(v, (int, float, str, bool, tuple, list, type(None))) or v is None:
            try:
                lit = _literal(v)
                ast.literal_eval(lit)
            except Exception:
                return False
            ctx.stmts.append(f"{name} = {lit}")
            ctx.defined.add(name)
            ctx.hoisted.append(name)
            return True
        return False

    def _emit_assign(self, e: _Entry, name: str, ctx: Ctx, depth: int, allow_self: bool = False) -> bool:
        stmt = e.stmt
        if stmt.value is None:
            return False
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        target = targets[0]
        if allow_self:
            ctx.defined.add(name)     # later statements of the chain may read it
        try:
            new_value = self.rewrite(stmt.value, e.scope, ctx, depth + 1)
        except _Unresolvable:
            if allow_self:
                ctx.defined.discard(name)
            return False
        if ctx.kind == "module":
            # module-level hoists must be import-safe
            fn = free_names(new_value)
            if not fn <= (self.module_names | _BUILTINS | self.module_ctx.defined):
                return False
        ctx.stmts.append(f"{ast.unparse(target)} = {ast.unparse(new_value)}")
        ctx.defined |= _store_names(target)
        ctx.hoisted.append(name)
        return True

    def _emit_param(self, e: _Entry, name: str, ctx: Ctx, depth: int) -> bool:
        for call in self.index.call_sites(e.func.name):
            arg = self._call_arg(call, e.func, name)
            if arg is None:
                continue
            cscope = self.index.scope_of(call)
            snapshot = (list(ctx.stmts), set(ctx.defined))
            try:
                new_value = self.rewrite(arg, cscope, ctx, depth + 1)
            except _Unresolvable:
                ctx.stmts, ctx.defined = snapshot
                continue
            if not (isinstance(new_value, ast.Name) and new_value.id == name):
                ctx.stmts.append(f"{name} = {ast.unparse(new_value)}")
            ctx.defined.add(name)
            ctx.hoisted.append(name)
            return True
        if e.param_default is not None:
            try:
                new_value = self.rewrite(e.param_default, self.index.module_scope, ctx, depth + 1)
            except _Unresolvable:
                return False
            ctx.stmts.append(f"{name} = {ast.unparse(new_value)}")
            ctx.defined.add(name)
            ctx.hoisted.append(name)
            return True
        return False

    # -- expression rewriting -----------------------------------------------
    def rewrite(self, expr: ast.AST, scope: Scope, ctx: Ctx, depth: int = 0) -> ast.AST:
        """Return a rewritten deep copy of `expr` (raises _Unresolvable)."""
        node = copy.deepcopy(expr)
        return _Rewriter(self, scope, ctx, depth).visit(node)


class _Rewriter(ast.NodeTransformer):
    def __init__(self, resolver: Resolver, scope: Scope, ctx: Ctx, depth: int):
        self.r = resolver
        self.index = resolver.index
        self.scope = scope
        self.ctx = ctx
        self.depth = depth
        self.local_bound: set[str] = set()

    def visit_Lambda(self, node):
        self.local_bound |= {a.arg for a in node.args.args}
        return self.generic_visit(node)

    def visit_ListComp(self, node):
        return self._comp(node)

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp
    visit_DictComp = visit_ListComp

    def _comp(self, node):
        for g in node.generators:
            self.local_bound |= _store_names(g.target)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.index.ns_names \
                and node.value.id not in self.ctx.defined and node.value.id not in self.local_bound:
            attr = node.attr
            if self.ctx.kind == "data" and attr.lower() in _ROOT_NAMES:
                self.ctx.notes.append(f"root:{node.value.id}.{attr}")
                return ast.copy_location(ast.Name(id="data_path", ctx=ast.Load()), node)
            if attr in self.index.argparse_defaults:
                self.ctx.notes.append(f"argparse:{attr}")
                return ast.copy_location(ast.Constant(value=self.index.argparse_defaults[attr]), node)
            raise _Unresolvable(f"{node.value.id}.{attr}")
        return self.generic_visit(node)

    def visit_Call(self, node):
        # path-argument rule (data ctx): unresolvable names passed to path functions
        d = _dotted(node.func)
        if self.ctx.kind == "data" and d in _PATH_FUNCS:
            new_args = []
            for a in node.args:
                if isinstance(a, ast.Name) and not self.r.available(a.id, self.ctx) \
                        and a.id not in self.local_bound \
                        and (a.id.lower() in _ROOT_NAMES or not self.r.can_resolve(a.id, self.scope, self.ctx)):
                    self.ctx.notes.append(f"path-arg:{a.id}")
                    new_args.append(ast.copy_location(ast.Name(id="data_path", ctx=ast.Load()), a))
                else:
                    new_args.append(a)
            node.args = new_args
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "cuda":
                if self.ctx.kind == "model":
                    return node.func.value
                return ast.copy_location(ast.Call(
                    func=ast.Attribute(value=node.func.value, attr="to", ctx=ast.Load()),
                    args=[ast.Name(id="device", ctx=ast.Load())], keywords=[]), node)
            if attr == "to" and self.ctx.kind == "model":
                return node.func.value
            if attr in {"DataParallel", "DistributedDataParallel"} and self.ctx.kind == "model" and node.args:
                return node.args[0]
        elif isinstance(node.func, ast.Name) and node.func.id in {"DataParallel", "DistributedDataParallel"} \
                and self.ctx.kind == "model" and node.args:
            return node.args[0]
        return node

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return node
        name = node.id
        if name in self.local_bound or self.r.available(name, self.ctx):
            return node
        if name == "device" and self.ctx.kind in {"train", "data"}:
            return node
        if self.r.materialize(name, self.scope, self.ctx, self.depth + 1):
            return node
        if self.ctx.kind == "data" and name.lower() in _ROOT_NAMES:
            return ast.copy_location(ast.Name(id="data_path", ctx=ast.Load()), node)
        raise _Unresolvable(name)


# ---------------------------------------------------------------------------
# Component discovery
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    var: str
    ctor: Optional[ast.AST]
    scope: Scope
    cls_name: str
    is_local_class: bool
    is_lightning: bool = False
    apply_calls: list = field(default_factory=list)
    cls_node: Optional[ast.ClassDef] = None


@dataclass
class LoopInfo:
    kind: str                                   # loop | lightning | none
    body: list = field(default_factory=list)
    scope: Optional[Scope] = None
    batch_target: Optional[ast.AST] = None
    index_names: list = field(default_factory=list)
    iter_expr: Optional[ast.AST] = None
    cls: Optional[ast.ClassDef] = None
    batch_param: str = "batch"
    node: Optional[ast.AST] = None


@dataclass
class DataCandidate:
    call: ast.Call
    scope: Scope
    kind: str          # torchvision | monai | torch | local | keras | sklearn
    name: str
    linked: bool = False
    rank: int = 5
    collate: Optional[tuple] = None   # (collate_fn expr, scope) of the feeding DataLoader


@dataclass
class LossInfo:
    name: str = ""
    kind: str = "class"
    expr_src: str = ""       # torch expression producing a callable loss (for the generic step)
    origin: str = "default"


def _is_model_class(cls: ast.ClassDef) -> tuple[bool, bool]:
    """(is_model, is_lightning)"""
    for b in cls.bases:
        last = _last_attr(b)
        if last in _MODEL_BASES:
            return True, "Lightning" in last
    return False, False


def _strip_wrappers(expr: ast.AST) -> ast.AST:
    """Strip .to(...)/.cuda()/.eval()/.train()/DataParallel(...) around a constructor call."""
    while True:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and \
                expr.func.attr in {"to", "cuda", "cpu", "eval", "train", "float", "double", "half"}:
            expr = expr.func.value
            continue
        if isinstance(expr, ast.Call) and _last_attr(expr) in {"DataParallel", "DistributedDataParallel"} and expr.args:
            expr = expr.args[0]
            continue
        return expr


def _is_model_ctor(call: ast.AST, local_classes: set[str]) -> Optional[str]:
    if not isinstance(call, ast.Call):
        return None
    f = call.func
    if isinstance(f, ast.Name) and f.id in local_classes:
        return f.id
    d = _dotted(f)
    if d:
        parts = d.split(".")
        if (len(parts) >= 2 and (parts[-2] in {"models", "nets", "networks"} or parts[0] in {"models", "nets"})) \
                or d.endswith("create_model") or d.endswith("Sequential") or d.startswith("monai.networks"):
            return d
        if len(parts) == 1 and parts[0] in local_classes:
            return d
    if isinstance(f, ast.Subscript) and _dotted(f.value).endswith("__dict__"):
        return _dotted(f.value)
    return None


def _backward_target(stmt: ast.AST) -> Optional[ast.AST]:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if isinstance(call.func, ast.Attribute):
            if call.func.attr == "backward":
                v = call.func.value
                if isinstance(v, ast.Call) and _last_attr(v) == "scale" and v.args:
                    return v.args[0]
                return v
            if call.func.attr == "manual_backward" and call.args:
                return call.args[0]
    return None


def _own_statements(body: list) -> list:
    """Statements of a loop body, recursing through If/With/Try but not nested loops."""
    out = []
    for s in body:
        out.append(s)
        if isinstance(s, ast.If):
            out += _own_statements(s.body) + _own_statements(s.orelse)
        elif isinstance(s, ast.With):
            out += _own_statements(s.body)
        elif isinstance(s, ast.Try):
            out += _own_statements(s.body)
            for h in s.handlers:
                out += _own_statements(h.body)
            out += _own_statements(s.orelse) + _own_statements(s.finalbody)
    return out


def find_training_loop(index: SourceIndex) -> LoopInfo:
    cands = []
    for node in ast.walk(index.tree):
        if isinstance(node, (ast.For, ast.While)):
            if any(_backward_target(s) is not None for s in _own_statements(node.body)):
                cands.append(node)
    if cands:
        def score(n):
            f = index.enclosing(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            fname = f.name.lower() if f else ""
            return (0 if "train" in fname or f is None else 1, n.lineno)
        loop = sorted(cands, key=score)[0]
        func = index.enclosing(loop, (ast.FunctionDef, ast.AsyncFunctionDef))
        cls = index.enclosing(loop, (ast.ClassDef,)) if func is not None else None
        scope = Scope(func, cls, before_lineno=loop.lineno)
        info = LoopInfo("loop", list(loop.body), scope, node=loop)
        if isinstance(loop, ast.For):
            info.iter_expr = loop.iter
            target = loop.target
            it = loop.iter
            if isinstance(it, ast.Call) and _last_attr(it) in {"enumerate"} and isinstance(target, ast.Tuple) and len(target.elts) == 2:
                info.index_names = sorted(_store_names(target.elts[0]))
                target = target.elts[1]
                info.iter_expr = it.args[0] if it.args else None
            elif isinstance(it, ast.Call) and _last_attr(it) in {"tqdm", "track", "progress_bar"} and it.args:
                info.iter_expr = it.args[0]
            info.batch_target = target
        return info
    # Lightning training_step
    for cls in index.classes.values():
        is_model, is_l = _is_model_class(cls)
        if not is_l:
            continue
        for m in cls.body:
            if isinstance(m, ast.FunctionDef) and m.name == "training_step":
                params = [a.arg for a in m.args.args if a.arg != "self"]
                info = LoopInfo("lightning", list(m.body), Scope(m, cls), cls=cls, node=m,
                                batch_param=params[0] if params else "batch")
                info.index_names = params[1:2]
                return info
    return LoopInfo("none")


def find_models(index: SourceIndex, resolver: Resolver, loop: LoopInfo) -> list[ModelInfo]:
    local = {}
    for name, cls in index.classes.items():
        is_model, is_l = _is_model_class(cls)
        if is_model:
            local[name] = (cls, is_l)
    local_names = set(local)

    # every instantiation site
    sites: dict[int, ModelInfo] = {}
    apply_calls: dict[str, list] = {}
    for node in ast.walk(index.tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            core = _strip_wrappers(node.value)
            cname = _is_model_ctor(core, local_names)
            if cname:
                var = node.targets[0].id
                sites[id(node)] = ModelInfo(var, core, index.scope_of(node), cname, cname in local_names,
                                            local.get(cname, (None, False))[1], cls_node=local.get(cname, (None,))[0])
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and \
                isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "apply" and \
                isinstance(node.value.func.value, ast.Name):
            apply_calls.setdefault(node.value.func.value.id, []).append(node.value)
    for m in sites.values():
        m.apply_calls = apply_calls.get(m.var, [])

    def resolve_called(name: str, scope: Scope, depth: int = 0) -> Optional[ModelInfo]:
        if depth > 4:
            return None
        for e in resolver._filter_conds(resolver.entries(name, scope)):
            if e.kind == "assign" and id(e.stmt) in sites:
                return sites[id(e.stmt)]
            if e.kind == "param":
                for call in index.call_sites(e.func.name):
                    arg = resolver._call_arg(call, e.func, name)
                    if isinstance(arg, ast.Name):
                        m = resolve_called(arg.id, index.scope_of(call), depth + 1)
                        if m:
                            return m
        return None

    found: list[ModelInfo] = []
    if loop.kind == "lightning":
        cls = loop.cls
        site = next((m for m in sites.values() if m.cls_name == cls.name), None)
        if site is None:
            site = ModelInfo("self", ast.parse(f"{cls.name}()").body[0].value, index.module_scope, cls.name, True, True, cls_node=cls)
        site.var = "self"
        site.is_lightning = True
        found.append(site)
        return found
    if loop.kind == "loop":
        seen = set()
        calls = [n for n in _walk_no_defs(ast.Module(body=loop.body, type_ignores=[]))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        for n in sorted(calls, key=lambda c: (c.lineno, c.col_offset)):
            if n.func.id not in seen:
                seen.add(n.func.id)
                m = resolve_called(n.func.id, loop.scope)
                if m and m not in found:
                    found.append(m)
    if found:
        return found
    # fallback: first instantiation of a local class, else any site, else last local class
    for m in sites.values():
        if m.is_local_class:
            return [m]
    if sites:
        return [next(iter(sites.values()))]
    if local:
        name, (cls, is_l) = list(local.items())[-1]
        return [ModelInfo("model", ast.parse(f"{name}()").body[0].value, index.module_scope, name, True, is_l, cls_node=cls)]
    return []


def _dataset_kind(index: SourceIndex, call: ast.Call, local_ds: set[str]) -> Optional[tuple[str, str]]:
    last = _last_attr(call)
    d = _dotted(call.func)
    root = d.split(".")[0] if d else ""
    origin = index.import_full.get(root, "")
    if isinstance(call.func, ast.Name) and last in local_ds:
        return "local", last
    if last in _TORCHVISION_DATASETS and ("torchvision" in origin or "torchvision" in d or root in {"dset", "datasets"} or root == last):
        return "torchvision", last
    if last in _MONAI_DATASETS and ("monai" in origin or "monai" in d):
        return "monai", last
    if last in _TORCH_DATASETS:
        return "torch", last
    if last == "load_data" and (index.is_keras_import(root) or "keras" in d or "tf" in root):
        parts = d.split(".")
        return "keras", parts[-2] if len(parts) >= 2 else "load_data"
    if last in {"image_dataset_from_directory", "text_dataset_from_directory"}:
        return "keras", last
    if _SKLEARN_LOADER_RE.match(last) and ("sklearn" in origin or "sklearn" in d):
        return "sklearn", last
    if last == "MNIST" and "lightning" in origin:
        return "torchvision", "MNIST"
    if last in _DATASET_CATALOGUE and last.endswith("DataModule"):
        return "lightning", last
    return None


def find_datasets(index: SourceIndex, resolver: Resolver, loop: LoopInfo) -> list[DataCandidate]:
    local_ds = set()
    for name, cls in index.classes.items():
        if any(_last_attr(b) in {"Dataset", "IterableDataset", "VisionDataset", "ImageFolder", "TensorDataset"} for b in cls.bases):
            local_ds.add(name)
    cands: list[DataCandidate] = []
    by_id: dict[int, DataCandidate] = {}
    for node in ast.walk(index.tree):
        if not isinstance(node, ast.Call):
            continue
        k = _dataset_kind(index, node, local_ds)
        if not k:
            continue
        encl_cls = index.enclosing(node, (ast.ClassDef,))
        if encl_cls is not None and encl_cls.name in local_ds:
            continue     # constructor call inside the wrapper dataset itself
        c = DataCandidate(node, index.scope_of(node), k[0], k[1])
        cands.append(c)
        by_id[id(node)] = c

    # data-flow link: training loop iterable -> DataLoader -> dataset
    def follow(expr: ast.AST, scope: Scope, depth: int = 0, collate=None):
        if depth > 6 or expr is None:
            return
        if isinstance(expr, ast.Call):
            if id(expr) in by_id:
                by_id[id(expr)].linked = True
                if collate is not None:
                    by_id[id(expr)].collate = collate
                return
            last = _last_attr(expr)
            if last in {"DataLoader", "random_split", "Subset", "ConcatDataset", "tqdm", "enumerate", "iter", "list", "cycle"}:
                arg = _kw(expr, "dataset")
                if arg is None and expr.args:
                    arg = expr.args[0]
                if last == "DataLoader" and _kw(expr, "collate_fn") is not None:
                    collate = (_kw(expr, "collate_fn"), scope)     # R2.5 batch-structure kwarg
                follow(arg, scope, depth + 1, collate)
            elif isinstance(expr.func, ast.Attribute) and last in {"train_dataloader", "val_dataloader", "take", "prefetch", "map", "batch", "shuffle", "cache", "repeat"}:
                follow(expr.func.value, scope, depth + 1, collate)
            return
        if isinstance(expr, ast.Attribute):
            follow(expr.value, scope, depth + 1, collate)
            return
        if isinstance(expr, ast.Subscript):
            follow(expr.value, scope, depth + 1, collate)
            return
        if isinstance(expr, ast.Name):
            for e in resolver._filter_conds(resolver.entries(expr.id, scope)):
                if e.kind == "assign":
                    follow(e.stmt.value, e.scope, depth + 1, collate)
                elif e.kind == "param":
                    for call in index.call_sites(e.func.name):
                        arg = resolver._call_arg(call, e.func, expr.id)
                        if arg is not None:
                            follow(arg, index.scope_of(call), depth + 1, collate)

    if loop.kind == "loop" and loop.iter_expr is not None:
        follow(loop.iter_expr, loop.scope)

    def rank(c: DataCandidate) -> int:
        if c.kind == "torchvision" and c.name in _DOWNLOADABLE:
            return 0
        if c.kind in {"local", "monai", "sklearn", "keras", "torch"}:
            return 1
        if c.kind == "torchvision" and c.name in _FOLDER_DATASETS:
            return 3
        if c.kind == "torchvision" and c.name == "FakeData":
            return 4
        return 2
    for c in cands:
        c.rank = rank(c)
    linked = [c for c in cands if c.linked]
    pool = linked if linked else cands
    pool.sort(key=lambda c: (c.rank, c.call.lineno))
    # de-duplicate identical constructor text
    seen, out = set(), []
    for c in pool:
        key = ast.unparse(c.call)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _loss_from_call(index: SourceIndex, resolver: Resolver, call: ast.Call, scope: Scope, depth: int = 0) -> Optional[LossInfo]:
    """Identify the loss behind a call expression."""
    if depth > 4:
        return None
    last = _last_attr(call)
    d = _dotted(call.func)
    if last in _LOSS_KIND:
        kind = _LOSS_KIND[last]
        if last in _TORCH_LOSS_FUNCS:
            return LossInfo(last, kind, f"F.{last}", "loop")
        if last in _TORCH_LOSS_CLASSES:
            return LossInfo(last, kind, f"nn.{last}()", "loop")
        return LossInfo(last, kind, "", "loop")
    # variable holding a criterion
    if isinstance(call.func, ast.Name):
        name = call.func.id
        for e in resolver._filter_conds(resolver.entries(name, scope)):
            if e.kind == "assign":
                core = _strip_wrappers(e.stmt.value)
                if isinstance(core, ast.Call):
                    li = _loss_from_call(index, resolver, core, e.scope, depth + 1)
                    if li:
                        if li.name in _TORCH_LOSS_CLASSES:
                            li.expr_src = f"nn.{li.name}()"
                        elif not li.expr_src:
                            try:
                                ctx = Ctx("train")
                                li.expr_src = ast.unparse(resolver.rewrite(core, e.scope, ctx))
                                li.expr_src = "\n".join(ctx.stmts + [li.expr_src]) if ctx.stmts else li.expr_src
                            except _Unresolvable:
                                li.expr_src = ""
                        return li
            elif e.kind == "param":
                for cs in index.call_sites(e.func.name):
                    arg = resolver._call_arg(cs, e.func, name)
                    if isinstance(arg, ast.Call):
                        li = _loss_from_call(index, resolver, _strip_wrappers(arg), index.scope_of(cs), depth + 1)
                        if li:
                            return li
                    elif isinstance(arg, ast.Name):
                        fake = ast.Call(func=ast.Name(id=arg.id, ctx=ast.Load()), args=[], keywords=[])
                        li = _loss_from_call(index, resolver, fake, index.scope_of(cs), depth + 1)
                        if li:
                            return li
        # local function used as a loss
        if name in index.functions:
            for n in ast.walk(index.functions[name]):
                if isinstance(n, ast.Call):
                    li = _loss_from_call(index, resolver, n, Scope(index.functions[name]), depth + 1)
                    if li:
                        return li
    # self.method(...) inside a class
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "self" and scope.cls is not None:
        for m in scope.cls.body:
            if isinstance(m, ast.FunctionDef) and m.name == call.func.attr:
                for n in ast.walk(m):
                    if isinstance(n, ast.Call):
                        li = _loss_from_call(index, resolver, n, Scope(m, scope.cls), depth + 1)
                        if li:
                            return li
    return None


def find_loss(index: SourceIndex, resolver: Resolver, loop: LoopInfo, loss_stmts: list) -> LossInfo:
    for stmt in loss_stmts:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call):
                li = _loss_from_call(index, resolver, n, loop.scope)
                if li:
                    return li
    # keras compile(loss=...)
    for n in ast.walk(index.tree):
        if isinstance(n, ast.Call) and _last_attr(n) == "compile":
            lv = _kw(n, "loss")
            if isinstance(lv, ast.Constant) and isinstance(lv.value, str) and lv.value in _LOSS_KIND:
                name = lv.value
                return LossInfo(name, _LOSS_KIND[name], f"F.{_KERAS_LOSS_TO_TORCH.get(name, 'cross_entropy')}", "keras.compile")
            if isinstance(lv, ast.Call) and _last_attr(lv) in _LOSS_KIND:
                name = _last_attr(lv)
                return LossInfo(name, _LOSS_KIND[name], f"F.{_KERAS_LOSS_TO_TORCH.get(name, 'cross_entropy')}", "keras.compile")
    # any known loss anywhere
    for n in ast.walk(index.tree):
        if isinstance(n, ast.Call):
            last = _last_attr(n)
            if last in _LOSS_KIND and last not in {"FocalLoss"}:
                kind = _LOSS_KIND[last]
                if last in _TORCH_LOSS_FUNCS:
                    return LossInfo(last, kind, f"F.{last}", "module-scan")
                if last in _TORCH_LOSS_CLASSES:
                    return LossInfo(last, kind, f"nn.{last}()", "module-scan")
                return LossInfo(last, kind, "", "module-scan")
    return LossInfo("cross_entropy", "class", "F.cross_entropy", "default")


def find_optimizer(index: SourceIndex, resolver: Resolver) -> dict:
    for n in ast.walk(index.tree):
        if isinstance(n, ast.Call) and _last_attr(n) in _OPTIMIZER_CLASSES:
            d = _dotted(n.func)
            root = d.split(".")[0] if d else ""
            if index.is_keras_import(root) or "keras" in d:
                lr = _kw(n, "learning_rate") or (n.args[0] if n.args else None)
                kws = f"lr={ast.unparse(lr)}" if lr is not None else ""
                return {"class": _last_attr(n), "kwargs_src": kws, "origin": "keras.optimizers", "dropped": []}
            scope = index.scope_of(n)
            kws = []
            positional = []
            dropped = []
            for i, a in enumerate(n.args[1:]):
                try:
                    positional.append(ast.unparse(resolver.rewrite(a, scope, Ctx("model"))))
                except _Unresolvable:
                    break
            for k in n.keywords:
                if k.arg is None:
                    continue
                try:
                    kws.append(f"{k.arg}={ast.unparse(resolver.rewrite(k.value, scope, Ctx('model')))}")
                except _Unresolvable:
                    dropped.append(k.arg)
            names = ["lr", "momentum", "weight_decay"]
            kws = [f"{names[i]}={p}" for i, p in enumerate(positional) if i < len(names)] + kws
            return {"class": _last_attr(n), "kwargs_src": ", ".join(kws), "origin": "torch",
                    "dropped": dropped}
    for n in ast.walk(index.tree):
        if isinstance(n, ast.Call) and _last_attr(n) == "compile":
            ov = _kw(n, "optimizer")
            if isinstance(ov, ast.Constant) and isinstance(ov.value, str):
                m = {"adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop", "adagrad": "Adagrad",
                     "adadelta": "Adadelta", "adamw": "AdamW", "nadam": "NAdam", "adamax": "Adamax"}
                return {"class": m.get(ov.value.lower(), "Adam"), "kwargs_src": "", "origin": "keras.compile"}
            if isinstance(ov, ast.Call) and _last_attr(ov) in _OPTIMIZER_CLASSES:
                lr = _kw(ov, "learning_rate") or (ov.args[0] if ov.args else None)
                kws = f"lr={ast.unparse(lr)}" if lr is not None else ""
                return {"class": _last_attr(ov), "kwargs_src": kws, "origin": "keras.compile"}
    return {"class": "AdamW", "kwargs_src": "", "origin": "default"}


# ---------------------------------------------------------------------------
# Shape inference (R4)
# ---------------------------------------------------------------------------

@dataclass
class ShapeSpec:
    input_shape: tuple = (64,)
    input_dtype: str = "float"
    target_kind: str = "class"
    num_classes: int = 10
    out_dim: int = 1
    target_channels: int = 1
    batch_format: str = "tuple"
    batch_keys: list = field(default_factory=lambda: ["x", "y"])
    evidence: list = field(default_factory=list)
    unresolved: bool = False


def _as_shape(v) -> Optional[tuple]:
    if isinstance(v, int) and not isinstance(v, bool):
        return (v,)
    if isinstance(v, (tuple, list)) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        return tuple(v)
    return None


def infer_shapes(index: SourceIndex, resolver: Resolver, models: list[ModelInfo],
                 datasets: list[DataCandidate], loop: LoopInfo, loss: LossInfo,
                 explicit: Optional[dict] = None) -> ShapeSpec:
    spec = ShapeSpec()
    explicit = explicit or {}
    ev = spec.evidence
    shape_expl: Optional[tuple] = None
    classes_expl: Optional[int] = None
    spatial_dims = 2

    def ceval(node, scope):
        try:
            return resolver.ceval(node, scope)
        except _Unresolvable:
            return None

    # keras / translator supplied
    if explicit.get("input_shape"):
        shape_expl = tuple(explicit["input_shape"])
        ev.append(f"explicit:{explicit.get('source', 'translator')}={shape_expl}")
    if explicit.get("num_outputs"):
        classes_expl = explicit["num_outputs"]
        ev.append(f"classes:{explicit.get('source', 'translator')}={classes_expl}")

    # FakeData
    for n in ast.walk(index.tree):
        if isinstance(n, ast.Call) and _last_attr(n) == "FakeData":
            sc = index.scope_of(n)
            img = _kw(n, "image_size") or (n.args[1] if len(n.args) > 1 else None)
            nc = _kw(n, "num_classes") or (n.args[2] if len(n.args) > 2 else None)
            s = _as_shape(ceval(img, sc)) if img is not None else None
            if s and shape_expl is None:
                shape_expl = s
                ev.append(f"explicit:FakeData={s}")
            c = ceval(nc, sc) if nc is not None else None
            if isinstance(c, int) and classes_expl is None:
                classes_expl = c
                ev.append(f"classes:FakeData={c}")

    # MONAI network kwargs
    monai_spatial = None
    for m in models:
        if isinstance(m.ctor, ast.Call) and not m.is_local_class:
            sd = _kw(m.ctor, "spatial_dims")
            ic = _kw(m.ctor, "in_channels")
            oc = _kw(m.ctor, "out_channels")
            if sd is not None:
                v = ceval(sd, m.scope)
                if isinstance(v, int):
                    spatial_dims = v
            if ic is not None and shape_expl is None:
                c = ceval(ic, m.scope)
                if isinstance(c, int):
                    size = None
                    for n in ast.walk(index.tree):
                        if isinstance(n, ast.Call) and _last_attr(n) in {"Resize", "Resized", "RandCropByPosNegLabeld", "RandSpatialCropd", "RandSpatialCrop", "CenterSpatialCropd", "CenterSpatialCrop", "SpatialPadd", "SpatialPad", "RandSpatialCropSamplesd", "ResizeWithPadOrCropd", "ResizeWithPadOrCrop"}:
                            arg = _kw(n, "spatial_size") or _kw(n, "roi_size") or (n.args[0] if n.args else None)
                            v = _as_shape(ceval(arg, index.scope_of(n))) if arg is not None else None
                            if v and len(v) in (1, spatial_dims):
                                size = v if len(v) == spatial_dims else v * spatial_dims
                                if -1 not in size:
                                    break
                    if size is None:
                        size = (32,) * spatial_dims
                        ev.append("default:monai-spatial=32")
                    shape_expl = (c,) + tuple(size)
                    ev.append(f"explicit:monai(in_channels,spatial)={shape_expl}")
            if oc is not None and classes_expl is None:
                v = ceval(oc, m.scope)
                if isinstance(v, int):
                    classes_expl = v
                    ev.append(f"classes:monai.out_channels={v}")
                    spec.target_channels = v
            for key in ("num_classes", "n_classes", "out_features", "num_outputs"):
                kv = _kw(m.ctor, key)
                if kv is not None and classes_expl is None:
                    v = ceval(kv, m.scope)
                    if isinstance(v, int):
                        classes_expl = v
                        ev.append(f"classes:ctor.{key}={v}")

    # model __init__ kwarg defaults such as img_shape=(1, 28, 28)
    primary = models[0] if models else None
    if primary is not None and primary.cls_node is not None:
        init = next((m for m in primary.cls_node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
        if init is not None:
            args = init.args
            n_def = len(args.args) - len(args.defaults)
            for i, a in enumerate(args.args):
                if i < n_def:
                    continue
                dflt = args.defaults[i - n_def]
                if a.arg in {"img_shape", "input_shape", "in_shape", "image_shape", "input_size", "input_dims"}:
                    v = _as_shape(ceval(dflt, index.module_scope))
                    if v and shape_expl is None:
                        shape_expl = v
                        ev.append(f"explicit:__init__.{a.arg}={v}")
                if a.arg in {"num_classes", "n_classes", "out_features", "output_dim", "n_outputs"} and classes_expl is None:
                    v = ceval(dflt, index.module_scope)
                    if isinstance(v, int):
                        classes_expl = v
                        ev.append(f"classes:__init__.{a.arg}={v}")
            # ctor call kwargs override
            if isinstance(primary.ctor, ast.Call):
                for k in primary.ctor.keywords:
                    if k.arg in {"img_shape", "input_shape", "in_shape", "image_shape", "input_size"}:
                        v = _as_shape(ceval(k.value, primary.scope))
                        if v:
                            shape_expl = v
                            ev.append(f"explicit:ctor.{k.arg}={v}")

    # torchvision transforms
    t_channels: list[int] = []
    t_sizes: list[tuple] = []
    for n in ast.walk(index.tree):
        if isinstance(n, ast.Call) and _last_attr(n) == "Compose" and n.args and isinstance(n.args[0], (ast.List, ast.Tuple)):
            sc = index.scope_of(n)
            last_size = None
            for t in n.args[0].elts:
                if not isinstance(t, ast.Call):
                    continue
                tn = _last_attr(t)
                if tn == "Normalize":
                    mean = _kw(t, "mean") or (t.args[0] if t.args else None)
                    v = ceval(mean, sc) if mean is not None else None
                    if isinstance(v, (tuple, list)):
                        t_channels.append(len(v))
                elif tn == "Grayscale":
                    k = _kw(t, "num_output_channels") or (t.args[0] if t.args else None)
                    v = ceval(k, sc) if k is not None else 1
                    t_channels.append(v if isinstance(v, int) else 1)
                elif tn in {"Resize", "CenterCrop", "RandomResizedCrop", "RandomCrop", "Resized"} and spatial_dims == 2:
                    s = _kw(t, "size") or (t.args[0] if t.args else None)
                    v = _as_shape(ceval(s, sc)) if s is not None else None
                    if v:
                        last_size = v if len(v) == 2 else (v[0], v[0])
            if last_size:
                t_sizes.append(last_size)
    for n in ast.walk(index.tree):   # Normalize outside Compose (imagenet style `normalize = ...`)
        if isinstance(n, ast.Call) and _last_attr(n) == "Normalize" and index.enclosing(n, (ast.Call,)) is None:
            mean = _kw(n, "mean") or (n.args[0] if n.args else None)
            v = ceval(mean, index.scope_of(n)) if mean is not None else None
            if isinstance(v, (tuple, list)):
                t_channels.append(len(v))

    # dataset catalogue
    cat_shape, cat_classes = None, None
    for c in datasets:
        if c.name in _DATASET_CATALOGUE:
            cat_shape, cat_classes = _DATASET_CATALOGUE[c.name]
            ev.append(f"catalogue:{c.name}={cat_shape},{cat_classes}")
            break
    if cat_shape is None:
        for n in ast.walk(index.tree):
            if isinstance(n, ast.Call) and _last_attr(n) in _DATASET_CATALOGUE and _last_attr(n) not in {"Dataset"}:
                cat_shape, cat_classes = _DATASET_CATALOGUE[_last_attr(n)]
                ev.append(f"catalogue:{_last_attr(n)}={cat_shape},{cat_classes}")
                break

    # model introspection
    m_channels, m_flat, m_classes, conv_dim = None, None, None, None
    if primary is not None and primary.cls_node is not None:
        init = next((m for m in primary.cls_node.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
        if init is not None:
            sc = Scope(init, primary.cls_node)
            convs, linears = [], []
            for n in ast.walk(init):
                if isinstance(n, ast.Call):
                    ln = _last_attr(n)
                    if ln in {"Conv1d", "Conv2d", "Conv3d", "LazyConv2d"} and n.args:
                        convs.append((ln, n))
                    elif ln == "Linear" and len(n.args) >= 2:
                        linears.append(n)
            convs.sort(key=lambda p: p[1].lineno)
            linears.sort(key=lambda p: p.lineno)
            if convs:
                ln, c0 = convs[0]
                conv_dim = int(ln[4]) if ln[4].isdigit() else 2
                v = ceval(c0.args[0], sc)
                if isinstance(v, int):
                    m_channels = v
                    ev.append(f"model:first-{ln}.in_channels={v}")
            elif linears:
                v = ceval(linears[0].args[0], sc)
                if isinstance(v, int):
                    m_flat = v
                    ev.append(f"model:first-Linear.in_features={v}")
            if linears:
                v = ceval(linears[-1].args[1], sc)
                if isinstance(v, int):
                    m_classes = v
                    ev.append(f"model:last-Linear.out_features={v}")

    # combine
    if shape_expl is not None:
        spec.input_shape = shape_expl
    else:
        ch = None
        if t_channels:
            ch = Counter(t_channels).most_common(1)[0][0]
            ev.append(f"transform:channels={ch}")
        elif m_channels is not None:
            ch = m_channels
        elif cat_shape is not None and len(cat_shape) > 1:
            ch = cat_shape[0]
        size = None
        if t_sizes:
            size = Counter(t_sizes).most_common(1)[0][0]
            ev.append(f"transform:size={size}")
        elif cat_shape is not None and len(cat_shape) > 1:
            size = tuple(cat_shape[1:])
        if ch is not None and (size is not None or conv_dim):
            if size is None:
                size = (32,) * (conv_dim or 2)
                ev.append("default:spatial=32")
            spec.input_shape = (ch,) + tuple(size)
        elif cat_shape is not None:
            spec.input_shape = tuple(cat_shape)
        elif m_flat is not None:
            spec.input_shape = (m_flat,)
        else:
            spec.input_shape = (64,)
            spec.unresolved = True
            ev.append("default:input=(64,)")
    if classes_expl is not None:
        spec.num_classes = classes_expl
    elif m_classes is not None:
        spec.num_classes = m_classes
    elif cat_classes is not None:
        spec.num_classes = cat_classes
    else:
        spec.num_classes = 10
        ev.append("default:classes=10")
    spec.target_kind = loss.kind
    if loss.kind == "regression":
        spec.out_dim = classes_expl or m_classes or 1
    if loss.kind == "segmentation" and classes_expl is not None:
        spec.target_channels = classes_expl

    # batch format from the loop body
    bname = None
    if loop.kind == "loop" and isinstance(loop.batch_target, ast.Name):
        bname = loop.batch_target.id
    elif loop.kind == "lightning":
        bname = loop.batch_param
    if bname:
        keys = []
        subs = [n for n in _walk_no_defs(ast.Module(body=loop.body, type_ignores=[])) if isinstance(n, ast.Subscript)]
        for n in sorted(subs, key=lambda x: (x.lineno, x.col_offset)):
            if isinstance(n.value, ast.Name) and n.value.id == bname \
                    and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) and n.slice.value not in keys:
                keys.append(n.slice.value)
        if keys:
            spec.batch_format = "dict"
            spec.batch_keys = keys[:2] if len(keys) >= 2 else keys + ["label"]
            ev.append(f"batch:dict{spec.batch_keys}")
    return spec


# ---------------------------------------------------------------------------
# Train-step extraction (R3)
# ---------------------------------------------------------------------------

@dataclass
class TrainStepResult:
    ok: bool
    body: list = field(default_factory=list)      # statement sources (unindented)
    prologue: list = field(default_factory=list)
    return_expr: str = "loss"
    loss_stmts: list = field(default_factory=list)  # AST statements assigning losses
    notes: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    reason: str = ""


def _inline_amp(body: list) -> list:
    out = []
    for s in body:
        if isinstance(s, ast.With) and all(_dotted(i.context_expr) in _AMP_CONTEXTS or _last_attr(i.context_expr) == "autocast" for i in s.items):
            out += _inline_amp(s.body)
        else:
            out.append(s)
    return out


def _is_scrub(stmt: ast.AST) -> bool:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        c = stmt.value
        if isinstance(c.func, ast.Attribute):
            if c.func.attr in _SCRUB_EXPR_ATTRS:
                return True
            if c.func.attr == "update" and not c.args and not c.keywords:
                return True
            if c.func.attr in _SCRUB_FUNC_NAMES:
                return True
        if isinstance(c.func, ast.Name) and c.func.id in _SCRUB_FUNC_NAMES:
            return True
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and _last_attr(stmt.value) in _SCRUB_ASSIGN_FROM:
        return True
    if isinstance(stmt, (ast.Break, ast.Continue, ast.Pass)):
        return True
    return False


class _BodyRewriter(ast.NodeTransformer):
    """Rename model variables, self -> model, .cuda() -> .to(device)."""

    def __init__(self, rename: dict[str, str]):
        self.rename = rename

    def visit_Name(self, node):
        if node.id in self.rename and isinstance(node.ctx, ast.Load):
            new = self.rename[node.id]
            if "." in new:
                base, attr = new.split(".", 1)
                return ast.copy_location(ast.Attribute(value=ast.Name(id=base, ctx=ast.Load()), attr=attr, ctx=ast.Load()), node)
            return ast.copy_location(ast.Name(id=new, ctx=ast.Load()), node)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            return ast.copy_location(ast.Call(
                func=ast.Attribute(value=node.func.value, attr="to", ctx=ast.Load()),
                args=[ast.Name(id="device", ctx=ast.Load())], keywords=[]), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to" and node.args:
            # X.to(device, non_blocking=True) / X.to(args.gpu) -> X.to(device)
            node.args = [ast.Name(id="device", ctx=ast.Load())]
            node.keywords = [k for k in node.keywords if k.arg not in {"non_blocking"}]
        return node


def _strip_item(expr: ast.AST) -> ast.AST:
    while isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr in {"item", "detach", "cpu", "float"} and not expr.args:
        expr = expr.func.value
    return expr


def extract_train_step(index: SourceIndex, resolver: Resolver, loop: LoopInfo,
                       rename: dict[str, str], module_names: set[str]) -> TrainStepResult:
    res = TrainStepResult(ok=False)
    if loop.kind not in {"loop", "lightning"}:
        res.reason = "no training loop / training_step found"
        return res
    body = _inline_amp(list(loop.body))
    if len(body) != len(loop.body):
        res.notes.append("R3.2 autocast block inlined")

    # --- losses -----------------------------------------------------------
    stores_by_stmt = [_store_names(s) for s in body]
    all_stores = set().union(*stores_by_stmt) if stores_by_stmt else set()
    loss_names: list[str] = []
    backward_idx: dict[str, int] = {}
    for i, s in enumerate(body):
        for sub in _own_statements([s]):
            tgt = _backward_target(sub)
            if tgt is None:
                continue
            names = [n.id for n in ast.walk(tgt) if isinstance(n, ast.Name) and n.id in all_stores]
            if isinstance(tgt, ast.Name):
                names = [tgt.id]
            if names and names[0] not in loss_names:
                loss_names.append(names[0])
                backward_idx[names[0]] = i
    return_expr_src = None
    if not loss_names and loop.kind == "lightning":
        for i, s in enumerate(body):
            if isinstance(s, ast.Return) and s.value is not None:
                v = s.value
                if isinstance(v, ast.Dict):
                    for k, val in zip(v.keys, v.values):
                        if isinstance(k, ast.Constant) and k.value == "loss":
                            v = val
                if isinstance(v, ast.Name):
                    loss_names = [v.id]
                    backward_idx[v.id] = i
                else:
                    return_expr_src = ast.unparse(v)
                    backward_idx["__return__"] = i
                break
    if not loss_names and return_expr_src is None:
        res.reason = "no backward()/manual_backward()/return-loss statement in the loop body"
        return res

    # cut after the last assignment of the last back-propagated loss
    cut = -1
    for name in loss_names:
        bidx = backward_idx[name]
        last_assign = -1
        for i in range(bidx, -1, -1):
            if name in stores_by_stmt[i]:
                last_assign = i
                break
        if last_assign < 0:
            res.reason = f"loss variable {name!r} is not assigned in the loop body"
            return res
        cut = max(cut, last_assign)
    if return_expr_src is not None:
        cut = max(cut, backward_idx["__return__"] - 1)
    kept = body[: cut + 1]
    if return_expr_src is None and len(loss_names) > 1:
        res.notes.append(f"R3.7 multi-loss loop: returning {' + '.join(loss_names)}")

    # scrub
    kept2 = []
    for s in kept:
        if _is_scrub(s):
            res.dropped.append(f"scrubbed: {ast.unparse(s).splitlines()[0]}")
            continue
        kept2.append(s)
    kept = kept2
    if len(kept2) != len(body[: cut + 1]):
        res.notes.append("R3.3 optimizer/backward/logging call sites scrubbed")

    # in-place -> out-of-place (single backward safety)
    kept3 = []
    for s in kept:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute) \
                and s.value.func.attr in _INPLACE_REWRITE and isinstance(s.value.func.value, ast.Name):
            x = s.value.func.value.id
            v = ast.unparse(s.value.args[0]) if s.value.args else "0"
            new = ast.parse(f"{x} = {_INPLACE_REWRITE[s.value.func.attr].format(x=x, v=v)}").body[0]
            kept3.append(new)
            res.notes.append(f"R3.6 in-place {x}.{s.value.func.attr}() rewritten out-of-place")
        else:
            kept3.append(s)
    kept = kept3

    # backward slice from the loss statements
    needed: set[str] = set(loss_names)
    if return_expr_src is not None:
        needed |= _load_names(ast.parse(return_expr_src, mode="eval"))
    keep_flags = [False] * len(kept)
    for i in range(len(kept) - 1, -1, -1):
        s = kept[i]
        stores = _store_names(s)
        mutates = set()
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute):
            base = s.value.func.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                mutates.add(base.id)
        if isinstance(s, ast.AugAssign):
            stores |= _store_names(s.target)
        if stores & needed or mutates & needed:
            keep_flags[i] = True
            needed |= free_names(s)
    sliced = [s for s, f in zip(kept, keep_flags) if f]
    if len(sliced) != len(kept):
        res.notes.append("R3.4 backward slice removed statements not feeding the loss")

    # prologue: batch binding
    prologue = []
    bound: set[str] = {"model", "batch", "optimizer", "config", "device"}
    if loop.kind == "loop":
        t = loop.batch_target
        if isinstance(t, ast.Name):
            if t.id != "batch":
                prologue.append(f"{t.id} = batch")
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            prologue.append(f"{ast.unparse(t)} = batch")
            bound |= _store_names(t)
        for n in loop.index_names:
            prologue.append(f"{n} = 0")
            bound.add(n)
    else:
        if loop.batch_param != "batch":
            prologue.append(f"{loop.batch_param} = batch")
        bound.add(loop.batch_param)
        for n in loop.index_names:
            prologue.append(f"{n} = 0")
            bound.add(n)

    # rewrite + free-name resolution
    ctx = Ctx("train", defined=set(bound) | set(rename.values()))
    ctx.defined.discard("self")
    hoist_ctx = Ctx("train", defined=set(bound))
    out_stmts: list[str] = []
    defined = set(bound)
    rw = _BodyRewriter(rename)
    for s in sliced:
        s2 = rw.visit(copy.deepcopy(s))
        ast.fix_missing_locations(s2)
        # strip .item()/.detach() on loss assignments
        if isinstance(s2, ast.Assign) and any(isinstance(t, ast.Name) and t.id in loss_names for t in s2.targets):
            s2.value = _strip_item(s2.value)
        fn = free_names(s2) - defined - module_names - set(rename.values())
        unresolved = []
        for name in sorted(fn):
            if name in ("model", "device"):
                continue
            if name in hoist_ctx.defined:
                continue
            if resolver.materialize(name, loop.scope, hoist_ctx):
                continue
            unresolved.append(name)
        if unresolved:
            res.dropped.append(f"unresolved {unresolved}: {ast.unparse(s2).splitlines()[0]}")
            if _store_names(s2) & set(loss_names):
                res.reason = f"loss statement depends on unresolvable names {unresolved}"
                res.dropped_loss = True
                return res
            continue
        out_stmts.append(ast.unparse(s2))
        defined |= _store_names(s2)
    for name in loss_names:
        if name not in defined:
            res.reason = f"loss variable {name!r} was dropped during extraction"
            return res
    if hoist_ctx.hoisted:
        res.notes.append(f"R3.5 hoisted from enclosing scopes: {sorted(set(hoist_ctx.hoisted))}")
    res.prologue = prologue + hoist_ctx.stmts
    res.body = out_stmts
    res.return_expr = return_expr_src or " + ".join(loss_names)
    res.loss_stmts = [s for s in sliced if _store_names(s) & set(loss_names)]
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# Keras -> torch translation (R5.1)
# ---------------------------------------------------------------------------

@dataclass
class KerasResult:
    ok: bool = False
    class_src: str = ""
    class_name: str = "TemplateKerasModel"
    input_shape: Optional[tuple] = None       # channels-first
    num_outputs: Optional[int] = None
    input_dtype: str = "float"
    fallback_reason: str = ""
    notes: list = field(default_factory=list)
    unresolved_shape: bool = False
    final_activation: str = ""
    helpers: set = field(default_factory=set)


_KERAS_ACT = {"relu": "nn.ReLU()", "sigmoid": "nn.Sigmoid()", "tanh": "nn.Tanh()",
              "softmax": "nn.Softmax(dim=-1)", "gelu": "nn.GELU()", "elu": "nn.ELU()",
              "selu": "nn.SELU()", "softplus": "nn.Softplus()", "leaky_relu": "nn.LeakyReLU(0.2)",
              "swish": "nn.SiLU()", "silu": "nn.SiLU()", "linear": "", None: ""}

_KERAS_HELPERS = {
    "rescale": '''class _TemplateRescale(nn.Module):
    """keras.layers.Rescaling equivalent."""
    def __init__(self, scale, offset=0.0):
        super().__init__()
        self.scale, self.offset = float(scale), float(offset)
    def forward(self, x):
        return x * self.scale + self.offset
''',
    "rnn": '''class _TemplateRNN(nn.Module):
    """keras LSTM/GRU equivalent (batch_first); returns sequences or the last step."""
    def __init__(self, rnn, return_sequences=False):
        super().__init__()
        self.rnn, self.return_sequences = rnn, return_sequences
    def forward(self, x):
        out, _ = self.rnn(x)
        return out if self.return_sequences else out[:, -1]
''',
    "cl1d": '''class _TemplateChannelsLast1d(nn.Module):
    """Apply a torch (N,C,L) module to keras-style (N,L,C) tensors."""
    def __init__(self, mod):
        super().__init__()
        self.mod = mod
    def forward(self, x):
        return self.mod(x.transpose(1, 2)).transpose(1, 2)
''',
}


def _pair(v, n=2):
    if isinstance(v, int):
        return (v,) * n
    if isinstance(v, (tuple, list)) and len(v) == n:
        return tuple(v)
    if isinstance(v, (tuple, list)) and len(v) == 1:
        return (v[0],) * n
    return None


def _conv_out(n, k, s, p):
    return (n + 2 * p - k) // s + 1


class KerasTranslator:
    def __init__(self, index: SourceIndex, resolver: Resolver):
        self.index = index
        self.r = resolver

    def ceval(self, node, scope):
        try:
            return self.r.ceval(node, scope)
        except _Unresolvable:
            return None

    def _shape_from_input_call(self, call: ast.Call, scope: Scope) -> tuple[Optional[tuple], bool]:
        sh = _kw(call, "shape") or _kw(call, "input_shape") or (call.args[0] if call.args else None)
        if sh is None:
            return None, True
        if isinstance(sh, (ast.Tuple, ast.List)):
            vals, unresolved = [], False
            for el in sh.elts:
                if isinstance(el, ast.Constant) and el.value is None:
                    vals.append(16)          # variable-length axis -> default length 16
                    unresolved = True
                    continue
                try:
                    v = self.r.ceval(el, scope)
                except _Unresolvable:
                    v = None
                if isinstance(v, int) and not isinstance(v, bool):
                    vals.append(v)
                else:
                    vals.append(32)          # data-derived size (e.g. vocabulary) -> default 32
                    unresolved = True
            return tuple(vals), unresolved
        v = self.ceval(sh, scope)
        s = _as_shape(v)
        if s is None:
            return None, True
        return s, False

    def translate(self) -> KerasResult:
        res = KerasResult()
        tree = self.index.tree
        # ---- input shapes ----
        inputs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _last_attr(n) in {"Input", "InputLayer"}]
        shapes = []
        for c in inputs:
            s, unres = self._shape_from_input_call(c, self.index.scope_of(c))
            if s is not None:
                shapes.append((s, unres))
        if not shapes:
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and _kw(n, "input_shape") is not None:
                    s, unres = self._shape_from_input_call(n, self.index.scope_of(n))
                    if s is not None:
                        shapes.append((s, unres))
                        break
        keras_shape, unres = shapes[0] if shapes else (None, True)
        res.unresolved_shape = unres or keras_shape is None
        if keras_shape is not None:
            res.input_shape = self._to_channels_first(keras_shape)
            res.notes.append(f"R5.1 keras Input shape {keras_shape} -> channels-first {res.input_shape}")
        # ---- layer sequence ----
        layers, reason = self._find_layers()
        if len(inputs) > 1:
            layers, reason = None, f"multi-input functional graph ({len(inputs)} keras.Input calls)"
        # last Dense units for the generic fallback
        dense_units = None
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and _last_attr(n) == "Dense":
                u = _kw(n, "units") or (n.args[0] if n.args else None)
                v = self.ceval(u, self.index.scope_of(n)) if u is not None else None
                if isinstance(v, int):
                    dense_units = v
        if dense_units is not None:
            res.num_outputs = dense_units
        if layers is None or keras_shape is None:
            res.fallback_reason = reason if layers is None else "keras Input shape not found"
            return res
        # ---- map ----
        try:
            body_lines, out_dim, helpers, dtype, final_act = self._map_layers(layers, keras_shape)
        except _Unresolvable as e:
            res.fallback_reason = f"unsupported keras layer/argument: {e}"
            return res
        res.helpers = helpers
        res.input_dtype = dtype
        res.num_outputs = out_dim
        res.final_activation = final_act
        res.ok = True
        lines = ["class TemplateKerasModel(nn.Module):",
                 '    """Translated from the Keras layer stack by rule R5.1 (static shape tracking,',
                 '    NHWC -> NCHW). Comments give the original layer and the tracked output shape."""',
                 "",
                 "    def __init__(self, num_outputs: int = %d):" % out_dim,
                 "        super().__init__()",
                 "        self.net = nn.Sequential("]
        for l in body_lines:
            lines.append("            " + l)
        lines += ["        )", "", "    def forward(self, x):", "        return self.net(x)"]
        res.class_src = "\n".join(lines)
        return res

    @staticmethod
    def _to_channels_first(shape: tuple) -> tuple:
        if len(shape) == 3:
            return (shape[2], shape[0], shape[1])
        if len(shape) == 4:
            return (shape[3], shape[0], shape[1], shape[2])
        return tuple(shape)

    def _find_layers(self):
        tree = self.index.tree
        # (a) Sequential([...])
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and _last_attr(n) == "Sequential" and n.args and isinstance(n.args[0], (ast.List, ast.Tuple)):
                items = [e for e in n.args[0].elts if isinstance(e, ast.Call) and _last_attr(e) not in {"Input", "InputLayer"}]
                if items:
                    return [(e, self.index.scope_of(e)) for e in items], ""
        # (b) model.add(...)
        adds = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and _last_attr(n.value) == "add" \
                    and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and n.value.args:
                adds.setdefault(n.value.func.value.id, []).append(n.value.args[0])
        for var, items in adds.items():
            items = [e for e in items if isinstance(e, ast.Call)]
            if items:
                return [(e, self.index.scope_of(e)) for e in items], ""
        # (c) linear functional chain
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and _last_attr(n) == "Model" and len(n.args) >= 2:
                container = self.index.enclosing(n, (ast.FunctionDef, ast.Module)) or tree
                body = container.body
                for s in body:
                    if isinstance(s, (ast.For, ast.While)) and any(isinstance(x, ast.Call) for x in ast.walk(s)):
                        return None, "non-linear functional graph (layers applied inside a loop)"
                chain, prev = [], None
                for s in body:
                    if not isinstance(s, ast.Assign) or len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                        continue
                    v = s.value
                    if isinstance(v, ast.Call) and _last_attr(v) in {"Input", "InputLayer"}:
                        prev = s.targets[0].id
                        continue
                    if prev is None:
                        continue
                    if isinstance(v, ast.Call) and isinstance(v.func, ast.Call):
                        if len(v.args) == 1 and isinstance(v.args[0], ast.Name) and v.args[0].id == prev:
                            chain.append((v.func, self.index.scope_of(s)))
                            prev = s.targets[0].id
                            continue
                        return None, "non-linear functional graph (layer applied to a non-chain tensor)"
                    if isinstance(v, ast.Call) and _last_attr(v) in {"add", "Add", "concatenate", "Concatenate", "Multiply", "multiply"}:
                        return None, "non-linear functional graph (residual/merge layer)"
                if chain:
                    return chain, ""
        return None, "no Sequential/add()/linear functional layer stack found"

    def _map_layers(self, layers, keras_shape):
        shape = list(keras_shape)          # keras order, without batch
        lines, helpers = [], set()
        dtype = "float"
        final_act = ""
        rank = len(shape)

        def act_line(name, layer_txt):
            if name in (None, "", "linear"):
                return
            if name not in _KERAS_ACT:
                raise _Unresolvable(f"activation {name}")
            lines.append(f"{_KERAS_ACT[name]},  # activation={name!r} of {layer_txt}")

        for i, (call, scope) in enumerate(layers):
            ln = _last_attr(call)
            txt = ast.unparse(call)
            if len(txt) > 70:
                txt = txt[:67] + "..."
            args = [self.ceval(a, scope) for a in call.args]
            kwargs = {k.arg: self.ceval(k.value, scope) for k in call.keywords if k.arg}
            act = kwargs.get("activation")
            is_last = i == len(layers) - 1
            if ln in {"Conv2D", "SeparableConv2D", "DepthwiseConv2D"} and rank == 3:
                H, W, C = shape
                filters = args[0] if args else kwargs.get("filters")
                k = _pair(args[1] if len(args) > 1 else kwargs.get("kernel_size"))
                s = _pair(args[2] if len(args) > 2 else kwargs.get("strides", 1))
                if ln == "DepthwiseConv2D":
                    k = _pair(args[0] if args else kwargs.get("kernel_size"))
                    filters = C * int(kwargs.get("depth_multiplier", 1))
                if filters is None or k is None or s is None:
                    raise _Unresolvable(f"{ln} args")
                pad = kwargs.get("padding", "valid")
                p = (k[0] // 2, k[1] // 2) if pad == "same" else (0, 0)
                bias = kwargs.get("use_bias", True)
                bias_txt = "" if bias else ", bias=False"
                nh, nw = _conv_out(H, k[0], s[0], p[0]), _conv_out(W, k[1], s[1], p[1])
                if ln == "Conv2D":
                    lines.append(f"nn.Conv2d({C}, {filters}, kernel_size={k}, stride={s}, padding={p}{bias_txt}),  # {txt} -> {(nh, nw, filters)}")
                elif ln == "SeparableConv2D":
                    lines.append(f"nn.Conv2d({C}, {C}, kernel_size={k}, stride={s}, padding={p}, groups={C}, bias=False),  # {txt} (depthwise)")
                    lines.append(f"nn.Conv2d({C}, {filters}, kernel_size=1{bias_txt}),  # {txt} (pointwise) -> {(nh, nw, filters)}")
                else:
                    lines.append(f"nn.Conv2d({C}, {filters}, kernel_size={k}, stride={s}, padding={p}, groups={C}{bias_txt}),  # {txt} -> {(nh, nw, filters)}")
                shape = [nh, nw, filters]
                act_line(act, ln)
            elif ln in {"MaxPooling2D", "MaxPool2D", "AveragePooling2D", "AvgPool2D"} and rank == 3:
                H, W, C = shape
                k = _pair(args[0] if args else kwargs.get("pool_size", 2))
                s = _pair(args[1] if len(args) > 1 else kwargs.get("strides")) or k
                pad = kwargs.get("padding", "valid")
                p = (k[0] // 2, k[1] // 2) if pad == "same" else (0, 0)
                nh, nw = _conv_out(H, k[0], s[0], p[0]), _conv_out(W, k[1], s[1], p[1])
                mod = "nn.MaxPool2d" if "Max" in ln else "nn.AvgPool2d"
                lines.append(f"{mod}(kernel_size={k}, stride={s}, padding={p}),  # {txt} -> {(nh, nw, C)}")
                shape = [nh, nw, C]
            elif ln in {"GlobalAveragePooling2D", "GlobalMaxPooling2D"} and rank == 3:
                mod = "nn.AdaptiveAvgPool2d(1)" if "Average" in ln else "nn.AdaptiveMaxPool2d(1)"
                lines.append(f"{mod},  # {txt}")
                lines.append(f"nn.Flatten(),  # -> ({shape[2]},)")
                shape = [shape[2]]
            elif ln == "ZeroPadding2D" and rank == 3:
                p = _pair(args[0] if args else kwargs.get("padding", 1))
                lines.append(f"nn.ZeroPad2d(({p[1]}, {p[1]}, {p[0]}, {p[0]})),  # {txt}")
                shape = [shape[0] + 2 * p[0], shape[1] + 2 * p[1], shape[2]]
            elif ln == "BatchNormalization":
                if rank == 3:
                    lines.append(f"nn.BatchNorm2d({shape[2]}),  # {txt}")
                elif rank == 1:
                    lines.append(f"nn.BatchNorm1d({shape[0]}),  # {txt}")
                else:
                    raise _Unresolvable("BatchNormalization on rank-%d input" % rank)
            elif ln == "Activation":
                name = args[0] if args else kwargs.get("activation")
                if is_last and name in {"softmax", "sigmoid"}:
                    final_act = name
                    lines.append(f"# {txt}: final {name} folded into the loss")
                else:
                    act_line(name, ln)
            elif ln in {"ReLU", "LeakyReLU", "ELU", "Softmax", "PReLU"}:
                m = {"ReLU": "nn.ReLU()", "ELU": "nn.ELU()", "PReLU": "nn.PReLU()",
                     "LeakyReLU": f"nn.LeakyReLU({args[0] if args else kwargs.get('negative_slope', kwargs.get('alpha', 0.3))})",
                     "Softmax": "nn.Softmax(dim=-1)"}[ln]
                if ln == "Softmax" and is_last:
                    final_act = "softmax"
                    lines.append(f"# {txt}: final softmax folded into the loss")
                else:
                    lines.append(f"{m},  # {txt}")
            elif ln in {"Dropout", "SpatialDropout2D", "SpatialDropout1D"}:
                rate = args[0] if args else kwargs.get("rate", 0.5)
                mod = "nn.Dropout2d" if ln == "SpatialDropout2D" else "nn.Dropout"
                lines.append(f"{mod}({rate}),  # {txt}")
            elif ln == "Flatten":
                n = int(math.prod(shape))
                lines.append(f"nn.Flatten(),  # {txt} -> ({n},)")
                shape = [n]
                rank = 1
            elif ln == "Dense":
                units = args[0] if args else kwargs.get("units")
                if not isinstance(units, int):
                    raise _Unresolvable("Dense units")
                if rank not in (1, 2):
                    n = int(math.prod(shape))
                    lines.append(f"nn.Flatten(),  # implicit flatten before Dense -> ({n},)")
                    shape, rank = [n], 1
                in_f = shape[-1]
                out_txt = "num_outputs" if is_last else str(units)
                lines.append(f"nn.Linear({in_f}, {out_txt}),  # {txt} -> {tuple(shape[:-1]) + (units,)}")
                shape = shape[:-1] + [units]
                if is_last and act in {"softmax", "sigmoid"}:
                    final_act = act
                    lines.append(f"# final activation {act!r} folded into the loss (logits are returned)")
                else:
                    act_line(act, ln)
            elif ln == "Rescaling":
                scale = args[0] if args else kwargs.get("scale", 1.0)
                offset = args[1] if len(args) > 1 else kwargs.get("offset", 0.0)
                helpers.add("rescale")
                lines.append(f"_TemplateRescale({scale}, {offset}),  # {txt}")
            elif ln == "Embedding":
                in_dim = args[0] if args else kwargs.get("input_dim")
                out_dim = args[1] if len(args) > 1 else kwargs.get("output_dim")
                if not isinstance(in_dim, int) or not isinstance(out_dim, int):
                    raise _Unresolvable("Embedding dims")
                dtype = "long"
                lines.append(f"nn.Embedding({in_dim}, {out_dim}),  # {txt} -> {tuple(shape) + (out_dim,)}")
                shape = list(shape) + [out_dim]
                rank = len(shape)
            elif ln in {"LSTM", "GRU", "SimpleRNN", "Bidirectional"}:
                bidir = False
                inner = call
                if ln == "Bidirectional":
                    inner = call.args[0] if call.args else None
                    if not isinstance(inner, ast.Call):
                        raise _Unresolvable("Bidirectional")
                    bidir = True
                    ln = _last_attr(inner)
                    args = [self.ceval(a, scope) for a in inner.args]
                    kwargs = {k.arg: self.ceval(k.value, scope) for k in inner.keywords if k.arg}
                units = args[0] if args else kwargs.get("units")
                if not isinstance(units, int) or rank != 2:
                    raise _Unresolvable(f"{ln} on rank-{rank} input")
                if kwargs.get("return_state"):
                    raise _Unresolvable(f"{ln}(return_state=True) (multi-output layer)")
                rs = bool(kwargs.get("return_sequences", False))
                mod = {"LSTM": "nn.LSTM", "GRU": "nn.GRU", "SimpleRNN": "nn.RNN"}[ln]
                helpers.add("rnn")
                out_units = units * (2 if bidir else 1)
                lines.append(f"_TemplateRNN({mod}({shape[1]}, {units}, batch_first=True{', bidirectional=True' if bidir else ''}), return_sequences={rs}),  # {txt}")
                shape = [shape[0], out_units] if rs else [out_units]
                rank = len(shape)
            elif ln in {"Conv1D", "MaxPooling1D", "AveragePooling1D"} and rank == 2:
                L, C = shape
                helpers.add("cl1d")
                if ln == "Conv1D":
                    filters = args[0] if args else kwargs.get("filters")
                    k = _pair(args[1] if len(args) > 1 else kwargs.get("kernel_size"), 1)
                    s = _pair(args[2] if len(args) > 2 else kwargs.get("strides", 1), 1)
                    if filters is None or k is None:
                        raise _Unresolvable("Conv1D args")
                    p = k[0] // 2 if kwargs.get("padding", "valid") == "same" else 0
                    nl = _conv_out(L, k[0], s[0], p)
                    lines.append(f"_TemplateChannelsLast1d(nn.Conv1d({C}, {filters}, kernel_size={k[0]}, stride={s[0]}, padding={p})),  # {txt} -> {(nl, filters)}")
                    shape = [nl, filters]
                    act_line(act, ln)
                else:
                    k = _pair(args[0] if args else kwargs.get("pool_size", 2), 1)
                    s = _pair(args[1] if len(args) > 1 else kwargs.get("strides"), 1) or k
                    nl = _conv_out(L, k[0], s[0], 0)
                    mod = "nn.MaxPool1d" if "Max" in ln else "nn.AvgPool1d"
                    lines.append(f"_TemplateChannelsLast1d({mod}({k[0]}, {s[0]})),  # {txt} -> {(nl, C)}")
                    shape = [nl, C]
            elif ln in {"GlobalAveragePooling1D", "GlobalMaxPooling1D"} and rank == 2:
                helpers.add("cl1d")
                mod = "nn.AdaptiveAvgPool1d(1)" if "Average" in ln else "nn.AdaptiveMaxPool1d(1)"
                lines.append(f"_TemplateChannelsLast1d({mod}),  # {txt}")
                lines.append(f"nn.Flatten(),  # -> ({shape[1]},)")
                shape = [shape[1]]
                rank = 1
            elif ln in {"RandomFlip", "RandomRotation", "RandomZoom", "RandomTranslation", "RandomContrast", "RandomCrop"}:
                lines.append(f"# {txt}: augmentation layer skipped (identity at inference)")
            else:
                raise _Unresolvable(f"{ln} on rank-{rank} input")
        out_dim = shape[-1]
        if lines and lines[-1].startswith("nn.Linear("):
            pass
        return lines, out_dim, helpers, dtype, final_act


# ---------------------------------------------------------------------------
# sklearn / XGBoost surrogate (R5.4)
# ---------------------------------------------------------------------------

_ESTIMATOR_SUFFIXES = ("Classifier", "Regressor", "SVC", "SVR", "Regression", "Perceptron",
                       "Ridge", "Lasso", "ElasticNet", "NB", "Booster")


@dataclass
class TabularResult:
    ok: bool = False
    estimator: str = ""
    task: str = "class"
    hidden_dims: tuple = (64, 64)
    standardize: bool = False
    class_src: str = ""
    notes: list = field(default_factory=list)


def find_estimator(index: SourceIndex, resolver: Resolver) -> TabularResult:
    res = TabularResult()
    for n in ast.walk(index.tree):
        if isinstance(n, ast.Call):
            last = _last_attr(n)
            d = _dotted(n.func)
            root = d.split(".")[0] if d else ""
            origin = index.import_full.get(root, "")
            if last.endswith(_ESTIMATOR_SUFFIXES) and any(p in origin or p in d for p in ("sklearn", "xgboost", "lightgbm", "catboost")):
                res.ok = True
                res.estimator = last
                res.task = "regression" if ("Regress" in last or last in {"SVR", "Ridge", "Lasso", "ElasticNet"}) else "class"
                h = _kw(n, "hidden_layer_sizes")
                if h is not None:
                    try:
                        v = resolver.ceval(h, index.scope_of(n))
                        s = _as_shape(v)
                        if s:
                            res.hidden_dims = s
                            res.notes.append(f"hidden_layer_sizes={s} mirrored from {last}")
                    except _Unresolvable:
                        pass
                break
    for n in ast.walk(index.tree):
        if isinstance(n, ast.Call) and _last_attr(n) in {"StandardScaler", "MinMaxScaler", "RobustScaler"}:
            res.standardize = True
    return res


def _surrogate_class_src(res: TabularResult, in_features: int, num_outputs: int) -> str:
    return f'''class TemplateTabularSurrogate(nn.Module):
    """TEMPLATE_FALLBACK = "surrogate": feed-forward stand-in for sklearn/XGBoost
    {res.estimator}(...) (rule R5.4). Tree/kernel estimators have no nn.Module form,
    so an MLP with hidden layers {tuple(res.hidden_dims)} is trained on the same tabular
    features with the matching objective."""

    def __init__(self, in_features: int = {in_features}, num_outputs: int = {num_outputs},
                 hidden_dims: tuple = {tuple(res.hidden_dims)}):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
'''


def _generic_mlp_src(input_shape: tuple, num_outputs: int, reason: str) -> str:
    n_in = int(math.prod(input_shape))
    return f'''class TemplateGenericMLP(nn.Module):
    """TEMPLATE_FALLBACK = "generic": the source model could not be translated by the
    rule set ({reason}). A two-layer MLP on the inferred input shape
    {tuple(input_shape)} stands in so that the FL pipeline can still be exercised."""

    def __init__(self, input_shape: tuple = {tuple(input_shape)}, num_outputs: int = {num_outputs}, hidden: int = 128):
        super().__init__()
        n_in = int(math.prod(input_shape))
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, num_outputs))

    def forward(self, x):
        out = self.net(x.float())
        return out.squeeze(-1) if out.shape[-1] == 1 else out
'''


# ---------------------------------------------------------------------------
# Lightning class cleanup (R5.2)
# ---------------------------------------------------------------------------

def _is_lightning_hook(name: str) -> bool:
    if name in {"forward", "__init__"}:
        return False
    return name.startswith(_LIGHTNING_HOOK_PREFIXES) or name.endswith(_LIGHTNING_HOOK_SUFFIXES)


class _LightningCleaner(ast.NodeTransformer):
    def __init__(self, index: SourceIndex):
        self.index = index
        self.notes: list[str] = []
        self._in_init = False
        self._init_params: list[str] = []

    def visit_ClassDef(self, node):
        new_bases = []
        for b in node.bases:
            if "Lightning" in _last_attr(b):
                new_bases.append(ast.parse("nn.Module", mode="eval").body)
                self.notes.append(f"R5.2 base {ast.unparse(b)} -> nn.Module")
            else:
                new_bases.append(b)
        node.bases = new_bases
        body = []
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and _is_lightning_hook(m.name):
                self.notes.append(f"R5.2 hook {m.name}() removed")
                continue
            body.append(self.visit(m))
        node.body = body or [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        self._in_init = node.name == "__init__"
        self._init_params = [a.arg for a in node.args.args if a.arg != "self"] + [a.arg for a in node.args.kwonlyargs]
        node = self.generic_visit(node)
        node.body = [s for s in node.body if s is not None] or [ast.Pass()]
        self._in_init = False
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "self":
            attr = node.value.func.attr
            if attr == "save_hyperparameters":
                ignore = _kw(node.value, "ignore")
                ign = set()
                if ignore is not None:
                    try:
                        v = const_eval(ignore, SourceIndex._no_lookup)
                        ign = set(v) if isinstance(v, (list, tuple, set)) else {v}
                    except _Unresolvable:
                        pass
                params = [p for p in self._init_params if p not in ign]
                kw = ", ".join(f"{p}={p}" for p in params)
                self.notes.append("R5.2 save_hyperparameters() -> SimpleNamespace hparams")
                return ast.parse(f"self.hparams = SimpleNamespace({kw})").body[0]
            if attr in _LIGHTNING_STRIP_CALLS:
                return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute) \
                and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "self" \
                and node.targets[0].attr in {"automatic_optimization", "example_input_array"}:
            return None
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr == "device":
            return ast.parse("next(self.parameters()).device", mode="eval").body
        return node


def clean_lightning_class(cls: ast.ClassDef, index: SourceIndex) -> tuple[str, list[str]]:
    cleaner = _LightningCleaner(index)
    node = cleaner.visit(copy.deepcopy(cls))
    ast.fix_missing_locations(node)
    return ast.unparse(node), cleaner.notes


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

_STD_IMPORTS = '''import math
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

try:
    from torchvision import datasets as _tvd, transforms as _tvt
except Exception:  # torchvision is optional for the synthetic path
    _tvd = _tvt = None
'''

_RUNTIME_HELPERS = '''
def _template_to_device(batch, device):
    """Move every tensor of a tuple/list/dict batch to `device`."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _template_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_template_to_device(v, device) for v in batch)
    return batch


def _template_unpack(batch):
    """(inputs, targets) from a tuple/list/dict batch."""
    if isinstance(batch, dict):
        vals = list(batch.values())
        return vals[0], (vals[1] if len(vals) > 1 else None)
    if isinstance(batch, (list, tuple)):
        return batch[0], (batch[1] if len(batch) > 1 else None)
    return batch, None


class _TemplateDictDataset(torch.utils.data.Dataset):
    """Synthetic dataset yielding dict batches with the keys the source loop uses."""

    def __init__(self, x, y, keys):
        self.x, self.y, self.keys = x, y, list(keys)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return {self.keys[0]: self.x[i], self.keys[1]: self.y[i]}
'''

_BUNDLE_CLASS = '''
class TemplateModelBundle(nn.Module):
    """R1.3: several networks are trained inside one loop (e.g. a GAN). They are
    bundled into one nn.Module so the FL runtime aggregates/optimises a single
    state_dict; forward() delegates to the network applied to the batch data."""

    def __init__(self, primary, **nets):
        super().__init__()
        self._primary = primary
        for k, v in nets.items():
            setattr(self, k, v)

    def forward(self, x):
        return getattr(self, self._primary)(x)
'''


@dataclass
class ConversionMeta:
    source: str = ""
    framework: str = ""
    detected: str = ""
    rules: list = field(default_factory=list)
    fallbacks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    model_fallback: str = "none"      # none | generic | surrogate
    train_step_mode: str = "extracted"
    data_candidates: int = 0
    shape: Optional[ShapeSpec] = None
    optimizer: dict = field(default_factory=dict)


class TemplateConverter:
    def __init__(self, source: str, path: str, framework: Optional[str] = None):
        self.index = SourceIndex(source, path)
        self.meta = ConversionMeta(source=path, framework=framework or "")
        self.detected = self._detect_framework()
        self.meta.detected = self.detected
        self.meta.framework = self.meta.framework or self.detected

    # ------------------------------------------------------------------
    def _detect_framework(self) -> str:
        pk = set(self.index.import_names.values())
        if pk & _KERAS_PKGS:
            return "keras"
        if pk & _LIGHTNING_PKGS:
            return "lightning"
        if "monai" in pk:
            return "monai"
        has_module_cls = any(_is_model_class(c)[0] for c in self.index.classes.values())
        if ("sklearn" in pk or "xgboost" in pk or "lightgbm" in pk) and not has_module_cls and "torch" not in pk:
            return "tabular"
        return "pytorch"

    def rule(self, s: str):
        if s not in self.meta.rules:
            self.meta.rules.append(s)

    def fallback(self, s: str):
        if s not in self.meta.fallbacks:
            self.meta.fallbacks.append(s)

    def warn(self, s: str):
        if s not in self.meta.warnings:
            self.meta.warnings.append(s)

    # ------------------------------------------------------------------
    def run(self) -> str:
        idx = self.index
        translate_keras = self.detected == "keras"
        # names available at module level of the output
        module_names = set()
        for bound, top in idx.import_names.items():
            if top in _LIGHTNING_PKGS and not idx.import_full.get(bound, "").endswith(".MNIST"):
                continue
            if top in _KERAS_PKGS and translate_keras:
                continue
            module_names.add(bound)
        module_names |= set(idx.functions) | set(idx.classes)
        module_names |= {"math", "os", "SimpleNamespace", "np", "torch", "nn", "F", "DataLoader",
                         "Dataset", "TensorDataset", "random_split", "_tvd", "_tvt"}
        resolver = Resolver(idx, module_names)
        self.resolver = resolver

        loop = find_training_loop(idx)
        models: list[ModelInfo] = []
        keras_res: Optional[KerasResult] = None
        tab_res: Optional[TabularResult] = None
        if translate_keras:
            keras_res = KerasTranslator(idx, resolver).translate()
        elif self.detected == "tabular":
            tab_res = find_estimator(idx, resolver)
        else:
            models = find_models(idx, resolver, loop)
        datasets = find_datasets(idx, resolver, loop)
        self.meta.data_candidates = len(datasets)

        # rename map for the loop body
        rename: dict[str, str] = {}
        if loop.kind == "lightning":
            rename["self"] = "model"
        elif loop.kind == "loop" and models:
            if len(models) == 1:
                if models[0].var != "model":
                    rename[models[0].var] = "model"
            else:
                for m in models:
                    rename[m.var] = f"model.{m.var}"

        ts = extract_train_step(idx, resolver, loop, rename, module_names) if loop.kind != "none" else TrainStepResult(False, reason="no training loop")
        loss = find_loss(idx, resolver, loop, ts.loss_stmts if ts.ok else [])
        explicit = {}
        if keras_res is not None and keras_res.input_shape:
            explicit = {"input_shape": keras_res.input_shape, "num_outputs": keras_res.num_outputs, "source": "keras.Input"}
        if tab_res is not None:
            for c in datasets:
                if c.name in _DATASET_CATALOGUE:
                    shp, ncls = _DATASET_CATALOGUE[c.name]
                    explicit = {"input_shape": shp, "num_outputs": ncls, "source": f"sklearn.{c.name}"}
                    break
        shape = infer_shapes(idx, resolver, models, datasets, loop, loss, explicit)
        if keras_res is not None and keras_res.unresolved_shape:
            shape.unresolved = True
        if keras_res is not None and keras_res.input_dtype == "long":
            shape.input_dtype = "long"
        self.meta.shape = shape
        self.meta.optimizer = find_optimizer(idx, resolver)

        # ---- sections ----
        sections: list[str] = []
        helper_classes: list[str] = []
        build_model_src = self._emit_build_model(models, keras_res, tab_res, shape, loss, helper_classes)
        data_src = self._emit_build_dataloader(datasets, shape, loss)
        train_src = self._emit_train_step(ts, loss, shape, loop)

        # ---- kept definitions (closure) ----
        kept_defs = self._kept_definitions("\n\n".join([build_model_src, data_src, train_src] + helper_classes), models)

        # ---- module-level constants needed by kept classes ----
        const_block = "\n".join(resolver.module_ctx.stmts)

        body_parts = [_RUNTIME_HELPERS]
        if const_block:
            body_parts.append("\n# Module-level constants folded from the source (R3.5/R6)\n" + const_block + "\n")
        if kept_defs:
            body_parts.append("\n# ---- definitions kept from the source ----\n" + "\n\n\n".join(kept_defs) + "\n")
        if helper_classes:
            body_parts.append("\n# ---- classes generated by the template converter ----\n" + "\n\n".join(helper_classes) + "\n")
        body_parts.append("\n# ---- FL client interface ----\n" + build_model_src + "\n\n" + data_src + "\n\n" + train_src + "\n")
        body_parts.append(self._emit_optimizer_helper())
        body = "\n".join(body_parts)

        imports = self._emit_imports(body, translate_keras)
        header = self._emit_header()
        future = ("\n".join(idx.future_imports) + "\n") if idx.future_imports else ""
        code = future + header + "\n" + _STD_IMPORTS + "\n" + imports + "\n" + body
        # final syntax check
        ast.parse(code)
        return code

    # ------------------------------------------------------------------
    def _emit_build_model(self, models, keras_res, tab_res, shape, loss, helper_classes) -> str:
        r = self.resolver
        out_dim = shape.num_classes
        if loss.kind == "binary":
            out_dim = 1
        elif loss.kind == "regression":
            out_dim = shape.out_dim
        elif loss.kind == "segmentation":
            out_dim = shape.target_channels
        doc = []
        lines = ["def build_model(config: dict) -> nn.Module:", "    __DOC__", '    kwargs = dict(config.get("model_kwargs", {}))']
        if keras_res is not None:
            if keras_res.ok:
                self.rule("R5.1 keras layer stack -> torch.nn.Sequential (static NHWC->NCHW shape tracking)")
                for h in sorted(keras_res.helpers):
                    helper_classes.append(_KERAS_HELPERS[h])
                helper_classes.append(keras_res.class_src)
                for n in keras_res.notes:
                    self.rule(n)
                if keras_res.final_activation:
                    self.rule(f"R5.1 final {keras_res.final_activation} folded into the loss (model returns logits)")
                doc.append("Model translated from the Keras layer stack (R5.1).")
                lines.append(f'    kwargs.setdefault("num_outputs", {keras_res.num_outputs})')
                lines.append("    return TemplateKerasModel(**kwargs)")
            else:
                self.fallback(f"generic model: {keras_res.fallback_reason}")
                self.meta.model_fallback = "generic"
                helper_classes.append(_generic_mlp_src(shape.input_shape, out_dim, keras_res.fallback_reason))
                doc.append("TEMPLATE_FALLBACK='generic' (see class docstring).")
                lines.append(f'    kwargs.setdefault("input_shape", {tuple(shape.input_shape)})')
                lines.append(f'    kwargs.setdefault("num_outputs", {out_dim})')
                lines.append("    return TemplateGenericMLP(**kwargs)")
        elif tab_res is not None:
            self.rule(f"R5.4 sklearn/XGBoost estimator {tab_res.estimator or '(none found)'} -> MLP surrogate on {shape.input_shape}")
            for n in tab_res.notes:
                self.rule("R5.4 " + n)
            self.fallback("surrogate model for a non-neural estimator")
            self.meta.model_fallback = "surrogate"
            helper_classes.append(_surrogate_class_src(tab_res, shape.input_shape[0], out_dim))
            doc.append("TEMPLATE_FALLBACK='surrogate' (see class docstring).")
            lines.append(f'    kwargs.setdefault("in_features", {shape.input_shape[0]})')
            lines.append(f'    kwargs.setdefault("num_outputs", {out_dim})')
            lines.append("    return TemplateTabularSurrogate(**kwargs)")
        elif not models:
            self.fallback("generic model: no nn.Module class or model constructor found")
            self.meta.model_fallback = "generic"
            helper_classes.append(_generic_mlp_src(shape.input_shape, out_dim, "no model found"))
            lines.append(f'    kwargs.setdefault("input_shape", {tuple(shape.input_shape)})')
            lines.append(f'    kwargs.setdefault("num_outputs", {out_dim})')
            lines.append("    return TemplateGenericMLP(**kwargs)")
        else:
            ctx = Ctx("model")
            built = []
            for m in models:
                ctor_src = self._rewrite_ctor(m, ctx, add_kwargs=(len(models) == 1))
                var = m.var if m.var not in {"self", "model"} or len(models) > 1 else "model"
                if len(models) == 1:
                    var = "model"
                built.append((var, m, ctor_src))
            body = []
            for s in ctx.stmts:
                body.append("    " + s)
            for var, m, ctor_src in built:
                body.append(f"    {var} = {ctor_src}")
                for ap in m.apply_calls:
                    fn = ap.args[0] if ap.args else None
                    if isinstance(fn, ast.Name) and fn.id in self.index.functions:
                        body.append(f"    {var}.apply({fn.id})")
                        self.rule(f"R1.4 kept post-construction call {m.var}.apply({fn.id})")
                where = f"{m.scope.func.name}()" if m.scope.func is not None else "module level"
                if m.is_lightning:
                    self.rule(f"R5.2 LightningModule {m.cls_name} used as the model (hooks stripped)")
                elif m.is_local_class:
                    self.rule(f"R1.1 model class {m.cls_name} (instantiated at {where} as {m.var})")
                else:
                    self.rule(f"R1.2 library model constructor kept verbatim: {m.cls_name} ({where})")
            if len(models) > 1:
                self.rule("R1.3 several networks trained in one loop -> TemplateModelBundle")
                helper_classes.append(_BUNDLE_CLASS)
                names = ", ".join(f"{v}={v}" for v, _, _ in built)
                primary = built[0][0]
                body.append(f'    return TemplateModelBundle("{primary}", {names})')
            else:
                body.append("    return model")
            lines += body
            if ctx.hoisted:
                self.rule(f"R3.5 constructor arguments resolved from enclosing scopes / argparse defaults: {sorted(set(ctx.hoisted))}")
        doc.append("Returns the training model; config['model_kwargs'] overrides constructor kwargs.")
        src = "\n".join(lines).replace("    __DOC__", '    """' + " ".join(doc) + '"""')
        return src

    def _rewrite_ctor(self, m: ModelInfo, ctx: Ctx, add_kwargs: bool = True) -> str:
        r = self.resolver
        ctor = m.ctor
        try:
            node = r.rewrite(ctor, m.scope, ctx)
        except _Unresolvable as e:
            # retry: drop keyword arguments that cannot be resolved, then positionals
            if isinstance(ctor, ast.Call):
                node = copy.deepcopy(ctor)
                kept_kw = []
                for k in node.keywords:
                    try:
                        k.value = r.rewrite(k.value, m.scope, ctx)
                        kept_kw.append(k)
                    except _Unresolvable:
                        self.warn(f"dropped unresolvable constructor kwarg {k.arg} of {m.cls_name} (class default used)")
                node.keywords = kept_kw
                kept_args = []
                for a in node.args:
                    try:
                        kept_args.append(r.rewrite(a, m.scope, ctx))
                    except _Unresolvable:
                        self.warn(f"dropped unresolvable positional constructor arg {ast.unparse(a)} of {m.cls_name}")
                        kept_args = []
                        break
                node.args = kept_args
                try:
                    node.func = r.rewrite(node.func, m.scope, ctx)
                except _Unresolvable:
                    self.warn(f"model constructor {ast.unparse(node.func)} could not be resolved")
            else:
                self.warn(f"model constructor could not be resolved: {e}")
                node = ctor
        if add_kwargs and isinstance(node, ast.Call) and not any(k.arg is None for k in node.keywords):
            node.keywords.append(ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load())))
        return ast.unparse(node)

    # ------------------------------------------------------------------
    def _emit_build_dataloader(self, datasets: list[DataCandidate], shape: ShapeSpec, loss: LossInfo) -> str:
        r = self.resolver
        idx = self.index
        lines = [
            'def build_dataloader(config: dict, split: str = "train") -> torch.utils.data.DataLoader:',
            '    """R2: try the dataset constructions found in the source (data root bound to',
            '    config["data_path"]); each is probed (len + first item) inside try/except. When',
            '    none works, synthetic data is used only if config["allow_synthetic_data"]."""',
            '    local = config.get("local", {})',
            '    batch_size = int(local.get("batch_size", config.get("batch_size", 16)))',
            '    num_workers = int(local.get("num_workers", 0))',
            '    data_path = str(config.get("data_path", "."))',
            '    seed = int(config.get("seed", 42))',
            '    dataset, errors, collate = None, [], None',
        ]
        n_emitted = 0
        for i, c in enumerate(datasets, 1):
            block = self._candidate_block(c, shape)
            if block is None:
                continue
            n_emitted += 1
            lines.append(f"    # candidate {n_emitted}: {c.kind} {c.name} (source line {c.call.lineno}{', feeds the training loader' if c.linked else ''})")
            lines.append("    if dataset is None:")
            lines.append("        try:")
            for s in block:
                lines.append("            " + s)
            lines.append("            _n = len(_ds)")
            lines.append('            if _n <= 0:')
            lines.append('                raise ValueError("empty dataset")')
            lines.append("            _ = _ds[0]")
            lines.append("            dataset = _ds")
            lines.append("        except Exception as e:  # noqa: BLE001")
            lines.append(f'            errors.append("candidate {n_emitted} ({c.name}): %s: %s" % (type(e).__name__, str(e)[:120]))')
        if n_emitted == 0:
            self.fallback("no usable dataset construction found; synthetic data only")
            self.warn("no dataset candidate could be emitted; build_dataloader relies on synthetic data")
        else:
            self.rule(f"R2.1 {n_emitted} dataset candidate(s) emitted{' (data-flow linked to the training loader)' if any(c.linked for c in datasets) else ''}")
        lines += [
            "    if dataset is None:",
            '        if not config.get("allow_synthetic_data", False):',
            '            raise FileNotFoundError(',
            '                "no dataset could be built from data_path=%r (%s) and allow_synthetic_data is False"',
            '                % (data_path, "; ".join(errors) or "no candidates"))',
            "        dataset = _template_synthetic_dataset(config, seed)",
            "    n_total = len(dataset)",
            "    n_val = max(1, int(0.1 * n_total)) if n_total >= 2 else 0",
            "    train_ds, val_ds = torch.utils.data.random_split(",
            "        dataset, [n_total - n_val, n_val], generator=torch.Generator().manual_seed(seed))",
            '    chosen = train_ds if split == "train" else val_ds',
            "    return torch.utils.data.DataLoader(",
            '        chosen, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers,',
            '        collate_fn=collate, drop_last=(split == "train" and len(chosen) > batch_size))',
        ]
        self.rule("R2.3 candidates probed in try/except; synthetic fallback gated on allow_synthetic_data")
        return "\n".join(lines) + "\n\n\n" + self._emit_synthetic(shape, loss)

    def _candidate_block(self, c: DataCandidate, shape: ShapeSpec) -> Optional[list[str]]:
        r = self.resolver
        ctx = Ctx("data", defined={"data_path", "batch_size", "seed", "split", "config", "num_workers", "local"})
        call = copy.deepcopy(c.call)
        if c.kind == "keras":
            if c.name in _DATASET_CATALOGUE and c.name.islower():
                cls = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST", "cifar10": "CIFAR10", "cifar100": "CIFAR100"}.get(c.name)
                if cls is None:
                    return None
                self.rule(f"R2.4 keras.datasets.{c.name}.load_data() -> torchvision.datasets.{cls}")
                return [f"_ds = _tvd.{cls}(data_path, train=True, download=True, transform=_tvt.ToTensor())"]
            if c.name == "image_dataset_from_directory":
                d = call.args[0] if call.args else _kw(call, "directory")
                size = _kw(call, "image_size")
                try:
                    d2 = r.rewrite(d, c.scope, ctx) if d is not None else None
                except _Unresolvable:
                    d2 = None
                if isinstance(d2, ast.Constant) and isinstance(d2.value, str):
                    root_src = f"os.path.join(data_path, {d2.value!r})"
                elif d2 is not None:
                    root_src = ast.unparse(d2)
                else:
                    root_src = "data_path"
                try:
                    sz = r.ceval(size, c.scope) if size is not None else None
                except _Unresolvable:
                    sz = None
                sz_txt = f"_tvt.Resize({tuple(sz) if isinstance(sz, (list, tuple)) else sz}), " if sz else ""
                self.rule("R2.4 keras image_dataset_from_directory -> torchvision ImageFolder(+Resize, ToTensor)")
                return ctx.stmts + [f"_ds = _tvd.ImageFolder({root_src}, transform=_tvt.Compose([{sz_txt}_tvt.ToTensor()]))"]
            return None
        if c.kind == "lightning":
            base = c.name.replace("DataModule", "")
            if base in _TORCHVISION_DATASETS:
                self.rule(f"R2.4 lightning demo {c.name} -> torchvision.datasets.{base}")
                return [f"_ds = _tvd.{base}(data_path, train=True, download=True, transform=_tvt.ToTensor())"]
            return None
        if c.kind == "sklearn":
            try:
                node = r.rewrite(call, c.scope, ctx)
            except _Unresolvable as e:
                self.warn(f"sklearn loader {c.name} not emitted: {e}")
                return None
            y_cast = ".float()" if shape.target_kind in {"binary", "regression"} else ".long()"
            std = self.detected == "tabular" and find_estimator(self.index, r).standardize
            lines = ctx.stmts + [
                f"_raw = {ast.unparse(node)}",
                '_X = np.asarray(getattr(_raw, "data", None) if hasattr(_raw, "data") else _raw[0], dtype="float32")',
                '_y = np.asarray(getattr(_raw, "target", None) if hasattr(_raw, "target") else _raw[1])',
            ]
            if std:
                lines.append("_X = (_X - _X.mean(axis=0)) / (_X.std(axis=0) + 1e-8)  # StandardScaler mirrored")
                self.rule("R2.4 StandardScaler mirrored as per-feature standardisation")
            lines.append(f"_ds = torch.utils.data.TensorDataset(torch.from_numpy(_X), torch.from_numpy(_y){y_cast})")
            self.rule(f"R2.4 sklearn.datasets.{c.name}() -> TensorDataset")
            return lines
        # torch / torchvision / monai / local dataset constructor
        root_done = False
        if c.kind == "torchvision" or c.kind == "local":
            for k in call.keywords:
                if k.arg in {"root", "directory", "data_dir", "data_root", "path", "data_path"} and isinstance(k.value, ast.Constant):
                    k.value = ast.Name(id="data_path", ctx=ast.Load())
                    root_done = True
            if not root_done and call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                call.args[0] = ast.Name(id="data_path", ctx=ast.Load())
                root_done = True
        try:
            node = r.rewrite(call, c.scope, ctx)
        except _Unresolvable as e:
            self.warn(f"dataset candidate {c.name} (line {c.call.lineno}) not emitted: unresolvable {e}")
            return None
        roots = sorted({n.split(":", 1)[1] for n in ctx.notes if n.startswith(("root:", "path-arg:"))})
        argp = sorted({n.split(":", 1)[1] for n in ctx.notes if n.startswith("argparse:")})
        if root_done or roots:
            self.rule("R2.2 data root bound to config['data_path'] (" + ", ".join(roots or ["literal root"]) + ")")
        if argp:
            self.rule(f"R3.5 argparse defaults folded in the data pipeline: {argp}")
        if ctx.hoisted:
            self.rule(f"R3.5 data pipeline definitions hoisted: {sorted(set(ctx.hoisted))}")
        lines = ctx.stmts + [f"_ds = {ast.unparse(node)}"]
        if c.collate is not None:
            cf_expr, cf_scope = c.collate
            try:
                cf = r.rewrite(cf_expr, cf_scope, ctx)
                lines += ctx.stmts[len(ctx.stmts):] + [f"collate = {ast.unparse(cf)}"]
                self.rule(f"R2.5 DataLoader collate_fn={ast.unparse(cf_expr)} preserved (batch structure)")
            except _Unresolvable as e:
                self.warn(f"collate_fn of the training DataLoader could not be resolved: {e}")
        return lines

    def _emit_synthetic(self, shape: ShapeSpec, loss: LossInfo) -> str:
        n_elem = int(math.prod(shape.input_shape))
        n_default = max(_SYNTHETIC_MIN_SAMPLES, min(_SYNTHETIC_MAX_SAMPLES, _SYNTHETIC_ELEMENT_BUDGET // max(1, n_elem)))
        self.rule("R4 synthetic shape " + f"{tuple(shape.input_shape)} / target={shape.target_kind}" + (f"[{shape.num_classes}]" if shape.target_kind == "class" else "") + " from: " + "; ".join(shape.evidence[:6]))
        if shape.unresolved:
            self.fallback("synthetic input shape could not be fully resolved from the source; defaults used")
        if shape.input_dtype == "long":
            x_line = f"    x = torch.randint(0, {max(2, shape.num_classes)}, (n, *input_shape), generator=g)"
        else:
            x_line = "    x = torch.randn((n, *input_shape), generator=g)"
        if shape.target_kind == "class":
            y_line = f"    y = torch.randint(0, {shape.num_classes}, (n,), generator=g)"
        elif shape.target_kind == "binary":
            y_line = "    y = torch.randint(0, 2, (n,), generator=g).float()"
        elif shape.target_kind == "regression":
            y_line = f"    y = torch.randn((n, {shape.out_dim}), generator=g)" if shape.out_dim > 1 else "    y = torch.randn((n,), generator=g)"
        else:
            y_line = f"    y = (torch.rand((n, {shape.target_channels}, *input_shape[1:]), generator=g) > 0.5).float()"
        ret = f"    return _TemplateDictDataset(x, y, {shape.batch_keys!r})" if shape.batch_format == "dict" else "    return torch.utils.data.TensorDataset(x, y)"
        if shape.batch_format == "dict":
            self.rule(f"R5.3 dict batches preserved with keys {shape.batch_keys}")
        return "\n".join([
            "def _template_synthetic_dataset(config: dict, seed: int = 42):",
            '    """R4 synthetic stand-in (only used when allow_synthetic_data=True).',
            f"    input {tuple(shape.input_shape)} ({shape.input_dtype}); target kind {shape.target_kind!r}.",
            "    Evidence: " + "; ".join(shape.evidence) + '"""',
            f'    input_shape = tuple(config.get("synthetic_input_shape", {tuple(shape.input_shape)}))',
            f'    n = int(config.get("synthetic_num_samples", {n_default}))',
            "    g = torch.Generator().manual_seed(seed)",
            x_line, y_line, ret,
        ])

    # ------------------------------------------------------------------
    def _emit_train_step(self, ts: TrainStepResult, loss: LossInfo, shape: ShapeSpec, loop: LoopInfo) -> str:
        lines = ["def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:"]
        if ts.ok:
            self.meta.train_step_mode = "extracted"
            where = "LightningModule.training_step" if loop.kind == "lightning" else f"training loop at line {loop.node.lineno}" + (f" in {loop.scope.func.name}()" if loop.scope and loop.scope.func else " (module level)")
            self.rule(f"R3.1 train_step extracted from the {where}")
            for n in ts.notes:
                self.rule(n)
            for d in ts.dropped:
                self.warn("train_step: " + d)
            lines.append(f'    """R3: forward + loss extracted from the {where}. The FL runtime performs')
            lines.append('    zero_grad/backward/step; this function must only return the loss tensor."""')
            lines.append("    device = next(model.parameters()).device")
            lines.append("    batch = _template_to_device(batch, device)")
            for s in ts.prologue:
                lines.append(_indent(s))
            for s in ts.body:
                lines.append(_indent(s))
            lines.append(f"    return {ts.return_expr}")
            return "\n".join(lines)
        # generic
        self.meta.train_step_mode = "generic"
        self.fallback(f"generic train_step ({ts.reason})")
        self.rule(f"R3.8 generic train_step with detected loss {loss.name} ({loss.origin})")
        lines.append('    """R3.8 generic forward/loss (the loop body could not be isolated: ' + ts.reason.replace('"', "'") + ').')
        lines.append('    The FL runtime performs zero_grad/backward/step."""')
        lines.append("    device = next(model.parameters()).device")
        lines.append("    x, y = _template_unpack(batch)")
        lines.append("    x = x.to(device)")
        lines.append("    y = y.to(device) if isinstance(y, torch.Tensor) else y")
        lines.append("    logits = model(x)")
        expr = loss.expr_src
        kind = loss.kind
        if kind == "class":
            fn = expr if expr.startswith(("F.", "nn.")) else "F.cross_entropy"
            lines.append(f"    loss = {fn}(logits, y.long())")
        elif kind == "binary":
            if loss.name in {"BCELoss", "binary_cross_entropy"}:
                lines.append("    loss = F.binary_cross_entropy(logits.reshape(-1).clamp(1e-6, 1 - 1e-6), y.float().reshape(-1))")
            else:
                lines.append("    loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), y.float().reshape(-1))")
        elif kind == "regression":
            fn = expr if expr.startswith(("F.", "nn.")) else "F.mse_loss"
            lines.append(f"    loss = {fn}(logits.reshape(logits.size(0), -1), y.float().reshape(y.size(0), -1))")
        else:
            if expr and "\n" not in expr:
                lines.append(f"    loss = {expr}(logits, y.float())")
            else:
                lines.append("    loss = F.binary_cross_entropy_with_logits(logits, y.float())")
        lines.append("    return loss")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _kept_definitions(self, emitted_code: str, models: list[ModelInfo]) -> list[str]:
        idx = self.index
        r = self.resolver
        try:
            needed = _load_names(ast.parse(emitted_code))
        except SyntaxError as e:  # should not happen; keep everything referenced by name as a fallback
            self.warn(f"internal: emitted code failed to parse during closure computation ({e})")
            needed = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", emitted_code))
        for m in models:
            if m.is_local_class:
                needed.add(m.cls_name)
        keep: dict[str, str] = {}
        order: dict[str, int] = {}
        queue = [n for n in needed if n in idx.functions or n in idx.classes]
        seen = set()
        while queue:
            n = queue.pop()
            if n in seen:
                continue
            seen.add(n)
            node = idx.classes.get(n) or idx.functions.get(n)
            if node is None:
                continue
            if isinstance(node, ast.ClassDef) and _is_model_class(node)[1]:
                src, notes = clean_lightning_class(node, idx)
                for x in notes:
                    self.rule(x)
                fn = _def_free_names(ast.parse(src))
            else:
                src = _src(idx.lines, node)
                fn = _def_free_names(node)
            keep[n] = src
            order[n] = node.lineno
            for f in fn:
                if f in idx.functions or f in idx.classes:
                    queue.append(f)
                elif f not in r.module_names and f not in _BUILTINS:
                    if r.materialize(f, idx.module_scope, r.module_ctx):
                        self.rule(f"R6 module-level constant {f} folded for {n}")
                    else:
                        self.warn(f"{n} references {f!r}, which could not be resolved at module level")
        return [keep[n] for n in sorted(keep, key=lambda k: order[k])]

    # ------------------------------------------------------------------
    def _emit_imports(self, body: str, translate_keras: bool) -> str:
        idx = self.index
        try:
            used = _load_names(ast.parse(body))
        except SyntaxError:
            used = set()
        out, dropped = [], []
        for node, conditional in idx.imports:
            top = ""
            if isinstance(node, ast.Import):
                names = []
                for a in node.names:
                    bound = a.asname or a.name.split(".")[0]
                    top = a.name.split(".")[0]
                    if top in _LIGHTNING_PKGS or (top in _KERAS_PKGS and translate_keras):
                        dropped.append(a.name)
                        continue
                    if bound in used:
                        names.append(a)
                if not names:
                    continue
                stmt = ast.Import(names=names)
            else:
                mod = node.module or ""
                top = mod.split(".")[0]
                if node.level:
                    continue
                if top in _LIGHTNING_PKGS:
                    for a in node.names:
                        if a.name == "MNIST" and (a.asname or "MNIST") in used:
                            out.append("try:\n    from torchvision.datasets import MNIST  # R2.4 lightning demo MNIST alias\nexcept Exception:\n    MNIST = None")
                            self.rule("R2.4 lightning.pytorch.demos MNIST -> torchvision.datasets.MNIST")
                    dropped.append(mod)
                    continue
                if top in _KERAS_PKGS and translate_keras:
                    dropped.append(mod)
                    continue
                names = [a for a in node.names if (a.asname or a.name) in used]
                if not names:
                    continue
                stmt = ast.ImportFrom(module=node.module, names=names, level=0)
            txt = ast.unparse(stmt)
            if conditional:
                bound = [a.asname or a.name.split(".")[0] for a in stmt.names]
                out.append("try:\n    " + txt + "\nexcept Exception:  # R6 conditional import in the source\n    " + " = ".join(bound) + " = None")
            else:
                out.append(txt)
        if dropped:
            self.rule("R6 unavailable/irrelevant framework imports dropped: " + ", ".join(sorted(set(dropped))))
        self.rule("R6 imports tree-shaken to the names referenced by the generated module")
        return "\n".join(dict.fromkeys(out))

    def _emit_optimizer_helper(self) -> str:
        o = self.meta.optimizer
        self.rule(f"R1 optimizer constructor detected: {o['class']}({o['kwargs_src']}) [{o['origin']}]")
        kw = o["kwargs_src"]
        ok = True
        try:
            ast.parse(f"f({kw})")
        except SyntaxError:
            ok = False
        body = [
            "",
            "def build_optimizer(model: nn.Module, config: dict):",
            '    """Optimizer as declared in the source (informational: the FL runtime owns the optimizer)."""',
            '    lr = config.get("learning_rate", None)',
            "    try:",
            f"        opt = torch.optim.{o['class']}(model.parameters(){', ' + kw if (kw and ok) else ''})",
            "    except Exception:",
            "        opt = torch.optim.AdamW(model.parameters(), lr=lr or 1e-3)",
            "    if lr is not None:",
            "        for g in opt.param_groups:",
            '            g["lr"] = lr',
            "    return opt",
            "",
        ]
        return "\n".join(body)

    # ------------------------------------------------------------------
    def _emit_header(self) -> str:
        m = self.meta
        lines = ["# " + "=" * 76,
                 "#  GENERATED BY autofl.converter.template_converter  (rule-based template converter;",
                 "#  NO LLM was involved in producing this file)",
                 f"#  Source    : {m.source}",
                 f"#  Framework : {m.framework} (detected: {m.detected})",
                 f"#  Model     : {m.model_fallback if m.model_fallback != 'none' else 'from source'};  train_step: {m.train_step_mode}",
                 "#",
                 "#  Rules fired:"]
        for r in m.rules:
            lines.append("#    - " + r)
        lines.append("#  Fallbacks used:")
        for f in (m.fallbacks or ["none"]):
            lines.append("#    - " + f)
        if m.warnings:
            lines.append("#  Warnings:")
            for w in m.warnings:
                lines.append("#    - " + w)
        lines.append("# " + "=" * 76)
        lines.append('TEMPLATE_CONVERTER = "autofl.converter.template_converter"')
        lines.append(f"TEMPLATE_FALLBACK = {m.model_fallback!r}")
        lines.append(f"TEMPLATE_TRAIN_STEP = {m.train_step_mode!r}")
        lines.append(f"TEMPLATE_RULES = {m.rules!r}")
        lines.append(f"TEMPLATE_FALLBACKS = {m.fallbacks!r}")
        lines.append(f"TEMPLATE_WARNINGS = {m.warnings!r}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_source(source: str, path: str = "<string>", framework: Optional[str] = None) -> tuple[str, ConversionMeta]:
    conv = TemplateConverter(source, path, framework)
    code = conv.run()
    return code, conv.meta


def convert(source_path: str | Path, output_path: str | Path | None = None,
            framework: Optional[str] = None, verbose: bool = True) -> Path:
    """Convert `source_path` into an FL client module at `output_path`
    (default: <stem>_fl_template.py next to the source). Returns the output path."""
    src = Path(source_path)
    out = Path(output_path) if output_path else src.parent / f"{src.stem}_fl_template.py"
    try:
        rel = str(src.resolve().relative_to(Path(__file__).resolve().parent.parent))
    except ValueError:
        rel = str(src)
    code, meta = convert_source(src.read_text(), rel, framework)
    out.write_text(code)
    if verbose:
        print(f"  [template] {src.name} -> {out.name}: model={meta.model_fallback}, train_step={meta.train_step_mode}, "
              f"rules={len(meta.rules)}, fallbacks={len(meta.fallbacks)}")
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Rule-based (template) converter for AutoFL")
    p.add_argument("source")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--framework", default=None)
    a = p.parse_args(argv)
    convert(a.source, a.output, a.framework)


if __name__ == "__main__":
    main()
