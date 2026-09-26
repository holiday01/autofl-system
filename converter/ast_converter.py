"""
AST-based converter: takes a standard PyTorch training script and generates
an FL-compatible client module exposing:
  - build_model(config) -> nn.Module
  - build_dataloader(config, split) -> DataLoader
  - train_step(model, batch, optimizer, config) -> loss

The converter does NOT require the original script to follow any specific
template — it uses heuristics to identify model class, dataset class,
training loop, and loss computation.
"""
import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# -----------------------------------------------------------------------
# Analysis result
# -----------------------------------------------------------------------

@dataclass
class AnalysisResult:
    model_class: str = ""
    dataset_class: str = ""
    loss_calls: list[str] = field(default_factory=list)
    optimizer_class: str = "AdamW"
    train_fn: str = ""
    imports: list[str] = field(default_factory=list)
    extra_classes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------
# AST helpers
# -----------------------------------------------------------------------

def _get_base_names(node: ast.ClassDef) -> list[str]:
    """Return list of base class name strings for a ClassDef."""
    names = []
    for b in node.bases:
        if isinstance(b, ast.Attribute):
            names.append(b.attr)
        elif isinstance(b, ast.Name):
            names.append(b.id)
    return names


def _find_classes(tree: ast.AST) -> list[ast.ClassDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def _find_functions(tree: ast.AST) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _find_assignments(tree: ast.AST, name: str) -> list[ast.Assign]:
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    results.append(node)
    return results


def _imports_to_source(tree: ast.AST) -> list[str]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node))
    return lines


# -----------------------------------------------------------------------
# Analyser
# -----------------------------------------------------------------------

class ScriptAnalyzer:
    """Analyse a PyTorch training script and extract key components."""

    MODEL_BASE_PATTERNS = {"Module", "nn.Module", "LightningModule"}
    DATASET_BASE_PATTERNS = {"Dataset", "IterableDataset", "TensorDataset",
                             "ImageFolder", "VisionDataset", "MNIST", "CIFAR10"}
    LOSS_FN_PATTERNS = {"CrossEntropyLoss", "BCELoss", "MSELoss", "NLLLoss",
                        "BCEWithLogitsLoss", "L1Loss", "HuberLoss",
                        "cross_entropy", "binary_cross_entropy", "mse_loss",
                        "nll_loss", "criterion"}

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)

    def analyse(self) -> AnalysisResult:
        result = AnalysisResult()
        result.imports = _imports_to_source(self.tree)

        classes = _find_classes(self.tree)
        for cls in classes:
            bases = _get_base_names(cls)
            if any(b in self.MODEL_BASE_PATTERNS for b in bases):
                if not result.model_class:
                    result.model_class = cls.name
                else:
                    result.extra_classes.append(cls.name)
            elif any(b in self.DATASET_BASE_PATTERNS for b in bases):
                if not result.dataset_class:
                    result.dataset_class = cls.name

        if not result.model_class and classes:
            result.model_class = classes[0].name
            result.warnings.append(
                f"Could not identify model class; guessing '{result.model_class}'"
            )

        # find loss references
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and node.id in self.LOSS_FN_PATTERNS:
                result.loss_calls.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in self.LOSS_FN_PATTERNS:
                result.loss_calls.append(node.attr)
        result.loss_calls = list(dict.fromkeys(result.loss_calls))

        # find optimizer
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                "Adam", "AdamW", "SGD", "RMSprop", "Adagrad"
            }:
                result.optimizer_class = node.attr
                break
            if isinstance(node, ast.Name) and node.id in {
                "Adam", "AdamW", "SGD", "RMSprop"
            }:
                result.optimizer_class = node.id
                break

        # find training function
        for fn in _find_functions(self.tree):
            if "train" in fn.name.lower() and fn.name != "__init__":
                result.train_fn = fn.name
                break

        return result


# -----------------------------------------------------------------------
# Code generator
# -----------------------------------------------------------------------

FL_CLIENT_TEMPLATE = '''\
{future_imports}"""
Auto-generated FL client module.
Original script: {original_path}

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
{imports}
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── Original source (unchanged) ────────────────────────────────────────
{original_source}

# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate the model. Override __init__ kwargs via config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {{}})
    return {model_class}(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for the requested split.
    Expects config to have 'data_path' and optionally 'val_ratio'.
    Client-local batch_size and num_workers are read from config['local'].
    """
    local = config.get("local", {{}})
    batch_size  = local.get("batch_size", config.get("batch_size", 16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    dataset_kwargs = config.get("dataset_kwargs", {{}})
    data_path = config.get("data_path", ".")
{dataset_init}
    val_ratio = config.get("val_ratio", 0.1)
    n_val = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )
    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Forward pass only — returns loss WITH grad attached.
    Backward pass and optimizer.step() are managed by the FL runtime
    (local_train in FLClient) to support gradient accumulation and AMP.
    The optimizer argument is accepted for API compatibility but ignored here.
    """
    local   = config.get("local", {{}})
    use_amp = local.get("use_amp", False)
    device  = next(model.parameters()).device

    # Move batch to device — supports (inputs, targets) or dict batches
    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {{k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {{type(batch)}}")

    with torch.autocast(device_type=device.type if hasattr(device, "type") else str(device),
                        enabled=use_amp):
        outputs = model(inputs)
{loss_computation}
    return loss
'''

DATASET_INIT_WITH_CLASS = """\
    full_dataset = {dataset_class}(root=data_path, **dataset_kwargs)"""

DATASET_INIT_GENERIC = """\
    # TODO: replace with your actual Dataset class
    # full_dataset = YourDataset(root=data_path, **dataset_kwargs)
    raise NotImplementedError(
        "build_dataloader: no Dataset class detected — "
        "edit the generated file and replace this block."
    )"""

LOSS_WITH_CRITERION = """\
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)"""

LOSS_GENERIC = """\
        # Auto-detected loss: {loss_calls}
        loss = {first_loss}(outputs, targets)"""


_KEEP_TOP_LEVEL = (
    ast.Import, ast.ImportFrom,
    ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
)


def _strip_import_unsafe_code(source: str) -> str:
    """
    Return a version of source safe to embed inside a generated module:
      - Keeps: imports, class defs, function defs
      - Removes: all other top-level executable statements
        (parse_args, print, assignments, if __name__ == '__main__', etc.)
      - Removes: from __future__ imports (hoisted to template top)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(
            l for l in source.splitlines()
            if not l.strip().startswith("from __future__")
        )

    keep_lines: set[int] = set()
    for node in tree.body:
        if isinstance(node, _KEEP_TOP_LEVEL):
            # exclude __future__ imports
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            for lineno in range(node.lineno, node.end_lineno + 1):
                keep_lines.add(lineno)

    lines = source.splitlines()
    return "\n".join(
        line for i, line in enumerate(lines, start=1)
        if i in keep_lines
    )


class FLClientGenerator:
    def __init__(self, analysis: AnalysisResult, original_source: str, original_path: str):
        self.a = analysis
        self.source = original_source
        self.path = original_path

    def generate(self) -> str:
        future_imports = [i for i in self.a.imports if i.startswith("from __future__")]
        regular_imports = [i for i in self.a.imports if not i.startswith("from __future__")]
        imports = "\n".join(regular_imports)
        future_block = "\n".join(future_imports) + "\n" if future_imports else ""

        # dataset init block
        if self.a.dataset_class:
            dataset_init = DATASET_INIT_WITH_CLASS.format(
                dataset_class=self.a.dataset_class
            )
        else:
            dataset_init = DATASET_INIT_GENERIC

        # loss block — prefer actual class names over variable names like 'criterion'
        VARIABLE_NAMES = {"criterion", "loss_fn", "loss_func"}
        KNOWN_CLASSES = {"CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss",
                         "MSELoss", "NLLLoss", "L1Loss", "HuberLoss"}
        KNOWN_FN = {"cross_entropy", "binary_cross_entropy", "mse_loss", "nll_loss"}

        if self.a.loss_calls:
            # pick best candidate: class name > functional > variable
            class_hits = [c for c in self.a.loss_calls if c in KNOWN_CLASSES]
            fn_hits    = [c for c in self.a.loss_calls if c in KNOWN_FN]

            if class_hits:
                first = class_hits[0]
                loss_block = LOSS_WITH_CRITERION if first in {
                    "CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss"
                } else LOSS_GENERIC.format(
                    loss_calls=self.a.loss_calls,
                    first_loss=f"nn.{first}()"
                )
            elif fn_hits:
                first = fn_hits[0]
                loss_block = LOSS_GENERIC.format(
                    loss_calls=self.a.loss_calls,
                    first_loss=f"nn.functional.{first}"
                )
            else:
                # Only variable names detected — default to CrossEntropy with warning
                loss_block = LOSS_WITH_CRITERION
        else:
            loss_block = """\
        # WARNING: no loss function detected — fill this in manually
        raise NotImplementedError("train_step: loss function not detected")"""

        clean_source = _strip_import_unsafe_code(self.source)

        code = FL_CLIENT_TEMPLATE.format(
            original_path=self.path,
            future_imports=future_block,
            imports=imports,
            original_source=clean_source,
            model_class=self.a.model_class or "nn.Module",
            dataset_init=dataset_init,
            loss_computation=loss_block,
        )
        return code


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def convert(source_path: str | Path, output_path: str | Path | None = None) -> Path:
    """
    Convert a PyTorch training script to an FL client module.

    Returns the output path.
    """
    src = Path(source_path)
    source = src.read_text()

    analyser = ScriptAnalyzer(source)
    analysis = analyser.analyse()

    for w in analysis.warnings:
        print(f"  [converter warning] {w}")

    generator = FLClientGenerator(analysis, source, str(src))
    fl_code = generator.generate()

    if output_path is None:
        output_path = src.parent / f"{src.stem}_fl_client.py"
    out = Path(output_path)
    out.write_text(fl_code)

    print(f"  [converter] {src.name} → {out.name}")
    print(f"    model_class   : {analysis.model_class or '(not found)'}")
    print(f"    dataset_class : {analysis.dataset_class or '(not found)'}")
    print(f"    optimizer     : {analysis.optimizer_class}")
    print(f"    loss_calls    : {analysis.loss_calls}")
    return out
