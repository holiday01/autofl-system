"""
LLM-based ML→FL converter.

Supported providers:
  - claude  : Anthropic Claude CLI (claude -p), no API key needed
  - gemini  : Google Gemini via google-generativeai; reads GOOGLE_API_KEY env var
  - ollama  : Local Ollama server at localhost:11434

Three skill levels:
  - zero_shot:   direct conversion request, no examples or spec
  - few_shot:    one complete ML→FL example pair provided
  - structured:  detailed system prompt with FL interface spec + rules
"""
import os
import subprocess
import textwrap
from enum import Enum
from pathlib import Path

CLAUDE_MODEL  = "claude-sonnet-4-6"
GEMINI_MODEL  = "gemini-2.5-flash"
OLLAMA_MODEL  = "llama3"   # override with OLLAMA_MODEL env var (e.g. deepseek-coder-v2)

# ── FL interface spec (used by the STRUCTURED skill) ─────────────────────────
#
# Two versions are kept for provenance:
#   v1 — the exact text used to generate every pre-cached *_fl_structured.py in
#        benchmarks/ (do not edit).
#   v2 — v1 plus one rule: the synthetic-data fallback must be gated on
#        config.get("allow_synthetic_data", False) and otherwise raise
#        FileNotFoundError (fail closed). v2 is the default for new conversions.

FL_INTERFACE_SPEC_V1 = """
The FL client module MUST expose exactly these three functions:

1. build_model(config: dict) -> torch.nn.Module
   - Instantiate and return the model.
   - Use config.get("model_kwargs", {}) for constructor args.

2. build_dataloader(config: dict, split: str = "train") -> DataLoader
   - Return a DataLoader for the requested split ("train" or "val").
   - Read batch_size from config.get("local", {}).get("batch_size", 16).
   - Read data_path from config.get("data_path", ".").
   - Use random_split to produce train/val subsets from a single dataset.
   - Always include a synthetic data fallback (torch.randn/randint) in case
     the real dataset is unavailable.

3. train_step(model, batch, optimizer, config: dict) -> torch.Tensor
   - Run ONE forward pass only. Return the loss tensor WITH grad attached.
   - Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
   - Move tensors to the device of the model parameters.
""".strip()

FL_INTERFACE_SPEC_V2 = """
The FL client module MUST expose exactly these three functions:

1. build_model(config: dict) -> torch.nn.Module
   - Instantiate and return the model.
   - Use config.get("model_kwargs", {}) for constructor args.

2. build_dataloader(config: dict, split: str = "train") -> DataLoader
   - Return a DataLoader for the requested split ("train" or "val").
   - Read batch_size from config.get("local", {}).get("batch_size", 16).
   - Read data_path from config.get("data_path", ".").
   - Use random_split to produce train/val subsets from a single dataset.
   - Include a synthetic data fallback (torch.randn/randint) for the case
     where the real dataset is unavailable, but it MUST be gated on
     config.get("allow_synthetic_data", False) (see Rules).

3. train_step(model, batch, optimizer, config: dict) -> torch.Tensor
   - Run ONE forward pass only. Return the loss tensor WITH grad attached.
   - Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
   - Move tensors to the device of the model parameters.
""".strip()

# Backwards-compatible alias: the name used before spec versioning existed
# refers to the v1 text (provenance of the cached benchmark outputs).
FL_INTERFACE_SPEC = FL_INTERFACE_SPEC_V1

STRUCTURED_RULES_V1 = """
Rules:
- Preserve the original model architecture exactly (copy the class).
- Keep all imports from the original script.
- build_dataloader must support both "train" and "val" splits using random_split.
- train_step must NOT call backward() or optimizer.step().
- If the original uses TensorFlow/Keras, convert the model to an equivalent torch.nn.Module.
- If the original uses PyTorch Lightning LightningModule, extract the underlying model and loss.
- If the original uses MONAI transforms, wrap them in the Dataset class.
- Output ONLY valid Python code. No markdown fences, no explanation.
""".strip()

STRUCTURED_RULES_V2 = STRUCTURED_RULES_V1 + """
- The synthetic-data fallback in build_dataloader MUST be gated on
  config.get('allow_synthetic_data', False). If real data is unavailable and
  that flag is False, raise FileNotFoundError with a clear message; never
  silently train on synthetic data.
"""
STRUCTURED_RULES_V2 = STRUCTURED_RULES_V2.strip()

SPEC_VERSIONS = {
    "v1": (FL_INTERFACE_SPEC_V1, STRUCTURED_RULES_V1),
    "v2": (FL_INTERFACE_SPEC_V2, STRUCTURED_RULES_V2),
}
DEFAULT_SPEC_VERSION = "v2"

# ── Few-shot example pair ────────────────────────────────────────────────────

_FEWSHOT_ORIGINAL = (Path(__file__).parent.parent / "examples" / "sample_training_script.py").read_text()
_FEWSHOT_FL = (Path(__file__).parent.parent / "examples" / "wsi_mlp_train_fl_client.py").read_text()


# Corrected-exemplar variant for the §5.7 robustness re-experiment.
# The original few-shot exemplar (used to generate the pre-cached _fl_few_shot.py
# outputs in benchmarks/) shipped with a `train_step` that called
# `loss.backward(); optimizer.step(); return loss.detach()` — a direct contract
# violation that the LLM frequently copied verbatim. The exemplar below is a
# minimal, contract-correct counterpart that preserves the exemplar structure
# (same model, dataset, build_model, build_dataloader) but has a clean
# `train_step`: one forward pass, return the raw loss tensor with grad attached,
# no .backward(), no optimizer.step(), no .detach(), no .item().
_FEWSHOT_FL_CORRECTED = textwrap.dedent('''
"""
Auto-generated FL client module (CORRECTED EXEMPLAR).
Original script: autofl/examples/wsi_mlp_train.py

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class WSIFeatureDataset(Dataset):
    """Load pre-extracted foundation model features (.npy) for cancer classification."""

    def __init__(self, root: str, feature_dim: int = 1024, num_classes: int = 9):
        self.root = root
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        npy_files, labels = [], []
        if os.path.isdir(root):
            for label_idx, subdir in enumerate(sorted(os.listdir(root))):
                subpath = os.path.join(root, subdir)
                if os.path.isdir(subpath):
                    for f in os.listdir(subpath):
                        if f.endswith(".npy"):
                            npy_files.append(os.path.join(subpath, f))
                            labels.append(label_idx)
        if npy_files:
            self.features = [np.load(p) for p in npy_files]
            self.labels = labels
        else:
            # synthetic fallback for testing
            self.features = [np.random.randn(feature_dim).astype(np.float32)
                             for _ in range(200)]
            self.labels = [i % num_classes for i in range(200)]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=9, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MLPClassifier(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    dataset_kwargs = config.get("dataset_kwargs", {})
    data_path = config.get("data_path", ".")
    full_dataset = WSIFeatureDataset(root=data_path, **dataset_kwargs)
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
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss
''').strip()


# ── Enums ────────────────────────────────────────────────────────────────────

class Skill(str, Enum):
    ZERO_SHOT           = "zero_shot"
    FEW_SHOT            = "few_shot"
    FEW_SHOT_CORRECTED  = "few_shot_corrected"
    STRUCTURED          = "structured"


class Provider(str, Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    OLLAMA = "ollama"


# ── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(
    source: str,
    skill: Skill,
    spec_version: str = DEFAULT_SPEC_VERSION,
) -> tuple[str | None, str]:
    """Return (system_prompt_or_None, user_message) for the given skill.

    ``spec_version`` selects the FL interface spec + rules used by the
    STRUCTURED skill ("v1" = provenance of cached outputs, "v2" = default).
    """
    if spec_version not in SPEC_VERSIONS:
        raise ValueError(f"Unknown spec_version {spec_version!r}; "
                         f"expected one of {sorted(SPEC_VERSIONS)}")
    spec_text, rules_text = SPEC_VERSIONS[spec_version]

    if skill == Skill.ZERO_SHOT:
        system = None
        user = (
            "Convert the following training script into a Federated Learning "
            "client module. The module must expose three functions: build_model(config), "
            "build_dataloader(config, split), and train_step(model, batch, optimizer, config). "
            "Return ONLY the Python code, no explanation.\n\n"
            f"```python\n{source}\n```"
        )

    elif skill == Skill.FEW_SHOT:
        system = None
        user = (
            "I will show you one example of converting a training script to an FL client "
            "module, then ask you to do the same for a new script.\n\n"
            "=== ORIGINAL SCRIPT ===\n"
            f"```python\n{_FEWSHOT_ORIGINAL}\n```\n\n"
            "=== FL CLIENT MODULE ===\n"
            f"```python\n{_FEWSHOT_FL}\n```\n\n"
            "Now convert this new script using the same pattern. "
            "Return ONLY the Python code, no explanation.\n\n"
            "=== NEW SCRIPT TO CONVERT ===\n"
            f"```python\n{source}\n```"
        )

    elif skill == Skill.FEW_SHOT_CORRECTED:
        # Same prompt scaffold as FEW_SHOT, but the exemplar's `train_step`
        # is contract-correct (no .backward(), no optimizer.step(),
        # no .detach(), no .item()). Used for the §5.7 robustness
        # re-experiment that isolates exemplar-quality effects from
        # prompting-strategy effects.
        system = None
        user = (
            "I will show you one example of converting a training script to an FL client "
            "module, then ask you to do the same for a new script.\n\n"
            "=== ORIGINAL SCRIPT ===\n"
            f"```python\n{_FEWSHOT_ORIGINAL}\n```\n\n"
            "=== FL CLIENT MODULE ===\n"
            f"```python\n{_FEWSHOT_FL_CORRECTED}\n```\n\n"
            "Now convert this new script using the same pattern. "
            "Return ONLY the Python code, no explanation.\n\n"
            "=== NEW SCRIPT TO CONVERT ===\n"
            f"```python\n{source}\n```"
        )

    elif skill == Skill.STRUCTURED:
        # NOTE: the textwrap.dedent(f-string) construction below is kept
        # verbatim from the version that generated the cached
        # *_fl_structured.py files, so spec_version="v1" reproduces that system
        # prompt byte-for-byte (including its indentation quirk: the
        # interpolated spec defeats dedent, so header/rule lines keep a
        # 12-space indent). v2 differs ONLY by the extra rule appended below.
        extra_rule = ""
        if spec_version == "v2":
            extra_rule = (
                "\n            - The synthetic-data fallback in build_dataloader MUST be "
                "gated on config.get('allow_synthetic_data', False). If real data is "
                "unavailable and that flag is False, raise FileNotFoundError with a "
                "clear message; never silently train on synthetic data."
            )
        system = textwrap.dedent(f"""
            You are an expert federated learning (FL) engineer.
            Your task is to convert standard ML training scripts into FL client modules.

            {spec_text}

            Rules:
            - Preserve the original model architecture exactly (copy the class).
            - Keep all imports from the original script.
            - build_dataloader must support both "train" and "val" splits using random_split.
            - train_step must NOT call backward() or optimizer.step().
            - If the original uses TensorFlow/Keras, convert the model to an equivalent torch.nn.Module.
            - If the original uses PyTorch Lightning LightningModule, extract the underlying model and loss.
            - If the original uses MONAI transforms, wrap them in the Dataset class.
            - Output ONLY valid Python code. No markdown fences, no explanation.{extra_rule}
        """).strip()
        user = f"Convert this training script to an FL client module:\n\n{source}"

    else:
        raise ValueError(f"Unknown skill: {skill}")

    return system, user


# ── Code-fence stripper ──────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        start = next((i + 1 for i, l in enumerate(lines) if l.startswith("```")), 1)
        end = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].startswith("```")), len(lines))
        raw = "\n".join(lines[start:end]).strip()
    return raw


# ── Backend callers ──────────────────────────────────────────────────────────

def _call_claude(system: str | None, user: str) -> str:
    # temperature is fixed at 1.0 by the claude CLI (no --temperature flag);
    # for reproducibility, generated files are stored in the repository.
    cmd = [
        "claude", "-p",
        "--model", CLAUDE_MODEL,
        "--output-format", "text",
    ]
    if system:
        cmd += ["--system-prompt", system]
    result = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={result.returncode}): {result.stderr[:500]}")
    return result.stdout


def _call_gemini(system: str | None, user: str) -> str:
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=0.0,  # deterministic output for reproducibility
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=config,
    )
    return response.text


def _call_ollama(system: str | None, user: str) -> str:
    import ollama
    model_name = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={"temperature": 0.0},  # deterministic output for reproducibility
    )
    return response["message"]["content"]


# ── Public API ───────────────────────────────────────────────────────────────

def convert_llm(
    source_path: str | Path,
    skill: Skill,
    output_path: str | Path | None = None,
    provider: Provider = Provider.CLAUDE,
    spec_version: str = DEFAULT_SPEC_VERSION,
) -> Path:
    """
    Convert a training script to an FL client module.

    Args:
        source_path:  Path to the original ML training script.
        skill:        Skill.ZERO_SHOT | FEW_SHOT | STRUCTURED
        output_path:  Where to write the FL client; defaults to
                      <stem>_fl_<skill>.py next to source.
        provider:     Provider.CLAUDE | GEMINI | OLLAMA
        spec_version: "v1" (prompt set that produced the cached benchmark
                      files) or "v2" (default; adds the allow_synthetic_data
                      gating rule). Only affects Skill.STRUCTURED.
    Returns:
        Path to the written FL client module.
    """
    src = Path(source_path)
    source = src.read_text()
    system, user = _build_prompt(source, skill, spec_version=spec_version)

    if provider == Provider.CLAUDE:
        raw = _call_claude(system, user)
    elif provider == Provider.GEMINI:
        raw = _call_gemini(system, user)
    elif provider == Provider.OLLAMA:
        raw = _call_ollama(system, user)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    code = _strip_fences(raw)

    if output_path is None:
        output_path = src.parent / f"{src.stem}_fl_{skill.value}.py"
    out = Path(output_path)
    out.write_text(code)
    return out
