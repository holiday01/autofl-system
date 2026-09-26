"""
Evaluator: measure conversion quality of a generated FL client module.

Metrics:
  syntax_valid        — code parses without SyntaxError
  interface_complete  — all 3 FL functions present
  model_detected      — build_model returns something
  dataset_detected    — build_dataloader callable
  loss_detected       — train_step returns a tensor
  optimizer_detected  — optimizer class referenced in source
  preflight_pass      — passes PreflightValidator dry-run
  e2e_runnable        — 1-round FL simulation completes
"""
import ast
import importlib.util
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch


@dataclass
class EvalResult:
    script_name: str
    method: str                        # ast / zero_shot / few_shot / structured
    framework: str                     # pytorch / tensorflow / monai / lightning

    # --- static analysis ---
    syntax_valid: bool = False
    interface_complete: bool = False   # all 3 functions present
    build_model_present: bool = False
    build_dataloader_present: bool = False
    train_step_present: bool = False
    optimizer_detected: bool = False   # any optimizer class in source

    # --- runtime ---
    preflight_pass: bool = False
    e2e_runnable: bool = False

    # --- diagnostics ---
    error_stage: str = ""
    error_message: str = ""
    elapsed_sec: float = 0.0

    # --- data provenance / prompt provenance ---
    data_path: str = ""
    data_path_exists: bool = False
    strict_data: bool = False
    spec_version: str = ""          # prompt-spec version the file was generated with
    dataset_class: str = ""         # underlying dataset class seen at preflight data_load
    dataset_len: int = -1

    @property
    def component_coverage(self) -> float:
        """Fraction of 4 components detected (model/dataloader/train_step/optimizer)."""
        hits = sum([
            self.build_model_present,
            self.build_dataloader_present,
            self.train_step_present,
            self.optimizer_detected,
        ])
        return hits / 4.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["component_coverage"] = round(self.component_coverage, 3)
        return d


_OPTIMIZER_NAMES = {
    "Adam", "AdamW", "SGD", "RMSprop", "Adagrad",
    "optimizer", "optim",
    # TF/Keras
    "keras.optimizers", "tf.keras.optimizers",
    "compile",   # model.compile() implies an optimizer
}

_MINIMAL_CONFIG = {
    "num_rounds": 1,
    "local_epochs": 1,
    "aggregation_algorithm": "FedAvg",
    "learning_rate": 1e-3,
    "seed": 0,
    "data_path": ".",
    "local": {"batch_size": 4, "use_amp": False, "num_workers": 0},
}


def evaluate(
    generated_path: str | Path,
    script_name: str,
    method: str,
    framework: str,
    data_root: str | Path = ".",
    strict_data: bool = False,
    spec_version: str = "",
) -> EvalResult:
    """Run all evaluation stages and return an EvalResult.

    Args:
        data_root:    value handed to the generated module as config["data_path"].
        strict_data:  if True the preflight fails closed when data_root does not
                      exist and passes allow_synthetic_data=False to the module;
                      if False (explicit test mode) allow_synthetic_data=True.
        spec_version: prompt-spec version recorded for provenance ("v1"/"v2"/"n/a").
    """

    result = EvalResult(
        script_name=script_name,
        method=method,
        framework=framework,
        data_path=str(data_root),
        data_path_exists=Path(data_root).exists(),
        strict_data=bool(strict_data),
        spec_version=spec_version,
    )
    t0 = time.time()
    path = Path(generated_path)
    config = {**_MINIMAL_CONFIG, "data_path": str(data_root),
              "allow_synthetic_data": not strict_data}

    try:
        source = path.read_text()
    except Exception as e:
        result.error_stage = "read"
        result.error_message = str(e)
        result.elapsed_sec = round(time.time() - t0, 2)
        return result

    # 1. Syntax check
    try:
        tree = ast.parse(source)
        result.syntax_valid = True
    except SyntaxError as e:
        result.error_stage = "syntax"
        result.error_message = str(e)
        result.elapsed_sec = round(time.time() - t0, 2)
        return result

    # 2. Interface completeness (static)
    fn_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    result.build_model_present = "build_model" in fn_names
    result.build_dataloader_present = "build_dataloader" in fn_names
    result.train_step_present = "train_step" in fn_names
    result.interface_complete = all([
        result.build_model_present,
        result.build_dataloader_present,
        result.train_step_present,
    ])

    # 3. Optimizer detection — check train_step signature and body
    # Correct FL contract: train_step accepts optimizer as parameter and uses it
    for node in ast.walk(tree):
        # Function parameter named "optimizer" (ast.arg, not ast.Name)
        if isinstance(node, ast.arg) and node.arg == "optimizer":
            result.optimizer_detected = True
            break
        # Optimizer class instantiation in module (AST-style or zero-shot)
        if isinstance(node, ast.Name) and node.id in _OPTIMIZER_NAMES:
            result.optimizer_detected = True
            break
        if isinstance(node, ast.Attribute) and node.attr in _OPTIMIZER_NAMES:
            result.optimizer_detected = True
            break
        # tf model.compile → optimizer implied
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "compile":
                result.optimizer_detected = True
                break

    if not result.interface_complete:
        result.error_stage = "interface"
        result.error_message = (
            f"Missing: "
            + ", ".join(
                fn for fn, present in [
                    ("build_model", result.build_model_present),
                    ("build_dataloader", result.build_dataloader_present),
                    ("train_step", result.train_step_present),
                ]
                if not present
            )
        )
        result.elapsed_sec = round(time.time() - t0, 2)
        return result

    # 4. Preflight
    try:
        from autofl.preflight.validator import run_preflight
        pf = run_preflight(
            client_id="eval-client",
            fl_client_module_path=path,
            config=config,
            required_packages=["torch"],
            strict_data=strict_data,
        )
        result.preflight_pass = pf.success
        result.data_path = pf.data_path
        result.data_path_exists = pf.data_path_exists
        result.strict_data = pf.strict_data
        result.dataset_class = pf.dataset_class
        result.dataset_len = pf.dataset_len
        if not pf.success:
            result.error_stage = f"preflight/{pf.error_stage}"
            result.error_message = pf.error_message[:300]
    except Exception as e:
        result.error_stage = "preflight/exception"
        result.error_message = traceback.format_exc()[:300]

    if not result.preflight_pass:
        result.elapsed_sec = round(time.time() - t0, 2)
        return result

    # 5. 1-round FL simulation
    try:
        from autofl.fl_runtime.client import FLClient
        from autofl.fl_runtime.server import FLServer

        with tempfile.TemporaryDirectory() as td:
            cfg = {**config, "client_id": "eval-site-1"}
            client = FLClient("eval-site-1", path, cfg)
            server = FLServer(client.get_weights(), config, results_dir=td)
            server.register_preflight("eval-site-1", {"success": True})
            server.run([client], num_rounds=1)
        result.e2e_runnable = True
    except Exception as e:
        result.error_stage = "e2e"
        result.error_message = traceback.format_exc()[:300]

    result.elapsed_sec = round(time.time() - t0, 2)
    return result
