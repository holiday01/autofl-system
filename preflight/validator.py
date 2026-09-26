"""
Pre-flight validator: run before FL starts to confirm the converted code
works on the local machine. Tests imports, data loading, and a 2-step
forward+backward pass. Reports a structured result back to the coordinator.

Stages (in order):
  1. import_check    — every required package imports AND the generated module
                       itself imports without error in an ISOLATED SUBPROCESS
                       (CUDA hidden, 120 s timeout). A failure here is reported
                       with the last traceback line, e.g.
                       "ModuleNotFoundError: No module named 'keras'".
  2. hardware_detect — local hardware profile + suggested local params.
  3. data_load       — build_dataloader(config, "train") yields one batch.
                       In strict mode (strict_data=True) the config carries
                       allow_synthetic_data=False and config["data_path"] must
                       exist on disk; otherwise allow_synthetic_data=True.
  4. forward_pass    — one no-grad forward pass through train_step.
  5. backward_pass   — two forward+backward+optimizer steps.
Stages 2-5 run in-process.
"""
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

# Suppress CUDA to avoid driver version errors on CPU-only benchmark runs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

IMPORT_CHECK_TIMEOUT_SEC = 120


class _StageFailed(Exception):
    """Raised by a stage that has already filled result.error_message."""


@dataclass
class PreflightResult:
    client_id: str
    success: bool
    stage_passed: str          # last successfully completed stage
    error_stage: str = ""      # which stage failed (empty = all passed)
    error_message: str = ""
    elapsed_sec: float = 0.0
    hardware_info: dict = field(default_factory=dict)
    local_config: dict = field(default_factory=dict)
    # --- data provenance (always recorded) ---
    data_path: str = ""
    data_path_exists: bool = False
    strict_data: bool = False
    # Diagnostics of what build_dataloader actually returned (best effort):
    # class name of the underlying dataset (Subset unwrapped) and its length.
    dataset_class: str = ""
    dataset_len: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def print_summary(self) -> None:
        status = "PASS" if self.success else f"FAIL at [{self.error_stage}]"
        print(f"  Preflight [{self.client_id}]: {status}  ({self.elapsed_sec:.1f}s)")
        if not self.success:
            print(f"    Error: {self.error_message.splitlines()[0] if self.error_message else ''}")


STAGES = [
    "import_check",
    "hardware_detect",
    "data_load",
    "forward_pass",
    "backward_pass",
]


# Code executed in the isolated subprocess for Stage 1. It mirrors
# _load_fl_module (spec_from_file_location + exec_module) exactly.
_SUBPROCESS_IMPORT_TEMPLATE = r"""
import importlib.util, sys
path = {path!r}
sys.path.insert(0, {parent!r})
spec = importlib.util.spec_from_file_location("fl_client_module", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
"""


class PreflightValidator:
    """
    Run a dry-run validation of the FL client module.

    The fl_client_module must expose:
      - build_model(config) -> torch.nn.Module
      - build_dataloader(config, split='train') -> DataLoader
      - train_step(model, batch, optimizer, config) -> loss (scalar tensor)
    """

    def __init__(
        self,
        client_id: str,
        fl_client_module_path: str | Path,
        config: dict,
        required_packages: list[str] | None = None,
        strict_data: bool = False,
    ):
        self.client_id = client_id
        self.module_path = Path(fl_client_module_path)
        self.strict_data = bool(strict_data)
        # The config handed to the generated module always carries an explicit
        # allow_synthetic_data flag: False in strict mode (fail closed), True
        # otherwise (explicit test mode).
        self.config = {**config, "allow_synthetic_data": not self.strict_data}
        self.required_packages = required_packages or ["torch"]

    def run(self) -> PreflightResult:
        t0 = time.time()
        data_path = str(self.config.get("data_path", ""))
        result = PreflightResult(
            client_id=self.client_id,
            success=False,
            stage_passed="",
            data_path=data_path,
            data_path_exists=bool(data_path) and Path(data_path).exists(),
            strict_data=self.strict_data,
        )

        for stage in STAGES:
            try:
                fn = getattr(self, f"_stage_{stage}")
                fn(result)
                result.stage_passed = stage
            except _StageFailed:
                result.error_stage = stage
                result.elapsed_sec = round(time.time() - t0, 2)
                return result
            except Exception as e:
                result.error_stage = stage
                result.error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                result.elapsed_sec = round(time.time() - t0, 2)
                return result

        result.success = True
        result.elapsed_sec = round(time.time() - t0, 2)
        return result

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_import_check(self, result: PreflightResult) -> None:
        """Stage 1: required packages import, then the generated module itself
        is imported in an isolated subprocess (CUDA hidden, 120 s timeout)."""
        failed = []
        for pkg in self.required_packages:
            try:
                importlib.import_module(pkg)
            except ImportError as e:
                failed.append(f"{pkg}: {e}")
        if failed:
            raise ImportError("Missing packages:\n" + "\n".join(failed))

        code = _SUBPROCESS_IMPORT_TEMPLATE.format(
            path=str(self.module_path), parent=str(self.module_path.parent)
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        # Preserve PYTHONPATH (plus the running interpreter's sys.path head so
        # `autofl.*` resolves the same way as in-process).
        extra = [p for p in sys.path[:3] if p]
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in [env.get("PYTHONPATH", "")] + extra if p]
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                timeout=IMPORT_CHECK_TIMEOUT_SEC,
                cwd=os.getcwd(),
            )
        except subprocess.TimeoutExpired:
            result.error_message = (
                f"TimeoutExpired: importing generated module exceeded "
                f"{IMPORT_CHECK_TIMEOUT_SEC}s in isolated subprocess"
            )
            raise _StageFailed()

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            lines = [l for l in stderr.splitlines() if l.strip()]
            last = lines[-1] if lines else f"subprocess exited with code {proc.returncode}"
            # error_message: last traceback line first, full stderr tail after.
            result.error_message = last + "\n" + stderr[-3000:]
            raise _StageFailed()

    def _stage_hardware_detect(self, result: PreflightResult) -> None:
        """Run hardware detection and store info + suggested params."""
        from autofl.hardware.detector import detect
        hw = detect()
        result.hardware_info = {
            "primary_gpu": hw.primary_gpu_name,
            "total_vram_mb": hw.total_vram_mb,
            "has_cuda": hw.has_cuda,
            "cpu_count": hw.cpu_count,
            "ram_gb": hw.ram_gb,
        }
        # Merge hardware suggestions into effective local config
        local = self.config.get("local", {})
        if "batch_size" not in local:
            local["batch_size"] = hw.suggested_batch_size
        if "use_amp" not in local:
            local["use_amp"] = hw.suggested_use_amp
        result.local_config = local

    def _stage_data_load(self, result: PreflightResult) -> None:
        """Try to load one batch from the training dataloader."""
        if self.strict_data:
            p = self.config.get("data_path", "")
            if not p or not Path(p).exists():
                raise FileNotFoundError(
                    f"strict mode: real data path {p!r} does not exist"
                )
        mod = self._load_fl_module()
        dl = mod.build_dataloader(self.config, split="train")
        self._record_dataset_info(result, dl)
        batch = next(iter(dl))
        if batch is None:
            raise ValueError("build_dataloader returned an empty batch")

    def _stage_forward_pass(self, result: PreflightResult) -> None:
        """One forward pass (no grad) — verifies model + loss computes."""
        import torch
        mod = self._load_fl_module()
        model = mod.build_model(self.config)
        device = "cuda" if result.hardware_info.get("has_cuda") else "cpu"
        model = model.to(device)
        model.eval()

        dl = mod.build_dataloader(self.config, split="train")
        batch = next(iter(dl))

        # Temporarily disable use_amp for the no-grad check
        cfg_copy = {**self.config, "local": {**self.config.get("local", {}), "use_amp": False}}
        with torch.no_grad():
            loss = mod.train_step(model, batch, optimizer=None, config=cfg_copy)
        if loss is None:
            raise ValueError("train_step returned None (expected scalar tensor)")

    def _stage_backward_pass(self, result: PreflightResult) -> None:
        """Two forward+backward steps to verify gradient flow."""
        import torch
        mod = self._load_fl_module()
        model = mod.build_model(self.config)
        device = "cuda" if result.hardware_info.get("has_cuda") else "cpu"
        model = model.to(device)
        model.train()

        # Disable AMP for preflight to keep the check simple and device-agnostic
        cfg_copy = {**self.config, "local": {**self.config.get("local", {}), "use_amp": False}}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dl = mod.build_dataloader(self.config, split="train")
        it = iter(dl)

        for _ in range(2):
            batch = next(it)
            opt.zero_grad()
            # train_step returns forward-only loss; backward managed here
            loss = mod.train_step(model, batch, optimizer=None, config=cfg_copy)
            loss.backward()
            opt.step()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_dataset_info(result: PreflightResult, dl) -> None:
        """Best-effort: record the class name / length of the dataset that
        build_dataloader actually wrapped (unwrapping torch Subset)."""
        try:
            ds = getattr(dl, "dataset", None)
            depth = 0
            while ds is not None and hasattr(ds, "dataset") and depth < 5:
                ds = ds.dataset
                depth += 1
            if ds is not None:
                result.dataset_class = type(ds).__name__
                try:
                    result.dataset_len = int(len(ds))
                except Exception:
                    result.dataset_len = -1
        except Exception:
            pass

    def _load_fl_module(self):
        """Dynamically import the generated FL client module."""
        sys.path.insert(0, str(self.module_path.parent))
        spec = importlib.util.spec_from_file_location(
            "fl_client_module", str(self.module_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def run_preflight(
    client_id: str,
    fl_client_module_path: str | Path,
    config: dict,
    required_packages: list[str] | None = None,
    report_path: str | Path | None = None,
    strict_data: bool = False,
) -> PreflightResult:
    """Convenience wrapper — runs validator, optionally saves JSON report."""
    v = PreflightValidator(
        client_id, fl_client_module_path, config, required_packages,
        strict_data=strict_data,
    )
    result = v.run()

    if report_path is not None:
        with open(report_path, "w") as f:
            f.write(result.to_json())

    result.print_summary()
    return result
