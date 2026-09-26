"""
Hierarchical config system:
  global_config (admin, locked) > local_config (client, only allowed_overrides)

Admin defines which params clients may override.
Clients use hardware detection to auto-fill those params.
"""
import copy
import os
from pathlib import Path
from typing import Any

import yaml


# Params clients are always allowed to tune (hardware-related).
HARDWARE_ADJUSTABLE = {
    "batch_size",
    "num_workers",
    "use_amp",
    "gradient_accumulation",
    "pin_memory",
    "prefetch_factor",
}

# Params that must stay global (never client-overridable).
LOCKED_PARAMS = {
    "learning_rate",
    "num_rounds",
    "local_epochs",
    "model_architecture",
    "aggregation_algorithm",
    "num_clients",
    "clients_per_round",
    "seed",
    "loss_function",
}


class ConfigError(ValueError):
    pass


class ConfigManager:
    def __init__(self, global_config_path: str | Path):
        self.global_path = Path(global_config_path)
        self.global_cfg: dict = self._load(self.global_path)
        self._validate_global()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build_client_config(
        self,
        local_config_path: str | Path | None = None,
        auto_hardware: bool = True,
    ) -> dict:
        """
        Merge global + local + hardware suggestions → final client config.

        Priority (highest → lowest):
          1. local_config (only for allowed_overrides keys)
          2. hardware auto-detection (only for HARDWARE_ADJUSTABLE keys not in local)
          3. global_config defaults
        """
        cfg = copy.deepcopy(self.global_cfg)
        allowed: set[str] = set(cfg.get("allowed_overrides", [])) | HARDWARE_ADJUSTABLE

        # Remove locked params from allowed (safety guard)
        allowed -= LOCKED_PARAMS

        # Hardware auto-suggestion
        if auto_hardware:
            from autofl.hardware.detector import detect
            hw = detect()
            hw_suggestions = {
                "batch_size": hw.suggested_batch_size,
                "num_workers": hw.suggested_num_workers,
                "use_amp": hw.suggested_use_amp,
                "gradient_accumulation": hw.suggested_gradient_accumulation,
            }
            for k, v in hw_suggestions.items():
                if k in allowed:
                    cfg.setdefault("local", {})[k] = v  # only as default; local file wins

        # Apply local overrides
        if local_config_path is not None:
            local = self._load(Path(local_config_path))
            violations = set(local.keys()) & LOCKED_PARAMS
            if violations:
                raise ConfigError(
                    f"Client config tries to override locked params: {violations}"
                )
            disallowed = set(local.keys()) - allowed - {"client_id", "data_path", "output_dir"}
            if disallowed:
                raise ConfigError(
                    f"Client config contains params not in allowed_overrides: {disallowed}"
                )
            cfg.setdefault("local", {}).update(local)

        return cfg

    def get_global(self, key: str, default: Any = None) -> Any:
        return self.global_cfg.get(key, default)

    def save_client_config(self, path: str | Path, cfg: dict) -> None:
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return data or {}

    def _validate_global(self) -> None:
        required = {"num_rounds", "local_epochs", "aggregation_algorithm"}
        missing = required - set(self.global_cfg.keys())
        if missing:
            raise ConfigError(f"Global config missing required keys: {missing}")

        overrides = set(self.global_cfg.get("allowed_overrides", []))
        conflict = overrides & LOCKED_PARAMS
        if conflict:
            raise ConfigError(
                f"allowed_overrides contains locked params (admin error): {conflict}"
            )

    def print_summary(self, client_cfg: dict) -> None:
        print("=" * 50)
        print("Effective Config")
        print("=" * 50)
        print("[GLOBAL - locked]")
        for k in sorted(LOCKED_PARAMS & set(self.global_cfg)):
            print(f"  {k:30s} = {self.global_cfg[k]}")
        print()
        print("[LOCAL - client-adjustable]")
        local = client_cfg.get("local", {})
        allowed = set(self.global_cfg.get("allowed_overrides", [])) | HARDWARE_ADJUSTABLE
        for k in sorted(allowed):
            val = local.get(k, client_cfg.get(k, "<not set>"))
            print(f"  {k:30s} = {val}")
        print("=" * 50)
