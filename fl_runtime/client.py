"""
FL Client base class. Works with any fl_client_module that exposes
build_model / build_dataloader / train_step.
"""
import copy
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class FLClient:
    def __init__(
        self,
        client_id: str,
        fl_module_path: str | Path,
        config: dict,
    ):
        self.client_id = client_id
        self.config = config
        self.mod = self._load_module(Path(fl_module_path))

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        self.model: Optional[nn.Module] = None
        self.amp_active: bool = False   # set by local_train (True only on CUDA)
        self._build_model()

    # ------------------------------------------------------------------
    # FL protocol
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in weights.items()},
            strict=True,
        )

    def local_train(self) -> tuple[dict[str, torch.Tensor], dict]:
        """Run local_epochs of training, return (updated_weights, metrics)."""
        local = self.config.get("local", {})
        local_epochs = self.config.get("local_epochs", 1)
        lr = self.config.get("learning_rate", 3e-4)
        grad_accum = local.get("gradient_accumulation", 1)
        use_amp = local.get("use_amp", False) and self.device.type == "cuda"

        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        dl = self.mod.build_dataloader(self.config, split="train")

        total_loss, n_steps, n_samples = 0.0, 0, 0
        t0 = time.time()

        for epoch in range(local_epochs):
            for step, batch in enumerate(dl):
                if step % grad_accum == 0:
                    opt.zero_grad()

                # train_step: forward only, returns loss with grad
                loss = self.mod.train_step(self.model, batch, None, self.config)
                total_loss += loss.item()
                n_steps += 1
                n_samples += self._batch_size(batch)

                if use_amp and scaler is not None:
                    scaler.scale(loss / grad_accum).backward()
                else:
                    (loss / grad_accum).backward()

                if (step + 1) % grad_accum == 0:
                    if use_amp and scaler is not None:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        opt.step()

        self.amp_active = bool(use_amp)
        metrics = {
            "client_id": self.client_id,
            "loss": round(total_loss / max(n_steps, 1), 6),
            "steps": n_steps,
            # Number of training examples actually consumed (sum of batch
            # sizes over all local epochs); used as n_k in sample-weighted FedAvg.
            "num_samples": n_samples,
            "amp_active": bool(use_amp),
            "device": str(self.device),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        return self.get_weights(), metrics

    def evaluate(self) -> dict:
        local = self.config.get("local", {})
        self.model.eval()
        dl = self.mod.build_dataloader(self.config, split="val")

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in dl:
                loss = self.mod.train_step(self.model, batch, None, self.config)
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                total += 1

        return {
            "client_id": self.client_id,
            "val_loss": round(total_loss / max(total, 1), 6),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        self.model = self.mod.build_model(self.config).to(self.device)

    @staticmethod
    def _batch_size(batch) -> int:
        """Best-effort number of examples in a batch (leading dim of the
        first tensor found). Returns 1 when no tensor is found so that the
        sample count degrades to the step count rather than zero."""
        if isinstance(batch, torch.Tensor):
            return int(batch.shape[0]) if batch.dim() > 0 else 1
        if isinstance(batch, dict):
            for v in batch.values():
                n = FLClient._batch_size(v)
                if n:
                    return n
            return 1
        if isinstance(batch, (list, tuple)):
            for v in batch:
                n = FLClient._batch_size(v)
                if n:
                    return n
            return 1
        try:
            return int(len(batch))
        except TypeError:
            return 1

    @staticmethod
    def _load_module(path: Path):
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("fl_client_module", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
