"""
Lightweight FL server / coordinator.

Supports: FedAvg with sample-count weighting
(w = sum_k (n_k / n) w_k, McMahan et al. 2017).
Collects preflight results before starting training.
"""
import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch


@dataclass
class RoundResult:
    round: int
    participating_clients: list[str]
    aggregated_loss: float
    elapsed_sec: float
    extra: dict = field(default_factory=dict)


class FLServer:
    def __init__(
        self,
        global_weights: dict[str, torch.Tensor],
        config: dict,
        results_dir: str | Path = "fl_results",
    ):
        self.weights = {k: v.cpu().clone() for k, v in global_weights.items()}
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.round_results: list[RoundResult] = []
        self.preflight_results: dict = {}

    # ------------------------------------------------------------------
    # Preflight collection
    # ------------------------------------------------------------------

    def register_preflight(self, client_id: str, result_dict: dict) -> None:
        self.preflight_results[client_id] = result_dict
        passed = result_dict.get("success", False)
        print(f"  Preflight [{client_id}]: {'PASS' if passed else 'FAIL'}")

    def ready_clients(self) -> list[str]:
        return [cid for cid, r in self.preflight_results.items() if r.get("success")]

    def save_preflight_report(self) -> Path:
        path = self.results_dir / "preflight_report.json"
        with open(path, "w") as f:
            json.dump(self.preflight_results, f, indent=2)
        return path

    # ------------------------------------------------------------------
    # Training coordination
    # ------------------------------------------------------------------

    def run_round(self, clients, round_num: int) -> RoundResult:
        """
        Send global weights to each client, collect local updates, aggregate.
        `clients` is a list of FLClient instances.

        Aggregation is sample-weighted FedAvg: each client's update and its
        reported loss are weighted by ``metrics["num_samples"]`` (the number of
        training examples it actually consumed this round).
        """
        t0 = time.time()
        all_weights = []
        all_losses = []
        all_num_samples = []
        participating = []
        per_client = {}

        for client in clients:
            client.set_weights(self.weights)
            updated_w, metrics = client.local_train()
            all_weights.append(updated_w)
            all_losses.append(metrics["loss"])
            all_num_samples.append(metrics.get("num_samples"))
            participating.append(client.client_id)
            per_client[client.client_id] = {
                "loss": metrics["loss"],
                "num_samples": metrics.get("num_samples"),
                "steps": metrics["steps"],
                "amp_active": metrics.get("amp_active"),
                "device": metrics.get("device"),
            }
            print(f"    [{client.client_id}] loss={metrics['loss']:.4f}  "
                  f"steps={metrics['steps']}  n={metrics.get('num_samples')}  "
                  f"{metrics['elapsed_sec']}s")

        num_samples = all_num_samples if all(n is not None for n in all_num_samples) else None
        self.weights = self._fedavg(all_weights, num_samples)
        agg_loss = self._weighted_mean(all_losses, num_samples)

        rr = RoundResult(
            round=round_num,
            participating_clients=participating,
            aggregated_loss=round(agg_loss, 6),
            elapsed_sec=round(time.time() - t0, 2),
            extra={
                "aggregation": "fedavg_sample_weighted" if num_samples is not None
                               else "fedavg_uniform",
                "client_num_samples": dict(zip(participating, all_num_samples)),
                "per_client": per_client,
            },
        )
        self.round_results.append(rr)
        return rr

    def run(self, clients, num_rounds: Optional[int] = None) -> None:
        total = num_rounds or self.config.get("num_rounds", 10)

        ready = self.ready_clients()
        if not ready:
            raise RuntimeError("No clients passed preflight — cannot start FL.")

        active_clients = [c for c in clients if c.client_id in ready]
        print(f"\nStarting FL: {total} rounds, {len(active_clients)} clients")
        print("=" * 50)

        for r in range(1, total + 1):
            print(f"\nRound {r}/{total}")
            rr = self.run_round(active_clients, r)
            print(f"  Aggregated loss: {rr.aggregated_loss:.4f}  ({rr.elapsed_sec}s)")

        self.save_results()
        print("\nFL training complete.")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalised_weights(count: int, num_samples) -> list[float]:
        """Return per-client coefficients that sum to 1.

        Sample-count weighting n_k / n when ``num_samples`` is given and valid;
        uniform 1/K otherwise (with a warning, because uniform averaging is only
        equal to FedAvg when every client holds the same number of examples).
        """
        if num_samples is None:
            warnings.warn(
                "FLServer._fedavg: num_samples not provided; falling back to "
                "UNIFORM averaging (1/K). This equals FedAvg only when all "
                "clients hold the same number of examples.",
                RuntimeWarning, stacklevel=3,
            )
            return [1.0 / count] * count
        if len(num_samples) != count:
            raise ValueError(
                f"num_samples has {len(num_samples)} entries for {count} clients"
            )
        ns = [float(n) for n in num_samples]
        if any(n < 0 for n in ns):
            raise ValueError(f"num_samples must be non-negative: {ns}")
        total = sum(ns)
        if total <= 0:
            warnings.warn(
                "FLServer._fedavg: all clients report num_samples=0; falling "
                "back to UNIFORM averaging (1/K).",
                RuntimeWarning, stacklevel=3,
            )
            return [1.0 / count] * count
        return [n / total for n in ns]

    def _fedavg(
        self,
        weight_list: list[dict],
        num_samples: Optional[list[int]] = None,
    ) -> dict[str, torch.Tensor]:
        """Sample-weighted FedAvg: w = sum_k (n_k / n) w_k.

        Args:
            weight_list: one state_dict per client (same keys, CPU tensors).
            num_samples: number of training examples consumed by each client
                this round. If None, uniform averaging is used and a
                RuntimeWarning is emitted.
        """
        if not weight_list:
            raise ValueError("No weights to aggregate")
        coef = self._normalised_weights(len(weight_list), num_samples)
        avg = {}
        for key in weight_list[0]:
            stacked = torch.stack([w[key].float() for w in weight_list])
            c = torch.tensor(coef, dtype=stacked.dtype).view(-1, *([1] * (stacked.dim() - 1)))
            merged = (stacked * c).sum(dim=0)
            # Preserve integer buffers (e.g. BatchNorm num_batches_tracked)
            if not torch.is_floating_point(weight_list[0][key]):
                merged = merged.round().to(weight_list[0][key].dtype)
            avg[key] = merged
        return avg

    def _weighted_mean(self, values: list[float], num_samples=None) -> float:
        coef = self._normalised_weights(len(values), num_samples)
        return float(sum(c * v for c, v in zip(coef, values)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, name: str = "global_model.pt") -> Path:
        path = self.results_dir / name
        torch.save(self.weights, path)
        return path

    def save_results(self) -> Path:
        path = self.results_dir / "fl_results.json"
        data = [asdict(r) for r in self.round_results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
