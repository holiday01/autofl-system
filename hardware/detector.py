"""
Detect local GPU hardware and suggest adaptive training parameters.
"""
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_mb: int
    vram_free_mb: int
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class HardwareProfile:
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_count: int = 1
    ram_gb: float = 0.0
    has_cuda: bool = False
    suggested_batch_size: int = 8
    suggested_num_workers: int = 2
    suggested_use_amp: bool = False
    suggested_gradient_accumulation: int = 1

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_mb for g in self.gpus)

    @property
    def primary_gpu_name(self) -> str:
        return self.gpus[0].name if self.gpus else "CPU"


def _query_nvidia_smi() -> list[GPUInfo]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                vram_mb=int(parts[2]),
                vram_free_mb=int(parts[3]),
                driver_version=parts[4],
            ))
        except ValueError:
            continue
    return gpus


def _suggest_params(gpus: list[GPUInfo]) -> dict:
    """Rule-based param suggestion from VRAM."""
    if not gpus:
        return {
            "batch_size": 4,
            "num_workers": 0,
            "use_amp": False,
            "gradient_accumulation": 1,
        }

    vram_mb = max(g.vram_mb for g in gpus)

    # batch_size ladder (conservative — leaves room for model + activations)
    if vram_mb >= 28_000:       # RTX 5090 / A100 40GB
        bs, amp, ga = 64, True, 1
    elif vram_mb >= 20_000:     # RTX 4090 / A100 20GB
        bs, amp, ga = 32, True, 1
    elif vram_mb >= 12_000:     # RTX 4080 / RTX 5070
        bs, amp, ga = 16, True, 1
    elif vram_mb >= 8_000:      # RTX 3070/4060
        bs, amp, ga = 8, True, 2
    elif vram_mb >= 4_000:
        bs, amp, ga = 4, False, 4
    else:
        bs, amp, ga = 2, False, 8

    import os
    cpu_count = os.cpu_count() or 1
    num_workers = min(4, cpu_count // 2)

    return {
        "batch_size": bs,
        "num_workers": num_workers,
        "use_amp": amp,
        "gradient_accumulation": ga,
    }


def detect() -> HardwareProfile:
    import os
    gpus = _query_nvidia_smi()

    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda and not gpus:
            # fallback via torch
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    vram_mb=props.total_memory // (1024 * 1024),
                    vram_free_mb=0,
                ))
    except ImportError:
        pass

    suggested = _suggest_params(gpus)

    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3) if _has_psutil() else 0.0

    return HardwareProfile(
        gpus=gpus,
        cpu_count=os.cpu_count() or 1,
        ram_gb=round(ram_gb, 1),
        has_cuda=has_cuda,
        suggested_batch_size=suggested["batch_size"],
        suggested_num_workers=suggested["num_workers"],
        suggested_use_amp=suggested["use_amp"],
        suggested_gradient_accumulation=suggested["gradient_accumulation"],
    )


def _has_psutil() -> bool:
    try:
        import psutil
        return True
    except ImportError:
        return False


def print_profile(profile: HardwareProfile) -> None:
    print("=" * 50)
    print("Hardware Profile")
    print("=" * 50)
    if profile.gpus:
        for g in profile.gpus:
            print(f"  GPU {g.index}: {g.name}  VRAM={g.vram_mb}MB  Free={g.vram_free_mb}MB")
    else:
        print("  No GPU detected (CPU mode)")
    print(f"  CPU cores : {profile.cpu_count}")
    print(f"  RAM       : {profile.ram_gb} GB")
    print(f"  CUDA      : {profile.has_cuda}")
    print()
    print("Suggested local overrides:")
    print(f"  batch_size           = {profile.suggested_batch_size}")
    print(f"  num_workers          = {profile.suggested_num_workers}")
    print(f"  use_amp              = {profile.suggested_use_amp}")
    print(f"  gradient_accumulation= {profile.suggested_gradient_accumulation}")
    print("=" * 50)


if __name__ == "__main__":
    print_profile(detect())
