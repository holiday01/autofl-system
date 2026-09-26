"""
Shared helpers for the Phase-2 experiment runners:
  eval/run_self_repair.py, eval/run_ablation.py, eval/run_repeated.py

All runners:
  * evaluate generated modules with autofl.eval.evaluator.evaluate (five-stage
    preflight, CPU-forced like the primary benchmark);
  * never overwrite archived v1 files; outputs go to results/<name>.csv and
    eval/_<name>/<provider>/...;
  * support --dry-run (no LLM call; the cached structured client of the same
    script is used as the fake response), --max-calls, --resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../autofl
sys.path.insert(0, str(ROOT.parent))                    # so `autofl.*` imports work

BENCHMARKS_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "results"
FRAMEWORK_DIRS = ["pytorch", "tensorflow", "monai", "lightning", "sklearn", "xgboost"]
DL_FRAMEWORKS = ["pytorch", "tensorflow", "monai", "lightning"]

USAGE_FIELDS = ["llm_" + f for f in ("provider", "model", "input_tokens", "output_tokens",
                                     "thinking_tokens", "cost_usd", "cost_source", "elapsed_sec",
                                     "temperature", "dry_run", "price_version")]


def iter_scripts(frameworks: list[str], only: list[str] | None = None):
    """Yield (framework, source_path) for original (non-_fl_) benchmark scripts."""
    for fw in frameworks:
        d = BENCHMARKS_DIR / fw
        if not d.exists():
            continue
        for p in sorted(d.glob("*.py")):
            if "_fl_" in p.name or p.name == "__init__.py":
                continue
            if only and p.stem not in only:
                continue
            yield fw, p


def cached_client(fw: str, stem: str, method: str) -> Path:
    return BENCHMARKS_DIR / fw / f"{stem}_fl_{method}.py"


def add_common_args(p: argparse.ArgumentParser, default_out: str, default_dir: str) -> None:
    p.add_argument("--provider", default="claude", choices=["claude", "gemini", "ollama"])
    p.add_argument("--frameworks", nargs="+", default=DL_FRAMEWORKS, choices=FRAMEWORK_DIRS)
    p.add_argument("--scripts", nargs="+", default=None,
                   help="script stems to include (default: all in --frameworks)")
    p.add_argument("--data-root", default=".", help="config['data_path'] handed to the module")
    p.add_argument("--strict", action="store_true", help="strict data mode (fail closed)")
    p.add_argument("--dry-run", action="store_true",
                   help="no LLM calls; cached structured client is used as the fake response")
    p.add_argument("--max-calls", type=int, default=None, help="hard cap on live LLM calls")
    p.add_argument("--resume", action="store_true", help="skip rows already in --out")
    p.add_argument("--out", default=str(RESULTS_DIR / default_out))
    p.add_argument("--outdir", default=str(ROOT / "eval" / default_dir))
    p.add_argument("--temperature", type=float, default=None,
                   help="Gemini/Ollama sampling temperature (None = provider default; "
                        "the Claude CLI has no temperature control)")


EVAL_TIMEOUT_SEC = int(os.environ.get("AUTOFL_EVAL_TIMEOUT", "1800"))


def evaluate_module(path: Path, script_name: str, method: str, framework: str,
                    data_root: str, strict: bool, spec_version: str,
                    timeout: int | None = None) -> dict:
    """Evaluate one generated module in a subprocess with a wall-clock timeout.

    A client whose one-round simulation would run for hours (for example a
    synthetic ImageNet loader on CPU) is reported as an `eval_timeout` row
    instead of blocking the experiment. The subprocess keeps the same
    CPU-forced settings as the primary benchmark.
    """
    cmd = [sys.executable, str(ROOT / "eval" / "eval_one.py"),
           "--path", str(path), "--script", script_name, "--method", method,
           "--framework", framework, "--data-root", str(data_root),
           "--spec-version", spec_version or ""]
    if strict:
        cmd.append("--strict")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    t = timeout or EVAL_TIMEOUT_SEC
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t, env=env, cwd=str(ROOT))
        line = next((l for l in r.stdout.splitlines() if l.startswith("@@EVAL_JSON@@")), None)
        if line is None:
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            raise RuntimeError(tail[-1] if tail else f"evaluator exited with {r.returncode}")
        d = json.loads(line[len("@@EVAL_JSON@@"):])
    except subprocess.TimeoutExpired:
        d = {"error_stage": "eval_timeout",
             "error_message": f"evaluation exceeded {t}s (killed)",
             "syntax_valid": True, "e2e_runnable": False, "preflight_pass": False,
             "data_path": str(data_root), "strict_data": bool(strict),
             "spec_version": spec_version or ""}
    except Exception as e:
        d = {"error_stage": "eval_error", "error_message": str(e)[:300],
             "e2e_runnable": False, "preflight_pass": False,
             "data_path": str(data_root), "strict_data": bool(strict),
             "spec_version": spec_version or ""}
    for k in ("script_name", "method", "framework"):   # identity keys are set by base_row
        d.pop(k, None)
    d["error_message"] = (d.get("error_message") or "").splitlines()[0][:300] \
        if d.get("error_message") else ""
    return d


def nval(v) -> str:
    """Normalize a cell-key value so that 0, "0" and "0.0" compare equal.

    A CSV that has been round-tripped through pandas writes integer columns
    holding any blank as floats ("0" becomes "0.0"), which silently broke
    --resume matching and re-evaluated every cell.
    """
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else repr(f)
    except (TypeError, ValueError):
        return s


def cell_key(*vals) -> tuple:
    return tuple(nval(v) for v in vals)


class CsvWriter:
    """Append-only CSV with a union header; rows may add fields in any order."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self.existing: list[dict] = []
        if self.path.exists():
            with open(self.path, newline="") as f:
                rd = csv.DictReader(f)
                self.existing = list(rd)
                for c in (rd.fieldnames or []):
                    if c not in self.fieldnames:
                        self.fieldnames.append(c)
        self._rewrite()

    def _rewrite(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in self.existing:
                w.writerow(r)

    def keys(self, cols: list[str]) -> set[tuple]:
        return {cell_key(*(r.get(c, "") for c in cols)) for r in self.existing}

    def append(self, row: dict) -> None:
        new_cols = [k for k in row if k not in self.fieldnames]
        if new_cols:
            self.fieldnames += new_cols
            self.existing.append(row)
            self._rewrite()
            return
        self.existing.append(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            w.writerow(row)


def rel(path: Path) -> str:
    """Path relative to the repo root when inside it, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def base_row(script: str, fw: str, method: str, provider: str, **extra) -> dict:
    r = {"script_name": script, "framework": fw, "method": method, "provider": provider,
         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    r.update(extra)
    return r


EVAL_FIELDS = ["syntax_valid", "interface_complete", "build_model_present",
               "build_dataloader_present", "train_step_present", "optimizer_detected",
               "component_coverage", "preflight_pass", "e2e_runnable", "error_stage",
               "error_message", "elapsed_sec", "data_path", "data_path_exists",
               "strict_data", "spec_version", "dataset_class", "dataset_len"]
