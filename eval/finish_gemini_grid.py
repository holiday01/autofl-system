#!/usr/bin/env python3
"""
Unattended driver that completes the Gemini repeated-sampling grid against a
free-tier quota of 20 generate_content requests per day.

Each pass runs eval/run_repeated.py with --resume, which regenerates only the
cells that are still missing. When the daily quota is exhausted the runner stops
cleanly (QuotaExhausted), and this driver sleeps until just after the quota
resets at midnight America/Los_Angeles, then tries again. It exits when the grid
is complete.

    nohup python eval/finish_gemini_grid.py > results/logs_v2/gemini_grid_driver.log 2>&1 &

State lives entirely in results/repeated_gemini.csv, so the driver can be killed
and restarted at any time.
"""
from __future__ import annotations

import datetime as dt
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "repeated_gemini.csv"
LOG = ROOT / "results" / "logs_v2" / "phase2_repeated_gemini.log"
SCRIPTS = ["dcgan_main", "imagenet_main", "mnist_main", "image_classification_from_scratch",
           "lstm_seq2seq", "mnist_convnet", "mednist_tutorial", "spleen_segmentation_3d",
           "backbone_image_classifier", "mnist_lite"]
METHODS = ["structured", "few_shot_corrected"]
SAMPLES = 5
PACIFIC = ZoneInfo("America/Los_Angeles")
DAILY_CALLS = 18          # leave headroom below the 20/day free-tier quota


def say(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%SZ}] {msg}", flush=True)


def remaining() -> int:
    if not CSV.exists():
        return len(SCRIPTS) * len(METHODS) * SAMPLES
    d = pd.read_csv(CSV)
    d = d[d.error_stage.astype(str) != "generation"]          # failed generations must be retried
    done = {(a, b, int(c)) for a, b, c in zip(d.script_name, d.method, d["sample"])}
    return sum(1 for c in itertools.product(SCRIPTS, METHODS, range(SAMPLES)) if c not in done)


def drop_generation_rows() -> None:
    if not CSV.exists():
        return
    d = pd.read_csv(CSV)
    m = d.error_stage.astype(str) == "generation"
    if m.any():
        say(f"dropping {int(m.sum())} rows that failed at generation so they are retried")
        d[~m].to_csv(CSV, index=False)


def seconds_to_reset() -> float:
    now = dt.datetime.now(PACIFIC)
    nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    return (nxt - now).total_seconds()


def one_pass() -> int:
    """Run the resumable runner once; return the number of cells it completed."""
    before = remaining()
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", AUTOFL_EVAL_TIMEOUT="2400")
    cmd = [sys.executable, str(ROOT / "eval" / "run_repeated.py"),
           "--provider", "gemini", "--methods", *METHODS,
           "--samples", str(SAMPLES), "--max-calls", str(DAILY_CALLS),
           "--out", str(CSV), "--outdir", str(ROOT / "eval" / "_repeated"), "--resume"]
    with open(LOG, "a") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    after = remaining()
    return before - after


def main() -> None:
    say(f"driver starting; {remaining()} of {len(SCRIPTS)*len(METHODS)*SAMPLES} cells remaining")
    idle_passes = 0
    while True:
        left = remaining()
        if left == 0:
            say("grid complete")
            return
        drop_generation_rows()
        say(f"pass starting; {left} cells remaining")
        got = one_pass()
        say(f"pass finished; {got} cells completed, {remaining()} remaining")
        if remaining() == 0:
            say("grid complete")
            return
        idle_passes = idle_passes + 1 if got == 0 else 0
        if idle_passes >= 3:
            say("three consecutive passes made no progress; stopping for inspection")
            return
        wait = seconds_to_reset()
        say(f"sleeping {wait/3600:.1f} h until the quota resets")
        time.sleep(wait + 60)


if __name__ == "__main__":
    main()
