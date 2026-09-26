"""
§5.7 Exemplar-correction re-experiment.

Goal: isolate prompting-strategy effects from exemplar-quality effects.

Background
----------
The pre-cached `_fl_few_shot.py` outputs in `benchmarks/{pytorch,tensorflow,monai,
lightning}/` were generated against an earlier version of the few-shot exemplar
(`examples/wsi_mlp_train_fl_client.py`) whose `train_step` violated the AutoFL
contract: it called `loss.backward(); optimizer.step(); return loss.detach()`.
The exemplar has since been corrected. A reviewer raised the concern that the
1/10 success rate reported for FEW_SHOT may reflect bad exemplar quality, not
the prompting strategy itself.

Methodology (dry-run analysis)
------------------------------
1. Detect contract violations statically in each existing `_fl_few_shot.py`'s
   `train_step` body: presence of `.backward()`, `optimizer.step()` (or
   `.step()` on an attribute named optimizer), `.detach()` on the returned
   loss, or `.item()` on the returned loss.
2. For each file with a violation, write a *patched* copy in which the
   violation is mechanically removed:  `loss.backward()` and `optimizer.step()`
   blocks (often gated by `if optimizer is not None:` in these outputs)
   are deleted, and `return loss.detach()` becomes `return loss`. Files
   already compliant are copied verbatim.
3. Run the standard `eval/evaluator.py` against each patched file, with the
   same `_MINIMAL_CONFIG` used by `run_benchmark.py`, so the comparison is
   apples-to-apples.

Output: results/exemplar_correction.csv with one row per script, columns:
  script, framework, original_violation_in_train_step,
  original_e2e_runnable (from results/benchmark.csv),
  corrected_e2e_runnable (from this run),
  cured_by_exemplar_fix, original_error_stage, corrected_error_stage.

This is a dry-run because we are NOT calling the LLM with the corrected
exemplar; instead we surgically remove the contract-violating lines from the
already-generated few-shot outputs. This is a sound *upper bound* on the
corrected-exemplar success rate: the LLM, given a corrected exemplar, would
in expectation produce at least as compliant a `train_step` as our mechanical
patch (which strictly removes violating statements but otherwise preserves
the LLM's choices). Failures that survive the patch (e.g. missing modules,
missing dataset paths) are independent of exemplar quality and would not
be cured by a corrected prompt either.
"""
import argparse
import ast
import csv
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent))

BENCHMARKS_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PRIMARY_SCRIPTS = [
    ("pytorch",    "dcgan_main"),
    ("pytorch",    "imagenet_main"),
    ("pytorch",    "mnist_main"),
    ("tensorflow", "image_classification_from_scratch"),
    ("tensorflow", "lstm_seq2seq"),
    ("tensorflow", "mnist_convnet"),
    ("monai",      "mednist_tutorial"),
    ("monai",      "spleen_segmentation_3d"),
    ("lightning",  "backbone_image_classifier"),
    ("lightning",  "mnist_lite"),
]


# ── Static violation detection ───────────────────────────────────────────────

@dataclass
class ViolationReport:
    has_backward_call: bool = False
    has_optimizer_step: bool = False
    returns_detached: bool = False
    returns_item: bool = False

    @property
    def any(self) -> bool:
        return any([self.has_backward_call, self.has_optimizer_step,
                    self.returns_detached, self.returns_item])

    def summary(self) -> str:
        flags = []
        if self.has_backward_call:  flags.append("backward")
        if self.has_optimizer_step: flags.append("opt.step")
        if self.returns_detached:   flags.append("return .detach()")
        if self.returns_item:       flags.append("return .item()")
        return ",".join(flags) if flags else "none"


def detect_violations(source: str) -> ViolationReport:
    """AST-walk the train_step body and report contract violations."""
    rep = ViolationReport()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return rep

    train_step_node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "train_step"),
        None,
    )
    if train_step_node is None:
        return rep

    for node in ast.walk(train_step_node):
        # *.backward()
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "backward"):
            rep.has_backward_call = True
        # optimizer.step() / scaler.step(optimizer)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "step"):
            # Heuristic: any .step() call inside train_step is a violation.
            # train_step never legitimately advances the optimizer or scaler.
            rep.has_optimizer_step = True
        # return X.detach()  / return X.item()
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == "detach":
                    rep.returns_detached = True
                if node.value.func.attr == "item":
                    rep.returns_item = True
    return rep


# ── Mechanical patch ─────────────────────────────────────────────────────────

def patch_source(source: str) -> str:
    """
    Surgically remove contract violations inside `train_step`.

    Strategy:
      - Find the `train_step` function span by line numbers (ast).
      - Within that span:
          * Drop any line containing `.backward()`, `optimizer.step()`,
            `scaler.step(`, `scaler.scale(`, `scaler.update()`,
            `scaler.unscale_(`, `optimizer.zero_grad`.
          * Drop any `if optimizer is not None:` block (and its body) — it
            only ever wraps the backward+step calls in these few-shot
            outputs. We delete the entire block.
          * Replace `return <expr>.detach()` with `return <expr>`.
          * Replace `return <expr>.item()` with `return <expr>`.
    """
    tree = ast.parse(source)
    train_step_node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "train_step"),
        None,
    )
    if train_step_node is None:
        return source

    start = train_step_node.lineno - 1            # inclusive (0-indexed)
    end   = train_step_node.end_lineno            # exclusive (1-indexed → exclusive)

    lines = source.splitlines()
    head, body, tail = lines[:start], lines[start:end], lines[end:]

    # 1. Identify line ranges of `if optimizer is not None:` blocks to delete.
    drop_ranges: list[tuple[int, int]] = []
    for node in ast.walk(train_step_node):
        if isinstance(node, ast.If):
            test = node.test
            # Match `if optimizer is not None`
            if (isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "optimizer"
                    and any(isinstance(op, ast.IsNot) for op in test.ops)):
                drop_ranges.append((node.lineno - 1, node.end_lineno))  # absolute
            # Match `if optimizer is not None and ...`
            elif (isinstance(test, ast.BoolOp)
                  and any(
                      isinstance(v, ast.Compare)
                      and isinstance(v.left, ast.Name)
                      and v.left.id == "optimizer"
                      and any(isinstance(op, ast.IsNot) for op in v.ops)
                      for v in test.values
                  )):
                drop_ranges.append((node.lineno - 1, node.end_lineno))

    drop_set: set[int] = set()
    for s, e in drop_ranges:
        for i in range(s, e):
            drop_set.add(i)

    new_body: list[str] = []
    for idx, line in enumerate(body, start=start):  # idx is absolute line index
        if idx in drop_set:
            continue
        stripped = line.strip()
        # Drop standalone backward/step/scaler calls (defensive — most are
        # already inside the `if optimizer is not None:` block we deleted).
        if any(tok in line for tok in (
            ".backward()", "optimizer.step(", "scaler.step(", "scaler.scale(",
            "scaler.update()", "scaler.unscale_(", "optimizer.zero_grad",
        )):
            continue
        # Replace `return X.detach()` with `return X` (and same for .item()).
        m = re.match(r'^(\s*return\s+.+?)\.detach\(\)\s*$', line)
        if m:
            line = m.group(1)
        else:
            m = re.match(r'^(\s*return\s+.+?)\.item\(\)\s*$', line)
            if m:
                line = m.group(1)
        new_body.append(line)

    return "\n".join(head + new_body + tail) + ("\n" if source.endswith("\n") else "")


# ── Per-script evaluation ────────────────────────────────────────────────────

@dataclass
class _SerializableResult:
    e2e_runnable: bool = False
    preflight_pass: bool = False
    error_stage: str = ""
    error_message: str = ""
    data_path: str = ""
    data_path_exists: bool = False
    strict_data: bool = False
    spec_version: str = ""
    dataset_class: str = ""
    dataset_len: int = -1
    elapsed_sec: float = 0.0


def evaluate_one(fl_path: Path, script_name: str, framework: str,
                 timeout_sec: int = 1800, data_root: str = ".",
                 strict_data: bool = False, spec_version: str = "v1"):
    """
    Run eval/evaluator.py in a subprocess so we can time-bound it.

    Some benchmark scripts (e.g. DCGAN with CIFAR10 download, ImageNet) can
    hang or run for many minutes inside `_stage_data_load` / e2e simulation;
    a hard timeout keeps the §5.7 re-experiment tractable. A timed-out script
    is recorded as a non-pass with error_stage='timeout'.
    """
    import json
    import subprocess
    import textwrap

    runner = textwrap.dedent(f"""
        import json, sys, traceback
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent.parent)!r})
        try:
            from autofl.eval.evaluator import evaluate
            r = evaluate(
                generated_path={str(fl_path)!r},
                script_name={script_name!r},
                method='few_shot_corrected',
                framework={framework!r},
                data_root={data_root!r},
                strict_data={strict_data!r},
                spec_version={spec_version!r},
            )
            payload = {{
                'e2e_runnable':   bool(r.e2e_runnable),
                'preflight_pass': bool(r.preflight_pass),
                'error_stage':    r.error_stage or '',
                'error_message':  (r.error_message or '')[:500],
                'data_path':        r.data_path,
                'data_path_exists': bool(r.data_path_exists),
                'strict_data':      bool(r.strict_data),
                'spec_version':     r.spec_version,
                'dataset_class':    r.dataset_class,
                'dataset_len':      int(r.dataset_len),
                'elapsed_sec':      float(r.elapsed_sec),
            }}
        except Exception as e:
            payload = {{
                'e2e_runnable': False, 'preflight_pass': False,
                'error_stage': 'evaluator_exception',
                'error_message': traceback.format_exc()[:500],
            }}
        print('__JSON__' + json.dumps(payload))
    """).strip()

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        out = proc.stdout
        for line in out.splitlines()[::-1]:
            if line.startswith("__JSON__"):
                d = json.loads(line[len("__JSON__"):])
                return _SerializableResult(**d)
        return _SerializableResult(
            error_stage="evaluator_no_output",
            error_message=(proc.stderr or out)[:500],
        )
    except subprocess.TimeoutExpired:
        return _SerializableResult(
            error_stage="timeout",
            error_message=f"evaluator exceeded {timeout_sec}s wall-time",
        )


# ── Original CSV lookup ──────────────────────────────────────────────────────

def load_original_few_shot_results() -> dict[tuple[str, str], dict]:
    """Map (framework, script_name) → row from results/benchmark.csv."""
    out: dict[tuple[str, str], dict] = {}
    csv_path = RESULTS_DIR / "benchmark.csv"
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == "few_shot":
                out[(row["framework"], row["script_name"])] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(RESULTS_DIR / "exemplar_correction.csv"))
    p.add_argument("--patched-dir", default=str(ROOT / "eval" / "_few_shot_corrected_patched"))
    p.add_argument("--reuse-patched", action="store_true",
                   help="Do not re-patch; evaluate the files already in --patched-dir.")
    p.add_argument("--data-root", default=".")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    patched_dir = Path(args.patched_dir)
    patched_dir.mkdir(parents=True, exist_ok=True)

    originals = load_original_few_shot_results()

    rows: list[dict] = []
    print(f"{'script':40s}  {'orig_viol':18s}  {'orig_e2e':10s}  {'corr_e2e':10s}  {'cured':6s}  corr_err_stage")
    print("-" * 120)

    for fw, name in PRIMARY_SCRIPTS:
        src = BENCHMARKS_DIR / fw / f"{name}_fl_few_shot.py"
        if not src.exists():
            print(f"  {fw}/{name}: MISSING")
            continue

        source = src.read_text()
        viol = detect_violations(source)

        # Write patched copy (or reuse the cached one verbatim)
        patched_path = patched_dir / f"{name}_fl_few_shot_corrected.py"
        if args.reuse_patched and patched_path.exists():
            pass
        else:
            patched_source = patch_source(source) if viol.any else source
            patched_path.write_text(patched_source)

        # Original outcome from benchmark.csv
        orig_row = originals.get((fw, name), {})
        orig_e2e = (orig_row.get("e2e_runnable", "False") == "True")
        orig_err_stage = orig_row.get("error_stage", "")

        # Evaluate patched
        result = evaluate_one(patched_path, name, fw, timeout_sec=args.timeout,
                              data_root=args.data_root, strict_data=args.strict,
                              spec_version="v1")
        corr_e2e = result.e2e_runnable
        corr_err_stage = result.error_stage

        cured = (not orig_e2e) and corr_e2e

        rows.append({
            "framework": fw,
            "script": name,
            "violations_in_original_train_step": viol.summary(),
            "original_e2e_runnable": orig_e2e,
            "original_error_stage": orig_err_stage,
            "corrected_e2e_runnable": corr_e2e,
            "corrected_error_stage": corr_err_stage,
            "cured_by_exemplar_fix": cured,
            "corrected_preflight_pass": result.preflight_pass,
            "corrected_error_message_head": (result.error_message or "")[:200].replace("\n", " | "),
            "data_path": result.data_path,
            "data_path_exists": result.data_path_exists,
            "strict_data": result.strict_data,
            "spec_version": result.spec_version,
            "dataset_class": result.dataset_class,
            "dataset_len": result.dataset_len,
            "elapsed_sec": result.elapsed_sec,
        })

        print(f"  {fw+'/'+name:38s}  {viol.summary():18s}  "
              f"{str(orig_e2e):10s}  {str(corr_e2e):10s}  {str(cured):6s}  {corr_err_stage}")

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {out_path}")

    # Summary
    n        = len(rows)
    orig_pass = sum(1 for r in rows if r["original_e2e_runnable"])
    corr_pass = sum(1 for r in rows if r["corrected_e2e_runnable"])
    print(f"\n=== Summary ===")
    print(f"Original few-shot e2e success: {orig_pass}/{n}")
    print(f"Corrected-exemplar few-shot e2e success: {corr_pass}/{n}")


if __name__ == "__main__":
    sys.exit(main())
