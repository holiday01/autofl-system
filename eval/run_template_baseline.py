"""
Template-converter baseline: convert every benchmark source with the rule-based
converter (converter/template_converter.py — no LLM) and evaluate the result
with the same evaluator/preflight/e2e pipeline as run_benchmark.py.

Usage:
  python eval/run_template_baseline.py [--output results/template_baseline.csv]
                                       [--frameworks pytorch tensorflow ...]
                                       [--only mnist_main dcgan_main ...]
                                       [--data-root .] [--strict] [--convert-only]

Outputs:
  benchmarks/<fw>/<stem>_fl_template.py      generated client modules
  results/template_baseline.csv              one row per script, run_benchmark.py
                                             columns + template provenance columns
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

# CPU-only, exactly like preflight/validator.py (which sets this at import time).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent))

BENCHMARKS_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FRAMEWORK_DIRS = ["pytorch", "tensorflow", "monai", "lightning", "sklearn", "xgboost"]

# Same columns as eval/run_benchmark.py (kept in sync by importing them) ...
from autofl.eval.run_benchmark import FIELDNAMES as _BENCH_FIELDNAMES, iter_scripts  # noqa: E402

# ... plus template provenance columns.
EXTRA_FIELDS = [
    "template_model", "template_train_step", "n_rules", "n_fallbacks",
    "rules_fired", "fallbacks_used", "converter_warnings", "convert_error",
]
FIELDNAMES = list(_BENCH_FIELDNAMES) + EXTRA_FIELDS


def _write_csv(out_path: Path, rows: list[dict]) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="AutoFL template-converter baseline")
    p.add_argument("--output", default=str(RESULTS_DIR / "template_baseline.csv"))
    p.add_argument("--frameworks", nargs="+", default=FRAMEWORK_DIRS, choices=FRAMEWORK_DIRS)
    p.add_argument("--only", nargs="*", default=None, help="script stems to run (default: all)")
    p.add_argument("--data-root", default=".")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--convert-only", action="store_true", help="generate files, skip evaluation")
    args = p.parse_args(argv)

    from autofl.converter.template_converter import convert_source
    from autofl.eval.evaluator import evaluate

    rows = []
    scripts = [(fw, s) for fw, s in iter_scripts(args.frameworks) if not args.only or s.stem in args.only]
    print(f"Template baseline: {len(scripts)} scripts  data_root={args.data_root!r}  strict={args.strict}")
    for fw, script in scripts:
        out = script.parent / f"{script.stem}_fl_template.py"
        print(f"\nScript: {fw}/{script.name}")
        t0 = time.time()
        meta, convert_error = None, ""
        try:
            code, meta = convert_source(script.read_text(), f"benchmarks/{fw}/{script.name}", fw)
            out.write_text(code)
            print(f"  [convert] -> {out.name}  model={meta.model_fallback} train_step={meta.train_step_mode} "
                  f"rules={len(meta.rules)} fallbacks={len(meta.fallbacks)} ({time.time() - t0:.1f}s)")
        except Exception as e:  # converter crash: record and emit a stub so evaluate() reports it
            convert_error = f"{type(e).__name__}: {e}"
            out.write_text(f"# template converter error: {convert_error}\n")
            print(f"  [convert] ERROR {convert_error}")
        if args.convert_only:
            continue
        result = evaluate(
            generated_path=out, script_name=script.stem, method="template", framework=fw,
            data_root=args.data_root, strict_data=args.strict, spec_version="n/a",
        )
        status = "OK" if result.e2e_runnable else (
            "preflight" if result.preflight_pass else (
                "interface" if result.interface_complete else ("syntax" if result.syntax_valid else "FAIL")))
        print(f"  [eval] {status}  stage={result.error_stage or '-'}  {result.elapsed_sec}s  "
              f"{(result.error_message or '').splitlines()[0][:110] if result.error_message else ''}")
        row = result.to_dict()
        row["provider"] = "none"
        row["template_model"] = meta.model_fallback if meta else ""
        row["template_train_step"] = meta.train_step_mode if meta else ""
        row["n_rules"] = len(meta.rules) if meta else 0
        row["n_fallbacks"] = len(meta.fallbacks) if meta else 0
        row["rules_fired"] = " | ".join(meta.rules) if meta else ""
        row["fallbacks_used"] = " | ".join(meta.fallbacks) if meta else ""
        row["converter_warnings"] = " | ".join(meta.warnings) if meta else ""
        row["convert_error"] = convert_error
        rows.append(row)
        _write_csv(Path(args.output), rows)      # incremental: a killed run keeps its rows

    if args.convert_only:
        return 0
    out_path = Path(args.output)
    _write_csv(out_path, rows)
    print(f"\nSaved {len(rows)} rows -> {out_path}")
    n_ok = sum(1 for r in rows if r["e2e_runnable"])
    n_pf = sum(1 for r in rows if r["preflight_pass"])
    print(f"\n{'script':36s} {'fw':11s} {'stage':22s} {'preflight':9s} {'e2e':5s} model/train_step")
    for r in rows:
        print(f"{r['script_name']:36s} {r['framework']:11s} {(r['error_stage'] or 'PASS'):22s} "
              f"{str(r['preflight_pass']):9s} {str(r['e2e_runnable']):5s} {r['template_model']}/{r['template_train_step']}")
    print(f"\npreflight_pass: {n_pf}/{len(rows)}   e2e_runnable: {n_ok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
