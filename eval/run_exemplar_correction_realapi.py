"""
§5.9 Exemplar-correction re-experiment — REAL-API run (replaces dry-run upper bound).

For each of the 10 primary scripts, calls the LLM provider live with
Skill.FEW_SHOT_CORRECTED (corrected exemplar) and evaluates the generated
FL client module with the standard evaluator. Output mirrors
results/exemplar_correction.csv but with provider/method='few_shot_corrected_realapi'.

Outputs:
  results/exemplar_correction_realapi.csv
  eval/_few_shot_corrected_realapi/<script>_fl_few_shot_corrected_<provider>.py
"""
import argparse
import csv
import json
import subprocess
import sys
import textwrap
import time
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


def load_original_few_shot_results() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    csv_path = RESULTS_DIR / "benchmark.csv"
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == "few_shot":
                out[(row["framework"], row["script_name"])] = row
    return out


def load_drun_results() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    csv_path = RESULTS_DIR / "exemplar_correction.csv"
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["framework"], row["script"])] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="gemini",
                   choices=["claude", "gemini", "ollama"])
    p.add_argument("--out", default=str(RESULTS_DIR / "exemplar_correction_realapi.csv"))
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "_few_shot_corrected_realapi"))
    p.add_argument("--retry", type=int, default=2,
                   help="Number of retries on API/network failure per script")
    p.add_argument("--cached-only", action="store_true",
                   help="Never call the LLM: evaluate the files already in --out-dir.")
    p.add_argument("--data-root", default=".")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.cached_only:
        from autofl.converter.llm_converter import convert_llm, Skill, Provider
        prov_map = {
            "claude": Provider.CLAUDE,
            "gemini": Provider.GEMINI,
            "ollama": Provider.OLLAMA,
        }
        provider_enum = prov_map[args.provider]

    originals = load_original_few_shot_results()
    drun = load_drun_results()

    rows: list[dict] = []
    print(f"Provider: {args.provider}  |  Skill: FEW_SHOT_CORRECTED")
    print(f"{'script':40s}  {'orig_e2e':10s}  {'drun_e2e':10s}  {'real_e2e':10s}  real_err_stage")
    print("-" * 110)

    for fw, name in PRIMARY_SCRIPTS:
        src = BENCHMARKS_DIR / fw / f"{name}.py"
        if not src.exists():
            print(f"  {fw}/{name}: MISSING source")
            continue

        out_path = out_dir / f"{name}_fl_few_shot_corrected_{args.provider}.py"

        # Live LLM conversion with retries (skipped in --cached-only mode)
        conv_err = ""
        if args.cached_only:
            if not out_path.exists():
                print(f"  {fw}/{name}: MISSING cached output {out_path}")
                continue
        for attempt in range(0 if args.cached_only else args.retry + 1):
            try:
                convert_llm(
                    source_path=src,
                    skill=Skill.FEW_SHOT_CORRECTED,
                    output_path=out_path,
                    provider=provider_enum,
                )
                conv_err = ""
                break
            except Exception as e:
                conv_err = f"{type(e).__name__}: {e}"[:300]
                if attempt < args.retry:
                    time.sleep(5 * (attempt + 1))
                else:
                    out_path.write_text(f"# conversion error: {conv_err}\n")

        # Evaluate
        result = evaluate_one(out_path, name, fw, timeout_sec=args.timeout,
                              data_root=args.data_root, strict_data=args.strict,
                              spec_version="v1")

        orig_row = originals.get((fw, name), {})
        orig_e2e = (orig_row.get("e2e_runnable", "False") == "True")

        drun_row = drun.get((fw, name), {})
        drun_e2e = (drun_row.get("corrected_e2e_runnable", "False") in ("True", "true", True))

        real_e2e = result.e2e_runnable

        rows.append({
            "framework": fw,
            "script": name,
            "provider": args.provider,
            "skill": "few_shot_corrected",
            "original_e2e_runnable": orig_e2e,
            "drun_corrected_e2e_runnable": drun_e2e,
            "realapi_e2e_runnable": real_e2e,
            "realapi_preflight_pass": result.preflight_pass,
            "realapi_error_stage": result.error_stage,
            "realapi_error_message_head": (result.error_message or "")[:200].replace("\n", " | "),
            "conversion_error": conv_err,
            "data_path": result.data_path,
            "data_path_exists": result.data_path_exists,
            "strict_data": result.strict_data,
            "spec_version": result.spec_version,
            "dataset_class": result.dataset_class,
            "dataset_len": result.dataset_len,
            "elapsed_sec": result.elapsed_sec,
        })

        print(f"  {fw+'/'+name:38s}  {str(orig_e2e):10s}  {str(drun_e2e):10s}  "
              f"{str(real_e2e):10s}  {result.error_stage}")

    out_csv = Path(args.out)
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows -> {out_csv}")

    n         = len(rows)
    orig_pass = sum(1 for r in rows if r["original_e2e_runnable"])
    drun_pass = sum(1 for r in rows if r["drun_corrected_e2e_runnable"])
    real_pass = sum(1 for r in rows if r["realapi_e2e_runnable"])
    print("\n=== Summary ===")
    print(f"Original few-shot e2e success:                 {orig_pass}/{n}")
    print(f"Dry-run corrected-exemplar e2e (upper bound):  {drun_pass}/{n}")
    print(f"Real-API corrected-exemplar e2e ({args.provider:7s}):  {real_pass}/{n}")


if __name__ == "__main__":
    sys.exit(main())
