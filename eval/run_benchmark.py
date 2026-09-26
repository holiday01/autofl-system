"""
Run the full benchmark: each script × each conversion method → EvalResult.

For LLM methods (zero_shot, few_shot, structured):
  - Default (--provider claude): uses pre-generated _fl_ files (no API calls).
  - With --provider gemini or --provider ollama: calls convert_llm() live to
    regenerate the FL client, then evaluates the freshly generated file.
  - With --regenerate: forces live conversion even for the claude provider.
For AST: the converter always runs on the original script.

Usage:
  python eval/run_benchmark.py [--output results/benchmark.csv]
                               [--methods ast zero_shot few_shot structured]
                               [--frameworks pytorch tensorflow monai lightning]
                               [--provider claude|gemini|ollama]
                               [--regenerate]
                               [--cached-only]          # never call an LLM
                               [--data-root .] [--strict]
                               [--spec-version v2]      # prompt spec for LIVE conversions

Cached-only mode (--cached-only): evaluates the pre-generated _fl_ files for
zero_shot/few_shot/structured and re-runs the deterministic, local AST
converter; any request that would require an LLM call aborts the run.
Cached LLM outputs are recorded with spec_version="v1" (the prompt set they
were generated with); AST rows are recorded as "n/a".
"""
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent))

BENCHMARKS_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FRAMEWORK_DIRS = ["pytorch", "tensorflow", "monai", "lightning", "sklearn", "xgboost"]

ALL_METHODS = ["ast", "zero_shot", "few_shot", "structured"]

FIELDNAMES = [
    "script_name", "framework", "method", "provider",
    "syntax_valid", "interface_complete",
    "build_model_present", "build_dataloader_present", "train_step_present",
    "optimizer_detected", "component_coverage",
    "preflight_pass", "e2e_runnable",
    "error_stage", "error_message", "elapsed_sec",
    # v2 provenance columns
    "data_path", "data_path_exists", "strict_data", "spec_version",
    "dataset_class", "dataset_len",
]

CACHED_SPEC_VERSION = "v1"   # prompt-spec version of the pre-generated _fl_ files


def iter_scripts(frameworks: list[str]):
    """Yield (framework, original_script_path) for each original (non-_fl_) script."""
    for fw in frameworks:
        d = BENCHMARKS_DIR / fw
        if not d.exists():
            print(f"  [skip] benchmarks/{fw}/ not found")
            continue
        for p in sorted(d.glob("*.py")):
            if "_fl_" not in p.name and p.name != "__init__.py":
                yield fw, p


def get_generated_path(
    script_path: Path,
    method: str,
    tmp_dir: Path,
    provider: str,
    regenerate: bool,
    cached_only: bool = False,
    spec_version: str = "v2",
) -> tuple[Path | None, str]:
    """
    Return (path to the file that should be evaluated, spec_version used).
    """
    if method == "ast":
        from autofl.converter.ast_converter import convert
        out = tmp_dir / f"{script_path.stem}_fl_ast.py"
        try:
            convert(script_path, out)
        except Exception as e:
            out.write_text(f"# converter error: {e}\n")
        return out, "n/a"

    # LLM methods ─────────────────────────────────────────────────────────────
    use_pregenerated = (provider == "claude") and not regenerate

    if use_pregenerated:
        expected = script_path.parent / f"{script_path.stem}_fl_{method}.py"
        return (expected if expected.exists() else None), CACHED_SPEC_VERSION

    if cached_only:
        raise RuntimeError(
            f"--cached-only: refusing live LLM conversion for "
            f"{script_path.name} / {method} / {provider}"
        )

    # Live conversion via LLM provider
    from autofl.converter.llm_converter import convert_llm, Skill, Provider
    skill_map = {
        "zero_shot":  Skill.ZERO_SHOT,
        "few_shot":   Skill.FEW_SHOT,
        "structured": Skill.STRUCTURED,
    }
    prov_map = {
        "claude": Provider.CLAUDE,
        "gemini": Provider.GEMINI,
        "ollama": Provider.OLLAMA,
    }
    out = tmp_dir / f"{script_path.stem}_fl_{method}_{provider}.py"
    try:
        convert_llm(
            source_path=script_path,
            skill=skill_map[method],
            output_path=out,
            provider=prov_map[provider],
            spec_version=spec_version,
        )
    except Exception as e:
        out.write_text(f"# conversion error: {e}\n")
    return out, spec_version


def main(argv=None):
    parser = argparse.ArgumentParser(description="AutoFL benchmark runner")
    parser.add_argument("--output", default=str(RESULTS_DIR / "benchmark.csv"))
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument(
        "--frameworks", nargs="+", default=FRAMEWORK_DIRS,
        choices=FRAMEWORK_DIRS,
    )
    parser.add_argument(
        "--provider", default="claude",
        choices=["claude", "gemini", "ollama"],
        help="LLM provider for live conversion (claude uses pre-generated files by default)",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Force live LLM conversion even for the claude provider",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing output CSV and skip already-completed (script, method, provider) pairs.",
    )
    parser.add_argument(
        "--cached-only", action="store_true",
        help="Never call an LLM: evaluate pre-generated _fl_ files + local AST converter only.",
    )
    parser.add_argument(
        "--data-root", default=".",
        help="Value passed to the generated module as config['data_path'] (default '.').",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict data mode: fail preflight if --data-root does not exist and "
             "pass allow_synthetic_data=False to the generated module.",
    )
    parser.add_argument(
        "--spec-version", default="v2", choices=["v1", "v2"],
        help="Prompt-spec version used for LIVE conversions (cached files are always v1).",
    )
    args = parser.parse_args(argv)
    if args.cached_only and (args.regenerate or args.provider != "claude"):
        parser.error("--cached-only is incompatible with --regenerate / non-claude providers")

    from autofl.eval.evaluator import evaluate

    out_path = Path(args.output)

    # Load existing rows when resuming
    existing_rows: list[dict] = []
    done_keys: set[tuple[str, str, str]] = set()
    if args.resume and out_path.exists():
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                done_keys.add((row["script_name"], row["method"], row.get("provider", "claude")))
        print(f"Resume mode: loaded {len(existing_rows)} existing rows, "
              f"skipping {len(done_keys)} (script, method, provider) triples.\n")

    scripts = list(iter_scripts(args.frameworks))
    total = len(scripts) * len(args.methods)
    print(f"Provider: {args.provider}  |  Regenerate: {args.regenerate}  |  "
          f"Cached-only: {args.cached_only}  |  data_root={args.data_root!r}  |  "
          f"strict={args.strict}")
    print(f"Running {len(scripts)} scripts × {len(args.methods)} methods = {total} evaluations")
    print(f"Output: {out_path}\n")

    new_rows: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for fw, script_path in scripts:
            print(f"Script: {fw}/{script_path.name}")

            for method in args.methods:
                print(f"  [{method}] ", end="", flush=True)

                key = (script_path.stem, method, args.provider)
                if key in done_keys:
                    print("SKIP (already in output)")
                    continue

                gen, spec_version = get_generated_path(
                    script_path, method, tmp_dir,
                    provider=args.provider,
                    regenerate=args.regenerate,
                    cached_only=args.cached_only,
                    spec_version=args.spec_version,
                )

                if gen is None:
                    print("SKIP (no pre-generated file found)")
                    continue

                result = evaluate(
                    generated_path=gen,
                    script_name=script_path.stem,
                    method=method,
                    framework=fw,
                    data_root=args.data_root,
                    strict_data=args.strict,
                    spec_version=spec_version,
                )

                status = "OK" if result.e2e_runnable else (
                    "preflight" if result.preflight_pass else (
                        "interface" if result.interface_complete else (
                            "syntax" if result.syntax_valid else "FAIL"
                        )
                    )
                )
                print(f"{status}  coverage={result.component_coverage:.2f}  {result.elapsed_sec}s")
                row = result.to_dict()
                row["provider"] = args.provider
                new_rows.append(row)

    # Write CSV (existing rows first, then new)
    all_rows = existing_rows + new_rows
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows ({len(new_rows)} new) → {out_path}")
    _print_summary(all_rows, args.methods, args.frameworks, args.provider)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _print_summary(rows: list[dict], methods: list[str], frameworks: list[str], provider: str):
    prows = [r for r in rows if r.get("provider", "claude") == provider]
    print(f"\n=== Summary [{provider}]: e2e_runnable rate ===")
    print(f"{'':20s}", end="")
    for m in methods:
        print(f"  {m:12s}", end="")
    print()

    for fw in frameworks:
        fw_rows = [r for r in prows if r["framework"] == fw]
        if not fw_rows:
            continue
        print(f"  {fw:18s}", end="")
        for m in methods:
            m_rows = [r for r in fw_rows if r["method"] == m]
            if not m_rows:
                print(f"  {'N/A':12s}", end="")
                continue
            rate = sum(_as_bool(r["e2e_runnable"]) for r in m_rows) / len(m_rows)
            print(f"  {rate:.0%} ({len(m_rows):2d})    ", end="")
        print()

    print(f"\n=== Summary [{provider}]: component_coverage (mean) ===")
    print(f"{'':20s}", end="")
    for m in methods:
        print(f"  {m:12s}", end="")
    print()
    for fw in frameworks:
        fw_rows = [r for r in prows if r["framework"] == fw]
        if not fw_rows:
            continue
        print(f"  {fw:18s}", end="")
        for m in methods:
            m_rows = [r for r in fw_rows if r["method"] == m]
            if not m_rows:
                print(f"  {'N/A':12s}", end="")
                continue
            mean_cov = sum(_as_float(r["component_coverage"]) for r in m_rows) / len(m_rows)
            print(f"  {mean_cov:.2f}          ", end="")
        print()


if __name__ == "__main__":
    sys.exit(main())
