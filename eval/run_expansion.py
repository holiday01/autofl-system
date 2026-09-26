"""
Benchmark-expansion runner: dev_new / holdout / adversarial strata
(benchmarks/expansion/**, manifest in results/expansion_manifest.csv).

Modeled on eval/run_repeated.py + eval/phase2_common.py.  For every manifest row
(filtered by --stratum / --frameworks / --scripts) and every requested strategy:

  template            rule-based converter (autofl.converter.template_converter.convert_source),
                      local, no LLM; provider column = "none"; one sample.
  zero_shot /
  few_shot_corrected /
  structured          prompts built by autofl.converter.llm_converter._build_prompt (byte-identical
                      to the archived benchmark prompts; structured uses --spec-version, default v2)
                      and sent through autofl.eval.llm_calls.generate_file with --provider,
                      --samples, --max-calls; --dry-run makes NO LLM call and uses the template
                      output of the same script as the fake response.

Every generated module is evaluated with phase2_common.evaluate_module (five-stage
preflight + 1-round FL simulation, CPU-forced, data_root default ".", strict False).
The evaluation runs in a child process with --eval-timeout (default 1800 s) so a
hanging download or a runaway script cannot block the run.

Outputs
  eval/_expansion/<stratum>/<provider>/<stem>_<method>_s<k>.py   generated clients
  eval/_expansion/<stratum>/<provider>/<stem>_<method>_s<k>.log  evaluator stdout/stderr
  results/expansion.csv (or --out)                               one row per generation

Typical use
  python eval/run_expansion.py --methods template --out results/expansion_template_baseline.csv
  python eval/run_expansion.py --stratum holdout --methods structured --provider claude --samples 3 --max-calls 100
  python eval/run_expansion.py --methods structured --dry-run          # pipeline test, zero LLM calls

The template rules are NOT adapted to the expansion scripts: a failure is a result.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# CPU-only, exactly like preflight/validator.py and eval/run_template_baseline.py.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_common import (ROOT, RESULTS_DIR, rel, base_row, evaluate_module,   # noqa: E402
                           CsvWriter, EVAL_FIELDS, USAGE_FIELDS)
sys.path.insert(0, str(ROOT.parent))
from autofl.converter.llm_converter import Skill, _build_prompt                 # noqa: E402
from autofl.eval.llm_calls import CallBudget, BudgetExceeded, generate_file      # noqa: E402

MANIFEST = RESULTS_DIR / "expansion_manifest.csv"
STRATA = ["dev_new", "holdout", "adversarial"]
METHODS = ["template", "zero_shot", "few_shot_corrected", "structured"]
TEMPLATE_FIELDS = ["template_model", "template_train_step", "n_rules", "n_fallbacks",
                   "rules_fired", "fallbacks_used", "converter_warnings", "convert_error"]
PROV_FIELDS = ["candidate_id", "commit_sha", "source_sha256", "difficulty", "post_cutoff", "invariant_stress"]
FIELDS = (["script_name", "stratum", "framework", "method", "provider", "sample", "generated_path", "timestamp"]
          + EVAL_FIELDS + USAGE_FIELDS + TEMPLATE_FIELDS + PROV_FIELDS)
EVAL_JSON_MARK = "EVALJSON "


# ── manifest ─────────────────────────────────────────────────────────────────

def read_manifest(path: Path = MANIFEST) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def iter_manifest(strata: list[str], frameworks: list[str] | None, scripts: list[str] | None,
                  path: Path = MANIFEST):
    """Yield manifest rows (dicts) in manifest order, filtered."""
    for r in read_manifest(path):
        if r["stratum"] not in strata:
            continue
        if frameworks and r["framework"] not in frameworks:
            continue
        if scripts and r["stem"] not in scripts:
            continue
        yield r


# ── template strategy (local, no LLM) ────────────────────────────────────────

def run_template(src: Path, out: Path, framework: str) -> dict:
    """Convert with the rule-based converter; always writes `out` (a stub on crash)."""
    from autofl.converter.template_converter import convert_source
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = {k: "" for k in TEMPLATE_FIELDS}
    try:
        code, meta = convert_source(src.read_text(), rel(src), framework)
        out.write_text(code)
        fields.update(template_model=meta.model_fallback, template_train_step=meta.train_step_mode,
                      n_rules=len(meta.rules), n_fallbacks=len(meta.fallbacks),
                      rules_fired=" | ".join(meta.rules), fallbacks_used=" | ".join(meta.fallbacks),
                      converter_warnings=" | ".join(meta.warnings), convert_error="")
    except Exception as e:                       # converter crash: record it, emit a stub so evaluate() reports it
        err = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
        out.write_text(f"# template converter error: {err}\n")
        fields.update(n_rules=0, n_fallbacks=0, convert_error=err)
    return fields


# ── evaluation in a child process (timeout-safe) ─────────────────────────────

def _empty_eval(stage: str, msg: str) -> dict:
    d = {k: "" for k in EVAL_FIELDS}
    for k in ("syntax_valid", "interface_complete", "build_model_present", "build_dataloader_present",
              "train_step_present", "optimizer_detected", "preflight_pass", "e2e_runnable",
              "data_path_exists", "strict_data"):
        d[k] = False
    d.update(component_coverage=0.0, error_stage=stage, error_message=msg[:300], elapsed_sec=0.0,
             dataset_len=-1)
    return d


EVAL_THREADS_DEFAULT = 4     # OMP/MKL threads per evaluation child (the host is shared with other runs)


def evaluate_with_timeout(path: Path, script: str, method: str, framework: str, data_root: str,
                          strict: bool, spec_version: str, timeout: float | None, log: Path,
                          threads: int = EVAL_THREADS_DEFAULT) -> dict:
    if not timeout or timeout <= 0:
        return evaluate_module(path, script, method, framework, data_root, strict, spec_version)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_eval-one", str(path), script, method,
           framework, data_root, "1" if strict else "0", spec_version]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "CUDA_VISIBLE_DEVICES": ""}
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                start_new_session=True, env=env)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # also kills DataLoader workers
            except ProcessLookupError:
                pass
            proc.wait()
            d = _empty_eval("timeout", f"evaluation exceeded --eval-timeout={timeout:.0f}s")
            d["elapsed_sec"] = round(time.time() - t0, 2)
            return d
    for line in reversed(log.read_text(errors="replace").splitlines()):
        if line.startswith(EVAL_JSON_MARK):
            return json.loads(line[len(EVAL_JSON_MARK):])
    tail = log.read_text(errors="replace").strip().splitlines()
    d = _empty_eval("eval_crash", f"child rc={proc.returncode}: {tail[-1] if tail else 'no output'}")
    d["elapsed_sec"] = round(time.time() - t0, 2)
    return d


def _static_stages(path: Path) -> dict:
    """Syntax + interface fields only (used when the runtime stages cannot report themselves)."""
    import ast
    d = _empty_eval("", "")
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        d.update(error_stage="syntax", error_message=str(e)[:300])
        return d
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    d.update(syntax_valid=True, build_model_present="build_model" in names,
             build_dataloader_present="build_dataloader" in names, train_step_present="train_step" in names)
    d["interface_complete"] = all((d["build_model_present"], d["build_dataloader_present"], d["train_step_present"]))
    d["component_coverage"] = round(sum((d["build_model_present"], d["build_dataloader_present"],
                                         d["train_step_present"])) / 4.0, 3)
    return d


def _eval_one_child(argv: list[str]) -> int:
    path, script, method, framework, data_root, strict, spec_version = argv
    # torch is imported lazily by evaluate_module, so these still take effect if the parent did not set them
    os.environ.setdefault("OMP_NUM_THREADS", str(EVAL_THREADS_DEFAULT))
    os.environ.setdefault("MKL_NUM_THREADS", str(EVAL_THREADS_DEFAULT))
    t0 = time.time()
    try:
        d = evaluate_module(Path(path), script, method, framework, data_root, strict == "1", spec_version)
    except SystemExit as e:
        # A generated client that calls argparse parse_args() (or sys.exit) at import time kills the
        # importing process: the preflight cannot report it, so it is recorded here as an import-time crash.
        d = _static_stages(Path(path))
        d.update(error_stage="preflight/import_exit",
                 error_message=f"SystemExit({e.code}) raised while evaluating the module (module-level "
                               f"parse_args()/sys.exit in the generated client; see the .log)",
                 elapsed_sec=round(time.time() - t0, 2), data_path=data_root,
                 data_path_exists=Path(data_root).exists(), strict_data=(strict == "1"),
                 spec_version=spec_version)
    print(EVAL_JSON_MARK + json.dumps(d, default=str), flush=True)
    return 0


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--_eval-one":
        return _eval_one_child(sys.argv[2:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--stratum", nargs="+", default=STRATA, choices=STRATA)
    p.add_argument("--frameworks", nargs="+", default=None,
                   help="manifest framework labels to include (default: all)")
    p.add_argument("--scripts", nargs="+", default=None, help="manifest stems to include (default: all)")
    p.add_argument("--methods", nargs="+", default=["template"], choices=METHODS)
    p.add_argument("--provider", default="claude", choices=["claude", "gemini", "ollama"])
    p.add_argument("--samples", type=int, default=1, help="generations per LLM cell (template: always 1)")
    p.add_argument("--spec-version", default="v2", choices=["v1", "v2"])
    p.add_argument("--data-root", default=".", help="config['data_path'] handed to the module")
    p.add_argument("--strict", action="store_true", help="strict data mode (fail closed)")
    p.add_argument("--dry-run", action="store_true",
                   help="no LLM calls; the template output of the same script is used as the fake response")
    p.add_argument("--max-calls", type=int, default=None, help="hard cap on live LLM calls")
    p.add_argument("--resume", action="store_true", help="skip rows already in --out")
    p.add_argument("--convert-only", action="store_true", help="generate files, skip evaluation")
    p.add_argument("--eval-timeout", type=float, default=1800.0,
                   help="seconds per evaluation in a child process (0 = in-process, no timeout)")
    p.add_argument("--eval-threads", type=int, default=EVAL_THREADS_DEFAULT,
                   help="OMP_NUM_THREADS/MKL_NUM_THREADS for each evaluation child")
    p.add_argument("--out", default=str(RESULTS_DIR / "expansion.csv"))
    p.add_argument("--outdir", default=str(ROOT / "eval" / "_expansion"))
    p.add_argument("--temperature", type=float, default=None,
                   help="Gemini/Ollama sampling temperature (None = provider default; Claude CLI: none)")
    a = p.parse_args(argv)

    budget = CallBudget(a.max_calls)
    w = CsvWriter(a.out, FIELDS)
    done = w.keys(["script_name", "stratum", "method", "provider", "sample"]) if a.resume else set()
    plan = list(iter_manifest(a.stratum, a.frameworks, a.scripts, Path(a.manifest)))
    n_llm = sum(1 for m in a.methods if m != "template")
    print(f"[expansion] strata={a.stratum} methods={a.methods} provider={a.provider} samples={a.samples} "
          f"scripts={len(plan)} -> {len(plan) * (n_llm * a.samples + ('template' in a.methods))} generations; "
          f"dry_run={a.dry_run} max_calls={a.max_calls} eval_timeout={a.eval_timeout}s "
          f"eval_threads={a.eval_threads} (one evaluation at a time)", flush=True)

    for row in plan:
        stem, stratum, fw = row["stem"], row["stratum"], row["framework"]
        src = ROOT / row["path"]
        source = src.read_text()
        prov = {"candidate_id": row.get("candidate_id", ""), "commit_sha": row.get("commit_sha", ""),
                "source_sha256": row.get("sha256", ""), "difficulty": row.get("difficulty", ""),
                "post_cutoff": row.get("post_cutoff", ""), "invariant_stress": row.get("invariant_stress", "")}
        template_out = Path(a.outdir) / stratum / "none" / f"{stem}_template_s0.py"

        for m in a.methods:
            provider = "none" if m == "template" else a.provider
            outdir = Path(a.outdir) / stratum / provider
            n_samples = 1 if m == "template" else a.samples
            for s in range(n_samples):
                if (stem, stratum, m, provider, str(s)) in done:
                    continue
                out = outdir / f"{stem}_{m}_s{s}.py"
                extra: dict = {}
                t0 = time.time()
                if m == "template":
                    extra = run_template(src, out, fw)
                    print(f"  [template] {stratum}/{fw}/{stem}: model={extra['template_model'] or '-'} "
                          f"train_step={extra['template_train_step'] or '-'} rules={extra['n_rules']} "
                          f"fallbacks={extra['n_fallbacks']} err={extra['convert_error'] or '-'} "
                          f"({time.time() - t0:.1f}s)", flush=True)
                else:
                    system, user = _build_prompt(source, Skill(m), spec_version=a.spec_version)
                    dry = None
                    if a.dry_run:
                        if not template_out.exists():
                            run_template(src, template_out, fw)
                        dry = template_out
                    try:
                        _, resp = generate_file(a.provider, system, user, out, temperature=a.temperature,
                                                budget=budget, dry_run_source=dry)
                    except BudgetExceeded as e:
                        print(f"[stop] {e}", flush=True)
                        return 0
                    except Exception as e:                # provider failure after retries: record and continue
                        print(f"  !! generation failed for {stem}/{m}: {str(e)[:200]}", flush=True)
                        w.append(base_row(stem, fw, m, provider, stratum=stratum, sample=s,
                                          error_stage="generation", error_message=str(e)[:300], **prov))
                        continue
                    extra = resp.usage_dict()
                if a.convert_only:
                    continue
                ev = evaluate_with_timeout(out, stem, m, fw, a.data_root, a.strict,
                                           a.spec_version if m == "structured" else "n/a",
                                           a.eval_timeout, out.with_suffix(".log"), a.eval_threads)
                w.append(base_row(stem, fw, m, provider, stratum=stratum, sample=s,
                                  generated_path=rel(out), **ev, **extra, **prov))
                status = "PASS" if ev.get("e2e_runnable") else (ev.get("error_stage") or "FAIL")
                print(f"  {stratum:11s} {stem:45s} {m:18s} s{s} -> {status:24s} "
                      f"{ev.get('elapsed_sec', '')}s  {(ev.get('error_message') or '')[:100]}", flush=True)

    print(f"[expansion] done. live LLM calls: {budget.calls}. rows -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
