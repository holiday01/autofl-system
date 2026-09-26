"""
Iterative self-repair baseline (reviewer-requested).

For a base prompting strategy WITHOUT the behavioral schema (zero_shot, few_shot,
few_shot_corrected), take the round-0 conversion, run the five-stage preflight,
and if it fails feed the failing stage + error message + current module back to
the same LLM asking for a fixed module, for up to --rounds rounds. No schema,
no invariants, and no exemplar beyond what the base strategy already had are
added at repair time: the only new information is the harness feedback.

Round 0 for --provider claude uses the archived benchmarks/*_fl_<base>.py file
(--round0 cached, default) so the repair loop starts from exactly the stored
generation the paper reports; --round0 live regenerates it.

Outputs: results/self_repair.csv (one row per script x base x round) and
eval/_self_repair/<provider>/<script>_<base>_r<k>.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from phase2_common import (ROOT, rel, cell_key, add_common_args, base_row, cached_client, evaluate_module,
                           iter_scripts, CsvWriter, EVAL_FIELDS, USAGE_FIELDS)
sys.path.insert(0, str(ROOT.parent))
from autofl.converter.llm_converter import Skill, _build_prompt          # noqa: E402
from autofl.eval.llm_calls import CallBudget, BudgetExceeded, QuotaExhausted, generate_file  # noqa: E402

REPAIR_TEMPLATE = """You previously converted a machine-learning training script into a Federated \
Learning client module that must expose three functions: build_model(config), \
build_dataloader(config, split), and train_step(model, batch, optimizer, config).

The module was executed by an automated validation harness and FAILED.
Failing stage: {stage}
Error:
{error}

=== ORIGINAL TRAINING SCRIPT ===
```python
{source}
```

=== CURRENT FL CLIENT MODULE (the one that failed) ===
```python
{current}
```

Fix the FL client module so that it passes the validation harness. \
Return ONLY the complete corrected Python module, no explanation."""

STAGE_HINTS = {
    # Only the harness's own stage names; no contract text is leaked.
    "syntax": "the module is not valid Python",
    "interface": "one of the three required functions is missing",
    "preflight/import_check": "importing the module raised an exception",
    "preflight/hardware_detect": "hardware detection failed",
    "preflight/data_load": "build_dataloader(config, 'train') did not yield a batch",
    "preflight/forward_pass": "train_step(model, batch, None, config) raised an exception",
    "preflight/backward_pass": "the harness could not run loss.backward() on the value returned by train_step",
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, "self_repair.csv", "_self_repair")
    p.add_argument("--base", nargs="+", default=["zero_shot", "few_shot"],
                   choices=["zero_shot", "few_shot", "few_shot_corrected"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--round0", default="cached", choices=["cached", "live"])
    p.add_argument("--seeds", type=int, default=1, help="independent repair trajectories per cell")
    p.add_argument("--with-hint", action="store_true",
                   help="add a one-line description of what the failing stage checks (still no contract)")
    a = p.parse_args(argv)

    budget = CallBudget(a.max_calls)
    outdir = Path(a.outdir) / a.provider
    w = CsvWriter(a.out, ["script_name", "framework", "method", "provider", "base", "seed",
                          "round", "round0_source", "generated_path", "timestamp"]
                  + EVAL_FIELDS + USAGE_FIELDS + ["cum_calls", "cum_cost_usd"])
    done = w.keys(["script_name", "base", "provider", "seed", "round"]) if a.resume else set()

    plan = list(iter_scripts(a.frameworks, a.scripts))
    print(f"[self-repair] provider={a.provider} bases={a.base} rounds={a.rounds} seeds={a.seeds} "
          f"scripts={len(plan)} dry_run={a.dry_run} max_calls={a.max_calls}")

    for fw, src in plan:
        source = src.read_text()
        for base in a.base:
            for seed in range(a.seeds):
                key_prefix = cell_key(src.stem, base, a.provider, seed)
                cum_calls, cum_cost = 0, 0.0
                # ---- round 0 ----
                r0 = outdir / f"{src.stem}_{base}_s{seed}_r0.py"
                r0_source = ""
                if key_prefix + cell_key(0) in done:
                    print(f"  skip {src.stem}/{base}/s{seed} (resume)")
                    continue
                if a.round0 == "cached" and a.provider == "claude" and cached_client(fw, src.stem, base).exists():
                    r0.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(cached_client(fw, src.stem, base), r0)
                    r0_source = "cached_v1"
                    usage = {}
                else:
                    system, user = _build_prompt(source, Skill(base))
                    try:
                        _, resp = generate_file(a.provider, system, user, r0, temperature=a.temperature,
                                                budget=budget, reuse_existing=a.resume,
                                                dry_run_source=cached_client(fw, src.stem, "structured") if a.dry_run else None)
                    except (BudgetExceeded, QuotaExhausted) as e:
                        print(f"[stop] {type(e).__name__}: {e}"); return
                    except Exception as e:
                        print(f"  !! generation failed for {src.stem}: {str(e)[:200]}", flush=True)
                        w.append(base_row(src.stem, fw, f"{base}+repair", a.provider, base=base, seed=seed,
                                          round=k if 'k' in dir() else 0, error_stage="generation",
                                          error_message=str(e)[:300]))
                        break
                    usage = resp.usage_dict(); r0_source = "live"
                    cum_calls += 0 if resp.dry_run else 1; cum_cost += resp.cost_usd or 0.0
                ev = evaluate_module(r0, src.stem, f"{base}+repair", fw, a.data_root, a.strict, "n/a")
                row = base_row(src.stem, fw, f"{base}+repair", a.provider, base=base, seed=seed, round=0,
                               round0_source=r0_source, generated_path=rel(r0),
                               cum_calls=cum_calls, cum_cost_usd=round(cum_cost, 6), **ev, **usage)
                w.append(row)
                print(f"  {src.stem:35s} {base:18s} s{seed} r0 -> {'PASS' if ev['e2e_runnable'] else ev['error_stage']}")
                current, stage, err = r0, ev["error_stage"], ev["error_message"]
                # ---- repair rounds ----
                for k in range(1, a.rounds + 1):
                    if ev["e2e_runnable"]:
                        break
                    hint = f" ({STAGE_HINTS.get(stage, '')})" if a.with_hint and stage in STAGE_HINTS else ""
                    user = REPAIR_TEMPLATE.format(stage=f"{stage}{hint}", error=err or "(no message)",
                                                  source=source, current=current.read_text())
                    rk = outdir / f"{src.stem}_{base}_s{seed}_r{k}.py"
                    try:
                        _, resp = generate_file(a.provider, None, user, rk, temperature=a.temperature,
                                                budget=budget, reuse_existing=a.resume,
                                                dry_run_source=cached_client(fw, src.stem, "structured") if a.dry_run else None)
                    except (BudgetExceeded, QuotaExhausted) as e:
                        print(f"[stop] {type(e).__name__}: {e}"); return
                    except Exception as e:
                        print(f"  !! generation failed for {src.stem}: {str(e)[:200]}", flush=True)
                        w.append(base_row(src.stem, fw, f"{base}+repair", a.provider, base=base, seed=seed,
                                          round=k if 'k' in dir() else 0, error_stage="generation",
                                          error_message=str(e)[:300]))
                        break
                    cum_calls += 0 if resp.dry_run else 1; cum_cost += resp.cost_usd or 0.0
                    ev = evaluate_module(rk, src.stem, f"{base}+repair", fw, a.data_root, a.strict, "n/a")
                    row = base_row(src.stem, fw, f"{base}+repair", a.provider, base=base, seed=seed, round=k,
                                   round0_source=r0_source, generated_path=rel(rk),
                                   cum_calls=cum_calls, cum_cost_usd=round(cum_cost, 6), **ev, **resp.usage_dict())
                    w.append(row)
                    print(f"  {'':35s} {'':18s} s{seed} r{k} -> {'PASS' if ev['e2e_runnable'] else ev['error_stage']}"
                          f"  [{resp.input_tokens}/{resp.output_tokens} tok, ${resp.cost_usd:.4f}]")
                    current, stage, err = rk, ev["error_stage"], ev["error_message"]
    print(f"[self-repair] done. live calls: {budget.calls}. rows -> {a.out}")


if __name__ == "__main__":
    main()
