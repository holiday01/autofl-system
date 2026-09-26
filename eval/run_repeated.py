"""
Repeated sampling (reviewer-requested): several independent generations per
script x strategy x provider so that run-to-run variation can be reported
instead of one stored generation per cell.

Methods: structured (spec v2, system prompt), few_shot_corrected (matched,
contract-correct exemplar), few_shot (archived defective exemplar), zero_shot.
Gemini/Ollama temperature defaults to the provider default (sampling); pass
--temperature 0 to reproduce the deterministic setting of the original runs.
The Claude CLI exposes no temperature control.

Outputs: results/repeated.csv and eval/_repeated/<provider>/<script>_<method>_s<k>.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phase2_common import (ROOT, rel, cell_key, add_common_args, base_row, cached_client, evaluate_module,
                           iter_scripts, CsvWriter, EVAL_FIELDS, USAGE_FIELDS)
sys.path.insert(0, str(ROOT.parent))
from autofl.converter.llm_converter import Skill, _build_prompt          # noqa: E402
from autofl.eval.llm_calls import CallBudget, BudgetExceeded, QuotaExhausted, generate_file  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, "repeated.csv", "_repeated")
    p.add_argument("--methods", nargs="+", default=["structured", "few_shot_corrected"],
                   choices=["structured", "few_shot_corrected", "few_shot", "zero_shot"])
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--spec-version", default="v2", choices=["v1", "v2"])
    a = p.parse_args(argv)

    budget = CallBudget(a.max_calls)
    outdir = Path(a.outdir) / a.provider
    w = CsvWriter(a.out, ["script_name", "framework", "method", "provider", "sample",
                          "generated_path", "timestamp"] + EVAL_FIELDS + USAGE_FIELDS)
    done = w.keys(["script_name", "method", "provider", "sample"]) if a.resume else set()
    plan = list(iter_scripts(a.frameworks, a.scripts))
    print(f"[repeated] provider={a.provider} methods={a.methods} samples={a.samples} scripts={len(plan)} "
          f"-> {len(plan)*len(a.methods)*a.samples} generations; dry_run={a.dry_run} max_calls={a.max_calls}")

    for fw, src in plan:
        source = src.read_text()
        for m in a.methods:
            system, user = _build_prompt(source, Skill(m), spec_version=a.spec_version)
            for s in range(a.samples):
                if cell_key(src.stem, m, a.provider, s) in done:
                    continue
                out = outdir / f"{src.stem}_{m}_s{s}.py"
                try:
                    _, resp = generate_file(a.provider, system, user, out, temperature=a.temperature,
                                            budget=budget, reuse_existing=a.resume,
                                            dry_run_source=cached_client(fw, src.stem, "structured") if a.dry_run else None)
                except (BudgetExceeded, QuotaExhausted) as e:
                    print(f"[stop] {type(e).__name__}: {e}"); return
                except Exception as e:                       # provider failure after retries: record and continue
                    print(f"  !! generation failed for {src.stem}: {str(e)[:200]}", flush=True)
                    w.append(base_row(src.stem, fw, locals().get("cond") or locals().get("m") or locals().get("base"),
                                      a.provider, sample=s, error_stage="generation", error_message=str(e)[:300]))
                    continue
                ev = evaluate_module(out, src.stem, m, fw, a.data_root, a.strict,
                                     a.spec_version if m == "structured" else "n/a")
                w.append(base_row(src.stem, fw, m, a.provider, sample=s,
                                  generated_path=rel(out), **ev, **resp.usage_dict()))
                print(f"  {src.stem:35s} {m:18s} s{s} -> {'PASS' if ev['e2e_runnable'] else ev['error_stage']}"
                      f"  [{resp.input_tokens}/{resp.output_tokens} tok, ${resp.cost_usd:.4f}]")
    print(f"[repeated] done. live calls: {budget.calls}. rows -> {a.out}")


if __name__ == "__main__":
    main()
