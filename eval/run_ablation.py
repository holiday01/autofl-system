"""
Same-provider prompt ablation (reviewer-requested): hold specification content
constant while varying message placement, exemplar presence and rule subsets.

Conditions (each x --samples generations per script):
  schema_sys            structured v2: full spec + rules as SYSTEM prompt, no exemplar (= paper's structured)
  schema_user           identical text placed in the USER message, no system prompt
  schema_sys_exemplar   schema_sys system prompt + the corrected few-shot exemplar in the user message
  exemplar_invariants   corrected exemplar + the four behavioral invariants as one-line prohibitions,
                        WITHOUT the framework-conversion rules (user message only)
  framework_rules_only  framework-conversion + formatting rules WITHOUT the behavioral invariants
  schema_sys_prohibit   schema_sys with I3 phrased as a prohibition ("do NOT return loss.detach()")
                        instead of the positive "return the loss WITH grad attached"

Outputs: results/ablation.csv and eval/_ablation/<provider>/<script>_<cond>_s<k>.py
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

INVARIANT_PROHIBITIONS = """Behavioral rules (a violation makes the module unusable by the FL runtime):
- train_step must NOT call loss.backward(), optimizer.step() or optimizer.zero_grad(); the FL runtime performs backpropagation and the optimizer update.
- train_step must return the loss tensor WITH grad attached; never return loss.detach(), loss.item() or a Python float.
- train_step must run exactly ONE forward pass and move the batch tensors to the device of the model parameters.
- build_dataloader must include a synthetic-data fallback (torch.randn/randint) gated on config.get('allow_synthetic_data', False); if real data is unavailable and that flag is False, raise FileNotFoundError."""

FRAMEWORK_RULES = """Rules:
- The module must expose exactly build_model(config), build_dataloader(config, split="train") and train_step(model, batch, optimizer, config).
- Preserve the original model architecture exactly (copy the class).
- Keep all imports from the original script.
- build_dataloader must support both "train" and "val" splits using random_split; read batch_size from config.get("local", {}).get("batch_size", 16) and data_path from config.get("data_path", ".").
- If the original uses TensorFlow/Keras, convert the model to an equivalent torch.nn.Module.
- If the original uses PyTorch Lightning LightningModule, extract the underlying model and loss.
- If the original uses MONAI transforms, wrap them in the Dataset class.
- Output ONLY valid Python code. No markdown fences, no explanation."""

POSITIVE_I3 = "Run ONE forward pass only. Return the loss tensor WITH grad attached."
PROHIBIT_I3 = ("Run ONE forward pass only. Do NOT return loss.detach(), loss.item() or any value "
               "without a grad_fn; return the loss tensor itself.")


def build_condition(cond: str, source: str) -> tuple[str | None, str]:
    sys_a, user_a = _build_prompt(source, Skill.STRUCTURED, spec_version="v2")
    if cond == "schema_sys":
        return sys_a, user_a
    if cond == "schema_user":
        return None, sys_a + "\n\n" + user_a
    if cond == "schema_sys_exemplar":
        _, user_fs = _build_prompt(source, Skill.FEW_SHOT_CORRECTED)
        return sys_a, user_fs
    if cond == "exemplar_invariants":
        _, user_fs = _build_prompt(source, Skill.FEW_SHOT_CORRECTED)
        return None, user_fs + "\n\n" + INVARIANT_PROHIBITIONS
    if cond == "framework_rules_only":
        user = ("Convert the following training script into a Federated Learning client module.\n\n"
                + FRAMEWORK_RULES + f"\n\n```python\n{source}\n```")
        return None, user
    if cond == "schema_sys_prohibit":
        assert POSITIVE_I3 in sys_a, "positive I3 wording not found in structured system prompt"
        return sys_a.replace(POSITIVE_I3, PROHIBIT_I3), user_a
    raise ValueError(cond)


CONDITIONS = ["schema_sys", "schema_user", "schema_sys_exemplar", "exemplar_invariants",
              "framework_rules_only", "schema_sys_prohibit"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, "ablation.csv", "_ablation")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--samples", type=int, default=1)
    a = p.parse_args(argv)

    budget = CallBudget(a.max_calls)
    outdir = Path(a.outdir) / a.provider
    w = CsvWriter(a.out, ["script_name", "framework", "method", "provider", "condition", "sample",
                          "system_words", "user_words", "generated_path", "timestamp"]
                  + EVAL_FIELDS + USAGE_FIELDS)
    done = w.keys(["script_name", "condition", "provider", "sample"]) if a.resume else set()
    plan = list(iter_scripts(a.frameworks, a.scripts))
    n_calls = len(plan) * len(a.conditions) * a.samples
    print(f"[ablation] provider={a.provider} conditions={a.conditions} samples={a.samples} "
          f"scripts={len(plan)} -> {n_calls} generations; dry_run={a.dry_run} max_calls={a.max_calls}")

    for fw, src in plan:
        source = src.read_text()
        for cond in a.conditions:
            system, user = build_condition(cond, source)
            for s in range(a.samples):
                if cell_key(src.stem, cond, a.provider, s) in done:
                    continue
                out = outdir / f"{src.stem}_{cond}_s{s}.py"
                try:
                    _, resp = generate_file(a.provider, system, user, out, temperature=a.temperature,
                                            budget=budget, reuse_existing=a.resume,
                                            dry_run_source=cached_client(fw, src.stem, "structured") if a.dry_run else None)
                except (BudgetExceeded, QuotaExhausted) as e:
                    print(f"[stop] {type(e).__name__}: {e}"); return
                except Exception as e:                       # provider failure after retries: record and continue
                    print(f"  !! generation failed for {src.stem}: {str(e)[:200]}", flush=True)
                    w.append(base_row(src.stem, fw, locals().get("cond") or locals().get("m") or locals().get("base"),
                                      a.provider, condition=cond, sample=s, error_stage="generation", error_message=str(e)[:300]))
                    continue
                ev = evaluate_module(out, src.stem, cond, fw, a.data_root, a.strict,
                                     "v2" if cond.startswith("schema") else "n/a")
                w.append(base_row(src.stem, fw, cond, a.provider, condition=cond, sample=s,
                                  system_words=len(system.split()) if system else 0,
                                  user_words=len(user.split()),
                                  generated_path=rel(out), **ev, **resp.usage_dict()))
                print(f"  {src.stem:35s} {cond:22s} s{s} -> {'PASS' if ev['e2e_runnable'] else ev['error_stage']}"
                      f"  [{resp.input_tokens}/{resp.output_tokens} tok, ${resp.cost_usd:.4f}]")
    print(f"[ablation] done. live calls: {budget.calls}. rows -> {a.out}")


if __name__ == "__main__":
    main()
