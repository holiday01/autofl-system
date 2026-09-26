"""
Repair Phase-2 rows whose 'syntax' failure was a code-fence extraction artifact
(response began with prose, then a fenced module). For repeated/ablation CSVs the
saved file is re-extracted and re-evaluated in place (no new LLM call); for the
self-repair CSV the whole cell is dropped so it reruns from round 0 under --resume.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from phase2_common import evaluate_module, ROOT
from autofl.eval.llm_calls import extract_code

def fenced(path):
    p = Path(path)
    return p.exists() and "```" in p.read_text()

for name in ["repeated_claude", "repeated_gemini", "ablation_claude"]:
    f = ROOT / "results" / f"{name}.csv"
    if not f.exists(): continue
    df = pd.read_csv(f)
    bad = df[(df.error_stage == "syntax") & df.generated_path.map(fenced)]
    print(name, "fence-artifact rows:", len(bad))
    for i, r in bad.iterrows():
        p = ROOT / r.generated_path if not str(r.generated_path).startswith("/") else Path(r.generated_path)
        raw = p.read_text(); p.with_suffix(".py.raw.txt").write_text(raw); p.write_text(extract_code(raw))
        spec = r.get("spec_version", "n/a"); spec = "n/a" if pd.isna(spec) else spec
        ev = evaluate_module(p, r.script_name, r.method, r.framework, str(r.data_path), bool(r.strict_data), spec)
        for k, v in ev.items():
            if k in df.columns: df.at[i, k] = v
        print("  re-evaluated", r.script_name, r.method, "->", "PASS" if ev["e2e_runnable"] else ev["error_stage"])
    df.to_csv(f, index=False)

f = ROOT / "results" / "self_repair_claude.csv"
if f.exists():
    df = pd.read_csv(f)
    bad_cells = set()
    for _, r in df[df.error_stage == "syntax"].iterrows():
        if fenced(ROOT / r.generated_path if not str(r.generated_path).startswith("/") else r.generated_path):
            bad_cells.add((r.script_name, r.base, r.seed))
    print("self_repair cells dropped:", bad_cells)
    keep = ~df.apply(lambda r: (r.script_name, r.base, r.seed) in bad_cells, axis=1)
    for _, r in df[~keep].iterrows():
        for suf in ("", ".raw.txt"):
            q = Path(ROOT / r.generated_path if not str(r.generated_path).startswith("/") else r.generated_path)
            q = Path(str(q) + suf)
            if q.exists(): q.unlink()
    df[keep].to_csv(f, index=False)
print("done")
