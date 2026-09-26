"""
Evaluate ONE generated module and print the EvalResult as JSON on stdout.

Used by phase2_common.evaluate_module so that a single pathological client
(for example a synthetic ImageNet loader that makes the one-round FL
simulation run for hours on CPU) can be killed by a wall-clock timeout
instead of blocking a whole experiment.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # parent of the package directory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--framework", required=True)
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--spec-version", default="")
    a = ap.parse_args()

    from autofl.eval.evaluator import evaluate
    res = evaluate(a.path, a.script, a.method, a.framework,
                   data_root=a.data_root, strict_data=a.strict, spec_version=a.spec_version)
    d = res.to_dict()
    d["component_coverage"] = res.component_coverage
    print("@@EVAL_JSON@@" + json.dumps(d))


if __name__ == "__main__":
    main()
