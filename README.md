# AutoFL: System and Benchmark Suite (v2.0)

Source code, benchmark suite, and generated client files for:

> Yen-Jung Chiu and Chao-Chun Chuang. *Schema-Augmented LLM Prompting for Converting ML Training Scripts to Federated Learning Clients.* ACM Transactions on Software Engineering and Methodology (resubmission under review, 2026).

- Concept DOI (always resolves to the latest version): https://doi.org/10.5281/zenodo.20156961
- Companion data deposit (result CSV/JSON, run notes, figures): https://doi.org/10.5281/zenodo.20156953 ([GitHub](https://github.com/holiday01/autofl-results))
- v1.0 (as submitted 2026-05-14): https://doi.org/10.5281/zenodo.20156962

## What changed in v2.0

Corrections found in the artifact audit for the resubmission:

- `fl_runtime/server.py`: FedAvg is weighted by client sample count (v1.0 averaged uniformly over unequal non-IID partitions). All simulations were rerun.
- `fl_runtime/run_simulation.py`: the IID simulation gives each client a disjoint shard (in v1.0 every client trained on the same data). Fixed seed 42.
- `preflight/validator.py`: Stage 1 imports the generated module in a fresh interpreter subprocess, so a missing dependency is a Stage 1 failure rather than a Stage 3 data failure. A `strict_data` mode fails closed when the real data path is absent instead of permitting the synthetic fallback; every run records whether real data was present.
- `converter/llm_converter.py`: the structured specification is versioned. `FL_INTERFACE_SPEC_V1` is the exact v1.0 text; v2 gates the synthetic fallback on `allow_synthetic_data`.
- `eval/non_iid_accuracy.py`: reports overall and macro-averaged accuracy (v1.0 labelled overall accuracy as macro).
- Run notes are generated from the run data (device, CUDA, AMP, seed), never edited by hand.

New in v2.0:

- `converter/template_converter.py`: a deterministic rule-based converter (six rule groups) used as the competent non-LLM baseline.
- `benchmarks/expansion/`: a frozen expansion benchmark (7 development, 21 holdout, 5 adversarial scripts) fetched at pinned upstream commits; verify with `cd benchmarks/expansion && sha256sum -c MANIFEST.sha256`.
- Runners for iterative self-repair, the prompt ablation, repeated sampling, the expansion benchmark, and source-to-converted differential tests (table below), plus `eval/check_consistency.py`, which recomputes the paper's numbers from the result CSVs and fails if any is missing from the manuscript source.
- The generated client files of every experiment (`benchmarks/<framework>/*_fl_<strategy>.py`, `eval/_self_repair/`, `eval/_ablation/`, `eval/_repeated/`, `eval/_expansion/`, `eval/_few_shot_corrected_*`).
- `Dockerfile` and `requirements-lock.txt` pinning the evaluation environment.

## Layout

```
converter/     AST rewriter, template converter, LLM converter (Claude CLI / Gemini / Ollama) with all prompts
preflight/     five-stage preflight validator (import / hardware / data / forward / backward)
fl_runtime/    FLClient, FLServer (sample-weighted FedAvg), non-IID partitioner, simulation runners
eval/          experiment runners, evaluator harness, table generators, consistency check; _*/ hold generated clients
hardware/      hardware detector and batch/learning-rate sweep
benchmarks/    13 primary input scripts with their converted clients; expansion/ = frozen expansion benchmark
examples/      few-shot exemplar files and example configs
config/        FL run configuration
tests/         unit tests
cli.py         python -m autofl {hardware,convert,preflight,run}
```

The code imports itself as the package `autofl`, so clone it into a directory of that name and run from inside it with its parent on `PYTHONPATH`:

```bash
git clone https://github.com/holiday01/autofl-system autofl
cd autofl && export PYTHONPATH=$(dirname "$PWD")
pip install torch torchvision && pip install -r requirements.txt
```

Or use the pinned environment: `docker build -t autofl:2.0 .` then `docker run --rm -it autofl:2.0` (add `--gpus all` for the simulations).

## Reproducing the paper

Place the result CSVs from the data deposit under `results/` and `fl_results/` to re-derive tables without new LLM calls. Commands are run from the repository root.

| Experiment | Command |
|---|---|
| Re-validate the archived generated clients (no LLM calls) | `python eval/run_benchmark.py --cached-only --strict --output results/benchmark_v2.csv` |
| Primary benchmark with fresh generations | `python eval/run_benchmark.py --regenerate --provider claude` |
| Template converter baseline | `python eval/run_template_baseline.py` |
| Iterative self-repair | `python eval/run_self_repair.py` |
| Prompt ablation | `python eval/run_ablation.py` |
| Repeated sampling | `python eval/run_repeated.py --provider claude` (or `gemini`) |
| Frozen expansion benchmark | `python eval/run_expansion.py` |
| Source-to-converted differential tests | `python eval/differential_test.py` |
| Exemplar correction (Gemini) | `python eval/run_exemplar_correction_realapi.py` |
| FL simulation (IID) | `python -m autofl.fl_runtime.run_simulation` |
| Non-IID simulation (Dirichlet 0.5, 0.1) | `python -m autofl.fl_runtime.run_simulation_non_iid` |
| Held-out accuracy per round | `python eval/non_iid_accuracy.py` |
| Non-DL proxy validation | `python eval/non_dl_proxy_validation.py` |
| Hardware sweep | `python hardware/param_sweep.py` |
| Phase-2 tables | `python eval/make_phase2_tables.py` |
| Consistency check against the manuscript | `python eval/check_consistency.py --tex <path to main_tosem.tex>` |

Each runner prints its options with `--help`. LLM runners need a provider: the `claude` command-line client for Claude, `GEMINI_API_KEY` for Gemini, or a local Ollama server for Llama 3.

The FL runtime is a self-contained FedAvg simulator. It does not integrate with Flower, FedML, or OpenFL, and it implements neither FedProx nor SCAFFOLD.

## License

MIT (see `LICENSE`).

## Citation

```bibtex
@misc{autofl_system_2026,
  title  = {AutoFL: System and Benchmark Suite},
  author = {Chiu, Yen-Jung and Chuang, Chao-Chun},
  year   = {2026},
  doi    = {10.5281/zenodo.20156961},
  note   = {Version 2.0. v1.0: 10.5281/zenodo.20156962}
}
```
