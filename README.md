# DVRPTW Benchmark

Production-style Python 3.12+ benchmark framework for static and dynamic VRPTW on Solomon RC-100 instances.

## Scope
- Static Solomon RC-100 evaluation.
- Dynamic simulation with progressive customer revelation, degree of dynamism `epsilon`, and snapshot re-optimization.
- Strategy families:
  - Heuristics/OR: OR-Tools, PyVRP, PMCA, custom GLS, optional Gurobi, optional LKH-3.
  - RL: RL4CO adapter, from-scratch GA baseline, from-scratch tabular Q-learning baseline.
  - Hybrid: RL warm-start + feasibility layer + GLS refinement.

## Assumptions
- Dynamic reveal policy:
  - `number_dynamic = round(epsilon * N_customers)`.
  - reveal times sampled uniformly over `[0, cutoff_time]`.
  - `cutoff_time = depot_due_time * cutoff_ratio`.
- Time-window adjustment:
  - `ready_time = max(original_ready, reveal_time)`.
  - `due_time = min(original_due, horizon)`.
  - scenario discarded if any dynamic customer has `ready_time > due_time`.
- Re-optimization runs on each reveal event with fixed budgets: `5, 10, 40, 120` seconds.

## Install
```bash
cd dvrptw-benchmark
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional extras:
```bash
pip install -e ".[gurobi]"
pip install -e ".[lkh3]"
pip install -e ".[mlflow]"
```

## Dataset
Place Solomon RC-100 files under:
`./dataset/solomon_rc100/`

Only `RC*.txt`-style instances are loaded.

## CLI
Heuristics:
```bash
python -m dvrptw_bench.cli.heuristics_cli list-instances --dataset-root ./dataset
python -m dvrptw_bench.cli.heuristics_cli solve-static --solver ortools --instance RC101.txt --time-limit 10
python -m dvrptw_bench.cli.heuristics_cli solve-dynamic --solver gls --instance RC101.txt --epsilon 0.5 --budget 10 --seed 1
```

RL:
```bash
python -m dvrptw_bench.cli.rl_cli train-ga --instance RC101.txt --time-limit 20
python -m dvrptw_bench.cli.rl_cli eval-dynamic --policy ga --instance RC101.txt --epsilon 0.5 --budget 10
```

Hybrid:
```bash
python -m dvrptw_bench.cli.hybrid_cli eval-hybrid --policy ga --instance RC101.txt --epsilon 0.5 --budget 10 --gls-time-share 0.9
```

Run full benchmark:
```bash
python -m dvrptw_bench.cli.run_all_cli run --dataset-root ./dataset --output-root ./outputs
```

## Outputs
`./outputs/<run_id>/` contains:
- `records.jsonl` incremental records.
- `records.parquet`, `summary.parquet`, `summary.csv`.
- `report.md`.
- plots (`png` + selected `svg`/`html`).

## Troubleshooting
- Gurobi license issues: verify license with `grbgetkey` and `gurobi_cl`.
- LKH-3 binary missing: install from official release and set `--lkh3-binary /path/to/LKH`.
- OR-Tools install quirks: use Python 3.12 wheel-compatible environment.
- RL/CUDA mismatch: force `--device cpu` if CUDA not available.

## Official documentation links
- NumPy: https://numpy.org/doc/
- SciPy: https://docs.scipy.org/doc/scipy/
- pandas: https://pandas.pydata.org/docs/
- PyArrow: https://arrow.apache.org/docs/python/
- Matplotlib: https://matplotlib.org/stable/
- Plotly: https://plotly.com/python/
- NetworkX: https://networkx.org/documentation/stable/
- Pydantic: https://docs.pydantic.dev/
- Hydra: https://hydra.cc/docs/intro/
- OmegaConf: https://omegaconf.readthedocs.io/
- Typer: https://typer.tiangolo.com/
- Rich: https://rich.readthedocs.io/
- tqdm: https://tqdm.github.io/
- OR-Tools: https://developers.google.com/optimization/routing
- PyVRP: https://pyvrp.org/
- Gurobi Python API: https://docs.gurobi.com/projects/optimizer/en/current/reference/python.html
- LKH-3: https://webhotel4.ruc.dk/~keld/research/LKH-3/
- PyTorch: https://pytorch.org/docs/stable/
- Lightning: https://lightning.ai/docs/pytorch/stable/
- RL4CO: https://ai4co.github.io/rl4co/ and https://rl4co.readthedocs.io/
- MLflow Tracking: https://mlflow.org/docs/latest/ml/tracking/

## License
MIT, see `LICENSE`.
