#!/usr/bin/env bash
set -euo pipefail

python -m dvrptw_bench.cli.heuristics_cli list-instances --dataset-root ./dataset
python -m dvrptw_bench.cli.heuristics_cli solve-static --solver pmca --instance RC101.txt --time-limit 5
