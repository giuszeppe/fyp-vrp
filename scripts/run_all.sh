#!/usr/bin/env bash
set -euo pipefail

python -m dvrptw_bench.cli.run_all_cli run \
  --dataset-root ./dataset \
  --output-root ./outputs \
  --verbose
