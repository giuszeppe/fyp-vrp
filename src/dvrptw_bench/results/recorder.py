"""Result recorder for JSONL and Parquet outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dvrptw_bench.common.serialization import append_jsonl
from dvrptw_bench.common.typing import ResultRecord


class Recorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "records.jsonl"
        self.records: list[ResultRecord] = []

    def log(self, record: ResultRecord) -> None:
        self.records.append(record)
        append_jsonl(self.jsonl_path, record.model_dump(mode="json"))

    def flush_parquet(self) -> Path:
        rows = [r.model_dump(mode="json") for r in self.records]
        df = pd.DataFrame(rows)
        out = self.run_dir / "records.parquet"
        df.to_parquet(out, index=False)
        return out
