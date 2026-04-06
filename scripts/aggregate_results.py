#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment summaries across run directories.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-csv", type=Path, default=Path("results") / "aggregated_results.csv")
    parser.add_argument("--output-json", type=Path, default=Path("results") / "aggregated_results.json")
    parser.add_argument("--best-csv", type=Path, default=Path("results") / "best_by_dataset.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_files = sorted(args.results_root.glob("*/summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No summary.csv files found under {args.results_root}")

    frames = []
    for summary_file in summary_files:
        frame = pd.read_csv(summary_file)
        frame["summary_file"] = str(summary_file.resolve())
        frame["run_directory"] = summary_file.parent.name
        frames.append(frame)

    aggregated = pd.concat(frames, ignore_index=True)
    aggregated = aggregated.sort_values(
        ["dataset", "macro_f1", "accuracy", "experiment_name"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output_csv, index=False)
    with args.output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(aggregated.to_dict(orient="records"), handle, indent=2, ensure_ascii=True)

    best = aggregated.groupby("dataset", as_index=False).first()
    best.to_csv(args.best_csv, index=False)

    print(f"Aggregated rows: {len(aggregated)}")
    print(f"Aggregated CSV: {args.output_csv.resolve()}")
    print(f"Aggregated JSON: {args.output_json.resolve()}")
    print(f"Best-by-dataset CSV: {args.best_csv.resolve()}")


if __name__ == "__main__":
    main()
