#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.runner import ExperimentRunner
from src.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular sentiment classification experiments from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "main_sentiment140_full.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Optional dataset filter. Repeatable.")
    parser.add_argument("--feature", action="append", default=[], help="Optional feature filter. Repeatable.")
    parser.add_argument("--model", action="append", default=[], help="Optional model filter. Repeatable.")
    parser.add_argument("--experiment", action="append", default=[], help="Optional experiment-name filter. Repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_yaml_config(args.config)
    runner = ExperimentRunner(
        config=config,
        config_path=config_path,
        experiment_filters={
            "datasets": set(args.dataset),
            "features": set(args.feature),
            "models": set(args.model),
            "experiments": set(args.experiment),
        },
    )
    outputs = runner.run()
    print(f"Run completed: {runner.run_id}")
    print(f"Summary CSV: {outputs['summary_csv']}")
    print(f"Summary JSON: {outputs['summary_json']}")
    print(f"Log file: {outputs['log_file']}")
    print(f"Resolved config: {outputs['resolved_config']}")


if __name__ == "__main__":
    main()
