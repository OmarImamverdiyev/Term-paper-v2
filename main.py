from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.runner import ExperimentRunner
from src.utils.config import load_yaml_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "main_sentiment140_full.yaml"


def main() -> None:
    config, config_path = load_yaml_config(DEFAULT_CONFIG)
    runner = ExperimentRunner(config=config, config_path=config_path)
    outputs = runner.run()
    print(f"Run completed: {runner.run_id}")
    print(f"Summary CSV: {outputs['summary_csv']}")
    print(f"Summary JSON: {outputs['summary_json']}")
    print(f"Log file: {outputs['log_file']}")
    print(f"Resolved config: {outputs['resolved_config']}")


if __name__ == "__main__":
    main()
