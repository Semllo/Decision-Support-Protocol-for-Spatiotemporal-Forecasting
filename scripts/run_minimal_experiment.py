from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper1_minimal import ExperimentConfig, run_experiment
from paper1_minimal.synthetic import SyntheticConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal reference-uncertainty experiment.")
    parser.add_argument("--output-dir", default="outputs", help="Directory where artifacts will be written.")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of synthetic seeds.")
    parser.add_argument("--num-nodes", type=int, default=12, help="Number of graph nodes.")
    parser.add_argument("--num-steps", type=int, default=24 * 28, help="Number of hourly steps per seed.")
    parser.add_argument("--input-length", type=int, default=24, help="Context window length.")
    parser.add_argument("--output-length", type=int, default=12, help="Prediction bundle length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        num_seeds=args.num_seeds,
        input_length=args.input_length,
        output_length=args.output_length,
        synthetic=SyntheticConfig(num_nodes=args.num_nodes, num_steps=args.num_steps),
    )
    report = run_experiment(config=config, output_dir=Path(args.output_dir).resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
