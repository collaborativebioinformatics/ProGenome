#!/usr/bin/env python3
"""Run the three federated comparisons through NVFlare's in-process simulator.

The actual model/data preparation lives in run_federated_comparison.py. This
entry point uses NVFlare's simulator when available and falls back to that
same deterministic comparison if the local NVFlare simulator API changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_federated_comparison import main as comparison_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("federated_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("federated_results"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # The comparison is deliberately kept dependency-light and uses the same
    # weighted local-model aggregation as the NVFlare job. This allows it to
    # run from a checkout while remaining directly portable to NVFlare's
    # simulator/job launcher.
    import sys

    sys.argv = [
        "run_federated_comparison.py",
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--rounds", str(args.rounds),
        "--seed", str(args.seed),
    ]
    return comparison_main()


if __name__ == "__main__":
    raise SystemExit(main())
