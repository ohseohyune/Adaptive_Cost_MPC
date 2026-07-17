"""Milestone 3: SE(3) interception and stabilization of a rotating box."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from control.squeeze import RotatingSideSqueezeConfig
from main_dynamic_box_squeeze import DynamicRunConfig, run_dynamic_side_squeeze


ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--log",
        default=str(ROOT / "sweep_results" / "rotating_box_squeeze.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = RotatingSideSqueezeConfig(random_seed=args.seed)
    summary = run_dynamic_side_squeeze(
        DynamicRunConfig(viewer=args.viewer, log_path=args.log, squeeze=config)
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if not summary.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
