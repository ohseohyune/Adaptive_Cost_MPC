"""Mode C: the frozen fixed-MPC baseline on the residual-zero evaluation set.

Separate from run_residual_zero_ablation.py because this controller is *not*
an AC-MPC checkpoint replay -- it has no actor at all (weight_delta_fraction=0)
and it uses FROZEN_PHASE_PRIORS, which differ from the source default priors
that AC-MPC's residual-zero mode falls back to. Running it on the identical
scenario sequence is what makes "is residual-zero the same controller as the
fixed baseline?" answerable with numbers rather than by inspection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.runtime_environment import (  # noqa: E402
    log_runtime_environment,
    validate_runtime_environment,
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.fixed_baseline_frozen import FROZEN_PHASE_PRIORS  # noqa: E402
from acmpc.main_acmpc_box_catch_curriculum import (  # noqa: E402
    CurriculumBoxCatchConfig,
    run_curriculum_box_catch,
)


def main() -> None:
    # Parent-side gate: refuse to spawn a fleet of hour-long jobs from a
    # non-canonical interpreter. Each child re-validates independently, since
    # the parent cannot vouch for an interpreter it does not run in.
    log_runtime_environment(
        "fixed-baseline eval", validate_runtime_environment(context="fixed-baseline eval runner")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "sweep_results" / "residual_zero_20260805" / "fixed_baseline",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=100_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--default-priors",
        action="store_true",
        help="use the source default phase priors instead of the frozen "
        "fixed-baseline overrides -- isolates how much of any gap between "
        "residual-zero AC-MPC and the fixed baseline is the prior alone.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_curriculum_box_catch(
        CurriculumBoxCatchConfig(
            episodes=args.episodes,
            random_seed=args.evaluation_seed,
            device=args.device,
            curriculum_mode="balanced",
            online_learning=False,
            # The robust zero-residual guarantee (see fixed_baseline_frozen's
            # docstring): a delta fraction of 0 cannot reintroduce a residual
            # even if a checkpoint were loaded.
            weight_delta_fraction=0.0,
            phase_priors=None if args.default_priors else FROZEN_PHASE_PRIORS,
            checkpoint_path=str(args.output_dir / "unused_checkpoint.pt"),
            load_checkpoint=False,
            log_path=str(args.output_dir / "result.json"),
            progress_every=args.episodes + 1,
            evaluation_every=0,
        )
    )
    print(f"wrote {args.output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
