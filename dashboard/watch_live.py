"""Native MuJoCo viewer window that tails a live_state.json a run_box_catch
episode is writing (see AcmpcBoxCatchConfig.live_state_path in
acmpc/main_acmpc_box_catch.py) -- run this in a terminal to watch a live
training run (or any standalone run_box_catch call) at full quality/rate.
Replaces the dashboard's old JPEG-over-HTTP viewport, which was too lossy
and too slow to poll (200ms, quality-70 480x270 JPEGs) to actually watch a
catch attempt; this instead re-poses a real GL window straight from qpos, no
network/encoding round trip in between.

Usage:
  python3 dashboard/watch_live.py                      # tails the newest dashboard/runs/*/live_state.json
  python3 dashboard/watch_live.py path/to/live_state.json
"""
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "model/robotis_ffw/scene_ffw_sg2_fixed_base_box_dynamic_squeeze.xml"
POLL_S = 0.02


def _newest_live_state() -> Path:
    candidates = sorted(
        ROOT.glob("dashboard/runs/*/live_state.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise SystemExit(
            "no dashboard/runs/*/live_state.json found -- start a run first "
            "(dashboard --mode real, or acmpc/main_acmpc_box_catch_curriculum.py "
            "--live-state-path <path>)"
        )
    return candidates[0]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _newest_live_state()
    print(f"watching {path}")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    last_mtime = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if path.exists():
                mtime = path.stat().st_mtime
                if mtime != last_mtime:
                    try:
                        qpos = json.loads(path.read_text())["qpos"]
                        n = min(len(qpos), model.nq)
                        data.qpos[:n] = qpos[:n]
                        mujoco.mj_forward(model, data)
                        viewer.sync()
                        last_mtime = mtime
                    except (OSError, ValueError, KeyError):
                        pass  # transient: file mid-write; retry next tick
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
