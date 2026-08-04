"""Runnable self-check for MockAdapter's state machine (dashboard/server.py).

No framework -- assert-based, exercises the same transitions the frontend
buttons trigger. Takes a few seconds (it waits through the same timers the
UI would see: INITIALIZING->RUNNING, EVALUATING->back, STOP sequence).

    python dashboard/test_adapter.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import MockAdapter  # noqa: E402


def wait_for(adapter, field, value, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if adapter.snapshot()[field] == value:
            return True
        time.sleep(0.05)
    return False


def main():
    a = MockAdapter()

    assert a.snapshot()["trainingState"] == "IDLE"

    # start -> INITIALIZING -> RUNNING
    result = a.handle_action("start", {"totalEpisodes": 50})
    assert result["ok"], result
    assert a.snapshot()["trainingState"] == "INITIALIZING"
    assert wait_for(a, "trainingState", "RUNNING"), "never left INITIALIZING"

    # duplicate start is rejected
    assert a.handle_action("start", {})["ok"] is False

    # let a few episodes tick so charts have data
    time.sleep(2.0)
    snap = a.snapshot()
    assert snap["episode"] > 0, "episode counter did not advance"
    telemetry = a.telemetry()
    assert len(telemetry["reward"]) == snap["episode"]
    assert "mpcMonitor" in telemetry and telemetry["mpcMonitor"]["reference"]

    # pause / resume
    assert a.handle_action("pause", {})["ok"]
    assert a.snapshot()["trainingState"] == "PAUSED"
    assert a.handle_action("resume", {})["ok"]
    assert a.snapshot()["trainingState"] == "RUNNING"

    # params: immediate applies live now; next_run-scoped stages without
    # touching the active run (batchSize isn't consumed by the mock loop,
    # only totalEpisodes/learningRate are reflected in top-level state)
    p = a.handle_params({"learningRate": 1e-4, "batchSize": 512})
    assert p["ok"] and p["applied"] == {"learningRate": 1e-4, "batchSize": 512}
    assert a.snapshot()["learningRate"] == 1e-4
    assert a.snapshot()["parameters"]["batchSize"]["value"] == 512

    # evaluate locks params, then returns to RUNNING
    assert a.handle_action("evaluate", {})["ok"]
    assert a.snapshot()["trainingState"] == "EVALUATING"
    assert a.handle_params({"learningRate": 5e-4})["ok"] is False
    assert wait_for(a, "trainingState", "RUNNING", timeout=4.0)

    # manual checkpoint save + load
    save = a.handle_action("save_checkpoint", {})
    assert save["ok"]
    name = save["checkpoint"]["filename"]
    assert name.startswith("actor_critic_mpc_ep_")
    assert a.handle_action("pause", {})["ok"]
    load = a.handle_action("load_checkpoint", {"filename": name})
    assert load["ok"], load

    # stop sequence walks all phases and lands on IDLE
    assert a.handle_action("stop", {})["ok"]
    assert a.snapshot()["trainingState"] == "STOPPING"
    assert wait_for(a, "trainingState", "IDLE", timeout=5.0)
    assert a.snapshot()["stopPhase"] is None

    print("OK -- all MockAdapter state transitions behaved as expected")


if __name__ == "__main__":
    main()
