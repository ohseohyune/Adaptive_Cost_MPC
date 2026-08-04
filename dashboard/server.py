"""Actor-Critic MPC Lab dashboard backend.

Serves dashboard/index.html plus the JSON + WebSocket API documented in
dashboard/API.md. Two interchangeable telemetry sources, chosen with
--mode:

  mock (default) -- self-driving simulated training run. No side effects,
                     safe to leave running; use this to develop/demo the UI.
  real            -- drives acmpc/main_acmpc_box_catch_curriculum.py as a
                     real subprocess (PAUSE/RESUME use SIGSTOP/SIGCONT,
                     STOP uses SIGINT) and reads its progress log + the
                     adaptive-cost-mpc-box-catch W&B project.

Usage:
  python dashboard/server.py                    # mock, http://localhost:8770
  python dashboard/server.py --mode real --port 8770
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
HTML_PATH = DASHBOARD_DIR / "index.html"
RUNS_DIR = DASHBOARD_DIR / "runs"

WANDB_PROJECT = "adaptive-cost-mpc-box-catch"  # control/mpc/wandb_logger.py:init_wandb default
COST_NAMES = ("object", "grasp", "compression", "velocity", "smoothness")  # matches wandb_logger._LOG_LABELS

TICK_S = 0.5
HISTORY_MAXLEN = 4000

TRAINING_STATES = (
    "IDLE", "INITIALIZING", "RUNNING", "PAUSED", "EVALUATING",
    "SAVING", "STOPPING", "COMPLETED", "ERROR",
)
STOP_PHASES = ("STOP_REQUESTED", "FINISH_UPDATE", "SAVING", "SYNCING_WANDB", "STOPPED")


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _ok(**extra):
    return {"ok": True, **extra}


def _err(message):
    return {"ok": False, "error": message}


def checkpoint_filename(episode: int, global_step: int) -> str:
    return f"actor_critic_mpc_ep_{episode:05d}_step_{global_step:08d}.pt"


def default_parameters() -> dict:
    return {
        "learningRate":     {"value": 3.0e-4,  "scope": "immediate", "group": "training"},
        "totalEpisodes":    {"value": 5000,    "scope": "next_run",  "group": "training"},
        "batchSize":        {"value": 256,     "scope": "next_run",  "group": "training"},
        "gamma":            {"value": 0.99,    "scope": "next_run",  "group": "training"},
        "tau":              {"value": 0.005,   "scope": "immediate", "group": "training"},
        "replayBufferSize": {"value": 200_000, "scope": "next_run",  "group": "training"},
        "actorUpdateFreq":  {"value": 2,       "scope": "next_run",  "group": "training"},
        "seed":             {"value": 7,       "scope": "next_run",  "group": "training"},
        "predictionHorizon":  {"value": 12,   "scope": "next_run",  "group": "mpc"},
        "controlHorizon":     {"value": 6,    "scope": "next_run",  "group": "mpc"},
        "numSamples":         {"value": 256,  "scope": "next_run",  "group": "mpc"},
        "optimIterations":    {"value": 4,    "scope": "immediate", "group": "mpc"},
        "temperature":        {"value": 0.65, "scope": "immediate", "group": "mpc"},
        "controlFrequency":   {"value": 50,   "scope": "next_run",  "group": "mpc"},
        **{
            f"costWeight_{name}": {"value": 1.0, "scope": "immediate", "group": "mpc_cost"}
            for name in COST_NAMES
        },
    }


def initial_state() -> dict:
    return {
        "backendConnected": True,
        "wandbConnected": False,
        "wandbRunUrl": None,
        "trainingState": "IDLE",
        "stopPhase": None,
        "episode": 0,
        "totalEpisodes": default_parameters()["totalEpisodes"]["value"],
        "globalStep": 0,
        "progress": 0.0,
        "successRate": 0.0,
        "successRateStatus": "BELOW_TARGET",
        "currentReward": 0.0,
        "bestReward": None,  # None, not -inf: -Infinity is not valid JSON (browser JSON.parse throws on it)
        "actorLoss": None,
        "criticLoss": None,
        "learningRate": default_parameters()["learningRate"]["value"],
        "mpcCost": None,
        "simulationFps": 0.0,
        "stage": "—",
        "checkpoint": {
            "lastSaved": None,
            "filename": None,
            "path": None,
            "episode": None,
            "globalStep": None,
            "isBest": False,
            "autoSaveEnabled": True,
            "lastResult": None,
        },
        "parameters": default_parameters(),
    }


def _success_status(history: list[float]) -> str:
    if not history:
        return "BELOW_TARGET"
    rate = history[-1]
    recent = history[-10:]
    stdev = statistics.pstdev(recent) if len(recent) > 1 else 0.0
    if rate >= 0.8:
        return "UNSTABLE" if stdev > 0.12 else "GOOD"
    if len(history) >= 6 and rate > history[-6] + 0.03:
        return "IMPROVING"
    return "BELOW_TARGET" if rate < 0.5 else "IMPROVING"


class MockAdapter:
    """Self-driving simulated AC-MPC training run. No filesystem/network side
    effects beyond writing placeholder checkpoint files under dashboard/runs/mock/.
    """

    SERIES = (
        "reward", "successRate", "actorLoss", "criticLoss", "learningRate",
        "mpcCost", "episodeLength", "entropy", "qValue", "actionMagnitude",
    )

    def __init__(self):
        self._lock = threading.RLock()
        self._state = initial_state()
        self._state["wandbConnected"] = True
        self._state["wandbRunUrl"] = f"https://wandb.ai/mock/{WANDB_PROJECT}/runs/mock-demo"
        self._history = {name: deque(maxlen=HISTORY_MAXLEN) for name in self.SERIES}
        self._success_hist: list[float] = []
        self._checkpoints: dict[str, dict] = {}
        self._tick_count = 0
        self._eval_return_state = "IDLE"
        self._ckpt_dir = RUNS_DIR / "mock" / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._loop, daemon=True).start()

    # -- read side -----------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return _sanitize(self._state)

    def telemetry(self) -> dict:
        with self._lock:
            out = {name: list(self._history[name]) for name in self.SERIES}
            out["mpcMonitor"] = self._mpc_monitor()
            return out

    def wandb_media(self) -> list:
        return []  # mock run never logs images; real adapter documents the hook

    # -- background tick -------------------------------------------------
    def _loop(self):
        while True:
            time.sleep(TICK_S)
            with self._lock:
                self._advance()

    def _advance(self):
        st = self._state
        st["simulationFps"] = round(_clip(st["simulationFps"] + random.uniform(-2, 2), 0, 63)
                                     if st["simulationFps"] else 58.0, 1)
        if st["trainingState"] not in ("RUNNING", "EVALUATING"):
            return
        self._tick_count += 1
        if st["trainingState"] == "RUNNING":
            st["globalStep"] += random.randint(180, 260)
            if self._tick_count % 3 == 0:
                self._complete_episode()
        elif st["trainingState"] == "EVALUATING":
            st["mpcCost"] = round(max(0.05, (st["mpcCost"] or 1.0) + random.gauss(0, 0.05)), 4)

    def _complete_episode(self):
        st = self._state
        st["episode"] += 1
        ep = st["episode"]
        st["progress"] = min(1.0, ep / max(1, st["totalEpisodes"]))

        reward_target = 6 + 12 * (1 - math.exp(-ep / 900))
        st["currentReward"] = round(reward_target + random.gauss(0, 1.8), 3)
        st["bestReward"] = st["currentReward"] if st["bestReward"] is None else max(st["bestReward"], st["currentReward"])

        sr_target = 0.9 / (1 + math.exp(-(ep - 500) / 220))
        sr = round(_clip(sr_target + random.gauss(0, 0.05), 0.0, 1.0), 4)
        st["successRate"] = sr
        self._success_hist.append(sr)
        st["successRateStatus"] = _success_status(self._success_hist)

        st["actorLoss"] = round(max(0.001, 0.6 * math.exp(-ep / 700) + random.gauss(0, 0.01)), 5)
        st["criticLoss"] = round(max(0.001, 1.1 * math.exp(-ep / 600) + random.gauss(0, 0.02)), 5)
        st["mpcCost"] = round(max(0.05, 3.2 * math.exp(-ep / 800) + random.gauss(0, 0.08)), 4)
        st["stage"] = ("warmup" if ep < 300 else "shape" if ep < 1200 else
                        "high_speed" if ep < 2600 else "full")

        entropy = max(0.02, 1.4 * math.exp(-ep / 1000) + random.gauss(0, 0.02))
        q_value = reward_target * 0.8 + random.gauss(0, 0.5)
        action_mag = _clip(0.6 + random.gauss(0, 0.08), 0, 1)
        ep_len = int(_clip(90 + random.gauss(0, 8), 40, 140))

        point = {"episode": ep, "step": st["globalStep"]}
        self._history["reward"].append({**point, "value": st["currentReward"]})
        self._history["successRate"].append({**point, "value": sr})
        self._history["actorLoss"].append({**point, "value": st["actorLoss"]})
        self._history["criticLoss"].append({**point, "value": st["criticLoss"]})
        self._history["learningRate"].append({**point, "value": st["learningRate"]})
        self._history["mpcCost"].append({**point, "value": st["mpcCost"]})
        self._history["episodeLength"].append({**point, "value": ep_len})
        self._history["entropy"].append({**point, "value": round(entropy, 4)})
        self._history["qValue"].append({**point, "value": round(q_value, 3)})
        self._history["actionMagnitude"].append({**point, "value": round(action_mag, 4)})

        if st["checkpoint"]["autoSaveEnabled"] and ep % 250 == 0:
            self._save_checkpoint(manual=False)
        if ep >= st["totalEpisodes"]:
            self._save_checkpoint(manual=False)
            st["trainingState"] = "COMPLETED"
            st["progress"] = 1.0

    def _mpc_monitor(self) -> dict:
        st = self._state
        n, horizon = 40, 10
        progress = st["progress"] or 0.0
        noise_scale = _clip(0.5 * (1 - progress) + 0.03, 0.03, 0.5)
        t = list(range(n))
        reference = [math.sin(x / 6) for x in t]
        actual = [reference[i] + random.gauss(0, noise_scale) for i in t]
        pred_from = max(0, n - horizon)
        predicted = [actual[pred_from]] if pred_from < n else []
        for i in range(pred_from + 1, n + horizon):
            drift = math.sin(i / 6) + random.gauss(0, noise_scale * 1.3)
            predicted.append(drift)
        control_error = (sum((a - r) ** 2 for a, r in zip(actual, reference)) / n) ** 0.5
        return {
            "t": t, "reference": reference, "actual": actual,
            "predictedFrom": pred_from, "predicted": predicted,
            "controlError": round(control_error, 4), "mpcCost": st["mpcCost"],
        }

    # -- checkpoint -----------------------------------------------------
    def _save_checkpoint(self, manual: bool):
        st = self._state
        name = checkpoint_filename(st["episode"], st["globalStep"])
        path = self._ckpt_dir / name
        try:
            path.write_bytes(b"")  # placeholder payload; mock has no real weights
            is_best = st["currentReward"] >= st["bestReward"] - 1e-9
            record = {
                "lastSaved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "filename": name,
                "path": str(path.relative_to(ROOT)),
                "episode": st["episode"],
                "globalStep": st["globalStep"],
                "isBest": is_best,
                "autoSaveEnabled": st["checkpoint"]["autoSaveEnabled"],
                "lastResult": "SUCCESS",
                "manual": manual,
            }
            self._checkpoints[name] = record
            st["checkpoint"] = record
        except OSError as exc:
            st["checkpoint"] = {**st["checkpoint"], "lastResult": "FAILED", "error": str(exc)}

    def list_checkpoints(self) -> list:
        with self._lock:
            return sorted(self._checkpoints.values(), key=lambda r: r["episode"], reverse=True)

    # -- actions ----------------------------------------------------------
    def handle_action(self, action: str, payload: dict) -> dict:
        with self._lock:
            st = self._state["trainingState"]
            if action == "start":
                if st not in ("IDLE", "COMPLETED", "ERROR"):
                    return _err(f"cannot start while {st}")
                total = payload.get("totalEpisodes") or self._state["parameters"]["totalEpisodes"]["value"]
                self._state.update({
                    "trainingState": "INITIALIZING", "episode": 0, "globalStep": 0,
                    "progress": 0.0, "totalEpisodes": int(total), "stage": "warmup",
                })
                self._state["bestReward"] = None
                self._success_hist.clear()
                threading.Timer(1.2, self._finish_init).start()
                return _ok()
            if action == "pause":
                if st != "RUNNING":
                    return _err("not running")
                self._state["trainingState"] = "PAUSED"
                return _ok()
            if action == "resume":
                if st != "PAUSED":
                    return _err("not paused")
                self._state["trainingState"] = "RUNNING"
                return _ok()
            if action == "evaluate":
                if st not in ("RUNNING", "PAUSED"):
                    return _err("start or resume training first")
                self._eval_return_state = st
                self._state["trainingState"] = "EVALUATING"
                threading.Timer(2.5, self._finish_eval).start()
                return _ok()
            if action == "save_checkpoint":
                if st in ("STOPPING",):
                    return _err("stop sequence in progress")
                self._save_checkpoint(manual=True)
                return _ok(checkpoint=self._state["checkpoint"])
            if action == "load_checkpoint":
                if st not in ("IDLE", "PAUSED", "COMPLETED"):
                    return _err("pause or stop training before loading a checkpoint")
                name = payload.get("filename")
                record = self._checkpoints.get(name)
                if not record:
                    return _err(f"unknown checkpoint '{name}'")
                self._state["episode"] = record["episode"]
                self._state["globalStep"] = record["globalStep"]
                self._state["checkpoint"] = {**record, "lastResult": "SUCCESS"}
                return _ok()
            if action == "stop":
                if st in ("IDLE", "STOPPING"):
                    return _err("nothing to stop")
                self._state["trainingState"] = "STOPPING"
                self._state["stopPhase"] = "STOP_REQUESTED"
                threading.Thread(target=self._run_stop_sequence, daemon=True).start()
                return _ok()
            if action == "set_autosave":
                self._state["checkpoint"]["autoSaveEnabled"] = bool(payload.get("enabled", True))
                return _ok()
            return _err(f"unknown action '{action}'")

    def _finish_init(self):
        with self._lock:
            if self._state["trainingState"] == "INITIALIZING":
                self._state["trainingState"] = "RUNNING"

    def _finish_eval(self):
        with self._lock:
            if self._state["trainingState"] == "EVALUATING":
                self._state["trainingState"] = self._eval_return_state

    def _run_stop_sequence(self):
        delays = {"FINISH_UPDATE": 0.6, "SAVING": 0.6, "SYNCING_WANDB": 0.5, "STOPPED": 0.4}
        for phase in ("FINISH_UPDATE", "SAVING", "SYNCING_WANDB", "STOPPED"):
            time.sleep(delays[phase])
            with self._lock:
                if phase == "SAVING":
                    self._save_checkpoint(manual=False)
                self._state["stopPhase"] = phase
        time.sleep(0.6)
        with self._lock:
            self._state["trainingState"] = "IDLE"
            self._state["stopPhase"] = None

    def handle_params(self, payload: dict) -> dict:
        # NEXT_RUN-scoped params stay editable at any time -- they only
        # stage a value for the *next* `start`, they never touch the
        # active run. Only EVALUATING blocks edits outright (spec: no
        # parameter changes while an evaluation rollout is in flight).
        with self._lock:
            if self._state["trainingState"] == "EVALUATING":
                return _err("parameters are locked during evaluation")
            st = self._state["trainingState"]
            applied, rejected = {}, {}
            for key, value in payload.items():
                spec = self._state["parameters"].get(key)
                if spec is None:
                    rejected[key] = "unknown parameter"
                    continue
                spec["value"] = value
                applied[key] = value
                if spec["scope"] == "immediate" and key == "learningRate":
                    self._state["learningRate"] = value
                if key == "totalEpisodes" and st in ("IDLE", "COMPLETED", "ERROR"):
                    self._state["totalEpisodes"] = int(value)
            return {"ok": not rejected, "applied": applied, "rejected": rejected}


def _sanitize(obj):
    """Recursively replace inf/-inf/nan floats with None.

    json.dumps() serializes these natively (as the literal tokens Infinity /
    -Infinity / NaN) instead of ever consulting a `default=` callback, since
    float is already a JSON-serializable type to the encoder -- so a
    `default=` fallback silently never fires for them. Those tokens are not
    valid JSON; browser JSON.parse throws on them. Sanitizing the object
    tree before dumps() is the only point that actually catches every
    source, including a malformed client payload like {"x": 1e999}.
    """
    if isinstance(obj, float):
        return None if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Real adapter: drives acmpc/main_acmpc_box_catch_curriculum.py.
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(
    r"episodes (?P<lo>\d+)-(?P<hi>\d+): success_rate=(?P<sr>[\d.]+) "
    r"mean_reward=(?P<reward>-?[\d.]+) mean_actor_loss=(?P<actor>[\d.]+) "
    r"mean_kl=(?P<kl>[\d.]+) stage=(?P<stage>\w+) "
    r"scheduler_success_rate=(?P<sched_sr>[\d.]+)"
)


class RealAdapter:
    """Drives the real curriculum trainer as a subprocess.

    Honesty note: the trainer does not print a per-episode global_step, so
    globalStep here is *estimated* as episode_count * an assumed
    steps-per-episode (see _STEPS_PER_EPISODE_ESTIMATE below), not read
    from the process. Every other live field (success_rate, reward, actor
    loss, stage) is parsed directly from the trainer's own progress line,
    so it is exact.
    """

    SCRIPT = ROOT / "acmpc" / "main_acmpc_box_catch_curriculum.py"
    _STEPS_PER_EPISODE_ESTIMATE = 120  # rough box-catch episode length; see episodeLength chart for the real value once eval data exists

    def __init__(self):
        self._lock = threading.RLock()
        self._state = initial_state()
        self._history = {name: deque(maxlen=HISTORY_MAXLEN) for name in MockAdapter.SERIES}
        self._success_hist: list[float] = []
        self._proc: subprocess.Popen | None = None
        self._run_dir: Path | None = None
        self._live_checkpoint_path: Path | None = None
        self._live_state_path: Path | None = None  # dashboard/watch_live.py tails this, not this server
        self._checkpoints: dict[str, dict] = {}
        self._api = None
        self._wandb_run = None
        threading.Thread(target=self._wandb_poll_loop, daemon=True).start()

    def snapshot(self) -> dict:
        with self._lock:
            self._state["backendConnected"] = True
            return _sanitize(self._state)

    def telemetry(self) -> dict:
        with self._lock:
            out = {name: list(self._history[name]) for name in MockAdapter.SERIES}
            out["mpcMonitor"] = {}  # real per-step trajectory needs an in-process hook; not wired yet
            return out

    def wandb_media(self) -> list:
        """Image files logged to the active W&B run, if any. Returns [] (not
        an error) when wandb is unavailable or the run has not logged any
        wandb.Image media -- true today since no caller in this repo logs
        images yet (see control/mpc/wandb_logger.py). Wiring an actual
        acmpc/*.py caller to log e.g. a trajectory plot via wandb.Image(...)
        is what would make this list non-empty.
        """
        if self._wandb_run is None:
            return []
        try:
            files = self._wandb_run.files()
            return [
                {"name": f.name, "url": f.url, "updatedAt": getattr(f, "updated_at", None)}
                for f in files
                if f.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
            ]
        except Exception:
            return []

    # -- subprocess control ------------------------------------------------
    def handle_action(self, action: str, payload: dict) -> dict:
        with self._lock:
            st = self._state["trainingState"]
            if action == "start":
                if st not in ("IDLE", "COMPLETED", "ERROR"):
                    return _err(f"cannot start while {st}")
                return self._start(payload)
            if action == "pause":
                if st != "RUNNING" or self._proc is None:
                    return _err("not running")
                if os.name != "posix":
                    return _err("pause/resume needs POSIX signals (SIGSTOP/SIGCONT)")
                try:
                    self._proc.send_signal(signal.SIGSTOP)
                except ProcessLookupError:
                    return _err("training process already exited")
                self._state["trainingState"] = "PAUSED"
                return _ok()
            if action == "resume":
                if st != "PAUSED" or self._proc is None:
                    return _err("not paused")
                try:
                    self._proc.send_signal(signal.SIGCONT)
                except ProcessLookupError:
                    return _err("training process already exited")
                self._state["trainingState"] = "RUNNING"
                return _ok()
            if action == "evaluate":
                return _err("evaluate is not wired for --mode real yet; run "
                             "acmpc/evaluate_bimanual_acmpc.py separately")
            if action == "save_checkpoint":
                return self._archive_checkpoint(manual=True)
            if action == "load_checkpoint":
                if st not in ("IDLE", "COMPLETED", "ERROR"):
                    return _err("stop training before loading a checkpoint")
                name = payload.get("filename")
                record = self._checkpoints.get(name)
                if not record:
                    return _err(f"unknown checkpoint '{name}'")
                self._state["parameters"]["_loadCheckpointPath"] = {
                    "value": record["path"], "scope": "next_run", "group": "training",
                }
                self._state["checkpoint"] = {**record, "lastResult": "SUCCESS"}
                return _ok(note="applied as NEXT RUN --checkpoint/--load-checkpoint")
            if action == "stop":
                if st in ("IDLE", "STOPPING") or self._proc is None:
                    return _err("nothing to stop")
                self._state["trainingState"] = "STOPPING"
                self._state["stopPhase"] = "STOP_REQUESTED"
                threading.Thread(target=self._run_stop_sequence, daemon=True).start()
                return _ok()
            if action == "set_autosave":
                self._state["checkpoint"]["autoSaveEnabled"] = bool(payload.get("enabled", True))
                return _ok()
            return _err(f"unknown action '{action}'")

    def _start(self, payload: dict) -> dict:
        params = self._state["parameters"]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = RUNS_DIR / run_id
        (self._run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        log_path = self._run_dir / "progress.json"

        # One-shot: a LOAD CHECKPOINT selection applies to exactly this
        # start, then clears -- otherwise every future run would silently
        # keep resuming from that same file. --checkpoint doubles as the
        # trainer's save-in-place path (see main_acmpc_box_catch_curriculum
        # .py's CurriculumBoxCatchConfig.checkpoint_path), so pointing it at
        # the archived file is what actually makes --load-checkpoint resume
        # from the file the user picked, instead of a fresh empty path.
        load_from = params.pop("_loadCheckpointPath", None)
        checkpoint_path = (ROOT / load_from["value"]) if load_from else (self._run_dir / "checkpoint.pt")
        self._live_checkpoint_path = checkpoint_path
        self._live_state_path = self._run_dir / "live_state.json"

        total = int(payload.get("totalEpisodes") or params["totalEpisodes"]["value"])
        cmd = [
            sys.executable, "-u", str(self.SCRIPT),
            "--episodes", str(total),
            "--seed", str(int(params["seed"]["value"])),
            "--device", str(payload.get("device", "cpu")),
            "--checkpoint", str(checkpoint_path),
            "--log", str(log_path),
            "--progress-every", "20",
            "--live-state-path", str(self._live_state_path),
        ]
        if load_from:
            cmd += ["--load-checkpoint"]
        if payload.get("useWandb", True):
            cmd += ["--use-wandb", "--wandb-run-name", f"dashboard-{run_id}"]

        try:
            self._proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
        except OSError as exc:
            self._state["trainingState"] = "ERROR"
            return _err(f"failed to launch trainer: {exc}")

        self._state.update({
            "trainingState": "INITIALIZING", "episode": 0, "globalStep": 0,
            "progress": 0.0, "totalEpisodes": total, "stage": "warmup",
            "wandbRunUrl": None, "wandbConnected": False,
        })
        self._state["bestReward"] = None
        self._success_hist.clear()
        self._run_wandb_name = f"dashboard-{run_id}"
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        return _ok(runId=run_id)

    def _read_stdout(self, proc: subprocess.Popen):
        first_line = True
        for line in proc.stdout:
            with self._lock:
                if first_line and self._state["trainingState"] == "INITIALIZING":
                    self._state["trainingState"] = "RUNNING"
                    first_line = False
                match = _PROGRESS_RE.search(line)
                if match:
                    self._apply_progress_line(match)
        proc.wait()  # stdout EOF fires slightly before the process is reaped -- returncode is None until this
        with self._lock:
            if self._state["trainingState"] not in ("STOPPING", "IDLE"):
                self._state["trainingState"] = "COMPLETED" if proc.returncode == 0 else "ERROR"

    def _apply_progress_line(self, m: re.Match):
        st = self._state
        hi = int(m.group("hi"))
        st["episode"] = hi
        st["globalStep"] = hi * self._STEPS_PER_EPISODE_ESTIMATE
        st["progress"] = min(1.0, hi / max(1, st["totalEpisodes"]))
        st["currentReward"] = round(float(m.group("reward")), 3)
        st["bestReward"] = st["currentReward"] if st["bestReward"] is None else max(st["bestReward"], st["currentReward"])
        sr = round(float(m.group("sr")), 4)
        st["successRate"] = sr
        self._success_hist.append(sr)
        st["successRateStatus"] = _success_status(self._success_hist)
        st["actorLoss"] = round(float(m.group("actor")), 5)
        st["stage"] = m.group("stage")
        point = {"episode": hi, "step": st["globalStep"]}
        self._history["reward"].append({**point, "value": st["currentReward"]})
        self._history["successRate"].append({**point, "value": sr})
        self._history["actorLoss"].append({**point, "value": st["actorLoss"]})
        self._history["learningRate"].append({**point, "value": st["learningRate"]})
        if st["checkpoint"]["autoSaveEnabled"]:
            self._archive_checkpoint(manual=False)

    def _archive_checkpoint(self, manual: bool) -> dict:
        st = self._state
        if self._run_dir is None:
            return _err("no active run")
        live_path = self._live_checkpoint_path
        if not live_path.exists():
            return _err("trainer has not written a checkpoint yet")
        name = checkpoint_filename(st["episode"], st["globalStep"])
        dest = self._run_dir / "checkpoints" / name
        try:
            dest.write_bytes(live_path.read_bytes())
            record = {
                "lastSaved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "filename": name,
                "path": str(dest.relative_to(ROOT)),
                "episode": st["episode"],
                "globalStep": st["globalStep"],
                # bestReward is still None before the first progress line
                # parses (see _apply_progress_line) -- a stop/save issued
                # before that boundary would otherwise crash this comparison.
                "isBest": st["bestReward"] is None or st["currentReward"] >= st["bestReward"] - 1e-9,
                "autoSaveEnabled": st["checkpoint"]["autoSaveEnabled"],
                "lastResult": "SUCCESS",
                "manual": manual,
            }
            self._checkpoints[name] = record
            st["checkpoint"] = record
            return _ok(checkpoint=record)
        except OSError as exc:
            st["checkpoint"] = {**st["checkpoint"], "lastResult": "FAILED", "error": str(exc)}
            return _err(str(exc))

    def list_checkpoints(self) -> list:
        with self._lock:
            return sorted(self._checkpoints.values(), key=lambda r: r["episode"], reverse=True)

    def _run_stop_sequence(self):
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        with self._lock:
            self._state["stopPhase"] = "STOP_REQUESTED"
        try:
            proc.send_signal(signal.SIGCONT)  # in case we stop from PAUSED
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        with self._lock:
            self._state["stopPhase"] = "FINISH_UPDATE"
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        with self._lock:
            self._state["stopPhase"] = "SAVING"
        self._archive_checkpoint(manual=False)
        with self._lock:
            self._state["stopPhase"] = "SYNCING_WANDB"
        time.sleep(0.5)  # the trainer's own wandb.finish() runs in its process, not ours
        with self._lock:
            self._state["stopPhase"] = "STOPPED"
        time.sleep(0.4)
        with self._lock:
            self._state["trainingState"] = "IDLE"
            self._state["stopPhase"] = None
            self._proc = None

    def handle_params(self, payload: dict) -> dict:
        with self._lock:
            if self._state["trainingState"] == "EVALUATING":
                return _err("parameters are locked during evaluation")
            applied, rejected = {}, {}
            for key, value in payload.items():
                spec = self._state["parameters"].get(key)
                if spec is None:
                    rejected[key] = "unknown parameter"
                    continue
                spec["value"] = value
                applied[key] = value
                if spec["scope"] == "immediate" and key == "learningRate":
                    self._state["learningRate"] = value
            return {"ok": not rejected, "applied": applied, "rejected": rejected}

    # -- W&B polling (independent of the subprocess; read-only) ------------
    def _wandb_poll_loop(self):
        try:
            import wandb  # noqa: F401
            self._api = wandb.Api()
        except Exception:
            self._api = None
            return
        while True:
            time.sleep(5.0)
            name = getattr(self, "_run_wandb_name", None)
            if not name or self._api is None:
                continue
            try:
                runs = self._api.runs(WANDB_PROJECT, filters={"display_name": name})
                run = next(iter(runs), None)
                with self._lock:
                    if run is not None:
                        self._wandb_run = run
                        self._state["wandbConnected"] = True
                        self._state["wandbRunUrl"] = run.url
                    else:
                        self._state["wandbConnected"] = False
            except Exception:
                with self._lock:
                    self._state["wandbConnected"] = False


# ---------------------------------------------------------------------------
# HTTP + hand-rolled WebSocket server
# ponytail: push-only WS (no ping/pong, no fragmentation, no client message
# parsing beyond the handshake) -- the `websockets` package isn't a project
# dependency and this dashboard only ever needs server->client telemetry
# push. Upgrade to `websockets`/`starlette` if bidirectional messaging or
# frame robustness is ever needed.
# ---------------------------------------------------------------------------

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()


def _ws_encode_text(payload: str) -> bytes:
    data = payload.encode("utf-8")
    length = len(data)
    header = bytearray([0x81])
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header += length.to_bytes(2, "big")
    else:
        header.append(127)
        header += length.to_bytes(8, "big")
    return bytes(header) + data


ADAPTER = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # ponytail: quiet by default; comment out to debug requests

    def _send_json(self, obj, status=200):
        body = json.dumps(_sanitize(obj)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        body = HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/ws":
            return self._handle_ws()
        if path in ("/", "/index.html"):
            return self._serve_html()
        if path == "/state":
            return self._send_json(ADAPTER.snapshot())
        if path == "/telemetry":
            return self._send_json(ADAPTER.telemetry())
        if path == "/checkpoints":
            return self._send_json(ADAPTER.list_checkpoints())
        if path == "/wandb/media":
            return self._send_json(ADAPTER.wandb_media())
        self.send_error(404)

    def do_POST(self):
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send_json(_err("invalid JSON body"), status=400)
        if path == "/action":
            result = ADAPTER.handle_action(body.get("action", ""), body.get("payload", {}))
            return self._send_json(result, status=200 if result.get("ok") else 409)
        if path == "/params":
            result = ADAPTER.handle_params(body)
            return self._send_json(result, status=200 if result.get("ok") else 409)
        self.send_error(404)

    def _handle_ws(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if not key or self.headers.get("Upgrade", "").lower() != "websocket":
            self.send_error(400, "expected a WebSocket upgrade request")
            return
        accept = _ws_accept_key(key)
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        self.wfile.write(response.encode())
        try:
            while True:
                payload = json.dumps(_sanitize(
                    {"type": "state", "data": ADAPTER.snapshot(), "telemetry": ADAPTER.telemetry()}
                ))
                self.wfile.write(_ws_encode_text(payload))
                time.sleep(TICK_S)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    global ADAPTER
    ADAPTER = MockAdapter() if args.mode == "mock" else RealAdapter()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Actor-Critic MPC Lab dashboard ({args.mode} mode) -- http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
