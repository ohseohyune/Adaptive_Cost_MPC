# Actor-Critic MPC Lab — dashboard API

Backend: `dashboard/server.py` (stdlib `http.server`, no framework). Frontend:
`dashboard/index.html` (single file, vanilla JS). WebSocket is tried first;
if it fails to connect or drops, the frontend falls back to polling
`/state` + `/telemetry` every 700ms and keeps retrying the WS every 3s.

```
python dashboard/server.py                    # mock mode, http://localhost:8770
python dashboard/server.py --mode real --port 8770
```

## GET /state

Full snapshot, also the shape pushed over WS as `data`.

```json
{
  "backendConnected": true,
  "wandbConnected": true,
  "wandbRunUrl": "https://wandb.ai/<entity>/adaptive-cost-mpc-box-catch/runs/abcd1234",
  "trainingState": "RUNNING",
  "stopPhase": null,
  "episode": 1250,
  "totalEpisodes": 5000,
  "globalStep": 450000,
  "progress": 0.25,
  "successRate": 0.82,
  "successRateStatus": "GOOD",
  "currentReward": 14.2,
  "bestReward": 18.7,
  "actorLoss": 0.031,
  "criticLoss": 0.118,
  "learningRate": 0.0003,
  "mpcCost": 2.44,
  "simulationFps": 58.4,
  "stage": "full",
  "checkpoint": {
    "lastSaved": "2026-07-25T19:10:03+00:00",
    "filename": "actor_critic_mpc_ep_01250_step_00450000.pt",
    "path": "dashboard/runs/20260725_190500/checkpoints/actor_critic_mpc_ep_01250_step_00450000.pt",
    "episode": 1250,
    "globalStep": 450000,
    "isBest": true,
    "autoSaveEnabled": true,
    "lastResult": "SUCCESS"
  },
  "parameters": {
    "learningRate": { "value": 0.0003, "scope": "immediate", "group": "training" },
    "totalEpisodes": { "value": 5000, "scope": "next_run", "group": "training" },
    "...": "see server.py:default_parameters() for the full set"
  }
}
```

`trainingState`: `IDLE | INITIALIZING | RUNNING | PAUSED | EVALUATING | SAVING | STOPPING | COMPLETED | ERROR`
`stopPhase` (only non-null while `STOPPING`): `STOP_REQUESTED | FINISH_UPDATE | SAVING | SYNCING_WANDB | STOPPED`
`successRateStatus`: `GOOD | IMPROVING | BELOW_TARGET | UNSTABLE`

Numbers are never `Infinity`/`NaN` — the server sanitizes those to `null`
before every response (`server.py:_sanitize`), since `-Infinity`/`NaN` are
not valid JSON and `JSON.parse` throws on them in every browser.

## GET /telemetry

Chart history. Each named series is a list of `{episode, step, value}`
points (newest last, capped at 4000). `mpcMonitor` is a single short window
for the MPC/Robot Monitor panel.

```json
{
  "reward": [{ "episode": 1, "step": 220, "value": 6.1 }, "..."],
  "successRate": ["..."],
  "actorLoss": ["..."], "criticLoss": ["..."], "learningRate": ["..."],
  "mpcCost": ["..."], "episodeLength": ["..."], "entropy": ["..."],
  "qValue": ["..."], "actionMagnitude": ["..."],
  "mpcMonitor": {
    "t": [0, 1, "...", 49],
    "reference": [0.0, 0.16, "..."],
    "actual": [0.02, 0.19, "..."],
    "predictedFrom": 30,
    "predicted": ["low-opacity horizon points, aligned to t[predictedFrom:]"],
    "controlError": 0.041,
    "mpcCost": 2.44
  }
}
```

## GET /checkpoints

List of saved checkpoint records (same shape as `state.checkpoint`),
newest episode first. Used to populate the LOAD CHECKPOINT picker.

## GET /wandb/media

`[{ "name": "trajectory_ep1250.png", "url": "https://...", "updatedAt": "..." }]`
— image files logged to the active W&B run via `wandb.Image(...)`. Empty
list (not an error) when W&B is unreachable or nothing has been logged as
an image yet — true today, since no caller in this repo logs images (see
`control/mpc/wandb_logger.py`). Frontend renders these URLs directly; the
dashboard never proxies the bytes.

## POST /action

```json
{ "action": "start", "payload": { "totalEpisodes": 5000 } }
```

`action` ∈ `start | pause | resume | evaluate | save_checkpoint |
load_checkpoint | stop | set_autosave`. Response: `{"ok": true, ...}` (HTTP
200) or `{"ok": false, "error": "..."}` (HTTP 409) — the caller decides
whether to show a retry affordance; the frontend always does.

| action | valid from | payload |
|---|---|---|
| `start` | IDLE, COMPLETED, ERROR | `{totalEpisodes?}` |
| `pause` | RUNNING | — |
| `resume` | PAUSED | — |
| `evaluate` | RUNNING, PAUSED | — |
| `save_checkpoint` | any except STOPPING | — |
| `load_checkpoint` | IDLE, PAUSED, COMPLETED | `{filename}` |
| `stop` | any except IDLE, STOPPING | — (drives the STOPPING sequence) |
| `set_autosave` | any | `{enabled: bool}` |

## POST /params

Body is `{paramKey: value, ...}` for any keys in `state.parameters`.
Response: `{"ok": bool, "applied": {...}, "rejected": {"key": "reason"}}`.

- `scope: "immediate"` params (learningRate, tau, MPC optimIterations/
  temperature/costWeight_*) take effect on the next control step.
- `scope: "next_run"` params (totalEpisodes, batchSize, gamma,
  replayBufferSize, actorUpdateFreq, seed, predictionHorizon,
  controlHorizon, numSamples, controlFrequency) are always accepted and
  staged, but only change behavior starting the next `start` — the
  frontend labels these `NEXT RUN` and never disables their inputs while
  training is active.
- All params are rejected while `trainingState == EVALUATING` (no config
  drift mid-rollout).

## WS /ws

Server pushes (no client→server messages needed or parsed):

```json
{ "type": "state", "data": { "...": "same shape as GET /state" }, "telemetry": { "...": "same shape as GET /telemetry" } }
```

Push-only, text frames, ~2/sec. `ponytail:` no ping/pong or fragmentation
handling — a closed socket is detected by the next failed `write()`, not by
reading close frames. Fine for one-way telemetry; swap in the `websockets`
package if the dashboard ever needs the server to receive messages.

## Mock vs real adapter

`--mode mock` (default): `MockAdapter` self-drives a plausible training
curve with no side effects beyond placeholder files under
`dashboard/runs/mock/`. Safe to leave running.

`--mode real`: `RealAdapter` launches
`acmpc/main_acmpc_box_catch_curriculum.py` as a subprocess per run (under
`dashboard/runs/<timestamp>/`), parses its `episodes N-M: success_rate=...`
progress line for live metrics, uses `SIGSTOP`/`SIGCONT` for pause/resume
and `SIGINT` (escalating to `SIGTERM`/`SIGKILL`) for stop, and polls
`wandb.Api()` for the matching run in the `adaptive-cost-mpc-box-catch`
project. Known gaps, called out in `server.py`'s `RealAdapter` docstring:
`globalStep` is estimated (episode × an assumed steps-per-episode — the
trainer doesn't print a real one), `evaluate` isn't wired to a rollout yet,
and `mpcMonitor` telemetry needs an in-process hook the trainer doesn't
expose today.

The dashboard itself has no simulation viewport (a JPEG-over-HTTP one was
tried and dropped -- too low quality/frequency to actually watch a catch).
`RealAdapter` still passes `--live-state-path <run_dir>/live_state.json` to
the trainer subprocess; run `python3 dashboard/watch_live.py` in a terminal
to tail it in a native MuJoCo viewer window instead.
