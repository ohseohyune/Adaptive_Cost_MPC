# PPO rollout flush is decoupled from episode boundaries, with truncation bootstrap

**Status**: accepted

**Context**: An episode runs hundreds of control steps (a 5s HOLD alone is ~500 steps). If the PPO Learner only updated once per episode, on-policy data would go stale for hundreds of steps before being used, and each batch would be enormous and low-frequency.

**Decision**: `rollout_size=16` (`main_acmpc_box_catch.py:268`) — the PPO Learner flushes and updates every 16 control steps, independent of episode boundaries (`len(rollout) >= config.rollout_size`, `:2691`). This yields dozens of updates per episode rather than one. GAE bootstraps `next_value = predict_value(observation)` from whatever observation is current at flush time — an explicit truncation bootstrap, not a bug — with `done` masking distinguishing true episode termination from mid-episode truncation.

Whether a flush is allowed to straddle an episode boundary is a separate, independently-tested axis: `accumulate_rollout_across_episodes` (`main_acmpc_box_catch_curriculum.py:100`, `main_acmpc_box_catch_prior_free_curriculum.py:103`), default `False`. With the default, buffers do not carry over between episodes — the only case a flush spans two episodes is an episode ending before 16 steps accumulate (early INTERCEPT failure).

**Considered / tested alternative**: `accumulate_rollout_across_episodes=True` ("diversity" arm) — see `sweep_results/batchdiag_*`.

**Consequences**: `rollout_size`, the truncation-bootstrap `next_value`, and `done`-masking are a matched triad — changing one without the others breaks the bootstrap's correctness.
