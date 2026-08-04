# HOLD → CAPTURE demotion uses shared dwell constant, not separate hysteresis thresholds

**Status**: accepted

**Context**: MuJoCo contact force can read a single-step zero or spike, causing `phase` to chatter between `HOLD` and `CAPTURE` if demotion reacted to a single bad sample. A more conventional fix is hysteresis — separate promotion/demotion force thresholds — but that introduces a tuning constant with no independent empirical grounding.

**Decision**: Demote `HOLD → CAPTURE` only when `both_pads_at_required_force` is false for `hold_entry_dwell_s` (0.05s) continuously — the same dwell constant already used for promotion (`main_acmpc_box_catch.py:2465-2489`). Rejected the hysteresis alternative for that reason.

**Consequences**: Demotion reaction is delayed up to 0.05s. Because `strict_stable_contact ⇒ both_pads_at_required_force`, the success hold-timer is already at 0 during any demotion window — demotion is structurally incapable of costing a success.

**Verified (2026-08-03)**, fixed baseline postfix_v4, full_wide_speed domain, 60 seeds:
- Demotions among successful episodes: 0/43
- Demotions that recovered to success: 0/9 (re-grasp path exists but never observed to pay off)
- Demotion chatter (>1 demotion in an episode): 0 — always exactly once when it fires

**Known unresolved side effect**: `exploring = online_learning and phase is not CatchPhase.HOLD` (`main_acmpc_box_catch.py:2734`) re-enables exploration noise during a demotion, which is the most unstable moment to inject noise. This is incidental to the demotion decision — the fix belongs on the exploration-noise gate, not here.
