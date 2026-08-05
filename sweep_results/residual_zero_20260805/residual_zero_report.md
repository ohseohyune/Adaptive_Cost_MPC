# Residual-zero ablation

Scenario pairing verified: identical mass/friction per episode index.

| Condition | Learned success | Zero success | Paired diff | CI95 | Actor 판정 |
|---|---:|---:|---:|---|---|
| D0 | 0.913 | 0.920 | -0.007 | [-0.017, +0.000] | 2: Actor 기여가 거의 없음 |
| D1 | 0.713 | 0.920 | -0.207 | [-0.257, -0.157] | 3: Actor가 성능을 악화 |
| D2 | 0.013 | 0.920 | -0.907 | [-0.940, -0.870] | 3: Actor가 성능을 악화 |
| D3 | 0.503 | 0.920 | -0.417 | [-0.477, -0.357] | 3: Actor가 성능을 악화 |
| M0 | 0.123 | 0.920 | -0.797 | [-0.843, -0.747] | 3: Actor가 성능을 악화 |

| Condition | Actor residual abs mean | Policy command delta (m/s) | Noise/signal | Δ success | Δ peak P95 |
|---|---:|---:|---:|---:|---:|
| D0 | 0.0198 | 0.00205 | 64.6 | -0.007 | +0.02 |
| D1 | 0.5305 | 0.07650 | 1.7 | -0.207 | +0.91 |
| D2 | 17.4941 | 1.33677 | 0.1 | -0.907 | +0.14 |
| D3 | 1.3439 | 0.49422 | 0.3 | -0.417 | -1.08 |
| M0 | 6.3958 | 0.74221 | 0.2 | -0.797 | +7.13 |

Noise/signal is E||u_sample - u_mean|| / ||u_mean - u_zero|| with E||u_sample - u_mean|| ~ 0.1323 m/s (std 0.03 x velocity limit 1.8 over 6 dims). Higher means the exploration noise dominates the learned residual's effect on the command.
