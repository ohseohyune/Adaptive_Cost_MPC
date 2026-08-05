# AC-MPC cumulative constraint ablation

## Final comparison

| Condition | Seeds | Success | ImpactSafe | StableHold | Net displacement | Path ratio |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 3/3 | 1.000 | 1.000 | 1.000 | 0.3893 | 164.1 |
| D1 | 3/3 | 0.733 | 0.933 | 0.747 | 4.4108 | 16.6 |
| D2 | 3/3 | 0.013 | 0.693 | 0.013 | 4.4349 | 14.9 |
| D3 | 3/3 | 0.487 | 0.687 | 0.487 | 4.1655 | 15.9 |
| M0 | 3/3 | 0.107 | 0.413 | 0.107 | 4.7396 | 16.8 |

## Incremental decisions

- D0 → D1 (cumulative cap 제거): success -0.267, ImpactSafe -0.067. 탈락 후보(≥5%p 하락).
- D1 → D2 (exp_residual + weight clip 제거): success -0.720, ImpactSafe -0.240. 탈락 후보(≥5%p 하락).
- D2 → D3 (online actor cap 제거): success +0.473, ImpactSafe -0.007. 5%p 비열화 기준 통과.
- D3 → M0 (target KL 제거): success -0.380, ImpactSafe -0.273. 탈락 후보(≥5%p 하락).

## Constraint activity

- D0: cumulative projections=9355.3, online caps=375.7, target-KL stops=0.0, KL projections=12.7, weight clips=0.0.
- D1: cumulative projections=0.0, online caps=395.0, target-KL stops=0.0, KL projections=205.7, weight clips=0.0.
- D2: cumulative projections=0.0, online caps=337.3, target-KL stops=0.0, KL projections=574.3, weight clips=0.0.
- D3: cumulative projections=0.0, online caps=0.0, target-KL stops=0.0, KL projections=661.0, weight clips=0.0.
- M0: cumulative projections=0.0, online caps=0.0, target-KL stops=0.0, KL projections=0.0, weight clips=0.0.

A constraint is called a learning bottleneck only when its removal both increases effective actor movement/residual differentiation and preserves or improves frozen task/safety metrics. Movement alone is not counted as improvement.
