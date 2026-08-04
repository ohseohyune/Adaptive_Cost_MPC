# AC-MPC Box Catch

Bimanual robot이 낙하/발사되는 박스를 MPC 기반 제어와 PPO로 학습된 cost-adaptation actor로 포획(catch)하는 시스템.

**Scope note**: 이 문서는 box-catch 계열만 다룸 (`main_acmpc_box_catch.py`, `OnlineActorCriticACMPC`, `DifferentiableBimanualMPC`, `AdaptiveCostActor`/`PriorFreeCostActor`, `CartesianImpedanceController`, `BallisticBoxPredictor`/`BoxFaceInterceptionPlanner`). `AdaptiveCostMPC`/`CartesianMPC`/`NeuralCostMap`/`PPOCostAdapter`/`box_squeeze/`는 별도 레거시 트랙이며 범위 밖.

## Language

**Cost Weights**:
Cost Predictor 신경망이 직접 출력하는 값. 5개 cost 항목 × horizon 스텝 shape.
_Avoid_: action (아래 Action과 구분할 것)

**Action**:
`DifferentiableBimanualMPC`가 Cost Weights로 QP를 풀어서 낸 Cartesian velocity. PPO가 log-prob 계산하고 최적화하는 실제 대상.
_Avoid_: cost weights, actor output (모두 별개 개념)

**Policy**:
Cost Predictor + MPC solve + log_std를 합친 전체 단위. Action 분포(mean + std)를 정의하며 PPO가 최적화하는 대상.
_Avoid_: Actor (모호함, 신경망만 가리키는지 전체인지 불분명하므로 좁은 의미로만 사용)

**Cost Predictor**:
Cost Weights를 출력하는 신경망 부분 (`AdaptiveCostActor`/`PriorFreeCostActor`). Policy의 서브컴포넌트.
_Avoid_: Actor (단독으로 쓰면 Policy 전체와 혼동)

**MPC**:
Cost Weights를 받아 QP를 풀어 velocity를 내는 미분가능 솔버 (`DifferentiableBimanualMPC`). Policy 안에 내장된 서브모듈.

**Critic**:
Observation 받아서 value를 추정하는 신경망 (`ValueCritic`). GAE 계산에 쓰임.

**PPO Learner**:
Policy + Critic + PPO update 로직을 모두 포함한 최상위 학습 단위 (`OnlineActorCriticACMPC`). `main_acmpc_box_catch.py`는 이 컴포넌트의 `.act()`/`.update()` 두 메서드만 호출.

**Impedance Controller**:
Action(velocity)을 실제 로봇 토크로 변환하는 저수준 실행기 (`CartesianImpedanceController`). PPO action space 밖 — gradient가 velocity에서 끊기고 numpy/MuJoCo 코드로 비학습. 단 게인은 고정값이 아니라 상태 의존 스케줄: `endpoint_error`/`stiffness_softening_distance` 기반 `adaptive_stiffness` (`main_acmpc_box_catch.py:2927-2958`), impact-window `contact_blend` smoothstep 램프로 소프트→풀 게인 보간 (`:2993-3000`). 이 출력(measured EE velocity, contact force)이 observation과 reward의 force 관련 항목을 직접 결정하므로, 비학습 하이퍼파라미터이자 reward 지형의 일부로 취급.
_Avoid_: 이 컴포넌트를 "고정 게인 실행기"로 단순화하지 말 것 — 게인 스케줄 자체가 학습 곡선에 영향을 준다.

**Ballistic Predictor**:
`BallisticBoxPredictor` + `BoxFaceInterceptionPlanner`를 합쳐 부르는 이름. 물리 기반(비학습) 예측기 — object 위치/속도로부터 catch-plane crossing time과 pad target을 계산.
- `remaining_ttc`만 observation에 들어감.
- `confidence`(predictor 쪽)는 observation에 없음 — 의도된 제외. 첫 4샘플(~40ms) 워밍업 카운터일 뿐이고 그 이후 상수 1.0로 고정돼 관측 신호로서 정보량이 없음. 대신 `precontact_confidence_min=0.75`로 `INTERCEPT→PRE_IMPACT` phase 게이트에서 "TTC를 아직 믿지 마라"는 락으로만 쓰임 — 거리 기반 fallback(`precontact_distance`)이 항상 병행되므로 confidence 하나에 게이트가 의존하지 않음.
_Avoid_: `BoxFaceInterceptionPlanner.confidence`와 predictor의 `confidence`를 혼동하지 말 것 — 이름은 같지만 planner 쪽은 unreachable 시 ×0.5 되는 reachability 신호로, 서로 다른 값. box-catch 게이트는 predictor의 confidence만 사용, planner confidence는 현재 무관.

**Episode Loop**:
`run_box_catch` 함수 전체를 가리키는 이름. MuJoCo 물리 스텝 + phase FSM + reward 계산 + PPO Learner 호출 + Impedance Controller 적용을 오케스트레이션하는 최상위 루프. 별도의 `gym.Env` 스타일 reset/step 추상화는 존재하지 않음 — MuJoCo는 물리 시뮬레이터 역할만 하고, 이 루프가 환경 역할까지 겸함.

**Phase**:
`CatchPhase`의 6개 값(INTERCEPT/PRE_IMPACT/CAPTURE/HOLD/SUCCESS/FAILED). Episode Loop의 제어 상태 — 모든 전환 로직이 이 이산값으로 돈다.

**Phase Encoding**:
Policy(정확히는 Cost Predictor)가 실제로 관측하는 phase 신호. 4개 control phase(INTERCEPT/PRE_IMPACT/CAPTURE/HOLD)에 대한 smoothstep 블렌딩 소프트 one-hot (`phase_blend_time_s=0.08s` 동안 이산 점프 없이 보간). 터미널 상태(SUCCESS/FAILED)는 표현되지 않음 — 그 상태에 도달하면 새 action을 계산하지 않으므로 인코딩할 필요가 없음.
_Avoid_: `CatchControlPhase`라는 enum 이름 자체를 도메인 용어로 쓰지 말 것 — Phase Encoding의 인덱스 공간일 뿐, 개념적으로 Phase Encoding과 분리된 것이 아니다.
**미검증 (재검토 필요, 2026-08-03)**: `phase_blend_time_s=0.08s`(=8 control step, `control_dt=0.01s`)는 hard one-hot 스위치 대비 불연속 스파이크를 줄인다는 직관으로 정해진 값이며, 실측 비교(hard switch vs 0.08s vs 다른 값) 없음. 또한 `PRE_IMPACT→CAPTURE`처럼 빠른 반응이 중요한 안전-critical 전환에도 동일하게 적용돼 반응 지연을 유발할 수 있음. Ablation으로 근거가 생기면 ADR로 승격할 것.

## MPC Cost 항목

Actor(Cost Predictor)가 5개 항목 각각의 weight를 per-horizon-step으로 조정한다 (`COST_NAMES`, `online_actor_critic.py:50`).

**Object**:
EE 중점을 ballistic-aware `center_ref`(object 위치 + 속도·lead_time + ½·중력·lead_time²)로 트래킹.

**Grasp**:
좌우 EE의 relative separation을 uncompressed `relative_ref`(목표 파지 간격)로 트래킹.

**Compression**:
relative separation을 압축된 reference(`relative_ref`를 `grasp_compression`만큼 줄인 값)로 트래킹. 힘을 직접 측정/추정하는 항목이 아니다.
_Avoid_: `force` — 코드의 dict/`COST_NAMES` 키가 여전히 `"force"`로 남아있는 레거시 이름(리네이밍 예정, 아래 참고). wandb/CSV 로깅은 이미 `"compression"`으로 relabel됨 (`_LOG_COST_LABELS`).

**Velocity**:
좌우 EE velocity를 `object_velocities` feedforward로 트래킹.

**Smoothness**:
Horizon 내 step-to-step velocity 변화(k=0은 `previous_velocity` 기준) 페널티.

**TODO (코드)**: `COST_NAMES`/cost dict의 `"force"` 키를 `"compression"`으로 리네이밍 예정 — 모든 참조 지점(`online_actor_critic.py`, `main_acmpc_box_catch.py`의 `_LOG_COST_LABELS` 포함) 동기화 필요.

**Phase Prior**:
현재 phase에 대응하는 cost weight 기준행 (`_BOX_CATCH_PHASE_PRIORS`). Phase Encoding과 동일한 `blend_beta`로 보간되므로, Policy가 관측하는 phase와 그에 짝지어진 cost weight 기준값이 항상 동기화되어 어긋나지 않는다.
