# Adaptive Cost MPC for FFW Bimanual Grasping

This workspace contains MuJoCo controllers for the ROBOTIS FFW dual-arm robot,
including an end-to-end dynamic bimanual grasp with online actor-critic cost
adaptation.

## Stationary box side-squeeze milestone

The first broad-pad squeeze milestone uses a normal handle-free box and keeps
all finger actuators open. Both arms approach fixed opposite faces, regulate
normal pressure with force admittance, then disable the box fixture. Success
requires frictional side contact to support the free box against gravity for
one second.

```bash
python box_squeeze/main_box_squeeze.py
python box_squeeze/main_box_squeeze.py --viewer
python tests/phase8_side_squeeze_test.py
```

Its implementation is split across `control/squeeze/`, `box_squeeze/main_box_squeeze.py`,
and `model/robotis_ffw/scene_ffw_sg2_fixed_base_box_squeeze.xml`.

## Dynamic ballistic box side-squeeze milestone

The second milestone releases the same handle-free box with a seeded random
linear velocity. A gravity-aware predictor estimates the trajectory, a
two-face planner places both broad pads at the predicted catch plane, and TTC
softens Cartesian stiffness before impact. A compliant contact pair plus a
first-contact force monitor keeps the initial pad force below the configured
limit before the controller ramps to the normal squeeze force.

```bash
python box_squeeze/main_dynamic_box_squeeze.py --seed 7
python box_squeeze/main_dynamic_box_squeeze.py --seed 7 --viewer
python tests/phase9_dynamic_box_squeeze_test.py
```

The validated launch envelope is configured in `DynamicSideSqueezeConfig`.
The current high-speed default is approximately 1.35–1.47 m/s toward the robot
(1.5x the original envelope), ±0.01
m/s laterally, with a fixed 0.56 s initial TTC. The launcher x position is
derived from randomized vx (about 1.06–1.12 m), and the box starts at z=0.15
m with vz derived (about 3.91 m/s) so it reaches z=0.80 m near the apex.
Both arms start from their actual home joint positions at the same simulation
step as box release; no ready-pose pre-positioning occurs in the default mode.
After capture stabilizes for 0.04 s, success now requires a further 5.0 s of
continuous safe bilateral hold. The dynamic episode timeout is 7.5 s so this
long hold can complete after the ballistic approach and impact transient. A
bounded tangent-plane anti-slip correction moves both pads together to restore
the captured box center during this extended hold without changing the initial
impact-force target.

During HOLD, the normal force is no longer a single fixed value. Each episode
computes the symmetric gravity-support bound `m|g|/(2 mu)` from the randomized
box mass and friction, then applies a contact-calibrated factor. The nominal
linear box starts at about 10.01 N per pad; rotating/random-shape trials start
at about 11.2 N per pad. Downward slip raises this setpoint temporarily, while
stable contact releases it back to the computed minimum. The soft impact
contact is also ramped over 0.5 s to a firmer hold contact, avoiding both the
impact spike of an instantaneous switch and long-term compliant creep.
Shape-conditioned impedance prevents the slip detector from compensating for
pose-control errors with excessive grip force: `flat` cuboids use 1200 N/m
tangential stiffness with the nominal rotational damping, while the other
randomized cuboids use stronger rotational damping. Normal force still starts
from the mass/friction minimum in both cases.

## Rotating box SE(3) stabilization milestone

The third milestone adds quaternion observations, filtered world angular
velocity prediction, and full SE(3) face targets. A constrained two-contact
wrench QP distributes object force and moment while enforcing a friction
pyramid and bounded soft-finger moments. Its slip and angular-velocity costs
are logged and used by the HOLD acceptance gate.

```bash
python box_squeeze/main_rotating_box_squeeze.py --seed 7
python box_squeeze/main_rotating_box_squeeze.py --seed 7 --viewer
python tests/phase10_rotating_box_squeeze_test.py
```

The high-speed validated angular-velocity envelope is deliberately
conservative: approximately ±0.005 rad/s about box x/y and ±0.025 rad/s about
box z. Combining the original maximum angular envelope with 1.5x translation
remains a later pad/contact-design task.

## Generalized curriculum AC-MPC milestone

The fourth milestone wraps the rotating-box catcher in a physical-domain
curriculum. Every episode samples all three box dimensions and decomposes them
into an overall-size term and x/y/z aspect ratios, as well as sampling mass,
sliding friction, linear launch velocity, and angular velocity. This produces
visibly deep, shallow, wide-grip, narrow-grip,
tall, flat, and near-balanced cuboids instead of merely scaling one fixed
shape. The sampled dimensions update
the MuJoCo collision geom and face sites; mass, box inertia, geom friction,
and both explicit pad/box contact-pair friction values are updated together.

A Gaussian actor observes the sampled domain and selects six bounded
residuals around the engineered Phase-3 costs: squeeze force, tangential
stiffness, angular damping, slip cost, angular-velocity cost, and wrench
tracking cost. The critic predicts the episode return. Training uses clipped
PPO and GAE; the physical controller and constrained wrench QP remain in the
loop for every rollout.

```bash
python box_squeeze/main_generalized_box_squeeze.py --episodes 12 --rollout-size 4 --device auto
python box_squeeze/main_generalized_box_squeeze.py --episodes 450 --rollout-size 12 --offline-training --curriculum-mode balanced --checkpoint checkpoints/generalized_shape_v3.pt
python box_squeeze/main_generalized_box_squeeze.py --episodes 300 --evaluation-suite --checkpoint checkpoints/generalized_shape_v3.pt --load-checkpoint --collision-mode full
python tests/phase11_generalized_acmpc_test.py
```

The default cuboid curriculum has three levels. Overall scale and shape/aspect
variation are represented separately in each sampled domain:

1. `warmup`: axis scales x/z ±10% and grip-width ±3%, mass 0.48–0.52 kg,
   friction 1.15–1.25;
2. `intermediate`: axis scales x/z ±22% and grip-width ±9%, mass
   0.42–0.58 kg, friction 1.00–1.40;
3. `full`: axis scales x/z ±35% and grip-width ±18%, mass 0.35–0.70 kg,
   friction 0.75–1.55.

Geometry type stays `box`: cylinders and spheres require different contact
targets and are intentionally outside this two-opposite-face controller.

`full` is intentionally a stress/training distribution rather than a claimed
100% capture envelope. The scheduler returns to `intermediate` when its
rolling success rate falls below the configured threshold.

Use `--curriculum-mode balanced` for offline training: it samples warmup,
intermediate, and full in round-robin order, so difficult stages cannot vanish
through repeated curriculum demotion. `--evaluation-suite` disables policy
updates, selects the same balanced stage coverage, and reports per-stage
success, safety, and mean-reward summaries. Change `--seed` for held-out test
suites.

It promotes only after sustained safe success and can demote after failures
or force-limit violations. Online updates require a minimum rollout, stop at
a small target KL, run only one PPO epoch, clip gradients, and project the
actor parameter change to a fixed norm. Actor outputs also have hard
per-cost multiplier limits, so online learning cannot remove the engineered
contact constraints. Use `--no-online-adaptation` for deterministic policy
evaluation, `--offline-training` for multi-epoch PPO training, and
`--checkpoint PATH` to save the actor/critic state. A deployment run can load
that policy with `--load-checkpoint` and retain the one-epoch online guard.

The implementation is in `control/squeeze/generalization.py`,
`control/mpc/ppo_cost_adapter.py`, and `box_squeeze/main_generalized_box_squeeze.py`.
CUDA is optional: `--device auto` uses it for the small PyTorch actor/critic
when available, while MuJoCo and the OSQP wrench allocator work on CPU.

### Dynamic collision modes

The dynamic runner supports three box/robot collision policies:

- `miss_backstop` (default): preserve the validated pad interception, then
  enable ordinary robot/floor collisions if the box passes behind the catch
  plane;
- `full`: enable robot/floor collision from launch. The duplicate link-7 and
  hand meshes are represented by the dedicated broad-pad collision boxes, so
  they are excluded from double contact;
- `pad_only`: reproduce the original simplified training environment.

The two hand-camera protrusions use dedicated collision proxies and bit 32.
They stay physical from launch in both `miss_backstop` and `full`, even though
overlapping link-7/hand meshes are suppressed. Only the intentionally
simplified `pad_only` mode allows the box to ignore them.

Use strict physical collision for visualization with
`--collision-mode full`. The broad pad remains the only contact model on each
distal hand assembly; proximal arm, body, and floor collisions stay active.

The default simultaneous mode opens the viewer at time zero: box fixture
release and home-to-catch arm tracking begin in the same simulation step.
`--show-prepare` applies only when `simultaneous_start=False` is selected from
Python for a legacy ready-pose experiment.

Checkpoints trained before the simultaneous/apex-launch change use a different
state distribution and should not be used for evaluation. Train a new file,
for example `checkpoints/generalized_highspeed.pt`.

## End-to-end AC-MPC demo

```bash
python acmpc/main_bimanual_acmpc.py --duration 6 --device auto
```

Add `--viewer` to visualize the run. `--device auto` uses CUDA when PyTorch can
access it and otherwise uses CPU. To verify the fixed safe MPC prior without
online updates:

```bash
python acmpc/main_bimanual_acmpc.py --duration 6 --device cpu --no-online-learning
```

The runtime pipeline is:

1. estimate the free object's position and velocity;
2. build a 24-dimensional contact and interception observation;
3. predict five horizon-wise MPC costs with a neural actor;
4. solve a differentiable Cartesian bimanual MPC;
5. track its bounded velocity command with two impedance controllers;
6. close both physical grippers and measure object-only contact forces;
7. update the actor and TD critic from the MuJoCo reward online;
8. lift/translate the object after a stable bilateral grasp.

The CPU/GPU implementation is in
`control/mpc/online_actor_critic.py`. It uses a small dense PyTorch quadratic
solve rather than requiring the reference project's drone-specific fused CUDA
iLQR kernels. The mathematical layer remains differentiable and the same code
runs on both devices.

## Validation

```bash
python tests/phase7_bimanual_acmpc_test.py
```

The test checks observation construction, gradients through MPC into the cost
actor, device selection, physical bilateral contact, manipulation state entry,
and nonzero online actor updates. Earlier component suites remain available as
`tests/phase1_torque_test.py` through `tests/phase6_neural_cost_test.py`.

Useful runtime options:

```text
--exploration-std 0.015   stochastic policy exploration during online learning
--checkpoint PATH         load an existing checkpoint and save updates on exit
--log PATH                write per-control-step costs, reward, force and TD data
```

## Main files

- `acmpc/main_bimanual_acmpc.py`: finite headless/viewer execution and online update loop
- `control/mpc/online_actor_critic.py`: actor, critic and differentiable MPC
- `model/robotis_ffw/scene_ffw_sg2_fixed_base_bimanual_dynamic.xml`: moving free object
- `tests/phase7_bimanual_acmpc_test.py`: end-to-end acceptance test

The implementation is inspired by the architecture in
[prisma-lab/CA-AC-MPC](https://github.com/prisma-lab/CA-AC-MPC), but is an
independent robot-specific implementation and does not copy its GPL source or
its quadrotor CUDA kernels.
