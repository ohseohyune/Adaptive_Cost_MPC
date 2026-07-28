"""Phase 11 acceptance tests for curriculum PPO cost adaptation."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.mpc import (
    PPOCostAdapter,
    PPOCostConfig,
    PPORolloutBuffer,
    build_generalization_observation,
    generalized_advantage_estimate,
)
from control.squeeze import (
    CurriculumScheduler,
    RotatingSideSqueezeConfig,
    apply_box_domain_randomization,
    default_curriculum,
)
from control.squeeze.generalization import NOMINAL_BOX_HALF_SIZE
from box_squeeze.main_generalized_box_squeeze import (
    GeneralizationRunConfig,
    _episode_config,
    run_generalized_box_squeeze,
)
from box_squeeze.main_dynamic_box_squeeze import (
    HAND_CAMERA_COLLISION_BIT,
    DynamicRunConfig,
    _disable_duplicate_end_effector_collisions,
    run_dynamic_side_squeeze,
)


SCENE = ROOT / "model/robotis_ffw/scene_ffw_sg2_fixed_base_box_dynamic_squeeze.xml"


def test_domain_randomization_changes_complete_physics() -> None:
    rng = np.random.default_rng(11)
    stage = default_curriculum()[-1]
    domain = stage.sample(rng, stage_index=2)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    apply_box_domain_randomization(model, domain)

    assert np.isclose(np.prod(domain.aspect_scale), 1.0)
    assert np.allclose(
        domain.half_size,
        NOMINAL_BOX_HALF_SIZE
        * domain.overall_size_scale
        * np.asarray(domain.aspect_scale),
    )
    assert domain.shape_family in {
        "balanced",
        "deep",
        "shallow",
        "wide_grip",
        "narrow_grip",
        "tall",
        "flat",
    }

    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_box")
    assert np.allclose(model.geom_size[geom], domain.half_size)
    assert np.allclose(model.geom_aabb[geom, :3], 0.0)
    assert np.allclose(model.geom_aabb[geom, 3:], domain.half_size)
    bvh_leaf = next(
        index
        for index in range(
            int(model.body_bvhadr[body]),
            int(model.body_bvhadr[body] + model.body_bvhnum[body]),
        )
        if int(model.bvh_nodeid[index]) == geom
    )
    assert np.allclose(model.bvh_aabb[bvh_leaf], model.geom_aabb[geom])
    assert np.isclose(model.body_mass[body], domain.mass)
    assert np.isclose(model.geom_friction[geom, 0], domain.friction)
    for name in ("dynamic_left_pad_box", "dynamic_right_pad_box"):
        pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
        assert np.allclose(model.pair_friction[pair, :2], domain.friction)
    hx, hy, hz = domain.half_size
    expected_inertia = domain.mass / 3.0 * np.array(
        [hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy]
    )
    assert np.allclose(model.body_inertia[body], expected_inertia)
    assert not np.allclose(np.asarray(domain.half_size), NOMINAL_BOX_HALF_SIZE)

    # Full mode selects ordinary robot collision bit 1, but removes duplicate
    # link-7/hand mesh contacts represented by the dedicated broad pads.
    assert (model.geom_conaffinity[geom] & 1) == 0
    model.geom_conaffinity[geom] |= 1
    pad_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("left_squeeze_pad", "right_squeeze_pad")
    }
    _disable_duplicate_end_effector_collisions(model, pad_ids)
    camera_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "left_hand_camera_collision",
            "right_hand_camera_collision",
        )
    }
    assert all(camera_id >= 0 for camera_id in camera_ids)
    assert all(
        (model.geom_contype[camera_id] & HAND_CAMERA_COLLISION_BIT) != 0
        for camera_id in camera_ids
    )
    model.geom_conaffinity[geom] |= HAND_CAMERA_COLLISION_BIT
    assert all(
        (model.geom_contype[camera_id] & model.geom_conaffinity[geom]) != 0
        for camera_id in camera_ids
    )
    link7_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_link7"
    )
    link6_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_link6"
    )
    assert all(
        (model.geom_contype[index] & 1) == 0
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) == link7_body and index not in pad_ids
    )
    assert any(
        (model.geom_contype[index] & 1) != 0
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) == link6_body
    )


def test_full_curriculum_generates_multiple_cuboid_shapes() -> None:
    rng = np.random.default_rng(23)
    full = default_curriculum()[-1]
    samples = [full.sample(rng, stage_index=2) for _ in range(80)]
    families = {sample.shape_family for sample in samples}
    dimensions = np.asarray([sample.half_size for sample in samples])
    assert len(families) >= 5
    assert np.ptp(dimensions[:, 0]) > 0.035
    assert np.ptp(dimensions[:, 1]) > 0.045
    assert np.ptp(dimensions[:, 2]) > 0.035


def test_shape_conditioned_impedance_avoids_force_substitution() -> None:
    rng = np.random.default_rng(29)
    domain = default_curriculum()[0].sample(rng, stage_index=0)
    base = RotatingSideSqueezeConfig()
    flat = _episode_config(
        base,
        replace(domain, shape_family="flat"),
        seed=1,
    )
    balanced = _episode_config(
        base,
        replace(domain, shape_family="balanced"),
        seed=2,
    )
    assert flat.tangential_stiffness >= 1200.0
    assert flat.rotational_damping == 20.0
    assert balanced.tangential_stiffness == base.tangential_stiffness
    assert balanced.rotational_damping == 30.0


def test_hand_camera_proxy_physically_blocks_box() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    box = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dynamic_box_geom"
    )
    camera = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_hand_camera_collision",
    )
    pads = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("left_squeeze_pad", "right_squeeze_pad")
    }
    _disable_duplicate_end_effector_collisions(model, pads)
    model.geom_conaffinity[box] |= HAND_CAMERA_COLLISION_BIT

    fixture = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_launch_fixture"
    )
    data.eq_active[fixture] = 0
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dynamic_box_joint"
    )
    qpos = int(model.jnt_qposadr[joint])
    data.qpos[qpos : qpos + 3] = data.geom_xpos[camera]
    data.qpos[qpos + 3 : qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    assert any(
        {int(contact.geom1), int(contact.geom2)} == {camera, box}
        for contact in data.contact[: data.ncon]
    )


def test_full_collision_mode_preserves_pad_capture() -> None:
    result = run_dynamic_side_squeeze(
        DynamicRunConfig(
            squeeze=RotatingSideSqueezeConfig(random_seed=7),
            collision_mode="full",
        )
    )
    assert result.success, result
    assert result.robot_collision_enabled
    assert result.first_nonpad_contact_time_s is None


def test_curriculum_promotes_and_demotes() -> None:
    scheduler = CurriculumScheduler(
        window=3,
        minimum_episodes_per_stage=3,
        promote_success_rate=2.0 / 3.0,
        demote_success_rate=1.0 / 3.0,
    )
    for outcome in (True, True, True):
        scheduler.record(outcome)
    assert scheduler.stage.name == "intermediate"
    for outcome in (False, False, False):
        scheduler.record(outcome, safety_violation=not outcome)
    assert scheduler.stage.name == "warmup"


def test_gae_matches_bootstrapped_definition() -> None:
    advantages, returns = generalized_advantage_estimate(
        rewards=np.array([1.0, 2.0]),
        values=np.array([0.5, 0.25]),
        dones=np.array([0.0, 1.0]),
        next_value=9.0,
        gamma=0.9,
        gae_lambda=0.8,
    )
    assert np.allclose(advantages, [1.985, 1.75], atol=1e-6)
    assert np.allclose(returns, [2.485, 2.0], atol=1e-6)


def test_online_ppo_update_is_parameter_limited() -> None:
    config = PPOCostConfig(
        device="cpu",
        seed=13,
        minimum_online_rollout=4,
        maximum_online_actor_delta=0.003,
    )
    adapter = PPOCostAdapter(config)
    rollout = PPORolloutBuffer()
    rng = np.random.default_rng(13)
    stage = default_curriculum()[0]
    for index in range(4):
        domain = stage.sample(rng, stage_index=0)
        observation = build_generalization_observation(
            domain, curriculum_stage_count=3
        )
        action = adapter.act(observation, training=True)
        rollout.add(
            observation=observation,
            action=action,
            reward=float(index),
            done=True,
            safety_violation=False,
        )
    update = adapter.update(rollout, online=True)
    assert update.applied
    assert update.actor_parameter_delta > 0.0
    assert update.actor_parameter_delta <= config.maximum_online_actor_delta + 1e-6
    assert np.isfinite(update.approximate_kl)


def test_offline_ppo_uses_multiple_epochs() -> None:
    config = PPOCostConfig(
        device="cpu",
        seed=17,
        training_epochs=3,
        minibatch_size=8,
    )
    adapter = PPOCostAdapter(config)
    rollout = PPORolloutBuffer()
    rng = np.random.default_rng(17)
    stage = default_curriculum()[0]
    for index in range(4):
        domain = stage.sample(rng, stage_index=0)
        observation = build_generalization_observation(
            domain, curriculum_stage_count=3
        )
        action = adapter.act(observation, training=True)
        rollout.add(
            observation=observation,
            action=action,
            reward=1.0 + 0.1 * index,
            done=True,
            safety_violation=False,
        )
    update = adapter.update(rollout, online=False)
    assert update.applied
    assert update.epochs == config.training_epochs


def test_randomized_catch_and_online_adaptation() -> None:
    summary = run_generalized_box_squeeze(
        GeneralizationRunConfig(
            episodes=4,
            rollout_size=4,
            random_seed=7,
            device="cpu",
            online_adaptation=True,
        )
    )
    assert summary.successes == 4, summary
    assert summary.safety_violations == 0
    assert summary.final_stage == "intermediate"
    assert summary.ppo_updates_applied == 1
    assert summary.actor_parameter_delta > 0.0
    for episode in summary.episode_summaries:
        assert 0.48 <= episode.domain.mass <= 0.52
        assert 1.15 <= episode.domain.friction <= 1.25
        assert episode.first_contact_peak_force_n <= 18.0
        assert episode.minimum_no_slip_force_per_pad_n > 0.0
        assert (
            episode.final_hold_force_per_pad_n
            >= episode.minimum_no_slip_force_per_pad_n
        )
        assert episode.final_hold_force_per_pad_n <= 24.0


def main() -> int:
    tests = [
        test_domain_randomization_changes_complete_physics,
        test_full_curriculum_generates_multiple_cuboid_shapes,
        test_shape_conditioned_impedance_avoids_force_substitution,
        test_hand_camera_proxy_physically_blocks_box,
        test_full_collision_mode_preserves_pad_capture,
        test_curriculum_promotes_and_demotes,
        test_gae_matches_bootstrapped_definition,
        test_online_ppo_update_is_parameter_limited,
        test_offline_ppo_uses_multiple_epochs,
        test_randomized_catch_and_online_adaptation,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - standalone acceptance runner
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
