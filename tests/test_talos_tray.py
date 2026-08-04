import re

import mujoco
import numpy as np
import pytest
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.terminations import nan_detection
from pal_mjlab.robots.pal_talos.talos_constants import (
  GRASPING_INIT_STATE,
  INIT_STATE,
  TALOS_FREE_TRAY_INIT_POS,
  TALOS_FT_SITE_BODIES,
  TALOS_FT_SITE_NAMES,
  TALOS_GRIPPER_CONTACT_BODY_NAMES,
  TALOS_GRIPPER_MAIN_JOINT_NAMES,
  TALOS_OVERGRIP_FRICTION,
  TALOS_TORQUE_SENSOR_JOINT_NAMES,
  TALOS_TRAY_BODY_NAME,
  TALOS_TRAY_HALF_SIZE,
  TALOS_TRAY_HANDLE_POSITIONS,
  TALOS_TRAY_MASS,
  TALOS_TRAY_PARENT_BODY_NAME,
  TALOS_TRAY_PAYLOAD_BODY_NAME,
  TALOS_TRAY_PAYLOAD_HALF_SIZE,
  TALOS_TRAY_PAYLOAD_INIT_POS,
  TALOS_TRAY_PAYLOAD_MASS,
  TALOS_TRAY_RIGHT_MOUNT_SITE_NAME,
  TALOS_TRAY_RIGHT_WRIST_SITE_NAME,
  TALOS_TRAY_SECONDARY_BODY_NAME,
  TALOS_WRIST_FT_SITE_NAMES,
  get_free_hand_tray_spec,
  get_free_tray_payload_spec,
  get_grasping_spec,
  get_spec,
  get_talos_free_hand_tray_cfg,
  get_talos_free_tray_payload_cfg,
  get_talos_grasping_robot_cfg,
  get_talos_tray_robot_cfg,
  get_tray_spec,
)
from pal_mjlab.tasks.velocity.talos.env_cfgs import (
  TALOS_FORCE_SENSOR_NAMES,
  TALOS_MASS_ESTIMATOR_HISTORY_LENGTH,
  TALOS_TORQUE_SENSOR_NAMES,
  TALOS_TRAY_PAYLOAD_ALPHA_RANGE,
  TALOS_TRAY_PAYLOAD_MASS_RANGE,
  TALOS_UNIFORM_MASS_DISTRIBUTION,
  pal_talos_estimated_mass_tray_flat_env_cfg,
  pal_talos_estimated_mass_zero_joint_torque_tray_flat_env_cfg,
  pal_talos_flat_env_cfg,
  pal_talos_free_payload_tray_flat_env_cfg,
  pal_talos_grasping_tray_flat_env_cfg,
  pal_talos_oracle_mass_tray_flat_env_cfg,
  pal_talos_payload_flat_env_cfg,
  pal_talos_random_mass_tray_flat_env_cfg,
  pal_talos_rough_env_cfg,
  pal_talos_tray_flat_env_cfg,
)
from pal_mjlab.tasks.velocity.talos.rl_cfg import (
  pal_talos_estimated_mass_tray_ppo_runner_cfg,
  pal_talos_estimated_mass_zero_joint_torque_tray_ppo_runner_cfg,
  pal_talos_free_payload_tray_ppo_runner_cfg,
  pal_talos_grasping_random_mass_tray_ppo_runner_cfg,
  pal_talos_grasping_tray_ppo_runner_cfg,
  pal_talos_oracle_mass_tray_ppo_runner_cfg,
  pal_talos_random_mass_tray_ppo_runner_cfg,
  pal_talos_tray_ppo_runner_cfg,
)


def _set_initial_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  data.qpos[:7] = (*INIT_STATE.pos, 1.0, 0.0, 0.0, 0.0)
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name is None:
      continue
    matches = [
      value
      for pattern, value in INIT_STATE.joint_pos.items()
      if re.fullmatch(pattern, joint_name)
    ]
    if matches:
      data.qpos[model.jnt_qposadr[joint_id]] = matches[-1]
  mujoco.mj_forward(model, data)


def test_tray_is_a_fixed_child_of_left_wrist() -> None:
  model = get_tray_spec().compile()
  tray_id = model.body(TALOS_TRAY_BODY_NAME).id
  left_wrist_id = model.body(TALOS_TRAY_PARENT_BODY_NAME).id

  assert model.body_parentid[tray_id] == left_wrist_id
  assert model.body_jntnum[tray_id] == 0
  assert model.body_mass[tray_id] == pytest.approx(TALOS_TRAY_MASS)

  base_geom = model.geom("hand_tray_base_collision")
  assert base_geom.type == mujoco.mjtGeom.mjGEOM_BOX
  assert tuple(model.geom_size[base_geom.id]) == TALOS_TRAY_HALF_SIZE

  for side, handle_pos in TALOS_TRAY_HANDLE_POSITIONS.items():
    handle = model.geom(f"hand_tray_{side}_handle_collision")
    assert handle.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    np.testing.assert_allclose(handle.pos, handle_pos, atol=1e-12)
    np.testing.assert_allclose(handle.quat, (1.0, 0.0, 0.0, 0.0), atol=1e-12)


def test_talos_has_four_force_torque_measurement_sites() -> None:
  model = get_spec().compile()
  for site_name, body_name in TALOS_FT_SITE_BODIES.items():
    site_id = model.site(site_name).id
    assert model.site_bodyid[site_id] == model.body(body_name).id


@pytest.mark.parametrize(
  ("body_name", "collision_name", "source_mesh_name"),
  (
    ("head_1_link", "head_1_collision", "head_1"),
    ("head_2_link", "head_2_collision", "head_2_default"),
    ("head_2_link", "orbbec_collision", "orbbec"),
  ),
)
def test_head_collisions_reuse_the_original_visual_meshes(
  body_name: str,
  collision_name: str,
  source_mesh_name: str,
) -> None:
  model = get_spec().compile()
  geom = model.geom(collision_name)

  assert geom.type == mujoco.mjtGeom.mjGEOM_MESH
  assert model.geom_bodyid[geom.id] == model.body(body_name).id
  assert model.geom_dataid[geom.id] == model.mesh(source_mesh_name).id
  assert model.geom_contype[geom.id] != 0
  assert model.geom_conaffinity[geom.id] != 0


def test_right_wrist_weld_is_aligned_at_initial_state() -> None:
  model = get_tray_spec().compile()
  data = mujoco.MjData(model)
  _set_initial_state(model, data)

  assert model.neq == 1
  assert model.eq_type[0] == mujoco.mjtEq.mjEQ_WELD
  assert model.eq_objtype[0] == mujoco.mjtObj.mjOBJ_SITE

  tray_mount_id = model.site(TALOS_TRAY_RIGHT_MOUNT_SITE_NAME).id
  wrist_mount_id = model.site(TALOS_TRAY_RIGHT_WRIST_SITE_NAME).id
  assert model.eq_obj1id[0] == tray_mount_id
  assert model.eq_obj2id[0] == wrist_mount_id
  np.testing.assert_allclose(
    data.site_xpos[tray_mount_id], data.site_xpos[wrist_mount_id], atol=1e-12
  )
  np.testing.assert_allclose(
    data.site_xmat[tray_mount_id], data.site_xmat[wrist_mount_id], atol=1e-12
  )


def test_tray_robot_cfg_uses_the_tray_spec() -> None:
  cfg = get_talos_tray_robot_cfg()
  assert cfg.spec_fn is get_tray_spec
  assert cfg.init_state is INIT_STATE
  model = cfg.spec_fn().compile()
  assert model.body(TALOS_TRAY_SECONDARY_BODY_NAME).id >= 0


def test_tray_task_has_separate_robot_and_experiment_configs() -> None:
  cfg = pal_talos_tray_flat_env_cfg()
  assert cfg.scene.entities["robot"].spec_fn is get_tray_spec
  assert pal_talos_tray_ppo_runner_cfg().experiment_name == "talos_tray_velocity"


def test_tray_payload_is_a_standalone_free_body() -> None:
  model = get_free_tray_payload_spec().compile()
  payload_id = model.body(TALOS_TRAY_PAYLOAD_BODY_NAME).id

  assert model.body_parentid[payload_id] == 0
  assert model.body_jntnum[payload_id] == 1
  joint_id = model.body_jntadr[payload_id]
  assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
  assert model.body_mass[payload_id] == pytest.approx(TALOS_TRAY_PAYLOAD_MASS)
  geom = model.geom("tray_payload_collision")
  assert tuple(model.geom_size[geom.id]) == TALOS_TRAY_PAYLOAD_HALF_SIZE

  cfg = get_talos_free_tray_payload_cfg()
  assert cfg.spec_fn is get_free_tray_payload_spec
  assert cfg.init_state.pos == TALOS_TRAY_PAYLOAD_INIT_POS


def test_grasping_model_restores_real_single_motor_coupling_per_hand() -> None:
  entity = Entity(get_talos_grasping_robot_cfg())
  model = entity.spec.compile()

  gripper_joint_names = tuple(
    name for name in entity.joint_names if name.startswith("gripper_")
  )
  assert len(gripper_joint_names) == 14
  assert model.neq == 12
  assert model.nu == 32
  assert tuple(model.actuator(i).name for i in range(model.nu - 2, model.nu)) == (
    *TALOS_GRIPPER_MAIN_JOINT_NAMES,
  )

  coupled_fingertips = set()
  for eq_id in range(model.neq):
    joint1 = model.joint(model.eq_obj1id[eq_id]).name
    joint2 = model.joint(model.eq_obj2id[eq_id]).name
    if "fingertip" in joint1:
      coupled_fingertips.add(joint1)
      assert joint2 in TALOS_GRIPPER_MAIN_JOINT_NAMES
      assert model.eq_data[eq_id, 1] == pytest.approx(-1.0)
  assert len(coupled_fingertips) == 6

  for side in ("left", "right"):
    for finger in (1, 2, 3):
      geom = model.geom(f"{side}_fingertip_{finger}_grasp_collision")
      assert geom.type == mujoco.mjtGeom.mjGEOM_MESH
      assert model.geom_condim[geom.id] == 4
      np.testing.assert_allclose(model.geom_friction[geom.id], TALOS_OVERGRIP_FRICTION)

  assert GRASPING_INIT_STATE.joint_pos["gripper_(left|right)_joint"] == -0.24
  assert GRASPING_INIT_STATE.joint_pos["gripper_.*_fingertip_.*_joint"] == 0.24


def test_contact_grasp_tray_is_free_and_has_no_weld() -> None:
  model = get_free_hand_tray_spec().compile()
  tray_id = model.body(TALOS_TRAY_BODY_NAME).id
  joint_id = model.body_jntadr[tray_id]
  assert model.body_parentid[tray_id] == 0
  assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
  assert model.neq == 0

  cfg = get_talos_free_hand_tray_cfg()
  assert cfg.spec_fn is get_free_hand_tray_spec
  assert cfg.init_state.pos == TALOS_FREE_TRAY_INIT_POS

  grasping_model = get_grasping_spec().compile()
  assert (
    mujoco.mj_name2id(grasping_model, mujoco.mjtObj.mjOBJ_BODY, TALOS_TRAY_BODY_NAME)
    == -1
  )


def test_contact_grasp_task_uses_deployable_actor_and_staged_commands() -> None:
  cfg = pal_talos_grasping_tray_flat_env_cfg()
  assert set(cfg.scene.entities) == {"robot", "payload", "tray"}
  assert cfg.scene.entities["robot"].spec_fn is get_grasping_spec
  assert cfg.scene.entities["tray"].spec_fn is get_free_hand_tray_spec
  assert "reset_gripper_joints" in cfg.events
  assert "payload_inertia" not in cfg.events

  actor_terms = cfg.observations["actor"].terms
  assert "tray_handle_offsets_w" not in actor_terms
  assert "tray_handle_relative_velocity_w" not in actor_terms
  assert "tray_projected_gravity" not in actor_terms
  assert (
    "gripper_(left|right)_joint"
    in actor_terms["joint_pos"].params["asset_cfg"].joint_names
  )

  critic_terms = cfg.observations["critic"].terms
  assert "tray_handle_offsets_w" in critic_terms
  assert "tray_handle_relative_velocity_w" in critic_terms
  assert "tray_projected_gravity" in critic_terms

  contact_sensor = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "gripper_tray_contact"
  )
  assert contact_sensor.primary.pattern == TALOS_GRIPPER_CONTACT_BODY_NAMES
  assert contact_sensor.secondary.entity == "tray"
  assert cfg.rewards["bilateral_gripper_contact"].weight > 0
  assert cfg.rewards["bilateral_gripper_contact"].params["contacts_per_hand"] == 2
  assert cfg.rewards["tray_grasp_pose"].weight > 0
  assert cfg.rewards["tray_grasp_slip"].weight > 0
  assert "tray_grasp_lost" in cfg.terminations
  assert cfg.terminations["tray_grasp_lost"].params["grace_period_s"] >= 0.48
  assert cfg.terminations["nan_state"].func is nan_detection
  for group_name in ("actor", "critic"):
    assert cfg.observations[group_name].nan_policy == "sanitize"
    assert not cfg.observations[group_name].nan_check_per_term

  stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert stages[0]["lin_vel_x"] == (0.0, 0.0)
  assert stages[0]["lin_vel_y"] == (0.0, 0.0)
  assert stages[0]["ang_vel_z"] == (0.0, 0.0)
  assert stages[1]["lin_vel_x"][1] > 0.0
  assert stages[1]["lin_vel_y"][1] > 0.0
  assert stages[-1]["lin_vel_x"][1] > stages[1]["lin_vel_x"][1]
  assert stages[-1]["lin_vel_y"][1] > stages[1]["lin_vel_y"][1]

  randomized = pal_talos_grasping_tray_flat_env_cfg(randomize_payload_mass=True)
  assert "payload_inertia" in randomized.events
  assert (
    pal_talos_grasping_tray_ppo_runner_cfg().experiment_name
    == "talos_contact_grasp_tray_fixed_2p5kg"
  )
  assert (
    pal_talos_grasping_random_mass_tray_ppo_runner_cfg().experiment_name
    == "talos_contact_grasp_tray_random_mass"
  )


def test_free_payload_tray_task_adds_state_and_balance_objectives() -> None:
  cfg = pal_talos_free_payload_tray_flat_env_cfg()
  assert set(cfg.scene.entities) == {"robot", "payload"}
  assert cfg.scene.entities["robot"].spec_fn is get_tray_spec
  assert cfg.scene.entities["payload"].spec_fn is get_free_tray_payload_spec

  actor_terms = cfg.observations["actor"].terms
  assert "payload_pos_t" not in actor_terms
  assert "payload_relative_velocity_t" not in actor_terms
  assert "payload_mass" not in actor_terms

  critic_terms = cfg.observations["critic"].terms
  assert critic_terms["payload_pos_t"] is not None
  assert critic_terms["payload_relative_velocity_t"] is not None
  assert critic_terms["payload_mass"] is not None

  assert cfg.events["reset_payload_on_tray"].mode == "reset"
  assert "payload_inertia" not in cfg.events
  assert cfg.rewards["tray_level"].weight > 0
  assert cfg.rewards["payload_position_on_tray"].weight > 0
  assert cfg.rewards["payload_relative_motion"].weight > 0
  assert cfg.rewards["wrist_load_balance"].weight > 0
  assert cfg.rewards["tray_tipping_moment"].weight > 0
  assert cfg.rewards["wrist_force_rate"].weight < 0
  assert cfg.rewards["track_linear_velocity"].weight == pytest.approx(4.0)
  assert cfg.rewards["planar_velocity_tracking_error"].weight < 0
  assert cfg.rewards["torso_height"].weight < 0
  for reward_name in (
    "tray_level",
    "payload_position_on_tray",
    "payload_relative_motion",
    "wrist_load_balance",
    "tray_tipping_moment",
  ):
    reward = cfg.rewards[reward_name]
    assert reward.params["tracking_command_name"] == "twist"
    assert reward.params["tracking_std"] == pytest.approx(0.5)
    assert reward.params["tracking_min_factor"] == pytest.approx(0.1)
  assert cfg.terminations["payload_dropped"] is not None
  assert (
    pal_talos_free_payload_tray_ppo_runner_cfg().experiment_name
    == "talos_free_payload_tray_velocity"
  )


def test_actor_contract_contains_only_hardware_available_payload_feedback() -> None:
  task_cfgs = (
    pal_talos_rough_env_cfg(),
    pal_talos_flat_env_cfg(),
    pal_talos_payload_flat_env_cfg(),
    pal_talos_tray_flat_env_cfg(),
    pal_talos_free_payload_tray_flat_env_cfg(),
  )
  actor_term_names = tuple(task_cfgs[0].observations["actor"].terms)
  assert all(
    tuple(cfg.observations["actor"].terms) == actor_term_names for cfg in task_cfgs
  )

  actor_terms = task_cfgs[-1].observations["actor"].terms
  assert actor_terms["imu_lin_acc"] is not None
  torque_term = actor_terms["joint_torque_sensors"]
  assert torque_term is not None
  assert torque_term.params["asset_cfg"].joint_names == TALOS_TORQUE_SENSOR_JOINT_NAMES
  assert torque_term.delay_max_lag == 1
  assert torque_term.clip == (-450.0, 450.0)

  for site_name in TALOS_FT_SITE_NAMES:
    for sensor_type in ("force", "torque"):
      term = actor_terms[f"{site_name}_{sensor_type}"]
      assert term is not None
      assert term.delay_max_lag == 1
      assert term.noise is not None

  play_actor_terms = (
    pal_talos_free_payload_tray_flat_env_cfg(play=True).observations["actor"].terms
  )
  assert play_actor_terms["joint_torque_sensors"].delay_max_lag == 0


def test_force_torque_sensors_cover_both_wrists_and_ankles() -> None:
  cfg = pal_talos_flat_env_cfg()
  sensor_cfgs = {sensor.prefixed_name: sensor for sensor in cfg.scene.sensors}
  assert len(TALOS_FORCE_SENSOR_NAMES) == len(TALOS_TORQUE_SENSOR_NAMES) == 4
  assert TALOS_WRIST_FT_SITE_NAMES == TALOS_FT_SITE_NAMES[:2]

  for sensor_name in TALOS_FORCE_SENSOR_NAMES:
    assert sensor_cfgs[sensor_name].sensor_type == "force"
  for sensor_name in TALOS_TORQUE_SENSOR_NAMES:
    assert sensor_cfgs[sensor_name].sensor_type == "torque"


def test_random_mass_baseline_only_privileges_the_critic() -> None:
  cfg = pal_talos_random_mass_tray_flat_env_cfg()
  inertia = cfg.events["payload_inertia"]
  assert inertia.mode == "startup"
  assert inertia.params["alpha_range"] == TALOS_TRAY_PAYLOAD_ALPHA_RANGE
  assert inertia.params["distribution"] is TALOS_UNIFORM_MASS_DISTRIBUTION
  assert "payload_mass" in cfg.observations["critic"].terms
  assert "payload_mass_oracle" not in cfg.observations["actor"].terms
  assert "mass_estimator" not in cfg.observations
  assert TALOS_TRAY_PAYLOAD_MASS_RANGE == (2.5, 30.0)
  assert (
    pal_talos_random_mass_tray_ppo_runner_cfg().experiment_name
    == "talos_random_mass_tray_critic_velocity"
  )


def test_randomized_payload_mass_is_uniform_in_kilograms() -> None:
  lower = torch.tensor(TALOS_TRAY_PAYLOAD_ALPHA_RANGE[0])
  upper = torch.tensor(TALOS_TRAY_PAYLOAD_ALPHA_RANGE[1])
  alpha = TALOS_UNIFORM_MASS_DISTRIBUTION.sample(lower, upper, (100_000,), "cpu")
  mass = TALOS_TRAY_PAYLOAD_MASS_RANGE[0] * torch.exp(2.0 * alpha)

  assert mass.min() >= TALOS_TRAY_PAYLOAD_MASS_RANGE[0]
  assert mass.max() <= TALOS_TRAY_PAYLOAD_MASS_RANGE[1]
  expected_mean = sum(TALOS_TRAY_PAYLOAD_MASS_RANGE) / 2.0
  assert mass.mean().item() == pytest.approx(expected_mean, abs=0.1)


def test_estimated_mass_task_uses_only_hardware_history_and_a_training_target() -> None:
  cfg = pal_talos_estimated_mass_tray_flat_env_cfg()
  estimator_group = cfg.observations["mass_estimator"]
  assert estimator_group.history_length == TALOS_MASS_ESTIMATOR_HISTORY_LENGTH
  assert estimator_group.flatten_history_dim
  assert "payload_mass" not in estimator_group.terms
  assert "payload_pos_t" not in estimator_group.terms
  assert "payload_relative_velocity_t" not in estimator_group.terms
  assert "joint_torque_sensors" in estimator_group.terms
  assert "left_wrist_ft_force" in estimator_group.terms
  assert "right_wrist_ft_force" in estimator_group.terms

  target_group = cfg.observations["payload_mass_target"]
  assert tuple(target_group.terms) == ("payload_mass",)
  assert not target_group.enable_corruption
  assert "payload_mass_target" not in cfg.observations["actor"].terms

  runner_cfg = pal_talos_estimated_mass_tray_ppo_runner_cfg()
  assert "PayloadMassEstimatorModel" in runner_cfg.actor.class_name
  assert "PayloadMassEstimatorPPO" in runner_cfg.algorithm.class_name
  assert runner_cfg.obs_groups["actor"] == ("actor",)
  assert runner_cfg.obs_groups["mass_estimator"] == ("mass_estimator",)


def test_estimator_zero_torque_ablation_preserves_observation_contract() -> None:
  baseline = pal_talos_estimated_mass_tray_flat_env_cfg()
  ablation = pal_talos_estimated_mass_zero_joint_torque_tray_flat_env_cfg()

  for group_name in ("actor", "mass_estimator"):
    baseline_group = baseline.observations[group_name]
    ablation_group = ablation.observations[group_name]
    assert tuple(ablation_group.terms) == tuple(baseline_group.terms)

    baseline_torque = baseline_group.terms["joint_torque_sensors"]
    ablation_torque = ablation_group.terms["joint_torque_sensors"]
    assert ablation_torque.params == baseline_torque.params
    assert ablation_torque.scale == baseline_torque.scale
    assert ablation_torque.clip == baseline_torque.clip
    assert ablation_torque.noise is None
    assert ablation_torque.func.__name__ == "zero_joint_torque_sensor"

  critic_torque = ablation.observations["critic"].terms["joint_torque_sensors"]
  assert critic_torque.func.__name__ == "joint_torque_sensor"
  assert (
    pal_talos_estimated_mass_zero_joint_torque_tray_ppo_runner_cfg().experiment_name
    == "talos_random_mass_tray_estimator_zero_joint_torque_h8"
  )


def test_oracle_task_exposes_only_scaled_true_mass_to_actor() -> None:
  cfg = pal_talos_oracle_mass_tray_flat_env_cfg()
  oracle = cfg.observations["actor"].terms["payload_mass_oracle"]
  assert oracle.clip == TALOS_TRAY_PAYLOAD_MASS_RANGE
  assert oracle.scale == pytest.approx(1.0 / TALOS_TRAY_PAYLOAD_MASS_RANGE[1])
  assert "payload_pos_t" not in cfg.observations["actor"].terms
  assert "payload_relative_velocity_t" not in cfg.observations["actor"].terms
  assert (
    pal_talos_oracle_mass_tray_ppo_runner_cfg().experiment_name
    == "talos_random_mass_tray_oracle_velocity"
  )
