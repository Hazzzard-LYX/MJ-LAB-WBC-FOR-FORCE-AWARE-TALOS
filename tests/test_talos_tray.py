import re

import mujoco
import numpy as np
import pytest
from pal_mjlab.robots.pal_talos.talos_constants import (
  INIT_STATE,
  TALOS_FT_SITE_BODIES,
  TALOS_FT_SITE_NAMES,
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
  get_free_tray_payload_spec,
  get_spec,
  get_talos_free_tray_payload_cfg,
  get_talos_tray_robot_cfg,
  get_tray_spec,
)
from pal_mjlab.tasks.velocity.talos.env_cfgs import (
  TALOS_FORCE_SENSOR_NAMES,
  TALOS_TORQUE_SENSOR_NAMES,
  pal_talos_flat_env_cfg,
  pal_talos_free_payload_tray_flat_env_cfg,
  pal_talos_payload_flat_env_cfg,
  pal_talos_rough_env_cfg,
  pal_talos_tray_flat_env_cfg,
)
from pal_mjlab.tasks.velocity.talos.rl_cfg import (
  pal_talos_free_payload_tray_ppo_runner_cfg,
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
  assert cfg.rewards["tray_level"].weight > 0
  assert cfg.rewards["payload_position_on_tray"].weight > 0
  assert cfg.rewards["payload_relative_motion"].weight > 0
  assert cfg.rewards["wrist_load_balance"].weight > 0
  assert cfg.rewards["tray_tipping_moment"].weight > 0
  assert cfg.rewards["wrist_force_rate"].weight < 0
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
