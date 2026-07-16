import re

import mujoco
import numpy as np
import pytest
from pal_mjlab.robots.pal_talos.talos_constants import (
  INIT_STATE,
  TALOS_TRAY_BODY_NAME,
  TALOS_TRAY_HALF_SIZE,
  TALOS_TRAY_HANDLE_POSITIONS,
  TALOS_TRAY_MASS,
  TALOS_TRAY_PARENT_BODY_NAME,
  TALOS_TRAY_RIGHT_MOUNT_SITE_NAME,
  TALOS_TRAY_RIGHT_WRIST_SITE_NAME,
  TALOS_TRAY_SECONDARY_BODY_NAME,
  get_talos_tray_robot_cfg,
  get_tray_spec,
)
from pal_mjlab.tasks.velocity.talos.env_cfgs import pal_talos_tray_flat_env_cfg
from pal_mjlab.tasks.velocity.talos.rl_cfg import pal_talos_tray_ppo_runner_cfg


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
