"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def imu_projected_gravity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Get projected gravity from IMU sensor orientation (accounts for IMU mounting)."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)

  # Get IMU orientation (already includes mounting offset)
  imu_quat = sensor.data  # or however you access orientation

  # Gravity in world frame
  gravity_w = torch.tensor([[0.0, 0.0, -1.0]], device=imu_quat.device).expand(
    imu_quat.shape[0], -1
  )
  # print(f"imu proj{quat_apply_inverse(imu_quat, gravity_w)}")
  # asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  # print(f"proj{asset.data.projected_gravity_b}")
  # Project to IMU frame (same as your C++ code)
  return quat_apply_inverse(imu_quat, gravity_w)


def payload_com_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Payload COM position relative to the robot root, in the root frame."""
  asset: Entity = env.scene[asset_cfg.name]
  if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 1:
    raise ValueError("payload_com_pos_b requires exactly one selected payload body.")

  payload_pos_w = asset.data.body_com_pos_w[:, asset_cfg.body_ids[0]]
  relative_pos_w = payload_pos_w - asset.data.root_link_pos_w
  return quat_apply_inverse(asset.data.root_link_quat_w, relative_pos_w)


def payload_mass(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the current payload mass in kilograms as a one-dimensional term."""
  asset: Entity = env.scene[asset_cfg.name]
  if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 1:
    raise ValueError("payload_mass requires exactly one selected payload body.")

  global_body_id = asset.indexing.body_ids[asset_cfg.body_ids[0]]
  masses = env.sim.model.body_mass[:, global_body_id]
  return masses.reshape(env.num_envs, 1)
