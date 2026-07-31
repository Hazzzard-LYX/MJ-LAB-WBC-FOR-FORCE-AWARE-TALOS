"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

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


def joint_torque_sensor(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Joint torque-sensor proxy for the selected instrumented TALOS joints.

  MuJoCo's actuator contribution in generalized joint coordinates is the
  closest model-side equivalent of TALOS's joint strain-gauge feedback.  The
  selected joint set intentionally excludes the uninstrumented head and wrist
  axes.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.qfrc_actuator[:, asset_cfg.joint_ids]


def zero_joint_torque_sensor(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return a zeroed joint-torque channel with the sensor's original shape.

  This supports torque-feedback ablations without changing the observation
  contract or downstream network dimensions.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.zeros_like(asset.data.qfrc_actuator[:, asset_cfg.joint_ids])


def force_torque_wrenches_w(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  force_sensor_names: tuple[str, ...],
  torque_sensor_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return site F/T measurements rotated into the world frame.

  Returns:
    A pair ``(force_w, torque_w)`` with shape ``[B, S, 3]``.  Sensor and
    ``asset_cfg.site_names`` ordering must match.
  """
  if len(force_sensor_names) != len(torque_sensor_names):
    raise ValueError("Force and torque sensor lists must have equal length.")

  site_ids = asset_cfg.site_ids
  if not isinstance(site_ids, list) or len(site_ids) != len(force_sensor_names):
    raise ValueError(
      "force_torque_wrenches_w requires one selected site per F/T sensor pair."
    )

  asset: Entity = env.scene[asset_cfg.name]
  force_s = torch.stack(
    tuple(_builtin_sensor_data(env, name) for name in force_sensor_names), dim=1
  )
  torque_s = torch.stack(
    tuple(_builtin_sensor_data(env, name) for name in torque_sensor_names), dim=1
  )
  site_quat_w = asset.data.site_quat_w[:, site_ids]
  flat_quat = site_quat_w.reshape(-1, 4)
  force_w = quat_apply(flat_quat, force_s.reshape(-1, 3)).reshape_as(force_s)
  torque_w = quat_apply(flat_quat, torque_s.reshape(-1, 3)).reshape_as(torque_s)
  return force_w, torque_w


def _builtin_sensor_data(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, BuiltinSensor):
    raise TypeError(
      f"Expected BuiltinSensor '{sensor_name}', got {type(sensor).__name__}."
    )
  return sensor.data


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
  body_id = _single_body_id(asset, asset_cfg, "payload_mass")
  global_body_id = asset.indexing.body_ids[body_id]
  masses = env.sim.model.body_mass[:, global_body_id]
  return masses.reshape(env.num_envs, 1)


def payload_pos_t(
  env: ManagerBasedRlEnv,
  tray_cfg: SceneEntityCfg,
  payload_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Payload COM position relative to the tray origin, in the tray frame."""
  tray: Entity = env.scene[tray_cfg.name]
  payload: Entity = env.scene[payload_cfg.name]
  tray_body_id = _single_body_id(tray, tray_cfg, "payload_pos_t")
  payload_body_id = _single_body_id(payload, payload_cfg, "payload_pos_t")

  tray_pos_w = tray.data.body_link_pos_w[:, tray_body_id]
  tray_quat_w = tray.data.body_link_quat_w[:, tray_body_id]
  payload_pos_w = payload.data.body_com_pos_w[:, payload_body_id]
  return quat_apply_inverse(tray_quat_w, payload_pos_w - tray_pos_w)


def payload_relative_velocity_t(
  env: ManagerBasedRlEnv,
  tray_cfg: SceneEntityCfg,
  payload_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Payload linear and angular velocity relative to the tray frame."""
  tray: Entity = env.scene[tray_cfg.name]
  payload: Entity = env.scene[payload_cfg.name]
  tray_body_id = _single_body_id(tray, tray_cfg, "payload_relative_velocity_t")
  payload_body_id = _single_body_id(payload, payload_cfg, "payload_relative_velocity_t")

  tray_pos_w = tray.data.body_link_pos_w[:, tray_body_id]
  tray_quat_w = tray.data.body_link_quat_w[:, tray_body_id]
  tray_vel_w = tray.data.body_link_vel_w[:, tray_body_id]
  payload_pos_w = payload.data.body_com_pos_w[:, payload_body_id]
  payload_vel_w = payload.data.body_com_vel_w[:, payload_body_id]

  offset_w = payload_pos_w - tray_pos_w
  tray_point_lin_vel_w = tray_vel_w[:, :3] + torch.cross(
    tray_vel_w[:, 3:], offset_w, dim=-1
  )
  relative_lin_vel_w = payload_vel_w[:, :3] - tray_point_lin_vel_w
  relative_ang_vel_w = payload_vel_w[:, 3:] - tray_vel_w[:, 3:]
  return torch.cat(
    (
      quat_apply_inverse(tray_quat_w, relative_lin_vel_w),
      quat_apply_inverse(tray_quat_w, relative_ang_vel_w),
    ),
    dim=-1,
  )


def tray_projected_gravity(
  env: ManagerBasedRlEnv,
  tray_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Unit gravity vector expressed in the tray frame."""
  tray: Entity = env.scene[tray_cfg.name]
  tray_body_id = _single_body_id(tray, tray_cfg, "tray_projected_gravity")
  tray_quat_w = tray.data.body_link_quat_w[:, tray_body_id]
  return quat_apply_inverse(tray_quat_w, tray.data.gravity_vec_w)


def _single_body_id(
  asset: Entity, asset_cfg: SceneEntityCfg, function_name: str
) -> int:
  if isinstance(asset_cfg.body_ids, list) and len(asset_cfg.body_ids) == 1:
    return asset_cfg.body_ids[0]
  if isinstance(asset_cfg.body_ids, slice) and asset.num_bodies == 1:
    return 0
  raise ValueError(f"{function_name} requires exactly one selected body.")
