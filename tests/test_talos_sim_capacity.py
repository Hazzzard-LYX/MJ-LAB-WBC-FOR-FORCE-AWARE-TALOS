from pal_mjlab.tasks.tracking.talos.env_cfgs import (
  pal_talos_flat_tracking_env_cfg,
)
from pal_mjlab.tasks.velocity.talos.env_cfgs import pal_talos_flat_env_cfg


def test_talos_velocity_uses_model_sized_contact_buffers() -> None:
  cfg = pal_talos_flat_env_cfg()

  assert cfg.sim.nconmax is None
  assert cfg.sim.contact_sensor_maxmatch == 500
  assert cfg.sim.mujoco.ccd_iterations == 500


def test_talos_tracking_uses_model_sized_contact_buffers() -> None:
  cfg = pal_talos_flat_tracking_env_cfg()

  assert cfg.sim.nconmax is None
  assert cfg.sim.contact_sensor_maxmatch == 500
  assert cfg.sim.mujoco.ccd_iterations == 500
