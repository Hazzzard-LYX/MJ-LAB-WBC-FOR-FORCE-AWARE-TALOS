from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.velocity.mdp.rewards import (
  command_tracking_gate,
  planar_velocity_tracking_error,
)


class _CommandManager:
  def __init__(self, command: torch.Tensor) -> None:
    self.command = command

  def get_command(self, name: str) -> torch.Tensor:
    assert name == "twist"
    return self.command


def _make_env(command: torch.Tensor, velocity: torch.Tensor) -> SimpleNamespace:
  robot = SimpleNamespace(
    data=SimpleNamespace(root_link_lin_vel_b=velocity),
  )
  return SimpleNamespace(
    scene={"robot": robot},
    command_manager=_CommandManager(command),
    extras={"log": {}},
  )


def test_command_tracking_gate_rewards_motion_and_standing_commands() -> None:
  command = torch.tensor(
    [
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ]
  )
  velocity = torch.tensor(
    [
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ]
  )
  env = _make_env(command, velocity)

  gate = command_tracking_gate(
    env,
    command_name="twist",
    std=0.5,
    min_factor=0.1,
  )

  assert gate[0] == pytest.approx(1.0)
  assert gate[1] == pytest.approx(0.1 + 0.9 * torch.exp(torch.tensor(-4.0)).item())
  assert gate[2] == pytest.approx(1.0)
  assert env.extras["log"]["Metrics/transport_tracking_gate"] == pytest.approx(
    gate.mean()
  )


def test_planar_velocity_tracking_error_does_not_saturate() -> None:
  command = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
  velocity = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]])
  env = _make_env(command, velocity)

  error = planar_velocity_tracking_error(
    env,
    command_name="twist",
    asset_cfg=SceneEntityCfg("robot"),
  )

  torch.testing.assert_close(error, torch.tensor([1.2, 0.0]))
  assert env.extras["log"]["Metrics/planar_velocity_tracking_error"] == (
    pytest.approx(0.6)
  )
