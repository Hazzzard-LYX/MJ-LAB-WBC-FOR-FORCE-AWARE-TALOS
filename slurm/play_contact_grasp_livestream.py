"""Visualize a contact-grasp TALOS checkpoint with the Viser web viewer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pal_mjlab.tasks  # noqa: F401  # Register project tasks.
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--task",
    default="Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp",
  )
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--forward-velocity", type=float, default=0.2)
  parser.add_argument("--port", type=int, default=18080)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  if args.num_envs <= 0:
    raise ValueError("num-envs must be positive")

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed

  command_cfg = env_cfg.commands["twist"]
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.ranges.lin_vel_x = (args.forward_velocity, args.forward_velocity)
  # The Viser command widget requires positive upper bounds.  Forward-only
  # sampling below still fixes lateral and yaw commands to zero.
  command_cfg.ranges.lin_vel_y = (0.0, 0.1)
  command_cfg.ranges.ang_vel_z = (0.0, 0.1)
  command_cfg.rel_standing_envs = 0.0
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_world_envs = 0.0
  command_cfg.rel_forward_envs = 1.0
  command_cfg.init_velocity_prob = 0.0

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  print(f"CHECKPOINT={args.checkpoint}", flush=True)
  print(f"LIVESTREAM_PORT={args.port}", flush=True)
  print(f"NUM_ENVS={args.num_envs}", flush=True)
  print(f"FORWARD_VELOCITY_MPS={args.forward_velocity:.6f}", flush=True)
  server = viser.ViserServer(
    host="0.0.0.0",
    port=args.port,
    label="talos-contact-grasp",
  )
  try:
    ViserPlayViewer(env, policy, viser_server=server).run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
