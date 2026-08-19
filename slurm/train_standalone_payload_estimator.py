"""Train a TALOS payload-state estimator from frozen-policy rollouts."""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pal_mjlab.tasks  # noqa: F401  # Register project tasks.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.tasks.velocity.talos.env_cfgs import (
  TALOS_TRAY_PAYLOAD_MASS_RANGE,
  TALOS_TRAY_PAYLOAD_POSITION_RANGE_T,
)
from pal_mjlab.tasks.velocity.talos.mass_estimation import (
  StandalonePayloadStateEstimator,
)

DEFAULT_TASK = (
  "Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-Random-Mass-State-Estimator"
)
STANDALONE_OBSERVATION_GROUP = "standalone_estimator"
TARGET_OBSERVATION_GROUP = "payload_state_target"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--policy-checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=1536)
  parser.add_argument("--steps", type=int, default=50_000)
  parser.add_argument("--history-length", type=int, default=32)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--validation-fraction", type=float, default=0.2)
  parser.add_argument("--learning-rate", type=float, default=3.0e-4)
  parser.add_argument("--mass-loss-weight", type=float, default=1.0)
  parser.add_argument("--position-loss-weight", type=float, default=1.0)
  parser.add_argument("--huber-beta", type=float, default=0.05)
  parser.add_argument("--gradient-clip", type=float, default=5.0)
  parser.add_argument("--log-every", type=int, default=100)
  parser.add_argument("--save-every", type=int, default=2_500)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if not args.policy_checkpoint.is_file():
    parser.error(f"policy checkpoint does not exist: {args.policy_checkpoint}")
  if args.num_envs < 2 or args.steps <= 0 or args.history_length <= 0:
    parser.error("num-envs, steps, and history-length must be positive")
  if not 0.0 < args.validation_fraction < 1.0:
    parser.error("validation-fraction must be between zero and one")
  if args.batch_size <= 0:
    parser.error("batch-size must be positive")
  return args


def configure_environment(args: argparse.Namespace):
  """Build noisy rollouts with a longer estimator-only history channel."""
  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed

  source_group = env_cfg.observations["mass_estimator"]
  standalone_group = deepcopy(source_group)
  standalone_group.history_length = args.history_length
  standalone_group.flatten_history_dim = True
  standalone_group.enable_corruption = True
  env_cfg.observations[STANDALONE_OBSERVATION_GROUP] = standalone_group

  # The behavior checkpoint already learned the final curriculum.  Sample all
  # command regimes immediately, with a standing subset that provides clean
  # gravity-loaded examples for identifying mass and centre of mass.
  if env_cfg.curriculum is not None:
    env_cfg.curriculum.pop("command_vel", None)
  command_cfg = env_cfg.commands["twist"]
  command_cfg.ranges.lin_vel_x = (-0.8, 1.2)
  command_cfg.ranges.lin_vel_y = (-0.4, 0.4)
  command_cfg.ranges.ang_vel_z = (-0.5, 0.5)
  command_cfg.rel_standing_envs = 0.25
  return env_cfg, agent_cfg


def finite_rows(observations: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  """Return a mask that excludes non-finite samples from supervision."""
  return torch.isfinite(observations).all(dim=-1) & torch.isfinite(target).all(dim=-1)


def save_checkpoint(
  path: Path,
  model: StandalonePayloadStateEstimator,
  optimizer: torch.optim.Optimizer,
  *,
  step: int,
  args: argparse.Namespace,
  metrics: dict[str, float],
) -> None:
  """Persist a self-describing standalone estimator checkpoint."""
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "format": "talos_standalone_payload_state_estimator_v1",
      "step": step,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "config": {
        "input_dim": model.input_dim,
        "hidden_dims": model.hidden_dims,
        "activation": model.activation,
        "history_length": args.history_length,
        "payload_mass_range": TALOS_TRAY_PAYLOAD_MASS_RANGE,
        "payload_position_range_t": TALOS_TRAY_PAYLOAD_POSITION_RANGE_T,
        "source_policy_checkpoint": str(args.policy_checkpoint),
        "task": args.task,
      },
      "metrics": metrics,
    },
    path,
  )


@torch.no_grad()
def validation_metrics(
  model: StandalonePayloadStateEstimator,
  observations: torch.Tensor,
  target: torch.Tensor,
  args: argparse.Namespace,
) -> dict[str, float]:
  """Measure estimator and constant-centre baselines on held-out environments."""
  model.eval()
  _, metrics = model.objective(
    observations,
    target,
    mass_loss_weight=args.mass_loss_weight,
    position_loss_weight=args.position_loss_weight,
    huber_beta=args.huber_beta,
  )
  mass_center = 0.5 * sum(TALOS_TRAY_PAYLOAD_MASS_RANGE)
  position_center = torch.tensor(
    [0.5 * sum(bounds) for bounds in TALOS_TRAY_PAYLOAD_POSITION_RANGE_T],
    device=target.device,
  )
  result = {f"val_{name}": float(value.item()) for name, value in metrics.items()}
  result["baseline_mass_mae_kg"] = float(
    torch.abs(target[:, :1] - mass_center).mean().item()
  )
  result["baseline_position_mae_m"] = float(
    torch.linalg.vector_norm(target[:, 1:] - position_center, dim=-1).mean().item()
  )
  return result


def main() -> None:
  args = parse_args()
  configure_torch_backends()
  torch.manual_seed(args.seed)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  metrics_path = args.output_dir / "metrics.jsonl"
  (args.output_dir / "config.json").write_text(
    json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
  )

  env_cfg, agent_cfg = configure_environment(args)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.policy_checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  behavior_policy = runner.get_inference_policy(device=args.device)
  for parameter in behavior_policy.parameters():
    parameter.requires_grad_(False)

  observations = env.get_observations().to(args.device)
  estimator_input = observations[STANDALONE_OBSERVATION_GROUP]
  input_dim = estimator_input.shape[-1]
  model = StandalonePayloadStateEstimator(
    input_dim,
    payload_mass_range=TALOS_TRAY_PAYLOAD_MASS_RANGE,
    payload_position_range_t=TALOS_TRAY_PAYLOAD_POSITION_RANGE_T,
  ).to(args.device)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5
  )

  generator = torch.Generator(device=args.device)
  generator.manual_seed(args.seed + 1)
  permutation = torch.randperm(args.num_envs, generator=generator, device=args.device)
  validation_count = max(1, round(args.num_envs * args.validation_fraction))
  validation_ids = permutation[:validation_count]
  training_ids = permutation[validation_count:]
  if len(training_ids) == 0:
    raise RuntimeError("No training environments remain after validation split.")

  print(
    "STANDALONE_ESTIMATOR_START "
    f"input_dim={input_dim} train_envs={len(training_ids)} "
    f"validation_envs={len(validation_ids)} history={args.history_length}",
    flush=True,
  )
  best_validation_loss = math.inf
  start_time = time.monotonic()
  last_metrics: dict[str, float] = {}

  try:
    for step in range(1, args.steps + 1):
      model.train()
      available = len(training_ids)
      if args.batch_size < available:
        sample = torch.randint(
          available,
          (args.batch_size,),
          generator=generator,
          device=args.device,
        )
        batch_ids = training_ids[sample]
      else:
        batch_ids = training_ids
      train_obs = observations[STANDALONE_OBSERVATION_GROUP][batch_ids].detach()
      train_target = observations[TARGET_OBSERVATION_GROUP][batch_ids].detach()
      valid = finite_rows(train_obs, train_target)
      train_obs = train_obs[valid]
      train_target = train_target[valid]
      if len(train_obs) == 0:
        raise RuntimeError(f"No finite estimator samples at step {step}.")

      model.update_normalization(train_obs)
      loss, train_metrics = model.objective(
        train_obs,
        train_target,
        mass_loss_weight=args.mass_loss_weight,
        position_loss_weight=args.position_loss_weight,
        huber_beta=args.huber_beta,
      )
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.gradient_clip
      )
      optimizer.step()

      with torch.inference_mode():
        actions = behavior_policy(observations)
        observations, _, _, _ = env.step(actions.to(env.device))
        observations = observations.to(args.device)

      should_log = step == 1 or step % args.log_every == 0 or step == args.steps
      if should_log:
        val_obs = observations[STANDALONE_OBSERVATION_GROUP][validation_ids]
        val_target = observations[TARGET_OBSERVATION_GROUP][validation_ids]
        valid = finite_rows(val_obs, val_target)
        val_obs = val_obs[valid]
        val_target = val_target[valid]
        if len(val_obs) == 0:
          raise RuntimeError(f"No finite validation samples at step {step}.")
        val_metrics = validation_metrics(model, val_obs, val_target, args)
        elapsed = time.monotonic() - start_time
        last_metrics = {
          "step": step,
          "elapsed_s": elapsed,
          "samples_seen": step * len(batch_ids),
          "gradient_norm": float(gradient_norm.item()),
          **{
            f"train_{name}": float(value.item())
            for name, value in train_metrics.items()
          },
          **val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
          stream.write(json.dumps(last_metrics, sort_keys=True) + "\n")
        print(
          "ESTIMATOR_STEP "
          f"step={step}/{args.steps} "
          f"train_mass_mae_kg={last_metrics['train_mass_mae_kg']:.4f} "
          f"val_mass_mae_kg={last_metrics['val_mass_mae_kg']:.4f} "
          f"val_position_mae_m={last_metrics['val_position_mae_m']:.4f} "
          f"baseline_mass_mae_kg={last_metrics['baseline_mass_mae_kg']:.4f} "
          f"elapsed_s={elapsed:.1f}",
          flush=True,
        )
        if last_metrics["val_loss"] < best_validation_loss:
          best_validation_loss = last_metrics["val_loss"]
          save_checkpoint(
            args.output_dir / "estimator_best.pt",
            model,
            optimizer,
            step=step,
            args=args,
            metrics=last_metrics,
          )

      if step % args.save_every == 0:
        save_checkpoint(
          args.output_dir / f"estimator_{step}.pt",
          model,
          optimizer,
          step=step,
          args=args,
          metrics=last_metrics,
        )
  finally:
    env.close()

  save_checkpoint(
    args.output_dir / "estimator_final.pt",
    model,
    optimizer,
    step=args.steps,
    args=args,
    metrics=last_metrics,
  )
  print(
    "STANDALONE_ESTIMATOR_OK "
    f"best_val_loss={best_validation_loss:.6f} "
    f"final_val_mass_mae_kg={last_metrics['val_mass_mae_kg']:.4f} "
    f"final_val_position_mae_m={last_metrics['val_position_mae_m']:.4f}",
    flush=True,
  )


if __name__ == "__main__":
  main()
