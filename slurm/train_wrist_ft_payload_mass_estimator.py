"""Train a mass-only estimator from frozen-policy wrist F/T rollouts."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import pal_mjlab.tasks  # noqa: F401  # Register project tasks.
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.tasks.velocity.talos.env_cfgs import TALOS_TRAY_PAYLOAD_MASS_RANGE
from pal_mjlab.tasks.velocity.talos.mass_estimation import (
  StandalonePayloadMassEstimator,
)

DEFAULT_TASK = (
  "Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-Random-Mass-Wrist-FT-Identification"
)
ESTIMATOR_OBSERVATION_GROUP = "wrist_ft_mass_estimator"
TARGET_OBSERVATION_GROUP = "payload_mass_target"


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
  parser.add_argument("--validation-samples", type=int, default=4096)
  parser.add_argument("--warmup-steps", type=int, default=32)
  parser.add_argument("--learning-rate", type=float, default=3.0e-4)
  parser.add_argument("--huber-beta", type=float, default=0.05)
  parser.add_argument("--gradient-clip", type=float, default=5.0)
  parser.add_argument("--log-every", type=int, default=500)
  parser.add_argument("--save-every", type=int, default=2500)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if not args.policy_checkpoint.is_file():
    parser.error(f"policy checkpoint does not exist: {args.policy_checkpoint}")
  positive = (
    args.num_envs,
    args.steps,
    args.history_length,
    args.batch_size,
    args.validation_samples,
  )
  if any(value <= 0 for value in positive) or args.num_envs < 2:
    parser.error(
      "environment, step, history, batch, and validation sizes must be positive"
    )
  if not 0.0 < args.validation_fraction < 1.0:
    parser.error("validation-fraction must be between zero and one")
  if args.warmup_steps < args.history_length:
    parser.error("warmup-steps must be at least history-length")
  return args


def configure_environment(args: argparse.Namespace):
  """Build noisy wrist F/T rollouts without an estimator in the behavior policy."""
  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed

  estimator_group = env_cfg.observations[ESTIMATOR_OBSERVATION_GROUP]
  estimator_group.history_length = args.history_length
  estimator_group.flatten_history_dim = True
  estimator_group.enable_corruption = True

  if env_cfg.curriculum is not None:
    env_cfg.curriculum.pop("command_vel", None)
  command_cfg = env_cfg.commands["twist"]
  command_cfg.ranges.lin_vel_x = (-0.8, 1.2)
  command_cfg.ranges.lin_vel_y = (-0.4, 0.4)
  command_cfg.ranges.ang_vel_z = (-0.5, 0.5)
  command_cfg.rel_standing_envs = 0.25
  return env_cfg, agent_cfg


def finite_rows(observations: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  """Return rows that contain finite F/T history and labels."""
  return torch.isfinite(observations).all(dim=-1) & torch.isfinite(target).all(dim=-1)


@torch.no_grad()
def rollout_step(env, behavior_policy, observations):
  """Advance the frozen behavior policy by one control step."""
  actions = behavior_policy(observations)
  observations, _, _, _ = env.step(actions.to(env.device))
  return observations


@torch.no_grad()
def collect_fixed_validation_set(
  env,
  behavior_policy,
  observations,
  validation_ids: torch.Tensor,
  *,
  warmup_steps: int,
  sample_count: int,
  device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Freeze a repeatable validation tensor before any estimator update."""
  for _ in range(warmup_steps):
    observations = rollout_step(env, behavior_policy, observations)

  observation_chunks: list[torch.Tensor] = []
  target_chunks: list[torch.Tensor] = []
  collected = 0
  while collected < sample_count:
    observations = rollout_step(env, behavior_policy, observations)
    val_obs = observations[ESTIMATOR_OBSERVATION_GROUP][validation_ids].to(device)
    val_target = observations[TARGET_OBSERVATION_GROUP][validation_ids].to(device)
    valid = finite_rows(val_obs, val_target)
    if torch.any(valid):
      observation_chunks.append(val_obs[valid].detach().clone())
      target_chunks.append(val_target[valid].detach().clone())
      collected += int(valid.sum().item())

  fixed_observations = torch.cat(observation_chunks, dim=0)[:sample_count]
  fixed_targets = torch.cat(target_chunks, dim=0)[:sample_count]
  return observations, fixed_observations, fixed_targets


def save_checkpoint(
  path: Path,
  model: StandalonePayloadMassEstimator,
  optimizer: torch.optim.Optimizer,
  *,
  step: int,
  args: argparse.Namespace,
  metrics: dict[str, float],
) -> None:
  """Persist a self-describing wrist-F/T mass estimator checkpoint."""
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "format": "talos_standalone_wrist_ft_payload_mass_estimator_v1",
      "step": step,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "config": {
        "input_dim": model.input_dim,
        "hidden_dims": model.hidden_dims,
        "activation": model.activation,
        "history_length": args.history_length,
        "payload_mass_range": TALOS_TRAY_PAYLOAD_MASS_RANGE,
        "source_policy_checkpoint": str(args.policy_checkpoint),
        "task": args.task,
        "input_group": ESTIMATOR_OBSERVATION_GROUP,
      },
      "metrics": metrics,
    },
    path,
  )


@torch.no_grad()
def validation_metrics(
  model: StandalonePayloadMassEstimator,
  observations: torch.Tensor,
  target: torch.Tensor,
  args: argparse.Namespace,
) -> dict[str, float]:
  """Evaluate on the same frozen samples and report calibration by mass bin."""
  model.eval()
  estimated_mass = model(observations)
  _, metrics = model.objective(observations, target, huber_beta=args.huber_beta)
  result = {f"val_{name}": float(value.item()) for name, value in metrics.items()}

  mass_center = 0.5 * sum(TALOS_TRAY_PAYLOAD_MASS_RANGE)
  result["baseline_mass_mae_kg"] = float(torch.abs(target - mass_center).mean().item())
  result["prediction_min_kg"] = float(estimated_mass.min().item())
  result["prediction_max_kg"] = float(estimated_mass.max().item())

  edges = (2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
  for lower, upper in zip(edges[:-1], edges[1:], strict=True):
    is_last = upper == edges[-1]
    mask = (target[:, 0] >= lower) & (
      target[:, 0] <= upper if is_last else target[:, 0] < upper
    )
    label = f"{lower:g}_{upper:g}kg"
    result[f"val_count_{label}"] = int(mask.sum().item())
    if torch.any(mask):
      result[f"val_mae_{label}"] = float(
        torch.abs(estimated_mass[mask] - target[mask]).mean().item()
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
  input_dim = observations[ESTIMATOR_OBSERVATION_GROUP].shape[-1]
  model = StandalonePayloadMassEstimator(
    input_dim,
    payload_mass_range=TALOS_TRAY_PAYLOAD_MASS_RANGE,
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
    "WRIST_FT_MASS_ESTIMATOR_START "
    f"input_dim={input_dim} train_envs={len(training_ids)} "
    f"validation_envs={len(validation_ids)} history={args.history_length}",
    flush=True,
  )
  observations, fixed_val_obs, fixed_val_target = collect_fixed_validation_set(
    env,
    behavior_policy,
    observations,
    validation_ids,
    warmup_steps=args.warmup_steps,
    sample_count=args.validation_samples,
    device=args.device,
  )
  print(
    "WRIST_FT_FIXED_VALIDATION_OK "
    f"samples={len(fixed_val_obs)} "
    f"baseline_mass_mae_kg="
    f"{torch.abs(fixed_val_target - 0.5 * sum(TALOS_TRAY_PAYLOAD_MASS_RANGE)).mean().item():.4f}",
    flush=True,
  )

  best_validation_mae = math.inf
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

      train_obs = observations[ESTIMATOR_OBSERVATION_GROUP][batch_ids].detach()
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
        huber_beta=args.huber_beta,
      )
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.gradient_clip
      )
      optimizer.step()

      with torch.inference_mode():
        observations = rollout_step(env, behavior_policy, observations).to(args.device)

      should_log = step == 1 or step % args.log_every == 0 or step == args.steps
      if should_log:
        val_metrics = validation_metrics(model, fixed_val_obs, fixed_val_target, args)
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
          "WRIST_FT_ESTIMATOR_STEP "
          f"step={step}/{args.steps} "
          f"train_mass_mae_kg={last_metrics['train_mass_mae_kg']:.4f} "
          f"val_mass_mae_kg={last_metrics['val_mass_mae_kg']:.4f} "
          f"val_r2={last_metrics['val_mass_r2']:.4f} "
          f"baseline_mass_mae_kg={last_metrics['baseline_mass_mae_kg']:.4f} "
          f"elapsed_s={elapsed:.1f}",
          flush=True,
        )
        if last_metrics["val_mass_mae_kg"] < best_validation_mae:
          best_validation_mae = last_metrics["val_mass_mae_kg"]
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
    "WRIST_FT_MASS_ESTIMATOR_OK "
    f"best_val_mass_mae_kg={best_validation_mae:.4f} "
    f"final_val_mass_mae_kg={last_metrics['val_mass_mae_kg']:.4f} "
    f"final_val_r2={last_metrics['val_mass_r2']:.4f}",
    flush=True,
  )


if __name__ == "__main__":
  main()
