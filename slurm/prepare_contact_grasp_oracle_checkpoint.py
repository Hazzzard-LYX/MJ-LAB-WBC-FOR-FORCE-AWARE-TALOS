"""Expand a contact-grasp actor checkpoint with a zero-effect oracle input."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _append_column(tensor: torch.Tensor, fill: float) -> torch.Tensor:
  column = torch.full(
    (*tensor.shape[:-1], 1),
    fill,
    dtype=tensor.dtype,
    device=tensor.device,
  )
  return torch.cat((tensor, column), dim=-1)


def expand_actor_oracle_input(checkpoint: dict[str, Any]) -> dict[str, Any]:
  """Append one actor input while preserving the source policy exactly.

  The new first-layer column and its Adam moments are zero, so the added true
  mass observation has no effect before continuation training learns to use it.
  """
  actor = checkpoint["actor_state_dict"]
  first_layer_key = "mlp.0.weight"
  first_layer = actor[first_layer_key]
  if first_layer.ndim != 2:
    raise ValueError(f"Expected a matrix at {first_layer_key}, got {first_layer.shape}.")
  source_input_dim = int(first_layer.shape[1])

  for key, fill in (
    ("obs_normalizer._mean", 0.0),
    ("obs_normalizer._var", 1.0),
    ("obs_normalizer._std", 1.0),
  ):
    value = actor[key]
    if value.shape[-1] != source_input_dim:
      raise ValueError(
        f"Actor {key} has width {value.shape[-1]}, expected {source_input_dim}."
      )
    actor[key] = _append_column(value, fill)
  actor[first_layer_key] = _append_column(first_layer, 0.0)

  optimizer = checkpoint.get("optimizer_state_dict")
  if isinstance(optimizer, dict):
    expanded_moments = 0
    for state in optimizer.get("state", {}).values():
      for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(name)
        if isinstance(value, torch.Tensor) and value.shape == first_layer.shape:
          state[name] = _append_column(value, 0.0)
          expanded_moments += 1
    if expanded_moments < 2:
      raise ValueError("Could not find both Adam moments for the actor input layer.")

  infos = checkpoint.get("infos")
  if not isinstance(infos, dict):
    infos = {}
    checkpoint["infos"] = infos
  source_iteration = int(checkpoint.get("iter", 0))
  infos["contact_grasp_oracle_input_expansion"] = {
    "source_actor_input_dim": source_input_dim,
    "target_actor_input_dim": source_input_dim + 1,
    "new_feature": "payload_mass_oracle",
    "initial_first_layer_weight": 0.0,
    "source_iteration": source_iteration,
  }
  checkpoint["iter"] = 0
  return checkpoint


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not args.source.is_file():
    raise FileNotFoundError(args.source)
  checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
  checkpoint = expand_actor_oracle_input(checkpoint)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint, args.output)
  actor = checkpoint["actor_state_dict"]
  print(f"SOURCE={args.source}")
  print(f"OUTPUT={args.output}")
  print(f"ACTOR_INPUT_DIM={actor['mlp.0.weight'].shape[1]}")


if __name__ == "__main__":
  main()
