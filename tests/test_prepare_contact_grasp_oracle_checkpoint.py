import importlib.util
from pathlib import Path

import torch

SCRIPT = (
  Path(__file__).parents[1] / "slurm" / "prepare_contact_grasp_oracle_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_contact_grasp_oracle_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_oracle_input_expansion_preserves_source_policy() -> None:
  source_dim = 4
  first_layer = torch.arange(12, dtype=torch.float32).reshape(3, source_dim)
  checkpoint = {
    "actor_state_dict": {
      "obs_normalizer._mean": torch.ones(1, source_dim),
      "obs_normalizer._var": torch.full((1, source_dim), 2.0),
      "obs_normalizer._std": torch.full((1, source_dim), 3.0),
      "mlp.0.weight": first_layer.clone(),
    },
    "optimizer_state_dict": {
      "state": {
        1: {
          "exp_avg": torch.ones_like(first_layer),
          "exp_avg_sq": torch.full_like(first_layer, 2.0),
        }
      }
    },
    "infos": {},
    "iter": 29_999,
  }

  expanded = MODULE.expand_actor_oracle_input(checkpoint)
  actor = expanded["actor_state_dict"]

  assert actor["mlp.0.weight"].shape == (3, source_dim + 1)
  torch.testing.assert_close(actor["mlp.0.weight"][:, :-1], first_layer)
  assert torch.count_nonzero(actor["mlp.0.weight"][:, -1]) == 0
  assert actor["obs_normalizer._mean"][0, -1] == 0.0
  assert actor["obs_normalizer._var"][0, -1] == 1.0
  assert actor["obs_normalizer._std"][0, -1] == 1.0
  for value in expanded["optimizer_state_dict"]["state"][1].values():
    assert value.shape == (3, source_dim + 1)
    assert torch.count_nonzero(value[:, -1]) == 0
  assert expanded["iter"] == 0
  assert (
    expanded["infos"]["contact_grasp_oracle_input_expansion"]["source_iteration"]
    == 29_999
  )
