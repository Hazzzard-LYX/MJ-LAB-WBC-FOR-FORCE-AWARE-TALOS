import torch
from pal_mjlab.tasks.velocity.talos.mass_estimation import (
  PayloadMassEstimatorModel,
  PayloadMassEstimatorPPO,
)
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


def _make_observations(batch_size: int = 4) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(batch_size, 7),
      "critic": torch.randn(batch_size, 9),
      "mass_estimator": torch.randn(batch_size, 40),
      "payload_mass_target": torch.linspace(2.5, 30.0, batch_size).unsqueeze(-1),
    },
    batch_size=[batch_size],
  )


def _make_model(obs: TensorDict) -> PayloadMassEstimatorModel:
  return PayloadMassEstimatorModel(
    obs=obs,
    obs_groups={"actor": ["actor"], "critic": ["critic"]},
    obs_set="actor",
    output_dim=3,
    hidden_dims=(16, 8),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    estimator_hidden_dims=(12, 6),
    payload_mass_range=(2.5, 30.0),
  )


def test_estimator_actor_outputs_actions_and_bounded_mass() -> None:
  obs = _make_observations()
  model = _make_model(obs)

  actions = model(obs)
  mass = model.estimate_payload_mass(obs)

  assert actions.shape == (4, 3)
  assert mass.shape == (4, 1)
  assert torch.all(mass >= 2.5)
  assert torch.all(mass <= 30.0)


def test_estimator_actor_never_reads_simulator_mass_target() -> None:
  obs = _make_observations()
  model = _make_model(obs)
  model.eval()
  actions_before = model(obs)

  changed_obs = obs.clone()
  changed_obs["payload_mass_target"] = torch.full((4, 1), 999.0)
  actions_after = model(changed_obs)

  torch.testing.assert_close(actions_before, actions_after)


def test_estimator_auxiliary_loss_backpropagates_to_estimator() -> None:
  obs = _make_observations()
  model = _make_model(obs)
  estimate = model.estimate_payload_mass_normalized(obs)
  target = (obs["payload_mass_target"] - 2.5) / (30.0 - 2.5)
  loss = torch.nn.functional.mse_loss(estimate, target)
  loss.backward()

  gradients = [
    parameter.grad
    for parameter in model.mass_estimator.parameters()
    if parameter.grad is not None
  ]
  assert gradients
  assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_estimator_export_contract_has_two_inputs_and_two_outputs() -> None:
  obs = _make_observations()
  model = _make_model(obs)
  exported = model.as_onnx(verbose=False)
  dummy_inputs = exported.get_dummy_inputs()
  actions, estimated_mass = exported(*dummy_inputs)

  assert exported.input_names == ["actor_obs", "proprioceptive_history"]
  assert exported.output_names == ["actions", "estimated_payload_mass_kg"]
  assert actions.shape == (1, 3)
  assert estimated_mass.shape == (1, 1)


def test_estimator_ppo_reports_supervised_mass_metrics() -> None:
  obs = _make_observations()
  actor = _make_model(obs)
  critic = MLPModel(
    obs=obs,
    obs_groups={"actor": ["actor"], "critic": ["critic"]},
    obs_set="critic",
    output_dim=1,
    hidden_dims=(16, 8),
  )
  storage = RolloutStorage(
    "rl",
    num_envs=4,
    num_transitions_per_env=2,
    obs=obs,
    actions_shape=[3],
  )
  algorithm = PayloadMassEstimatorPPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
  )

  for _ in range(2):
    algorithm.act(obs)
    next_obs = _make_observations()
    algorithm.process_env_step(
      next_obs,
      rewards=torch.randn(4),
      dones=torch.zeros(4, dtype=torch.long),
      extras={},
    )
    obs = next_obs
  algorithm.compute_returns(obs)
  losses = algorithm.update()

  assert losses["mass_estimator"] >= 0.0
  assert losses["mass_estimator_mae_kg"] >= 0.0
