import torch
from pal_mjlab.tasks.velocity.talos.mass_estimation import (
  PayloadMassEstimatorModel,
  PayloadMassEstimatorPPO,
  PayloadStateEstimatorModel,
  PayloadStateEstimatorPPO,
  StandalonePayloadStateEstimator,
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
      "payload_state_target": torch.cat(
        (
          torch.linspace(2.5, 30.0, batch_size).unsqueeze(-1),
          torch.tensor([0.05, -0.10, 0.15]).repeat(batch_size, 1),
        ),
        dim=-1,
      ),
      "estimated_payload_state": torch.zeros(batch_size, 4),
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


def _make_state_model(obs: TensorDict) -> PayloadStateEstimatorModel:
  return PayloadStateEstimatorModel(
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
    payload_position_range_t=((-0.38, 0.38), (-0.49, 0.49), (-0.05, 0.40)),
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


def test_state_estimator_conditions_actor_and_exports_physical_state() -> None:
  obs = _make_observations()
  model = _make_state_model(obs)

  actions = model(obs)
  mass_kg, position_t = model.estimate_payload_state(obs)

  assert actions.shape == (4, 3)
  assert mass_kg.shape == (4, 1)
  assert position_t.shape == (4, 3)
  assert torch.all((mass_kg >= 2.5) & (mass_kg <= 30.0))
  lower = torch.tensor([-0.38, -0.49, -0.05])
  upper = torch.tensor([0.38, 0.49, 0.40])
  assert torch.all(position_t >= lower)
  assert torch.all(position_t <= upper)

  exported = model.as_onnx(verbose=False)
  export_actions, export_mass, export_position = exported(*exported.get_dummy_inputs())
  assert export_actions.shape == (1, 3)
  assert export_mass.shape == (1, 1)
  assert export_position.shape == (1, 3)
  assert exported.output_names == [
    "actions",
    "estimated_payload_mass_kg",
    "estimated_payload_position_t_m",
  ]


def test_state_estimator_supervises_mass_and_position_and_injects_critic() -> None:
  obs = _make_observations()
  actor = _make_state_model(obs)
  actor(obs)
  estimator_loss, metrics = actor.estimator_objective(obs, "payload_state_target")
  estimator_loss.backward()

  assert set(metrics) == {
    "payload_state_estimator",
    "mass_estimator_mae_kg",
    "position_estimator_mae_m",
  }
  assert any(
    parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
    for parameter in actor.mass_estimator.parameters()
  )

  critic = MLPModel(
    obs=obs,
    obs_groups={
      "actor": ["actor"],
      "critic": ["critic", "estimated_payload_state"],
    },
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
  algorithm = PayloadStateEstimatorPPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
  )
  actions = algorithm.act(obs)

  assert actions.shape == (4, 3)
  assert actor.latest_payload_state_normalized is not None
  torch.testing.assert_close(
    obs["estimated_payload_state"],
    actor.latest_payload_state_normalized.detach(),
  )


def test_standalone_estimator_has_independent_supervised_gradients() -> None:
  batch_size = 16
  observations = torch.randn(batch_size, 24)
  target = torch.cat(
    (
      torch.linspace(2.5, 30.0, batch_size).unsqueeze(-1),
      torch.stack(
        (
          torch.linspace(-0.2, 0.2, batch_size),
          torch.linspace(0.3, -0.3, batch_size),
          torch.full((batch_size,), 0.15),
        ),
        dim=-1,
      ),
    ),
    dim=-1,
  )
  model = StandalonePayloadStateEstimator(
    observations.shape[-1],
    hidden_dims=(16, 8),
  )
  model.update_normalization(observations)

  loss, metrics = model.objective(observations, target)
  loss.backward()
  mass_kg, position_t = model(observations)

  assert set(metrics) == {
    "loss",
    "mass_loss",
    "position_loss",
    "mass_mae_kg",
    "position_mae_m",
  }
  assert mass_kg.shape == (batch_size, 1)
  assert position_t.shape == (batch_size, 3)
  assert torch.all((mass_kg >= 2.5) & (mass_kg <= 30.0))
  assert any(
    parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
    for parameter in model.network.parameters()
  )


def test_standalone_estimator_state_dict_includes_input_normalization() -> None:
  model = StandalonePayloadStateEstimator(12, hidden_dims=(8, 4))
  observations = torch.randn(32, 12) + 3.0
  model.update_normalization(observations)

  restored = StandalonePayloadStateEstimator(12, hidden_dims=(8, 4))
  restored.load_state_dict(model.state_dict())

  torch.testing.assert_close(
    restored.obs_normalizer.mean,
    model.obs_normalizer.mean,
  )
  torch.testing.assert_close(restored(observations)[0], model(observations)[0])
