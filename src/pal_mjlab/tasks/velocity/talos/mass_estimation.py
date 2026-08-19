"""Auxiliary payload-mass estimation for TALOS tray transport."""

from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.utils import compile_model
from tensordict import TensorDict


class PayloadMassEstimatorModel(MLPModel):
  """Actor that estimates payload mass from proprioceptive history."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    estimator_obs_group: str = "mass_estimator",
    estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    estimator_activation: str = "elu",
    estimator_obs_normalization: bool = True,
    payload_mass_range: tuple[float, float] = (2.5, 30.0),
    cnn_cfg: dict | None = None,
    rnn_type: str | None = None,
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
  ) -> None:
    del rnn_hidden_dim, rnn_num_layers
    if cnn_cfg is not None or rnn_type is not None:
      raise ValueError("The payload-mass estimator actor supports MLP inputs only.")
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    if estimator_obs_group not in obs:
      raise ValueError(
        f"Estimator observation group '{estimator_obs_group}' is missing."
      )
    estimator_obs = obs[estimator_obs_group]
    if len(estimator_obs.shape) != 2:
      raise ValueError(
        "PayloadMassEstimatorModel requires a flattened 1D estimator history, "
        f"got {estimator_obs.shape}."
      )
    mass_min, mass_max = payload_mass_range
    if mass_min < 0.0 or mass_max <= mass_min:
      raise ValueError(f"Invalid payload mass range: {payload_mass_range}.")

    self.estimator_obs_group = estimator_obs_group
    self.estimator_obs_dim = estimator_obs.shape[-1]
    self.estimator_obs_normalization = estimator_obs_normalization
    self.estimator_obs_normalizer: nn.Module
    if estimator_obs_normalization:
      self.estimator_obs_normalizer = EmpiricalNormalization(self.estimator_obs_dim)
    else:
      self.estimator_obs_normalizer = nn.Identity()
    self.mass_estimator = MLP(
      self.estimator_obs_dim,
      1,
      estimator_hidden_dims,
      estimator_activation,
      last_activation="sigmoid",
    )
    self.payload_mass_min = float(mass_min)
    self.payload_mass_max = float(mass_max)
    self.payload_mass_span = self.payload_mass_max - self.payload_mass_min

    # The policy receives the hardware actor contract plus one normalized,
    # estimated-mass scalar. It never receives the simulator mass target.
    self.mlp = MLP(self.obs_dim + 1, output_dim, hidden_dims, activation)
    if self.distribution is not None:
      self.mlp = MLP(
        self.obs_dim + 1,
        self.distribution.input_dim,
        hidden_dims,
        activation,
      )
      self.distribution.init_mlp_weights(self.mlp)

    self.latest_mass_estimate_normalized: torch.Tensor | None = None

  def estimate_payload_mass_normalized(self, obs: TensorDict) -> torch.Tensor:
    """Estimate payload mass on the unit interval from hardware observations."""
    estimator_obs = self.estimator_obs_normalizer(obs[self.estimator_obs_group])
    estimate = self.mass_estimator(estimator_obs)
    self.latest_mass_estimate_normalized = estimate
    return estimate

  def estimate_payload_mass(self, obs: TensorDict) -> torch.Tensor:
    """Estimate payload mass in kilograms."""
    estimate = self.estimate_payload_mass_normalized(obs)
    return self.payload_mass_min + self.payload_mass_span * estimate

  def estimator_objective(
    self,
    obs: TensorDict,
    target_group: str,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the supervised estimator loss and reporting metrics."""
    if self.latest_mass_estimate_normalized is None:
      raise RuntimeError("The actor must run before its estimator objective.")
    mass_estimate = self.latest_mass_estimate_normalized
    target_mass = obs[target_group]
    target_normalized = (target_mass - self.payload_mass_min) / self.payload_mass_span
    target_normalized = target_normalized.clamp(0.0, 1.0)
    estimator_loss = F.mse_loss(mass_estimate, target_normalized)
    estimator_mae_kg = F.l1_loss(
      self.payload_mass_min + self.payload_mass_span * mass_estimate,
      target_mass,
    )
    return estimator_loss, {
      "mass_estimator": estimator_loss,
      "mass_estimator_mae_kg": estimator_mae_kg,
    }

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    """Append the learned mass estimate to the deployable actor observation."""
    actor_latent = super().get_latent(obs, masks, hidden_state)
    mass_estimate = self.estimate_payload_mass_normalized(obs)
    return torch.cat((actor_latent, mass_estimate), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    """Update actor and estimator normalization statistics."""
    super().update_normalization(obs)
    if self.estimator_obs_normalization:
      self.estimator_obs_normalizer.update(obs[self.estimator_obs_group])

  def as_jit(self) -> nn.Module:
    """Return an exportable deterministic actor and estimator."""
    return _ExportablePayloadMassEstimator(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    """Return an ONNX wrapper with actor and history inputs."""
    del verbose
    return _ExportablePayloadMassEstimator(self)


class _ExportablePayloadMassEstimator(nn.Module):
  """Deployment wrapper that exposes actions and estimated mass."""

  is_recurrent: bool = False

  def __init__(self, model: PayloadMassEstimatorModel) -> None:
    super().__init__()
    self.actor_obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.estimator_obs_normalizer = copy.deepcopy(model.estimator_obs_normalizer)
    self.mass_estimator = copy.deepcopy(model.mass_estimator)
    self.policy = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.actor_obs_dim = model.obs_dim
    self.estimator_obs_dim = model.estimator_obs_dim
    self.payload_mass_min = model.payload_mass_min
    self.payload_mass_span = model.payload_mass_span

  def forward(
    self,
    actor_obs: torch.Tensor,
    proprioceptive_history: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic actions and the estimated payload mass in kg."""
    actor_latent = self.actor_obs_normalizer(actor_obs)
    estimator_latent = self.estimator_obs_normalizer(proprioceptive_history)
    mass_normalized = self.mass_estimator(estimator_latent)
    policy_input = torch.cat((actor_latent, mass_normalized), dim=-1)
    actions = self.deterministic_output(self.policy(policy_input))
    mass_kg = self.payload_mass_min + self.payload_mass_span * mass_normalized
    return actions, mass_kg

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return representative inputs for ONNX tracing."""
    return (
      torch.zeros(1, self.actor_obs_dim),
      torch.zeros(1, self.estimator_obs_dim),
    )

  @property
  def input_names(self) -> list[str]:
    return ["actor_obs", "proprioceptive_history"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "estimated_payload_mass_kg"]

  @torch.jit.export
  def reset(self) -> None:
    """Reset recurrent state (no-op for the feed-forward history encoder)."""


class PayloadStateEstimatorModel(PayloadMassEstimatorModel):
  """Actor conditioned on estimated payload mass and tray-frame position."""

  def __init__(
    self,
    *args,
    payload_position_range_t: tuple[
      tuple[float, float],
      tuple[float, float],
      tuple[float, float],
    ] = ((-0.38, 0.38), (-0.49, 0.49), (-0.05, 0.40)),
    critic_estimate_group: str = "estimated_payload_state",
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    position_min = torch.tensor([bounds[0] for bounds in payload_position_range_t])
    position_max = torch.tensor([bounds[1] for bounds in payload_position_range_t])
    if torch.any(position_max <= position_min):
      raise ValueError(f"Invalid payload position range: {payload_position_range_t}.")
    self.register_buffer("payload_position_min_t", position_min)
    self.register_buffer("payload_position_max_t", position_max)
    self.register_buffer(
      "payload_position_center_t", 0.5 * (position_min + position_max)
    )
    self.register_buffer(
      "payload_position_half_span_t", 0.5 * (position_max - position_min)
    )
    self.critic_estimate_group = critic_estimate_group

    # Replace the mass-only head with one raw mass logit and three position
    # logits.  Mass is mapped through sigmoid and position through tanh.
    self.mass_estimator = MLP(
      self.estimator_obs_dim,
      4,
      kwargs.get("estimator_hidden_dims", (256, 128)),
      kwargs.get("estimator_activation", "elu"),
    )
    hidden_dims = kwargs.get("hidden_dims", (512, 256, 128))
    activation = kwargs.get("activation", "elu")
    policy_output_dim = (
      self.distribution.input_dim
      if self.distribution is not None
      else kwargs["output_dim"]
    )
    self.mlp = MLP(self.obs_dim + 4, policy_output_dim, hidden_dims, activation)
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

    self.latest_payload_state_normalized: torch.Tensor | None = None

  def estimate_payload_state_normalized(self, obs: TensorDict) -> torch.Tensor:
    """Estimate normalized mass in ``[0, 1]`` and position in ``[-1, 1]``."""
    estimator_obs = self.estimator_obs_normalizer(obs[self.estimator_obs_group])
    raw_estimate = self.mass_estimator(estimator_obs)
    estimate = torch.cat(
      (torch.sigmoid(raw_estimate[:, :1]), torch.tanh(raw_estimate[:, 1:])),
      dim=-1,
    )
    self.latest_payload_state_normalized = estimate
    self.latest_mass_estimate_normalized = estimate[:, :1]
    return estimate

  def estimate_payload_state(
    self,
    obs: TensorDict,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return estimated mass in kilograms and position in metres."""
    estimate = self.estimate_payload_state_normalized(obs)
    mass_kg = self.payload_mass_min + self.payload_mass_span * estimate[:, :1]
    position_t = (
      self.payload_position_center_t
      + self.payload_position_half_span_t * estimate[:, 1:]
    )
    return mass_kg, position_t

  def estimate_payload_mass_normalized(self, obs: TensorDict) -> torch.Tensor:
    """Return the mass component of the joint state estimate."""
    return self.estimate_payload_state_normalized(obs)[:, :1]

  def estimate_payload_mass(self, obs: TensorDict) -> torch.Tensor:
    """Return the estimated payload mass in kilograms."""
    estimate = self.estimate_payload_mass_normalized(obs)
    return self.payload_mass_min + self.payload_mass_span * estimate

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    """Append estimated mass and position to the torque-free actor input."""
    actor_latent = MLPModel.get_latent(self, obs, masks, hidden_state)
    state_estimate = self.estimate_payload_state_normalized(obs)
    return torch.cat((actor_latent, state_estimate), dim=-1)

  def inject_critic_estimate(
    self,
    obs: TensorDict,
    *,
    use_latest: bool,
  ) -> None:
    """Insert a detached estimator output for the critic observation set."""
    estimate = self.latest_payload_state_normalized if use_latest else None
    if estimate is None:
      estimate = self.estimate_payload_state_normalized(obs)
    obs[self.critic_estimate_group] = estimate.detach()

  def estimator_objective(
    self,
    obs: TensorDict,
    target_group: str,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise mass and tray-frame payload position jointly."""
    if self.latest_payload_state_normalized is None:
      raise RuntimeError("The actor must run before its estimator objective.")
    target = obs[target_group]
    target_mass = target[:, :1]
    target_position_t = target[:, 1:]
    target_mass_normalized = (
      target_mass - self.payload_mass_min
    ) / self.payload_mass_span
    target_position_normalized = (
      target_position_t - self.payload_position_center_t
    ) / self.payload_position_half_span_t
    target_normalized = torch.cat(
      (
        target_mass_normalized.clamp(0.0, 1.0),
        target_position_normalized.clamp(-1.0, 1.0),
      ),
      dim=-1,
    )
    estimate = self.latest_payload_state_normalized
    estimator_loss = F.mse_loss(estimate, target_normalized)
    estimated_mass_kg = self.payload_mass_min + self.payload_mass_span * estimate[:, :1]
    estimated_position_t = (
      self.payload_position_center_t
      + self.payload_position_half_span_t * estimate[:, 1:]
    )
    mass_mae_kg = F.l1_loss(estimated_mass_kg, target_mass)
    position_mae_m = torch.linalg.vector_norm(
      estimated_position_t - target_position_t,
      dim=-1,
    ).mean()
    return estimator_loss, {
      "payload_state_estimator": estimator_loss,
      "mass_estimator_mae_kg": mass_mae_kg,
      "position_estimator_mae_m": position_mae_m,
    }

  def as_jit(self) -> nn.Module:
    """Return an exportable deterministic state-conditioned actor."""
    return _ExportablePayloadStateEstimator(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    """Return an ONNX wrapper with actor and history inputs."""
    del verbose
    return _ExportablePayloadStateEstimator(self)


class _ExportablePayloadStateEstimator(nn.Module):
  """Deployment wrapper exposing actions, mass, and tray-frame position."""

  is_recurrent: bool = False

  def __init__(self, model: PayloadStateEstimatorModel) -> None:
    super().__init__()
    self.actor_obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.estimator_obs_normalizer = copy.deepcopy(model.estimator_obs_normalizer)
    self.payload_state_estimator = copy.deepcopy(model.mass_estimator)
    self.policy = copy.deepcopy(model.mlp)
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None
      else nn.Identity()
    )
    self.actor_obs_dim = model.obs_dim
    self.estimator_obs_dim = model.estimator_obs_dim
    self.payload_mass_min = model.payload_mass_min
    self.payload_mass_span = model.payload_mass_span
    self.register_buffer(
      "payload_position_center_t", model.payload_position_center_t.detach().clone()
    )
    self.register_buffer(
      "payload_position_half_span_t",
      model.payload_position_half_span_t.detach().clone(),
    )

  def forward(
    self,
    actor_obs: torch.Tensor,
    proprioceptive_history: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return actions, estimated mass in kg, and position in metres."""
    actor_latent = self.actor_obs_normalizer(actor_obs)
    estimator_latent = self.estimator_obs_normalizer(proprioceptive_history)
    raw_estimate = self.payload_state_estimator(estimator_latent)
    estimate = torch.cat(
      (torch.sigmoid(raw_estimate[:, :1]), torch.tanh(raw_estimate[:, 1:])),
      dim=-1,
    )
    actions = self.deterministic_output(
      self.policy(torch.cat((actor_latent, estimate), dim=-1))
    )
    mass_kg = self.payload_mass_min + self.payload_mass_span * estimate[:, :1]
    position_t = (
      self.payload_position_center_t
      + self.payload_position_half_span_t * estimate[:, 1:]
    )
    return actions, mass_kg, position_t

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return representative inputs for ONNX tracing."""
    return (
      torch.zeros(1, self.actor_obs_dim),
      torch.zeros(1, self.estimator_obs_dim),
    )

  @property
  def input_names(self) -> list[str]:
    return ["actor_obs", "proprioceptive_history"]

  @property
  def output_names(self) -> list[str]:
    return [
      "actions",
      "estimated_payload_mass_kg",
      "estimated_payload_position_t_m",
    ]

  @torch.jit.export
  def reset(self) -> None:
    """Reset recurrent state (no-op for the feed-forward history encoder)."""


class StandalonePayloadStateEstimator(nn.Module):
  """Supervised payload-state estimator with no policy-gradient path.

  This module is intentionally independent from :class:`MLPModel`.  It can be
  trained from frozen-policy rollouts and later copied into a deployable actor.
  Mass and position use separate losses so the three position coordinates do
  not dilute the single mass target.
  """

  def __init__(
    self,
    input_dim: int,
    *,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    payload_mass_range: tuple[float, float] = (2.5, 30.0),
    payload_position_range_t: tuple[
      tuple[float, float],
      tuple[float, float],
      tuple[float, float],
    ] = ((-0.38, 0.38), (-0.49, 0.49), (-0.05, 0.40)),
  ) -> None:
    super().__init__()
    if input_dim <= 0:
      raise ValueError(f"input_dim must be positive, got {input_dim}.")
    mass_min, mass_max = payload_mass_range
    if mass_min < 0.0 or mass_max <= mass_min:
      raise ValueError(f"Invalid payload mass range: {payload_mass_range}.")
    position_min = torch.tensor(
      [bounds[0] for bounds in payload_position_range_t], dtype=torch.float32
    )
    position_max = torch.tensor(
      [bounds[1] for bounds in payload_position_range_t], dtype=torch.float32
    )
    if torch.any(position_max <= position_min):
      raise ValueError(f"Invalid payload position range: {payload_position_range_t}.")

    self.input_dim = input_dim
    self.hidden_dims = tuple(hidden_dims)
    self.activation = activation
    self.obs_normalizer = EmpiricalNormalization(input_dim)
    self.network = MLP(input_dim, 4, self.hidden_dims, activation)
    self.register_buffer("payload_mass_min", torch.tensor(float(mass_min)))
    self.register_buffer("payload_mass_span", torch.tensor(float(mass_max - mass_min)))
    self.register_buffer("payload_position_min_t", position_min)
    self.register_buffer("payload_position_max_t", position_max)
    self.register_buffer(
      "payload_position_center_t", 0.5 * (position_min + position_max)
    )
    self.register_buffer(
      "payload_position_half_span_t", 0.5 * (position_max - position_min)
    )

  @torch.no_grad()
  def update_normalization(self, observations: torch.Tensor) -> None:
    """Update input statistics from training environments only."""
    self.obs_normalizer.update(observations)

  def estimate_normalized(self, observations: torch.Tensor) -> torch.Tensor:
    """Return normalized mass in ``[0, 1]`` and position in ``[-1, 1]``."""
    raw_estimate = self.network(self.obs_normalizer(observations))
    return torch.cat(
      (torch.sigmoid(raw_estimate[:, :1]), torch.tanh(raw_estimate[:, 1:])),
      dim=-1,
    )

  def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return payload mass in kilograms and tray-frame position in metres."""
    estimate = self.estimate_normalized(observations)
    mass_kg = self.payload_mass_min + self.payload_mass_span * estimate[:, :1]
    position_t = (
      self.payload_position_center_t
      + self.payload_position_half_span_t * estimate[:, 1:]
    )
    return mass_kg, position_t

  def objective(
    self,
    observations: torch.Tensor,
    target: torch.Tensor,
    *,
    mass_loss_weight: float = 1.0,
    position_loss_weight: float = 1.0,
    huber_beta: float = 0.05,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute separately weighted normalized supervision and physical MAEs."""
    if target.shape[-1] != 4:
      raise ValueError(f"Expected a four-dimensional target, got {target.shape}.")
    estimate = self.estimate_normalized(observations)
    target_mass = target[:, :1]
    target_position_t = target[:, 1:]
    target_mass_normalized = (
      (target_mass - self.payload_mass_min) / self.payload_mass_span
    ).clamp(0.0, 1.0)
    target_position_normalized = (
      (target_position_t - self.payload_position_center_t)
      / self.payload_position_half_span_t
    ).clamp(-1.0, 1.0)

    mass_loss = F.smooth_l1_loss(
      estimate[:, :1], target_mass_normalized, beta=huber_beta
    )
    position_loss = F.smooth_l1_loss(
      estimate[:, 1:], target_position_normalized, beta=huber_beta
    )
    loss = mass_loss_weight * mass_loss + position_loss_weight * position_loss

    estimated_mass_kg = self.payload_mass_min + self.payload_mass_span * estimate[:, :1]
    estimated_position_t = (
      self.payload_position_center_t
      + self.payload_position_half_span_t * estimate[:, 1:]
    )
    return loss, {
      "loss": loss.detach(),
      "mass_loss": mass_loss.detach(),
      "position_loss": position_loss.detach(),
      "mass_mae_kg": F.l1_loss(estimated_mass_kg, target_mass).detach(),
      "position_mae_m": torch.linalg.vector_norm(
        estimated_position_t - target_position_t, dim=-1
      )
      .mean()
      .detach(),
    }


class PayloadMassEstimatorPPO(PPO):
  """PPO with supervised payload-mass estimation as an auxiliary objective."""

  def __init__(
    self,
    *args,
    estimator_loss_coef: float = 1.0,
    estimator_target_group: str = "payload_mass_target",
    **kwargs,
  ) -> None:
    if kwargs.get("symmetry_cfg") is not None:
      raise ValueError("Symmetry augmentation is not supported by estimator PPO.")
    super().__init__(*args, **kwargs)
    if not isinstance(self.actor, PayloadMassEstimatorModel):
      raise TypeError(
        "PayloadMassEstimatorPPO requires PayloadMassEstimatorModel as actor."
      )
    self.estimator_loss_coef = estimator_loss_coef
    self.estimator_target_group = estimator_target_group

  def _prepare_critic_observations(
    self,
    obs: TensorDict,
    *,
    use_latest: bool,
  ) -> None:
    """Hook for estimator variants that condition the critic."""
    del obs, use_latest

  def update(self) -> dict[str, float]:
    """Optimize PPO and the supervised estimator on the same rollout batches."""
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_estimator_metrics: dict[str, float] = {}
    mean_rnd_loss = 0.0 if self.rnd else None

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      assert batch.observations is not None
      original_batch_size = batch.observations.batch_size[0]
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          assert batch.advantages is not None
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      estimator_loss, estimator_metrics = self.actor.estimator_objective(
        batch.observations,
        self.estimator_target_group,
      )
      self._prepare_critic_observations(batch.observations, use_latest=True)

      assert batch.actions is not None
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      distribution_params = tuple(
        parameter[:original_batch_size]
        for parameter in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          assert batch.old_distribution_params is not None
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            learning_rate = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(learning_rate, src=0)
            self.learning_rate = learning_rate.item()
          for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate

      assert batch.advantages is not None
      assert batch.old_actions_log_prob is not None
      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      assert batch.values is not None
      assert batch.returns is not None
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
        + self.estimator_loss_coef * estimator_loss
      )
      rnd_loss = (
        self.rnd.compute_loss(batch.observations[:original_batch_size])
        if self.rnd
        else None
      )

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        assert rnd_loss is not None
        rnd_loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      for metric_name, metric_value in estimator_metrics.items():
        mean_estimator_metrics[metric_name] = (
          mean_estimator_metrics.get(metric_name, 0.0) + metric_value.item()
        )
      if mean_rnd_loss is not None:
        assert rnd_loss is not None
        mean_rnd_loss += rnd_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    losses = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
    }
    losses.update(
      {
        metric_name: metric_total / num_updates
        for metric_name, metric_total in mean_estimator_metrics.items()
      }
    )
    if mean_rnd_loss is not None:
      losses["rnd"] = mean_rnd_loss / num_updates
    self.storage.clear()
    return losses

  def compile(self, mode: str | None = None) -> None:
    """Compile actor and critic while retaining custom model type information."""
    self.actor = compile_model(self._raw_actor, mode)
    self.critic = compile_model(self._raw_critic, mode)

  def reduce_parameters(self) -> None:
    """Average actor, critic, and optional RND gradients across GPUs."""
    all_parameters = chain(self.actor.parameters(), self.critic.parameters())
    if self.rnd:
      all_parameters = chain(all_parameters, self.rnd.parameters())
    parameters = list(all_parameters)
    gradients = [
      parameter.grad.view(-1) for parameter in parameters if parameter.grad is not None
    ]
    all_gradients = torch.cat(gradients)
    torch.distributed.all_reduce(all_gradients, op=torch.distributed.ReduceOp.SUM)
    all_gradients /= self.gpu_world_size
    offset = 0
    for parameter in parameters:
      if parameter.grad is not None:
        count = parameter.numel()
        parameter.grad.data.copy_(
          all_gradients[offset : offset + count].view_as(parameter.grad.data)
        )
        offset += count


class PayloadStateEstimatorPPO(PayloadMassEstimatorPPO):
  """Estimator PPO that gives the same learned payload state to both networks."""

  def __init__(
    self,
    *args,
    estimator_target_group: str = "payload_state_target",
    **kwargs,
  ) -> None:
    super().__init__(
      *args,
      estimator_target_group=estimator_target_group,
      **kwargs,
    )
    if not isinstance(self.actor, PayloadStateEstimatorModel):
      raise TypeError(
        "PayloadStateEstimatorPPO requires PayloadStateEstimatorModel as actor."
      )

  def _prepare_critic_observations(
    self,
    obs: TensorDict,
    *,
    use_latest: bool,
  ) -> None:
    self.actor.inject_critic_estimate(obs, use_latest=use_latest)

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Estimate state once, then use it for the actor and critic rollout."""
    self.transition.hidden_states = (
      self.actor.get_hidden_state(),
      self.critic.get_hidden_state(),
    )
    self.transition.actions = self.actor(obs, stochastic_output=True).detach()
    self._prepare_critic_observations(obs, use_latest=True)
    self.transition.values = self.critic(obs).detach()
    self.transition.actions_log_prob = self.actor.get_output_log_prob(
      self.transition.actions
    ).detach()
    self.transition.distribution_params = tuple(
      parameter.detach() for parameter in self.actor.output_distribution_params
    )
    self.transition.observations = obs
    return self.transition.actions

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Inject the next-state estimate before normalizer updates."""
    with torch.no_grad():
      self._prepare_critic_observations(obs, use_latest=False)
    super().process_env_step(obs, rewards, dones, extras)

  def compute_returns(self, obs: TensorDict) -> None:
    """Bootstrap the critic from the estimator-conditioned final state."""
    with torch.no_grad():
      self._prepare_critic_observations(obs, use_latest=False)
    super().compute_returns(obs)
