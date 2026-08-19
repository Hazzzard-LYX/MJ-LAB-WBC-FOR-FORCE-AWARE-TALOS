"""RL configuration for PAL Robotics' Talos velocity task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from .env_cfgs import (
  TALOS_MASS_ESTIMATOR_HISTORY_LENGTH,
  TALOS_TRAY_PAYLOAD_MASS_RANGE,
  TALOS_TRAY_PAYLOAD_POSITION_RANGE_T,
)


@dataclass
class PayloadMassEstimatorModelCfg(RslRlModelCfg):
  """Configuration for the deployable estimator-conditioned actor."""

  class_name: str = (
    "pal_mjlab.tasks.velocity.talos.mass_estimation:PayloadMassEstimatorModel"
  )
  estimator_obs_group: str = "mass_estimator"
  estimator_hidden_dims: tuple[int, ...] = (256, 128)
  estimator_activation: str = "elu"
  estimator_obs_normalization: bool = True
  payload_mass_range: tuple[float, float] = TALOS_TRAY_PAYLOAD_MASS_RANGE


@dataclass
class PayloadMassEstimatorPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO configuration with a supervised payload-mass auxiliary loss."""

  class_name: str = (
    "pal_mjlab.tasks.velocity.talos.mass_estimation:PayloadMassEstimatorPPO"
  )
  estimator_loss_coef: float = 1.0
  estimator_target_group: str = "payload_mass_target"


@dataclass
class PayloadStateEstimatorModelCfg(PayloadMassEstimatorModelCfg):
  """Configuration for mass-and-position estimator conditioning."""

  class_name: str = (
    "pal_mjlab.tasks.velocity.talos.mass_estimation:PayloadStateEstimatorModel"
  )
  payload_position_range_t: tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
  ] = TALOS_TRAY_PAYLOAD_POSITION_RANGE_T
  critic_estimate_group: str = "estimated_payload_state"


@dataclass
class PayloadStateEstimatorPpoAlgorithmCfg(PayloadMassEstimatorPpoAlgorithmCfg):
  """PPO configuration sharing the learned state with actor and critic."""

  class_name: str = (
    "pal_mjlab.tasks.velocity.talos.mass_estimation:PayloadStateEstimatorPPO"
  )
  estimator_target_group: str = "payload_state_target"


def pal_talos_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PAL Talos velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="talos_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def pal_talos_payload_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a separate runner configuration for payload-conditioned policies."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.experiment_name = "talos_payload_velocity"
  return cfg


def pal_talos_tray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a separate runner configuration for tray-carrying policies."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.experiment_name = "talos_tray_velocity"
  return cfg


def pal_talos_free_payload_tray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a separate runner configuration for free payload tray transport."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.experiment_name = "talos_free_payload_tray_velocity"
  return cfg


def pal_talos_random_mass_tray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the critic-privileged random-payload baseline runner."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.experiment_name = "talos_random_mass_tray_critic_velocity"
  return cfg


def pal_talos_estimated_mass_tray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the learned payload-mass estimator experiment runner."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.actor = PayloadMassEstimatorModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  cfg.algorithm = PayloadMassEstimatorPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
  )
  cfg.obs_groups = {
    "actor": ("actor",),
    "critic": ("critic",),
    "mass_estimator": ("mass_estimator",),
  }
  cfg.experiment_name = (
    f"talos_random_mass_tray_estimator_h{TALOS_MASS_ESTIMATOR_HISTORY_LENGTH}"
  )
  return cfg


def pal_talos_estimated_mass_zero_joint_torque_tray_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """Create the estimator runner for the zero-joint-torque ablation."""
  cfg = pal_talos_estimated_mass_tray_ppo_runner_cfg()
  cfg.experiment_name = (
    f"talos_random_mass_tray_estimator_zero_joint_torque_h"
    f"{TALOS_MASS_ESTIMATOR_HISTORY_LENGTH}"
  )
  return cfg


def pal_talos_estimated_payload_state_tray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the torque-private mass-and-position estimator runner."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.actor = PayloadStateEstimatorModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  cfg.algorithm = PayloadStateEstimatorPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
  )
  cfg.obs_groups = {
    "actor": ("actor",),
    "critic": ("critic", "estimated_payload_state"),
    "mass_estimator": ("mass_estimator",),
  }
  cfg.experiment_name = (
    f"talos_random_mass_tray_state_estimator_h{TALOS_MASS_ESTIMATOR_HISTORY_LENGTH}"
  )
  cfg.max_iterations = 40_000
  return cfg


def pal_talos_oracle_mass_tray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the true-mass actor-observation upper-bound runner."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.experiment_name = "talos_random_mass_tray_oracle_velocity"
  return cfg


def pal_talos_grasping_tray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the staged fixed-light-payload contact-grasp runner."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.experiment_name = "talos_contact_grasp_tray_fixed_2p5kg"
  cfg.max_iterations = 40_000
  return cfg


def pal_talos_grasping_random_mass_tray_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the 2.5--30 kg continuation runner for a learned grasp."""
  cfg = pal_talos_ppo_runner_cfg()
  cfg.experiment_name = "talos_contact_grasp_tray_random_mass"
  cfg.max_iterations = 40_000
  return cfg


def pal_talos_grasping_payload_state_estimator_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  """Create the torque-private estimator runner for contact-only grasping."""
  cfg = pal_talos_estimated_payload_state_tray_ppo_runner_cfg()
  cfg.experiment_name = (
    f"talos_contact_grasp_tray_random_mass_state_estimator_h"
    f"{TALOS_MASS_ESTIMATOR_HISTORY_LENGTH}"
  )
  return cfg
