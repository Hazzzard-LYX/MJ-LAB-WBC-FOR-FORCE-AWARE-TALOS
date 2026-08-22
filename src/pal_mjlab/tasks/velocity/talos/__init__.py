from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_talos_estimated_mass_tray_flat_env_cfg,
  pal_talos_estimated_mass_zero_joint_torque_tray_flat_env_cfg,
  pal_talos_estimated_payload_state_tray_flat_env_cfg,
  pal_talos_flat_env_cfg,
  pal_talos_free_payload_tray_flat_env_cfg,
  pal_talos_grasping_estimated_payload_state_tray_flat_env_cfg,
  pal_talos_grasping_tray_flat_env_cfg,
  pal_talos_grasping_wrist_ft_mass_identification_tray_flat_env_cfg,
  pal_talos_oracle_mass_tray_flat_env_cfg,
  pal_talos_payload_flat_env_cfg,
  pal_talos_random_mass_tray_flat_env_cfg,
  pal_talos_rough_env_cfg,
  pal_talos_tray_flat_env_cfg,
)
from .rl_cfg import (
  pal_talos_estimated_mass_tray_ppo_runner_cfg,
  pal_talos_estimated_mass_zero_joint_torque_tray_ppo_runner_cfg,
  pal_talos_estimated_payload_state_tray_ppo_runner_cfg,
  pal_talos_free_payload_tray_ppo_runner_cfg,
  pal_talos_grasping_payload_state_estimator_ppo_runner_cfg,
  pal_talos_grasping_random_mass_tray_ppo_runner_cfg,
  pal_talos_grasping_tray_ppo_runner_cfg,
  pal_talos_oracle_mass_tray_ppo_runner_cfg,
  pal_talos_payload_ppo_runner_cfg,
  pal_talos_ppo_runner_cfg,
  pal_talos_random_mass_tray_ppo_runner_cfg,
  pal_talos_tray_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pal-Talos",
  env_cfg=pal_talos_rough_env_cfg(),
  play_env_cfg=pal_talos_rough_env_cfg(play=True),
  rl_cfg=pal_talos_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos",
  env_cfg=pal_talos_flat_env_cfg(),
  play_env_cfg=pal_talos_flat_env_cfg(play=True),
  rl_cfg=pal_talos_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Payload",
  env_cfg=pal_talos_payload_flat_env_cfg(),
  play_env_cfg=pal_talos_payload_flat_env_cfg(play=True),
  rl_cfg=pal_talos_payload_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Tray",
  env_cfg=pal_talos_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray",
  env_cfg=pal_talos_free_payload_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_free_payload_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_free_payload_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray-Fixed-2p5kg",
  env_cfg=pal_talos_free_payload_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_free_payload_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_free_payload_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Critic",
  env_cfg=pal_talos_random_mass_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_random_mass_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_random_mass_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Estimator",
  env_cfg=pal_talos_estimated_mass_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_estimated_mass_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_estimated_mass_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Estimator-Zero-Torque",
  env_cfg=pal_talos_estimated_mass_zero_joint_torque_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_estimated_mass_zero_joint_torque_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_estimated_mass_zero_joint_torque_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-State-Estimator",
  env_cfg=pal_talos_estimated_payload_state_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_estimated_payload_state_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_estimated_payload_state_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Oracle",
  env_cfg=pal_talos_oracle_mass_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_oracle_mass_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_oracle_mass_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp",
  env_cfg=pal_talos_grasping_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_grasping_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_grasping_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-Random-Mass",
  env_cfg=pal_talos_grasping_tray_flat_env_cfg(randomize_payload_mass=True),
  play_env_cfg=pal_talos_grasping_tray_flat_env_cfg(
    play=True, randomize_payload_mass=True
  ),
  rl_cfg=pal_talos_grasping_random_mass_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-Random-Mass-State-Estimator"
  ),
  env_cfg=pal_talos_grasping_estimated_payload_state_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_grasping_estimated_payload_state_tray_flat_env_cfg(play=True),
  rl_cfg=pal_talos_grasping_payload_state_estimator_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-"
    "Random-Mass-Wrist-FT-Identification"
  ),
  env_cfg=pal_talos_grasping_wrist_ft_mass_identification_tray_flat_env_cfg(),
  play_env_cfg=pal_talos_grasping_wrist_ft_mass_identification_tray_flat_env_cfg(
    play=True
  ),
  rl_cfg=pal_talos_grasping_random_mass_tray_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
