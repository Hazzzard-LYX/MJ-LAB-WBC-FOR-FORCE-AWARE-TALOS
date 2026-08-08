"""PAL Robotics Talos velocity tracking environment configurations."""

import math
from copy import deepcopy
from dataclasses import replace

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots import (
  TALOS_ACTION_SCALE,
  TALOS_FT_SITE_NAMES,
  TALOS_GRASPING_ACTION_SCALE,
  TALOS_GRIPPER_CONTACT_BODY_NAMES,
  TALOS_PAYLOAD_BODY_NAME,
  TALOS_TORQUE_SENSOR_JOINT_NAMES,
  TALOS_TRAY_BODY_NAME,
  TALOS_TRAY_PAYLOAD_BODY_NAME,
  TALOS_WRIST_FT_SITE_NAMES,
  get_talos_free_hand_tray_cfg,
  get_talos_free_tray_payload_cfg,
  get_talos_grasping_robot_cfg,
  get_talos_payload_robot_cfg,
  get_talos_robot_cfg,
  get_talos_tray_robot_cfg,
)
from pal_mjlab.tasks.velocity import mdp as pal_mdp

TALOS_PAYLOAD_MASS_RANGE = (2.0, 25.0)
TALOS_TRAY_PAYLOAD_MASS_RANGE = (2.5, 30.0)
# Bounds used to normalize the estimated payload COM position in the tray
# frame.  The horizontal limits match the payload-drop termination envelope;
# the vertical interval includes a settled cube and brief contact transients.
TALOS_TRAY_PAYLOAD_POSITION_RANGE_T = (
  (-0.38, 0.38),
  (-0.49, 0.49),
  (-0.05, 0.40),
)
TALOS_MASS_ESTIMATOR_HISTORY_LENGTH = 8
TALOS_MASS_ESTIMATOR_TERM_NAMES = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "imu_lin_acc",
  "joint_torque_sensors",
  "left_wrist_ft_force",
  "left_wrist_ft_torque",
  "right_wrist_ft_force",
  "right_wrist_ft_torque",
)
TALOS_PAYLOAD_POS_RANGES = {
  0: (0.20, 0.55),
  1: (-0.20, 0.20),
  2: (0.05, 0.30),
}

# dr.pseudo_inertia scales mass by exp(2 * alpha). Convert the desired
# absolute mass bounds into alpha bounds relative to the nominal 10 kg cube.
TALOS_PAYLOAD_ALPHA_RANGE = (
  0.5 * math.log(TALOS_PAYLOAD_MASS_RANGE[0] / 10.0),
  0.5 * math.log(TALOS_PAYLOAD_MASS_RANGE[1] / 10.0),
)
TALOS_TRAY_PAYLOAD_ALPHA_RANGE = (
  0.5 * math.log(TALOS_TRAY_PAYLOAD_MASS_RANGE[0] / 2.5),
  0.5 * math.log(TALOS_TRAY_PAYLOAD_MASS_RANGE[1] / 2.5),
)


def _sample_uniform_mass_alpha(
  lower: torch.Tensor,
  upper: torch.Tensor,
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  """Sample pseudo-inertia alpha values that produce uniform physical mass."""
  lower_scale = torch.exp(2.0 * lower)
  upper_scale = torch.exp(2.0 * upper)
  mass_scale = lower_scale + (upper_scale - lower_scale) * torch.rand(
    shape, device=device
  )
  return 0.5 * torch.log(mass_scale)


TALOS_UNIFORM_MASS_DISTRIBUTION = dr.Distribution(
  name="uniform_physical_mass",
  sample=_sample_uniform_mass_alpha,
)

TALOS_FORCE_SENSOR_NAMES = tuple(
  f"robot/{site_name}_force" for site_name in TALOS_FT_SITE_NAMES
)
TALOS_TORQUE_SENSOR_NAMES = tuple(
  f"robot/{site_name}_torque" for site_name in TALOS_FT_SITE_NAMES
)
TALOS_WRIST_FORCE_SENSOR_NAMES = TALOS_FORCE_SENSOR_NAMES[:2]
TALOS_WRIST_TORQUE_SENSOR_NAMES = TALOS_TORQUE_SENSOR_NAMES[:2]


def _talos_force_torque_sensor_cfgs() -> tuple[BuiltinSensorCfg, ...]:
  sensors: list[BuiltinSensorCfg] = []
  for site_name in TALOS_FT_SITE_NAMES:
    site_ref = ObjRef(type="site", name=site_name, entity="robot")
    sensors.extend(
      (
        BuiltinSensorCfg(
          name=f"{site_name}_force",
          sensor_type="force",
          obj=site_ref,
        ),
        BuiltinSensorCfg(
          name=f"{site_name}_torque",
          sensor_type="torque",
          obj=site_ref,
        ),
      )
    )
  return tuple(sensors)


def _add_talos_hardware_observations(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """Install the fixed, real-hardware-compatible TALOS actor contract."""
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  max_sensor_lag = 0 if play else 1

  # Existing encoder and IMU/state-estimator channels gain bounded latency and
  # saturation in the actor only.  The critic remains instantaneous and clean.
  sensor_limits = {
    "base_lin_vel": (-10.0, 10.0),
    "base_ang_vel": (-20.0, 20.0),
    "projected_gravity": (-1.0, 1.0),
    "joint_pos": (-4.0, 4.0),
    "joint_vel": (-50.0, 50.0),
  }
  for term_name, clip in sensor_limits.items():
    actor_terms[term_name] = replace(
      actor_terms[term_name],
      clip=clip,
      delay_min_lag=0,
      delay_max_lag=max_sensor_lag,
      delay_hold_prob=0.8,
      delay_update_period=5,
    )

  actor_terms["imu_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
    clip=(-100.0, 100.0),
    scale=0.1,
    delay_min_lag=0,
    delay_max_lag=max_sensor_lag,
    delay_hold_prob=0.8,
    delay_update_period=5,
  )
  critic_terms["imu_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
    scale=0.1,
  )

  instrumented_joints = SceneEntityCfg(
    "robot",
    joint_names=TALOS_TORQUE_SENSOR_JOINT_NAMES,
    preserve_order=True,
  )
  actor_terms["joint_torque_sensors"] = ObservationTermCfg(
    func=pal_mdp.joint_torque_sensor,
    params={"asset_cfg": instrumented_joints},
    noise=Unoise(n_min=-1.0, n_max=1.0),
    clip=(-450.0, 450.0),
    scale=0.01,
    delay_min_lag=0,
    delay_max_lag=max_sensor_lag,
    delay_hold_prob=0.8,
    delay_update_period=5,
  )
  critic_terms["joint_torque_sensors"] = ObservationTermCfg(
    func=pal_mdp.joint_torque_sensor,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=TALOS_TORQUE_SENSOR_JOINT_NAMES,
        preserve_order=True,
      )
    },
    scale=0.01,
  )

  for site_name in TALOS_FT_SITE_NAMES:
    for sensor_type, noise, clip, scale in (
      ("force", 2.0, (-2000.0, 2000.0), 0.01),
      ("torque", 0.2, (-300.0, 300.0), 0.05),
    ):
      term_name = f"{site_name}_{sensor_type}"
      sensor_name = f"robot/{term_name}"
      actor_terms[term_name] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": sensor_name},
        noise=Unoise(n_min=-noise, n_max=noise),
        clip=clip,
        scale=scale,
        delay_min_lag=0,
        delay_max_lag=max_sensor_lag,
        delay_hold_prob=0.8,
        delay_update_period=5,
      )
      critic_terms[term_name] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": sensor_name},
        scale=scale,
      )


def pal_talos_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics Talos rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()

  # TALOS can generate substantially more contacts than the generic velocity
  # environment, especially with full-body self-collision enabled. Let
  # MuJoCo-Warp size its contact buffer from the compiled model and retain more
  # contact-sensor matches for reliable batched simulation.
  cfg.sim.nconmax = None
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.mujoco.ccd_iterations = 500

  cfg.scene.entities = {"robot": get_talos_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = ("left_foot_collision", "right_foot_collision")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(leg_left_6_link|leg_right_6_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  body_ground_cfg = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(leg_left_4_link|leg_right_4_link|torso_2_link|arm_left_7_link|arm_right_7_link|arm_left_5_link|arm_right_5_link|)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),  # Set per-robot: frame and pattern.
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )
  cfg.scene.sensors = (
    feet_ground_cfg,
    self_collision_cfg,
    body_ground_cfg,
    foot_height_scan,
    *_talos_force_torque_sensor_cfgs(),
  )
  _add_talos_hardware_observations(cfg, play=play)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = TALOS_ACTION_SCALE

  cfg.viewer.body_name = "torso_2_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["actor"].terms["height_scan"] = None
  cfg.observations["critic"].terms["height_scan"] = None

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_2_link",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r"leg_.*_3_.*": 0.3,  # pitch
    r"leg_.*_2_.*": 0.15,  # roll
    r"leg_.*_1_.*": 0.15,
    r"leg_.*_4_.*": 0.35,  # knee
    r"leg_.*_5_.*": 0.25,
    r"leg_.*_6_.*": 0.1,
    # Waist.
    r".*torso_2.*": 0.1,  # pitch
    r".*torso_1.*": 0.2,  # yaw
    r".*head.*": 0.1,
    # Arms.
    r"arm_.*_1_.*": 0.15,  # yaw
    r"arm_.*_2_.*": 0.15,  # roll
    r"arm_.*_3_.*": 0.1,  # yaw
    r"arm_.*_4_.*": 0.15,  # elbow
    r"arm_.*_5_.*": 0.1,  # elbow
    r"arm_.*_6_.*": 0.1,  # wrist
    r"arm_.*_7_.*": 0.2,  # wrist
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r"leg_.*_3_.*": 0.5,  # pitch
    r"leg_.*_2_.*": 0.2,  # roll
    r"leg_.*_1_.*": 0.2,
    r"leg_.*_4_.*": 0.6,
    r"leg_.*_5_.*": 0.35,
    r"leg_.*_6_.*": 0.15,
    # Waist.
    r".*torso_2.*": 0.2,  # pitch
    r".*torso_1.*": 0.3,  # yaw
    r".*head.*": 0.1,
    # Arms.
    r"arm_.*_1_.*": 0.2,  # yaw
    r"arm_.*_2_.*": 0.2,  # roll
    r"arm_.*_3_.*": 0.1,  # yaw
    r"arm_.*_4_.*": 0.35,  # elbow
    r"arm_.*_5_.*": 0.1,  # elbow
    r"arm_.*_6_.*": 0.1,  # wrist
    r"arm_.*_7_.*": 0.2,  # wrist
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_2_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_2_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  cfg.terminations["illegal_contacts"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "body_ground_contact"},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def pal_talos_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Talos flat terrain velocity configuration."""
  cfg = pal_talos_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def pal_talos_payload_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create flat velocity tracking with a rigid payload fixed to TALOS's torso."""
  cfg = pal_talos_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_talos_payload_robot_cfg()}

  payload_cfg = SceneEntityCfg(
    "robot",
    body_names=(TALOS_PAYLOAD_BODY_NAME,),
  )
  payload_terms = {
    "payload_com_pos_b": ObservationTermCfg(
      func=pal_mdp.payload_com_pos_b,
      params={"asset_cfg": payload_cfg},
    ),
    "payload_mass": ObservationTermCfg(
      func=pal_mdp.payload_mass,
      params={"asset_cfg": payload_cfg},
      clip=(0.0, 50.0),
    ),
  }
  cfg.observations["critic"].terms.update(payload_terms)

  # Sample one physically consistent payload variant per parallel environment.
  # Parameters stay fixed within an environment, avoiding costly inertial model
  # recomputation at every episode reset while covering the distribution densely
  # during large batched training.
  cfg.events["payload_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": payload_cfg,
      "alpha_range": TALOS_PAYLOAD_ALPHA_RANGE,
    },
  )
  cfg.events["payload_position"] = EventTermCfg(
    mode="startup",
    func=dr.body_pos,
    params={
      "asset_cfg": payload_cfg,
      "operation": "abs",
      "ranges": TALOS_PAYLOAD_POS_RANGES,
    },
  )

  return cfg


def pal_talos_tray_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create flat velocity tracking with an empty tray fixed to both wrists."""
  cfg = pal_talos_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_talos_tray_robot_cfg()}
  return cfg


def pal_talos_free_payload_tray_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create tray transport with a completely free cube placed at its center."""
  cfg = pal_talos_tray_flat_env_cfg(play=play)
  cfg.scene.entities["payload"] = get_talos_free_tray_payload_cfg()

  tray_cfg = SceneEntityCfg("robot", body_names=(TALOS_TRAY_BODY_NAME,))
  payload_cfg = SceneEntityCfg("payload", body_names=(TALOS_TRAY_PAYLOAD_BODY_NAME,))

  # Keep the robot at its calibrated carrying pose so the independently reset
  # payload starts exactly over the tray in every parallel environment.
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["reset_base"].params["velocity_range"] = {}
  cfg.events["reset_payload_on_tray"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_root_state_uniform,
    params={
      "asset_cfg": payload_cfg,
      "pose_range": {},
      "velocity_range": {},
    },
  )

  payload_terms = {
    "payload_pos_t": ObservationTermCfg(
      func=pal_mdp.payload_pos_t,
      params={"tray_cfg": tray_cfg, "payload_cfg": payload_cfg},
    ),
    "payload_relative_velocity_t": ObservationTermCfg(
      func=pal_mdp.payload_relative_velocity_t,
      params={"tray_cfg": tray_cfg, "payload_cfg": payload_cfg},
      clip=(-20.0, 20.0),
    ),
    "payload_mass": ObservationTermCfg(
      func=pal_mdp.payload_mass,
      params={"asset_cfg": payload_cfg},
      clip=(0.0, 50.0),
    ),
  }
  cfg.observations["critic"].terms.update(payload_terms)

  wrist_sensor_site_cfg = SceneEntityCfg(
    "robot",
    site_names=TALOS_WRIST_FT_SITE_NAMES,
    preserve_order=True,
  )
  transport_tracking_gate = {
    "tracking_command_name": "twist",
    "tracking_std": 0.5,
    "tracking_min_factor": 0.1,
  }
  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["planar_velocity_tracking_error"] = RewardTermCfg(
    func=pal_mdp.planar_velocity_tracking_error,
    weight=-2.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["torso_height"] = RewardTermCfg(
    func=pal_mdp.torso_height,
    weight=-0.5,
    params={"z_des": 1.0, "std": 0.1},
  )
  cfg.rewards["tray_level"] = RewardTermCfg(
    func=pal_mdp.tray_level_reward,
    weight=2.0,
    params={
      "std": 0.25,
      "tray_cfg": tray_cfg,
      **transport_tracking_gate,
    },
  )
  cfg.rewards["payload_position_on_tray"] = RewardTermCfg(
    func=pal_mdp.payload_position_on_tray_reward,
    weight=3.0,
    params={
      "std": 0.12,
      "desired_pos_t": (0.0, 0.0, 0.1325),
      "tray_cfg": tray_cfg,
      "payload_cfg": payload_cfg,
      **transport_tracking_gate,
    },
  )
  cfg.rewards["payload_relative_motion"] = RewardTermCfg(
    func=pal_mdp.payload_relative_motion_reward,
    weight=2.0,
    params={
      "lin_vel_std": 0.5,
      "ang_vel_std": 1.0,
      "tray_cfg": tray_cfg,
      "payload_cfg": payload_cfg,
      **transport_tracking_gate,
    },
  )
  cfg.rewards["wrist_load_balance"] = RewardTermCfg(
    func=pal_mdp.wrist_load_balance_reward,
    weight=1.0,
    params={
      "std": 0.25,
      "sensor_site_cfg": wrist_sensor_site_cfg,
      "force_sensor_names": TALOS_WRIST_FORCE_SENSOR_NAMES,
      "torque_sensor_names": TALOS_WRIST_TORQUE_SENSOR_NAMES,
      **transport_tracking_gate,
    },
  )
  cfg.rewards["tray_tipping_moment"] = RewardTermCfg(
    func=pal_mdp.tray_tipping_moment_reward,
    weight=1.0,
    params={
      "std": 20.0,
      "tray_cfg": tray_cfg,
      "sensor_site_cfg": SceneEntityCfg(
        "robot",
        site_names=TALOS_WRIST_FT_SITE_NAMES,
        preserve_order=True,
      ),
      "force_sensor_names": TALOS_WRIST_FORCE_SENSOR_NAMES,
      "torque_sensor_names": TALOS_WRIST_TORQUE_SENSOR_NAMES,
      **transport_tracking_gate,
    },
  )
  cfg.rewards["wrist_force_rate"] = RewardTermCfg(
    func=pal_mdp.wrist_force_rate_penalty,
    weight=-0.02,
    params={
      "force_sensor_names": TALOS_WRIST_FORCE_SENSOR_NAMES,
      "max_force_rate": 5000.0,
    },
  )
  cfg.terminations["payload_dropped"] = TerminationTermCfg(
    func=pal_mdp.payload_dropped,
    params={
      "min_z_t": -0.05,
      "max_abs_x_t": 0.38,
      "max_abs_y_t": 0.49,
      "tray_cfg": tray_cfg,
      "payload_cfg": payload_cfg,
    },
  )
  return cfg


def pal_talos_random_mass_tray_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the critic-privileged random-mass tray transport baseline."""
  cfg = pal_talos_free_payload_tray_flat_env_cfg(play=play)
  payload_cfg = SceneEntityCfg("payload", body_names=(TALOS_TRAY_PAYLOAD_BODY_NAME,))
  cfg.events["payload_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": payload_cfg,
      "alpha_range": TALOS_TRAY_PAYLOAD_ALPHA_RANGE,
      "distribution": TALOS_UNIFORM_MASS_DISTRIBUTION,
    },
  )
  return cfg


def pal_talos_estimated_mass_tray_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create random-mass transport with a deployable learned mass estimator."""
  cfg = pal_talos_random_mass_tray_flat_env_cfg(play=play)
  actor_group = cfg.observations["actor"]
  estimator_terms = {
    name: deepcopy(actor_group.terms[name]) for name in TALOS_MASS_ESTIMATOR_TERM_NAMES
  }
  cfg.observations["mass_estimator"] = ObservationGroupCfg(
    terms=estimator_terms,
    concatenate_terms=True,
    enable_corruption=actor_group.enable_corruption,
    history_length=TALOS_MASS_ESTIMATOR_HISTORY_LENGTH,
    flatten_history_dim=True,
  )

  payload_cfg = SceneEntityCfg("payload", body_names=(TALOS_TRAY_PAYLOAD_BODY_NAME,))
  cfg.observations["payload_mass_target"] = ObservationGroupCfg(
    terms={
      "payload_mass": ObservationTermCfg(
        func=pal_mdp.payload_mass,
        params={"asset_cfg": payload_cfg},
        clip=TALOS_TRAY_PAYLOAD_MASS_RANGE,
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg


def pal_talos_estimated_mass_zero_joint_torque_tray_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the estimator experiment with deployable joint torque fixed to zero."""
  cfg = pal_talos_estimated_mass_tray_flat_env_cfg(play=play)
  for group_name in ("actor", "mass_estimator"):
    torque_term = cfg.observations[group_name].terms["joint_torque_sensors"]
    cfg.observations[group_name].terms["joint_torque_sensors"] = replace(
      torque_term,
      func=pal_mdp.zero_joint_torque_sensor,
      noise=None,
    )
  return cfg


def pal_talos_estimated_payload_state_tray_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Estimate payload mass and tray-frame position from torque-aware history.

  Joint torque is available only to the estimator.  Actor and critic consume
  the estimator's normalized four-dimensional output instead of torque or
  simulator payload state.  Ground-truth mass and position remain isolated in
  a training-only target group for the supervised auxiliary objective.
  """
  cfg = pal_talos_estimated_mass_tray_flat_env_cfg(play=play)

  # The estimator group was deep-copied before these terms are removed, so it
  # retains noisy deployable torque history while actor and critic do not.
  cfg.observations["actor"].terms.pop("joint_torque_sensors")
  cfg.observations["critic"].terms.pop("joint_torque_sensors")

  # Do not let the critic bypass the estimator through privileged object state.
  for term_name in (
    "payload_pos_t",
    "payload_relative_velocity_t",
    "payload_mass",
  ):
    cfg.observations["critic"].terms.pop(term_name)

  payload_cfg = SceneEntityCfg("payload", body_names=(TALOS_TRAY_PAYLOAD_BODY_NAME,))
  tray_cfg = SceneEntityCfg("robot", body_names=(TALOS_TRAY_BODY_NAME,))
  cfg.observations.pop("payload_mass_target")
  cfg.observations["payload_state_target"] = ObservationGroupCfg(
    terms={
      "payload_mass": ObservationTermCfg(
        func=pal_mdp.payload_mass,
        params={"asset_cfg": payload_cfg},
        clip=TALOS_TRAY_PAYLOAD_MASS_RANGE,
      ),
      "payload_pos_t": ObservationTermCfg(
        func=pal_mdp.payload_pos_t,
        params={"tray_cfg": tray_cfg, "payload_cfg": payload_cfg},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["estimated_payload_state"] = ObservationGroupCfg(
    terms={
      "estimated_payload_state": ObservationTermCfg(
        func=pal_mdp.zero_payload_state,
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg


def pal_talos_oracle_mass_tray_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create random-mass transport with true payload mass exposed to the actor."""
  cfg = pal_talos_random_mass_tray_flat_env_cfg(play=play)
  payload_cfg = SceneEntityCfg("payload", body_names=(TALOS_TRAY_PAYLOAD_BODY_NAME,))
  cfg.observations["actor"].terms["payload_mass_oracle"] = ObservationTermCfg(
    func=pal_mdp.payload_mass,
    params={"asset_cfg": payload_cfg},
    clip=TALOS_TRAY_PAYLOAD_MASS_RANGE,
    scale=1.0 / TALOS_TRAY_PAYLOAD_MASS_RANGE[1],
  )
  return cfg


def pal_talos_grasping_tray_flat_env_cfg(
  play: bool = False,
  randomize_payload_mass: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Learn contact-based tray grasping and velocity tracking.

  Unlike the legacy tray task, the tray is a separate free body and neither
  wrist is welded to it.  The actor receives only deployable robot channels;
  tray pose, slip, contacts, and payload state are privileged training signals.
  """
  cfg = pal_talos_free_payload_tray_flat_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_talos_grasping_robot_cfg()
  cfg.scene.entities["tray"] = get_talos_free_hand_tray_cfg()

  tray_cfg = SceneEntityCfg("tray", body_names=(TALOS_TRAY_BODY_NAME,))
  payload_cfg = SceneEntityCfg("payload", body_names=(TALOS_TRAY_PAYLOAD_BODY_NAME,))
  grasp_site_cfg = SceneEntityCfg(
    "robot",
    site_names=("left_grasp_center", "right_grasp_center"),
    preserve_order=True,
  )
  handle_site_cfg = SceneEntityCfg(
    "tray",
    site_names=(
      "hand_tray_left_handle_grasp",
      "hand_tray_right_handle_grasp",
    ),
    preserve_order=True,
  )

  # Repoint inherited transport terms from the old robot-attached tray to the
  # independent tray entity.
  for term_name in (
    "payload_pos_t",
    "payload_relative_velocity_t",
  ):
    cfg.observations["critic"].terms[term_name].params["tray_cfg"] = tray_cfg
  for reward_name in (
    "tray_level",
    "payload_position_on_tray",
    "payload_relative_motion",
    "tray_tipping_moment",
  ):
    cfg.rewards[reward_name].params["tray_cfg"] = tray_cfg
  cfg.terminations["payload_dropped"].params["tray_cfg"] = tray_cfg

  cfg.events["reset_free_tray"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_root_state_uniform,
    params={
      "asset_cfg": tray_cfg,
      "pose_range": {},
      "velocity_range": {},
    },
  )
  cfg.events["gripper_overgrip_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_grasp_collision",)),
      "operation": "abs",
      "ranges": (1.0, 2.0),
      "shared_random": True,
    },
  )
  cfg.events["tray_overgrip_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "tray", geom_names=("hand_tray_.*_handle_collision",)
      ),
      "operation": "abs",
      "ranges": (1.0, 2.0),
      "shared_random": True,
    },
  )
  if randomize_payload_mass:
    cfg.events["payload_inertia"] = EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": payload_cfg,
        "alpha_range": TALOS_TRAY_PAYLOAD_ALPHA_RANGE,
        "distribution": TALOS_UNIFORM_MASS_DISTRIBUTION,
      },
    )

  grasp_contact_cfg = ContactSensorCfg(
    name="gripper_tray_contact",
    primary=ContactMatch(
      mode="body",
      pattern=TALOS_GRIPPER_CONTACT_BODY_NAMES,
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="body",
      pattern=TALOS_TRAY_BODY_NAME,
      entity="tray",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=cfg.decimation,
  )
  cfg.scene.sensors = (*cfg.scene.sensors, grasp_contact_cfg)

  # Do not expose passive distal-link coordinates or simulated tray state to
  # the actor.  The commanded main gripper encoders are available on hardware.
  deployable_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=(
      "arm_.*_joint",
      "leg_.*_joint",
      "head_.*_joint",
      "torso_.*_joint",
      "gripper_(left|right)_joint",
    ),
    preserve_order=True,
  )
  locomotion_joint_cfg = SceneEntityCfg(
    "robot",
    joint_names=(
      "arm_.*_joint",
      "leg_.*_joint",
      "head_.*_joint",
      "torso_.*_joint",
    ),
    preserve_order=True,
  )
  for group_name in ("actor", "critic"):
    for term_name in ("joint_pos", "joint_vel"):
      cfg.observations[group_name].terms[term_name].params["asset_cfg"] = (
        deployable_joint_cfg
      )
  cfg.rewards["pose"].params["asset_cfg"] = locomotion_joint_cfg
  cfg.events["reset_robot_joints"].params["asset_cfg"] = locomotion_joint_cfg
  cfg.events["reset_gripper_joints"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_joints_by_offset,
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=("gripper_.*_joint",)),
    },
  )

  cfg.observations["critic"].terms.update(
    {
      "tray_handle_offsets_w": ObservationTermCfg(
        func=pal_mdp.tray_handle_offsets_w,
        params={
          "grasp_site_cfg": grasp_site_cfg,
          "handle_site_cfg": handle_site_cfg,
        },
        clip=(-0.5, 0.5),
      ),
      "tray_handle_relative_velocity_w": ObservationTermCfg(
        func=pal_mdp.tray_handle_relative_velocity_w,
        params={
          "grasp_site_cfg": grasp_site_cfg,
          "handle_site_cfg": handle_site_cfg,
        },
        clip=(-5.0, 5.0),
      ),
      "tray_projected_gravity": ObservationTermCfg(
        func=pal_mdp.tray_projected_gravity,
        params={"tray_cfg": tray_cfg},
      ),
    }
  )

  cfg.rewards["bilateral_gripper_contact"] = RewardTermCfg(
    func=pal_mdp.bilateral_gripper_contact_reward,
    weight=4.0,
    # The three-finger linkage normally loads two or more collision links per
    # side around the cylindrical handle; do not require every distal mesh to
    # touch simultaneously.
    params={"sensor_name": grasp_contact_cfg.name, "contacts_per_hand": 2},
  )
  cfg.rewards["tray_grasp_pose"] = RewardTermCfg(
    func=pal_mdp.tray_grasp_pose_reward,
    weight=6.0,
    params={
      "std": 0.06,
      "grasp_site_cfg": grasp_site_cfg,
      "handle_site_cfg": handle_site_cfg,
    },
  )
  cfg.rewards["tray_grasp_slip"] = RewardTermCfg(
    func=pal_mdp.tray_grasp_slip_reward,
    weight=3.0,
    params={
      "std": 0.25,
      "grasp_site_cfg": grasp_site_cfg,
      "handle_site_cfg": handle_site_cfg,
    },
  )
  cfg.terminations["tray_grasp_lost"] = TerminationTermCfg(
    func=pal_mdp.tray_grasp_lost,
    params={
      "max_handle_error": 0.16,
      # A low absolute floor catches a truly dropped tray.  Normal lowering of
      # both loaded arms is handled by posture/height rewards, not mislabeled
      # as loss of grasp while the handles remain between the fingers.
      "min_tray_height": 0.30,
      # One PPO rollout is 0.48 s.  Keep at least one complete rollout of dense
      # grasp feedback before failed exploratory actions can end the episode.
      "grace_period_s": 0.75,
      "grasp_site_cfg": grasp_site_cfg,
      "handle_site_cfg": handle_site_cfg,
      "tray_cfg": tray_cfg,
    },
  )
  # Contact-rich gripper exploration can occasionally drive one batched world
  # into a non-finite MuJoCo state.  Reset only that world before corrupted
  # physics reaches the next policy observation.  RewardManager sanitizes the
  # terminal-step reward, and the observation policy below is a final backstop
  # for derived sensor channels.
  cfg.terminations["nan_state"] = TerminationTermCfg(func=mdp.nan_detection)
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].nan_policy = "sanitize"
    cfg.observations[group_name].nan_check_per_term = False

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = TALOS_GRASPING_ACTION_SCALE

  # Phase 1 establishes a bilateral grasp while standing; later stages add
  # walking and turning without removing the grasp-related learning signal.
  if not play:
    assert cfg.curriculum is not None
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (0.0, 0.0),
            "lin_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
          },
          {
            "step": 3000 * 24,
            "lin_vel_x": (-0.15, 0.35),
            "lin_vel_y": (-0.10, 0.10),
            "ang_vel_z": (-0.15, 0.15),
          },
          {
            "step": 8000 * 24,
            "lin_vel_x": (-0.4, 0.8),
            "lin_vel_y": (-0.25, 0.25),
            "ang_vel_z": (-0.35, 0.35),
          },
          {
            "step": 15000 * 24,
            "lin_vel_x": (-0.8, 1.2),
            "lin_vel_y": (-0.40, 0.40),
            "ang_vel_z": (-0.5, 0.5),
          },
        ],
      },
    )
  return cfg
