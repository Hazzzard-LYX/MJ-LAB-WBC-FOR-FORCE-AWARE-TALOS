import math

import mujoco
import pytest
from pal_mjlab.robots.pal_talos.talos_constants import (
  TALOS_PAYLOAD_BODY_NAME,
  TALOS_PAYLOAD_HALF_SIZE,
  TALOS_PAYLOAD_MASS,
  TALOS_PAYLOAD_PARENT_BODY_NAME,
  TALOS_PAYLOAD_POS,
  get_payload_spec,
)
from pal_mjlab.tasks.velocity.talos.env_cfgs import (
  TALOS_PAYLOAD_ALPHA_RANGE,
  TALOS_PAYLOAD_MASS_RANGE,
  TALOS_PAYLOAD_POS_RANGES,
  pal_talos_payload_flat_env_cfg,
)
from pal_mjlab.tasks.velocity.talos.rl_cfg import pal_talos_payload_ppo_runner_cfg


def test_payload_is_a_fixed_child_of_torso() -> None:
  model = get_payload_spec().compile()
  payload_id = model.body(TALOS_PAYLOAD_BODY_NAME).id
  parent_id = model.body(TALOS_PAYLOAD_PARENT_BODY_NAME).id

  assert model.body_parentid[payload_id] == parent_id
  assert model.body_jntnum[payload_id] == 0
  assert model.body_mass[payload_id] == TALOS_PAYLOAD_MASS
  assert tuple(model.body_pos[payload_id]) == TALOS_PAYLOAD_POS

  geom = model.geom("front_payload_collision")
  assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
  assert tuple(model.geom_size[geom.id]) == TALOS_PAYLOAD_HALF_SIZE


def test_payload_state_is_privileged_to_the_critic() -> None:
  cfg = pal_talos_payload_flat_env_cfg()

  actor_terms = cfg.observations["actor"].terms
  assert "payload_com_pos_b" not in actor_terms
  assert "payload_mass" not in actor_terms

  critic_terms = cfg.observations["critic"].terms
  assert critic_terms["payload_com_pos_b"] is not None
  assert critic_terms["payload_mass"] is not None

  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.spec_fn is get_payload_spec


def test_payload_training_uses_a_separate_experiment_directory() -> None:
  assert pal_talos_payload_ppo_runner_cfg().experiment_name == "talos_payload_velocity"


def test_payload_domain_randomization_is_enabled() -> None:
  cfg = pal_talos_payload_flat_env_cfg()

  inertia = cfg.events["payload_inertia"]
  assert inertia.mode == "startup"
  assert inertia.params["alpha_range"] == TALOS_PAYLOAD_ALPHA_RANGE

  position = cfg.events["payload_position"]
  assert position.mode == "startup"
  assert position.params["operation"] == "abs"
  assert position.params["ranges"] == TALOS_PAYLOAD_POS_RANGES

  randomized_bounds = tuple(
    TALOS_PAYLOAD_MASS * math.exp(2.0 * alpha) for alpha in TALOS_PAYLOAD_ALPHA_RANGE
  )
  assert randomized_bounds == pytest.approx(TALOS_PAYLOAD_MASS_RANGE)
