import mujoco
from pal_mjlab.robots.pal_talos.talos_constants import (
  TALOS_PAYLOAD_BODY_NAME,
  TALOS_PAYLOAD_HALF_SIZE,
  TALOS_PAYLOAD_MASS,
  TALOS_PAYLOAD_PARENT_BODY_NAME,
  TALOS_PAYLOAD_POS,
  get_payload_spec,
)
from pal_mjlab.tasks.velocity.talos.env_cfgs import pal_talos_payload_flat_env_cfg
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


def test_payload_observations_are_added_to_actor_and_critic() -> None:
  cfg = pal_talos_payload_flat_env_cfg()

  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert terms["payload_com_pos_b"] is not None
    assert terms["payload_mass"] is not None

  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.spec_fn is get_payload_spec


def test_payload_training_uses_a_separate_experiment_directory() -> None:
  assert pal_talos_payload_ppo_runner_cfg().experiment_name == "talos_payload_velocity"
