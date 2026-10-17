"""Task resets must not spawn free objects inside other geometry.

Each case was a reset whose stiff contacts ejected objects on the first steps:
cheese graters spawned inside the inhand_transfer table, the chess tin inside
the board (and pieces inside the tin walls or each other), and a lying bottle's
cap inside the room wall. Requires the task assets (``prepare.py --sim``).
"""

import logging

import mujoco
import numpy as np
import pytest

import abc_sim
from abc_sim.initial_penetration import (
    free_object_penetrations,
    warn_on_free_object_penetrations,
)
from abc_sim.randomization.assets import inhand
from abc_sim.randomization.core import RandomizationState

PASSIVE_STEPS = 500  # 1 s at 2 ms
MAX_SPEED_M_S = 1.0


def _make_env(task: str):
    try:
        return abc_sim.make_env(task=task, render_cameras=False)
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"{task} assets unavailable: {error}")


def _max_free_body_speeds(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, float]:
    """Step from rest with the arms held where they are; peak speed per free body."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for actuator in range(model.nu):
        if model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT:
            joint = model.actuator_trnid[actuator, 0]
            data.ctrl[actuator] = data.qpos[model.jnt_qposadr[joint]]
    free = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint): int(model.jnt_dofadr[joint])
        for joint in range(model.njnt)
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
        and data.qpos[model.jnt_qposadr[joint] + 2] > -0.25
    }
    peak = dict.fromkeys(free, 0.0)
    for _ in range(PASSIVE_STEPS):
        mujoco.mj_step(model, data)
        for name, dof in free.items():
            peak[name] = max(peak[name], float(np.linalg.norm(data.qvel[dof : dof + 3])))
    return peak


def _assert_clean_spawn(env) -> None:
    penetrations = [item.describe() for item in free_object_penetrations(env.model, env.data)]
    assert penetrations == []
    peak = _max_free_body_speeds(env.model, env.data.qpos)
    assert {name: speed for name, speed in peak.items() if speed > MAX_SPEED_M_S} == {}


@pytest.fixture(scope="module")
def inhand_env():
    env = _make_env("inhand_transfer_item_to_other_side")
    env.reset(seed=1)
    return env


def _grater_variants() -> list[str]:
    try:
        return [path.name for path in inhand._inhand_get_variants("cheese_grater")]
    except FileNotFoundError:
        return []


@pytest.mark.parametrize("scale", [0.9, 1.2])
@pytest.mark.parametrize("variant", _grater_variants() or ["missing"])
def test_every_cheese_grater_rests_on_the_table(inhand_env, variant, scale):
    if variant == "missing":
        pytest.skip("inhand_transfer assets unavailable")
    # The legacy fixed spawn height; apply() must re-derive it from the mesh.
    state = RandomizationState(
        seed=0,
        object_states={"task_object_joint": {"pos": [0.6, 0.25, inhand._OBJ_Z], "quat": [1, 0, 0, 0]}},
        scale_states={"task_object_joint": scale},
        metadata={"category": "cheese_grater", "variant": variant, "side": "left"},
    )
    inhand_env._task_randomizer.apply(inhand_env.model, inhand_env.data, state)
    _assert_clean_spawn(inhand_env)


def test_inhand_reset_records_the_resting_height(inhand_env):
    for seed in (20260513, 20260541, 20260528):  # two graters, one water bottle
        inhand_env.reset(seed=seed)
        recorded = inhand_env._last_randomization.object_states["task_object_joint"]["pos"]
        joint = mujoco.mj_name2id(inhand_env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
        address = inhand_env.model.jnt_qposadr[joint]
        np.testing.assert_allclose(inhand_env.data.qpos[address : address + 3], recorded)
        _assert_clean_spawn(inhand_env)


@pytest.mark.parametrize(
    "seed",
    [
        # tin_setup: tin inside the board
        20260568, 20260590, 20260599,
        # tin_setup: pieces inside the tin walls (and the tin in the board)
        20260581, 20260602,
        # knocked_setup: knocked pieces inside each other
        20260517, 20260543, 20260549,
    ],
)
def test_chess_resets_without_overlaps(seed):
    env = _make_env("set_up_chess_pieces_on_board")
    env.reset(seed=seed)
    _assert_clean_spawn(env)


@pytest.mark.parametrize("seed", [20260552, 20260591, 20260602, 20260605, 20260516, 20260525])
def test_throw_bottles_spawn_outside_the_walls(seed):
    env = _make_env("throw_plastic_bottles_in_bin")
    env.reset(seed=seed)
    _assert_clean_spawn(env)


def test_reset_warns_about_a_spawn_overlap(inhand_env, caplog):
    joint = mujoco.mj_name2id(inhand_env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
    address = inhand_env.model.jnt_qposadr[joint]
    inhand_env.data.qpos[address + 2] -= 0.05
    with caplog.at_level(logging.WARNING, logger="abc_sim.initial_penetration"):
        found = warn_on_free_object_penetrations(inhand_env.model, inhand_env.data)
    assert found and found[0].other_geom == "table_plane"
    assert "Free object spawned inside other geometry" in caplog.text
