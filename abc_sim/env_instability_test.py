"""Real MuJoCo auto-reset is a terminal error, never a continuing episode."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_sim.env import MuJoCoYAMEnv


def make_env():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="nut" pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".01" mass=".01"/>
      </body></worldbody></mujoco>
    """)
    env = object.__new__(MuJoCoYAMEnv)
    env.model, env.data = model, mujoco.MjData(model)
    env._control_decimation = 17
    env._task_runtime = None
    return env


@pytest.mark.parametrize("time_s", [0.0, 7.65])
def test_real_bad_acceleration_auto_reset_raises_terminal_error(time_s):
    env = make_env()
    env.data.qpos[:3] = [0.52, 0.06, 0.76]
    env.data.time = time_s
    env.data.xfrc_applied[1, 3] = 1e30
    after_step_calls = []
    env._task_runtime = SimpleNamespace(
        before_step=lambda *_: None,
        after_step=lambda *_: after_step_calls.append(True),
    )
    with pytest.raises(mujoco.FatalError, match="mjWARN_BADQACC at DOF 3") as error:
        env.advance_physics()
    assert f"time {time_s:.9g} -> 0.002" in str(error.value)
    assert env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 1
    # MuJoCo reset silently; the environment catches it before runtime/scoring.
    np.testing.assert_allclose(env.data.qpos[:2], env.model.qpos0[:2])
    assert env.data.time == env.model.opt.timestep
    assert not after_step_calls


def test_normal_physics_steps_advance_without_false_instability():
    env = make_env()
    env.advance_physics(steps=100)
    assert env.data.time == pytest.approx(100 * env.model.opt.timestep)
    assert np.all(np.isfinite(env.data.qpos))
    for warning in (
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC,
    ):
        assert env.data.warning[warning].number == 0
