"""Exercise contact-gated countdown scoring with real, asset-free MuJoCo data."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_sim.env import MuJoCoYAMEnv
from abc_sim.task_eval.ball_tray import BallTrayBalancingEvaluator
from abc_sim.task_runtime.ball_tray import BallTrayBalancingRuntime
from abc_sim.task_specs import get_task_spec


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="tray"><freejoint/>
        <site name="ball_tray_center_site"/>
        <geom name="ball_tray_left_handle_collision" type="sphere" size=".015" pos="-.16 0 0"/>
        <geom name="ball_tray_right_handle_collision" type="sphere" size=".015" pos=".16 0 0"/>
      </body>
      <body name="ball_tray_ball" pos="0 0 .022">
        <freejoint name="ball_tray_ball_joint"/><geom type="sphere" size=".022"/>
      </body>
      <body name="left_link_left_finger" mocap="true" pos="-.16 0 .02">
        <geom type="sphere" size=".015"/>
      </body>
      <body name="right_link_left_finger" mocap="true" pos=".16 0 .02">
        <geom type="sphere" size=".015"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    runtime = BallTrayBalancingRuntime()
    runtime.bind(model, data)
    runtime.after_reset(
        model,
        data,
        randomization=SimpleNamespace(
            metadata={"balance_duration_s": 1},
            seed=42,
        ),
    )
    evaluator = BallTrayBalancingEvaluator(
        model=model, spec=get_task_spec("ball_tray_balancing")
    )
    evaluator.bind_runtime(runtime)
    return model, data, runtime, evaluator


def sample(scene, time_s, *, held=True):
    model, data, runtime, evaluator = scene
    data.time = time_s
    data.mocap_pos[:] = (
        [[-0.16, 0, 0.02], [0.16, 0, 0.02]] if held else [[1, 1, 1], [1, -1, 1]]
    )
    mujoco.mj_forward(model, data)
    runtime.after_step(model, data)
    return evaluator.evaluate_model_data(model, data).to_info(squeeze=True)


def test_passive_tray_never_succeeds(scene):
    for time_s in (0, 1, 12, 120):
        result = sample(scene, time_s, held=False)
        assert result["current_balanced"]
        assert not result["success"]
        assert not result["ever_success"]
    assert result["success_semantics"] == "held_countdown_complete"


def test_sustained_grasp_completes_the_countdown(scene):
    assert not sample(scene, 0)["success"]
    assert scene[2].debug_state()["both_posts_gripped"]
    assert not sample(scene, 0.99)["ever_success"]
    assert sample(scene, 1)["ever_success"]


def test_release_resets_countdown_and_regrasp_needs_full_duration(scene):
    sample(scene, 0)
    sample(scene, 0.5)
    assert not sample(scene, 0.7, held=False)["success"]
    assert not scene[2].debug_state()["started"]
    assert not sample(scene, 1.2)["success"]
    assert not sample(scene, 2.19)["success"]
    assert sample(scene, 2.2)["success"]


def test_contact_flicker_within_grace_keeps_countdown(scene):
    sample(scene, 0)
    sample(scene, 0.9)
    assert sample(scene, 1, held=False)["success"]


@pytest.mark.parametrize("offset, position", [(0, 0.3), (2, 0.0)])
def test_ball_failure_cannot_be_repaired_for_success(scene, offset, position):
    sample(scene, 0)
    _model, data, _, evaluator = scene
    original_position = data.qpos[evaluator._ball_qpos_addr + offset]
    data.qpos[evaluator._ball_qpos_addr + offset] = position
    assert sample(scene, 0.5)["ever_failed"]
    data.qpos[evaluator._ball_qpos_addr + offset] = original_position
    result = sample(scene, 1)
    assert result["countdown_complete"]
    assert result["current_balanced"]
    assert not result["ever_success"]


def test_two_grippers_touching_the_same_handle_do_not_start(scene):
    model, data, runtime, evaluator = scene
    data.mocap_pos[:] = [[-0.16, 0, 0.02], [-0.16, 0, 0.02]]
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    runtime.after_step(model, data)
    assert not runtime.debug_state()["both_posts_gripped"]
    assert not runtime.debug_state()["started"]
    assert not evaluator.evaluate_model_data(model, data).scalar_success()


def test_qpos_without_runtime_does_not_earn_a_pass(scene):
    model, data, _, _ = scene
    evaluator = BallTrayBalancingEvaluator(
        model=model, spec=get_task_spec("ball_tray_balancing")
    )
    result = evaluator.evaluate_qpos_batch(data.qpos[None])
    assert not np.any(result.success)
    assert not np.any(result.metrics["ever_success"])


def test_environment_binds_runtime_to_its_evaluator(scene):
    model, data, _, _ = scene
    env = object.__new__(MuJoCoYAMEnv)
    env.model, env.data = model, data
    env._task_request = None
    env.set_task("ball_tray_balancing")
    assert env._task_evaluator._runtime is env._task_runtime
    env._task_runtime.after_reset(model, data)
    assert not env.evaluate_task().scalar_success()


@pytest.mark.parametrize("reload_method", ["model", "xml"])
def test_environment_rebinds_runtime_after_model_reload(scene, tmp_path, reload_method):
    model, data, _, _ = scene
    env = object.__new__(MuJoCoYAMEnv)
    env.model, env.data = model, data
    env._task_request = None
    env._physics_dt = 0.002
    env._render_cameras_flag = False
    env._camera_provider = None
    # This minimal model intentionally has no YAM arm joints.
    env._build_index_maps = lambda: None
    env.set_task("ball_tray_balancing")
    if reload_method == "model":
        env.reload_from_model(model)
    else:
        xml_path = tmp_path / "scene.xml"
        mujoco.mj_saveLastXML(str(xml_path), model)
        env.reload_from_xml(xml_path.read_text())
    assert env._task_evaluator._runtime is env._task_runtime
    env._task_runtime.after_reset(
        env.model,
        env.data,
        randomization=SimpleNamespace(
            metadata={"balance_duration_s": 1},
            seed=42,
        ),
    )
    case = env.model, env.data, env._task_runtime, env._task_evaluator
    assert not sample(case, 0)["success"]
    assert sample(case, 1)["success"]
