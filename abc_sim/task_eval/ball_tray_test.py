"""Exercise contact-gated countdown scoring with real, asset-free MuJoCo data."""

import xml.etree.ElementTree as ET
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
      <geom name="table_plane" type="plane" size="1 1 .01" pos="0 0 -.008"/>
      <body name="tray"><freejoint/>
        <site name="ball_tray_center_site"/>
        <geom name="ball_tray_floor_collision" type="box" size=".153 .162 .004" pos="0 0 -.004"/>
        <geom name="ball_tray_left_handle_collision" type="sphere" size=".015" pos="-.16 0 0"/>
        <geom name="ball_tray_right_handle_collision" type="sphere" size=".015" pos=".16 0 0"/>
      </body>
      <body name="ball_tray_ball" pos="0 0 .022">
        <freejoint name="ball_tray_ball_joint"/><geom name="ball_tray_ball_geom" type="sphere" size=".022"/>
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


def sample(scene, time_s, *, held=True, lifted=True):
    model, data, runtime, evaluator = scene
    data.time = time_s
    tray_height = 0.02 if lifted else 0.0
    data.qpos[evaluator._ball_qpos_addr + 2] += tray_height - data.qpos[2]
    data.qpos[2] = tray_height
    data.mocap_pos[:] = (
        [[-0.16, 0, tray_height + 0.02], [0.16, 0, tray_height + 0.02]]
        if held
        else [[1, 1, 1], [1, -1, 1]]
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
    assert result["success_semantics"] == "held_lifted_countdown_then_survive_timeout"


def test_sustained_grasp_completes_the_countdown(scene):
    assert not sample(scene, 0)["success"]
    assert scene[2].debug_state()["both_posts_gripped"]
    assert not sample(scene, 0.99)["ever_success"]
    assert sample(scene, 1)["ever_success"]


def test_handles_without_lifting_never_start_countdown(scene):
    for time_s in (0, 1, 12, 120):
        result = sample(scene, time_s, lifted=False)
        assert result["handles_held"]
        assert not result["tray_lifted"]
        assert result["tray_table_contact"]
        assert not result["countdown_complete"]
        assert not result["success"]


def test_returning_tray_to_table_requires_new_lifted_countdown(scene):
    sample(scene, 0)
    assert not sample(scene, 0.8, lifted=False)["success"]
    assert not scene[2].debug_state()["started"]
    assert not sample(scene, 1)["success"]
    assert not sample(scene, 1.99)["success"]
    assert sample(scene, 2)["success"]


def test_completed_countdown_allows_setdown_but_later_loss_revokes_success(scene):
    sample(scene, 0)
    assert sample(scene, 1)["countdown_qualified"]
    settled = sample(scene, 2, held=False, lifted=False)
    assert settled["success"] and settled["ever_success"]
    assert not settled["countdown_complete"]
    assert not settled["timed_out"]
    scene[1].qpos[scene[3]._ball_qpos_addr] = 0.3
    lost = sample(scene, 3, held=False, lifted=False)
    assert lost["countdown_qualified"]
    assert lost["ever_failed"]
    assert not lost["success"] and not lost["ever_success"]


@pytest.mark.parametrize("start_s, expected", [(29.0, True), (29.1, False)])
def test_late_countdown_must_finish_within_timeout(scene, start_s, expected):
    sample(scene, start_s)
    result = sample(scene, 31)
    assert result["timed_out"]
    assert result["countdown_qualified"] == expected
    assert result["success"] == expected


def test_timeout_and_lift_configuration_resets_between_episodes(scene):
    model, data, runtime, evaluator = scene
    mujoco.mj_forward(model, data)
    runtime.after_reset(
        model,
        data,
        randomization=SimpleNamespace(
            metadata={"timeout_s": 2, "min_lift_height_m": 0.03}, seed=0
        ),
    )
    assert not sample(scene, 0)["tray_lifted"]
    assert sample(scene, 2)["timed_out"]
    data.qpos[2] = 0
    data.qpos[evaluator._ball_qpos_addr + 2] = 0.022
    data.time = 10
    mujoco.mj_forward(model, data)
    runtime.after_reset(model, data)
    assert runtime.debug_state()["timeout_s"] == 30
    assert runtime.debug_state()["min_lift_height_m"] == 0.001
    assert not sample(scene, 39.9)["timed_out"]
    assert sample(scene, 40)["timed_out"]


@pytest.mark.parametrize(
    "options",
    [
        {"min_lift_height_m": 0},
        {"min_lift_height_m": float("nan")},
        {"timeout_s": -1},
        {"timeout_s": float("inf")},
    ],
)
def test_runtime_rejects_invalid_lift_and_timeout_options(scene, options):
    with pytest.raises(ValueError):
        BallTrayBalancingRuntime(**options)
    model, data, runtime, _ = scene
    with pytest.raises(ValueError):
        runtime.after_reset(
            model, data, randomization=SimpleNamespace(metadata=options)
        )


def test_height_gain_alone_cannot_earn_countdown_with_remaining_table_contact(
    scene, tmp_path
):
    model, _, runtime, evaluator = scene
    # An elevated table still physically supports the lifted tray.
    path = tmp_path / "elevated_table.xml"
    mujoco.mj_saveLastXML(str(path), model)
    xml = ET.fromstring(path.read_text())
    xml.find("worldbody/geom[@name='table_plane']").set("pos", "0 0 .01201")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    runtime.bind(model, data)
    runtime.after_reset(model, data)
    result = sample((model, data, runtime, evaluator), 0)
    assert result["tray_height_gain_m"] >= 0.001
    assert result["tray_table_contact"]
    assert not result["tray_lifted"]
    assert not runtime.debug_state()["started"]


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


@pytest.mark.parametrize("scale, yaw", [(1.0, 0.0), (1.2, 0.7), (0.8, -0.6)])
def test_physical_floor_bounds_are_distinct_from_safe_region_live_and_batch(
    scene, scale, yaw
):
    model, data, _, evaluator = scene
    floor = evaluator._floor_geom_id
    model.geom_size[floor] *= scale
    model.geom_quat[floor] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    model.geom_size[evaluator._ball_geom_id, 0] *= scale
    evaluator._usable_half_extents *= scale
    mujoco.mj_forward(model, data)
    rotation = data.geom_xmat[floor].reshape(3, 3).copy()
    center = data.geom_xpos[floor].copy()
    half = model.geom_size[floor].copy()
    radius = model.geom_size[evaluator._ball_geom_id, 0]
    # A sphere on the physical edge remains valid even outside the centering site.
    inside = center + rotation @ [half[0] * 0.99, half[1] * 0.8, half[2] + radius]
    data.qpos[evaluator._ball_qpos_addr : evaluator._ball_qpos_addr + 3] = inside
    mujoco.mj_forward(model, data)
    live = evaluator.evaluate_model_data(model, data)
    assert live.metrics["inside_tray"][0]
    assert not live.metrics["inside_safe_region"][0]
    assert not live.metrics["ever_failed"][0]
    np.testing.assert_allclose(
        live.metrics["physical_floor_half_extents_m"][0], half[:2]
    )
    batch = evaluator.evaluate_qpos_batch(data.qpos[None])
    assert batch.metrics["inside_tray"][0]
    assert not batch.metrics["ever_failed"][0]
    # Center-outside is loss even while the sphere still overlaps the edge.
    outside = center + rotation @ [half[0] + radius / 2, 0, half[2] + radius]
    data.qpos[evaluator._ball_qpos_addr : evaluator._ball_qpos_addr + 3] = outside
    mujoco.mj_forward(model, data)
    assert evaluator.evaluate_model_data(model, data).metrics["ever_failed"][0]
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).metrics["ever_failed"][0]


def test_drop_uses_actual_floor_plane_and_scaled_sphere(scene):
    model, data, _, evaluator = scene
    floor = evaluator._floor_geom_id
    model.geom_pos[floor, 2] = 0.1
    model.geom_size[evaluator._ball_geom_id, 0] = 0.03
    top = 0.1 + model.geom_size[floor, 2]
    # A sphere underneath the tray cannot qualify through cap overlap.
    for height, dropped in [(top + 0.03, False), (top, True), (top - 0.001, True)]:
        data.qpos[evaluator._ball_qpos_addr + 2] = height
        mujoco.mj_forward(model, data)
        live = evaluator.evaluate_model_data(model, data)
        assert bool(live.metrics["ball_dropped"][0]) == dropped
        batch = evaluator.evaluate_qpos_batch(data.qpos[None])
        assert bool(batch.metrics["ball_dropped"][0]) == dropped


def test_environment_binds_runtime_to_its_evaluator(scene):
    model, data, _, _ = scene
    env = object.__new__(MuJoCoYAMEnv)
    env.model, env.data = model, data
    env._task_request = None
    env.set_task("ball_tray_balancing")
    assert env._task_evaluator._runtime is env._task_runtime
    env._task_runtime.after_reset(model, data)
    assert not env.evaluate_task().scalar_success()


@pytest.mark.parametrize("terminate_on_success", [False, True])
@pytest.mark.parametrize(
    "qualified, failed, timed_out",
    [
        (True, False, False),
        (True, True, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
    ],
)
def test_gym_episode_status_respects_ball_timeout_and_terminal_failure(
    terminate_on_success, qualified, failed, timed_out
):
    env = object.__new__(MuJoCoYAMEnv)
    env.terminate_on_success = terminate_on_success
    env.max_episode_steps = None
    env.cur_step = 1
    info = {
        "task_success": qualified and not failed,
        "task_eval": {
            "success_semantics": "held_lifted_countdown_then_survive_timeout",
            "ever_failed": failed,
            "timed_out": timed_out,
        },
    }
    terminated, truncated = env._episode_status(info)
    assert terminated == (failed or (terminate_on_success and qualified and timed_out))
    assert truncated == (timed_out and not terminated)


def test_gym_terminal_failure_stops_count_box_without_success_opt_in():
    env = object.__new__(MuJoCoYAMEnv)
    env.terminate_on_success = False
    env.max_episode_steps = None
    env.cur_step = 1
    assert env._episode_status(
        {
            "task_success": False,
            "task_eval": {
                "success_semantics": "final_state_until_terminal_failure",
                "ever_failed": True,
            },
        }
    ) == (True, False)


@pytest.mark.parametrize("stop_reason", ["timeout", "loss"])
def test_gym_chunk_stops_on_real_ball_deadline_or_loss(scene, stop_reason):
    model, data, runtime, evaluator = scene
    sample(scene, 0)
    sample(scene, 1)
    env = object.__new__(MuJoCoYAMEnv)
    env.model, env.data = model, data
    env._task_evaluator = evaluator
    env._task_runtime = runtime
    env.terminate_on_success = True
    env.max_episode_steps = None
    env.cur_step = 0
    env.chunk_dim = 4
    env.single_timestep_action_dim = 14

    def apply_action(_action, *, render_obs):
        env.cur_step += 1
        if stop_reason == "loss" and env.cur_step == 2:
            data.qpos[evaluator._ball_qpos_addr] = 0.3
        sample(scene, 28 + env.cur_step)
        return {"state": data.qpos.copy()}

    env._apply_single_action = apply_action
    _, history, _, terminated, truncated, info = env.step_chunk(np.zeros((4, 14)))
    assert env.cur_step == 2
    assert len(history["state"]) == 2
    if stop_reason == "loss":
        assert terminated and not truncated and info["task_eval"]["ever_failed"]
    else:
        assert terminated and not truncated and info["task_eval"]["timed_out"]


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
