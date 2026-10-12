"""Committed opaque-box insertion ends failed worlds without full sim assets."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_minimal.batched_eval import over, world_eval
from abc_minimal.eval_policy import rollout_over, world_success_flags
from abc_sim.batched_env import PerWorldEvaluators
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.task_eval.count_box import CountIntoOpaqueBoxEvaluator
from abc_sim.task_specs import get_task_spec


def directive(target=1):
    return SimpleNamespace(
        metadata={
            "target_count": target,
            "eligible_objects": ["a", "b"],
            "objects": [{"name": name} for name in ("a", "b", "wrong")],
        }
    )


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="opaque_count_box" pos=".70 0 .75">
        <geom type="box" size=".15 .15 .01"/>
        <geom type="box" pos="-.16 0 .07" size=".01 .16 .08"/>
        <geom type="box" pos=".16 0 .07" size=".01 .16 .08"/>
        <geom type="box" pos="0 -.16 .07" size=".16 .01 .08"/>
        <geom type="box" pos="0 .16 .07" size=".16 .01 .08"/>
        <geom type="box" pos="0 0 .15" size=".16 .16 .01"/>
      </body>
      <body name="a" pos=".40 -.1 .78"><freejoint name="a_joint"/><geom type="sphere" size=".02"/></body>
      <body name="b" pos=".40 0 .78"><freejoint name="b_joint"/><geom type="sphere" size=".02"/></body>
      <body name="wrong" pos=".40 .1 .78"><freejoint name="wrong_joint"/><geom type="sphere" size=".02"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    spec = get_task_spec("count_into_opaque_box")
    evaluator = CountIntoOpaqueBoxEvaluator(model=model, spec=spec)
    evaluator.configure_from_randomization(model, directive())
    return model, data, evaluator


def put(model, qpos, name, *, inside=True, z=0.78):
    adr = int(model.joint(name + "_joint").qposadr[0])
    y = {"a": -0.08, "b": 0, "wrong": 0.08}[name]
    qpos[adr : adr + 3] = [0.70 if inside else 0.40, y, z]


def sample(scene):
    model, data, evaluator = scene
    mujoco.mj_forward(model, data)
    return evaluator.evaluate_qpos_batch(data.qpos[None]).to_info(squeeze=True)


@pytest.mark.parametrize(
    "names, reason",
    [
        (("a", "b"), "eligible_overfill"),
        (("wrong",), "ineligible_object"),
        (("a", "b", "wrong"), "overfill_and_ineligible"),
    ],
)
def test_invalid_insertion_latches_terminal_failure(scene, names, reason):
    model, data, _ = scene
    for name in names:
        put(model, data.qpos, name)
    result = sample(scene)
    assert result["terminal_failure"] and result["ever_failed"]
    assert result["failure_reason"] == reason
    assert not result["success"] and not result["ever_success"]
    assert rollout_over(result)
    assert world_success_flags(result) == (False, False)


def test_exact_count_then_overfill_stops_and_cannot_recover(scene):
    model, data, _ = scene
    put(model, data.qpos, "a")
    result = sample(scene)
    assert result["success"] and result["ever_correct"]
    assert not rollout_over(result)  # Correctness remains final-state semantics.
    put(model, data.qpos, "b")
    result = sample(scene)
    assert result["ever_correct"] and result["ever_failed"]
    assert rollout_over(result)
    put(model, data.qpos, "b", inside=False)
    result = sample(scene)
    assert result["num_eligible_in_box"] == 1
    assert result["ever_correct"] and result["ever_failed"]
    assert not result["success"] and not result["ever_success"]
    assert result["failure_reason"] == "eligible_overfill"
    assert result["reward"] == 0


def test_outside_and_on_lid_transient_is_not_committed_insertion(scene):
    model, data, evaluator = scene
    put(model, data.qpos, "a")
    for name in ("b", "wrong"):
        put(model, data.qpos, name, z=evaluator._box_ceiling_z + 0.03)
    result = sample(scene)
    assert result["success"] and not result["ever_failed"]
    for name in ("b", "wrong"):
        put(model, data.qpos, name, inside=False)
    assert not sample(scene)["ever_failed"]
    put(model, data.qpos, "b")
    assert sample(scene)["ever_failed"]


def test_failure_is_reset_for_new_episode(scene):
    model, data, evaluator = scene
    put(model, data.qpos, "wrong")
    assert sample(scene)["ever_failed"]
    evaluator.reset(nworld=1)
    mujoco.mj_resetData(model, data)
    result = sample(scene)
    assert not result["ever_failed"] and result["failure_reason"] == ""
    assert not result["ever_correct"]


def test_batched_worlds_stop_independently_and_freeze_failure_reason(scene):
    model, data, _ = scene
    evaluators = PerWorldEvaluators(model, get_task_spec("count_into_opaque_box"), 2)
    evaluators.reset(model, [directive(), directive(2)])
    qpos = np.repeat(data.qpos[None], 2, axis=0)
    keys = evaluators.per_world_keys(qpos)
    assert {"ever_failed", "terminal_failure", "failure_reason"} <= keys
    for name in ("a", "b"):
        put(model, qpos[0], name)
    put(model, qpos[1], "a")
    result = evaluators.evaluate_qpos_batch(qpos)
    np.testing.assert_array_equal(over(result, 2), [True, False])
    first = world_eval(result, 0, keys)
    second = world_eval(result, 1, keys)
    assert first["ever_failed"] and first["failure_reason"] == "eligible_overfill"
    assert not second["ever_failed"] and second["failure_reason"] == ""
    put(model, qpos[0], "wrong")
    assert (
        evaluators.evaluate_qpos_batch(qpos).metrics["failure_reason"][0]
        == "eligible_overfill"
    )


@pytest.mark.parametrize("reload_method", ["model", "xml"])
def test_model_reload_clears_failed_episode_and_reconfigures_directive(
    scene, tmp_path, reload_method
):
    model, data, _ = scene
    env = object.__new__(MuJoCoYAMEnv)
    env.model, env.data = model, data
    env._task_request = None
    env._physics_dt = 0.002
    env._render_cameras_flag = False
    env._camera_provider = None
    env._build_index_maps = lambda: None
    env.set_task("count_into_opaque_box")
    env._task_evaluator.configure_from_randomization(model, directive())
    put(model, data.qpos, "wrong")
    assert env.evaluate_task().metrics["ever_failed"][0]
    if reload_method == "model":
        env.reload_from_model(model)
    else:
        path = tmp_path / "scene.xml"
        mujoco.mj_saveLastXML(str(path), model)
        env.reload_from_xml(path.read_text())
    env._task_evaluator.configure_from_randomization(env.model, directive())
    result = env.evaluate_task().to_info(squeeze=True)
    assert result["goal_configured"]
    assert not result["ever_failed"] and result["failure_reason"] == ""
    put(env.model, env.data.qpos, "a")
    assert env.evaluate_task().scalar_success()


def test_unconfigured_result_keeps_per_world_failure_schema(scene):
    model, data, _ = scene
    evaluator = CountIntoOpaqueBoxEvaluator(
        model=model, spec=get_task_spec("count_into_opaque_box")
    )
    result = evaluator.evaluate_qpos_batch(np.repeat(data.qpos[None], 2, axis=0))
    assert not result.metrics["goal_configured"]
    assert result.metrics["ever_failed"].shape == (2,)
    assert result.metrics["failure_reason"].tolist() == ["", ""]
