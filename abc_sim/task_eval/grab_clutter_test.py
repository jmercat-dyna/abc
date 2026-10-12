"""Physical containment, irreversible insertion and actual per-world timing."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_sim.batched_env import BatchedWarpYAMEnv, PerWorldEvaluators
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.grab_clutter import GrabClutterTargetInBoxEvaluator
from abc_sim.task_specs import get_task_spec


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="grab_clutter_target_box"><freejoint/>
        <geom name="grab_clutter_target_box_floor" type="box" size=".075 .070 .006" pos="0 0 .006"/>
        <geom type="box" size=".005 .070 .052" pos=".075 0 .052"/>
        <geom type="box" size=".005 .070 .052" pos="-.075 0 .052"/>
        <geom type="box" size=".075 .005 .052" pos="0 .070 .052"/>
        <geom type="box" size=".075 .005 .052" pos="0 -.070 .052"/>
      </body>
      <body name="target_object" pos="0 0 .022"><freejoint name="target_object_joint"/>
        <geom name="target" type="box" size=".01 .01 .01"/>
      </body>
      <body name="clutter_obj_01" pos=".3 0 .022"><freejoint name="clutter_obj_01_joint"/>
        <geom type="box" size=".01 .01 .01"/>
      </body>
      <body name="left_link_left_finger" mocap="true" pos="1 1 1">
        <geom type="sphere" size=".01"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    evaluator = GrabClutterTargetInBoxEvaluator(
        model=model, spec=get_task_spec("grab_specific_object_from_clutter")
    )
    mujoco.mj_forward(model, data)
    return model, data, evaluator


def score(scene, time, *, target=None, wrong=None, held=False):
    model, data, evaluator = scene
    for index, position in enumerate((target, wrong)):
        if position is not None:
            address = evaluator._addresses[index]
            data.qpos[address : address + 3] = position
    data.time = time
    data.mocap_pos[0] = (
        data.qpos[evaluator._addresses[0] : evaluator._addresses[0] + 3] + [0, 0, 0.015]
        if held
        else [1, 1, 1]
    )
    mujoco.mj_forward(model, data)
    return evaluator.evaluate_model_data(model, data).to_info(squeeze=True)


def test_released_supported_placement_waits_exact_fixed_deadline(scene):
    assert score(scene, 7)["settling_deadline_s"] == 10
    assert not score(scene, 9.999)["success"]
    assert score(scene, 10)["success"]


def test_wrong_object_failure_is_immediate_irreversible_even_with_correct_target(scene):
    result = score(scene, 0, wrong=[0.025, 0, 0.05])
    assert result["terminal_failure"] and not result["success"]
    assert result["failure_reason"] == "wrong_object_in_box"
    result = score(scene, 5, wrong=[0.3, 0, 0.022])
    assert result["terminal_failure"]
    assert result["wrong_objects_ever_in_box"] == ["clutter_obj_01"]
    assert not result["wrong_objects_in_box"]


@pytest.mark.parametrize(
    "position", [[0, 0, 0.115], [0.1, 0, 0.05], [0, 0.08, 0.05], [0, 0, -0.02]]
)
def test_wrong_objects_on_rim_above_outside_or_under_box_do_not_fail(scene, position):
    assert not score(scene, 0, wrong=position)["terminal_failure"]


def test_correct_held_or_hovering_target_cannot_start_deadline(scene):
    assert np.isnan(score(scene, 0, held=True)["settling_started_s"])
    assert np.isnan(score(scene, 2, target=[0, 0, 0.07])["settling_started_s"])
    assert score(scene, 4, target=[0, 0, 0.022])["settling_deadline_s"] == 7


def test_removal_does_not_restart_first_deadline_and_fails_when_expired(scene):
    score(scene, 0)
    assert not score(scene, 2, target=[0.3, 0, 0.022])["terminal_failure"]
    result = score(scene, 3)
    assert (
        result["terminal_failure"]
        and result["failure_reason"] == "target_missing_after_settling"
    )
    assert not score(scene, 5, target=[0, 0, 0.022])["success"]


def test_target_can_return_before_fixed_deadline(scene):
    score(scene, 0)
    score(scene, 1, target=[0.3, 0, 0.022])
    assert score(scene, 3, target=[0, 0, 0.022])["success"]


def test_clockless_calls_never_manufacture_settling_time(scene):
    _, data, evaluator = scene
    for _ in range(100):
        result = evaluator.evaluate_qpos_batch(data.qpos[None])
    assert not result.scalar_success()
    assert np.isnan(result.metrics["settling_started_s"][0])
    assert not result.metrics["clock_available"]
    evaluator.evaluate_qpos_batch_at_time(data.qpos[None], [10])
    assert evaluator.evaluate_qpos_batch_at_time(data.qpos[None], [13]).scalar_success()


def test_per_world_actual_time_different_placement_times(scene):
    model, data, _ = scene
    evaluators = PerWorldEvaluators(
        model, get_task_spec("grab_specific_object_from_clutter"), 2
    )
    poses = np.repeat(data.qpos[None], 2, axis=0)
    evaluators.evaluate_qpos_batch(poses, time_s=[2, 9])
    result = evaluators.evaluate_qpos_batch(poses, time_s=[5, 11])
    assert result.success.tolist() == [True, False]
    assert result.metrics["settling_deadline_s"].tolist() == [5, 12]
    assert evaluators.evaluate_qpos_batch(poses, time_s=[5, 12]).success.all()


def test_clock_backward_is_rejected_and_reset_clears_failure(scene):
    _, _, evaluator = scene
    score(scene, 5, wrong=[0, 0, 0.05])
    with pytest.raises(ValueError, match="backward"):
        score(scene, 4)
    evaluator.reset()
    assert not score(scene, 0, wrong=[0.3, 0, 0.022])["terminal_failure"]
    assert score(scene, 3)["success"]


def test_rotated_translated_crate_uses_collision_local_coordinates(scene):
    _, data, evaluator = scene
    data.qpos[:3] = [1, 2, 0.5]
    angle = np.pi / 4
    data.qpos[3:7] = [np.cos(angle), 0, 0, np.sin(angle)]
    assert not score(scene, 0, target=[1, 2, 0.522], wrong=[1, 2.03, 0.55])["success"]
    assert evaluator._ever_failed[0]


def test_real_warp_clock_forwarding_and_old_evaluator_compatibility(scene):
    _, data, evaluator = scene
    env = BatchedWarpYAMEnv.__new__(BatchedWarpYAMEnv)
    env.qpos = data.qpos[None]
    env._evaluator = evaluator
    env._runtime = SimpleNamespace(
        d_warp=SimpleNamespace(time=SimpleNamespace(numpy=lambda: np.asarray([4.0])))
    )
    assert env.evaluate().metrics["settling_deadline_s"].tolist() == [7]
    env._runtime.d_warp.time.numpy = lambda: np.asarray([7.0])
    assert env.evaluate().scalar_success()

    class OldEvaluator:
        def evaluate_qpos_batch(self, qpos):
            return TaskEvalResult(
                np.zeros(len(qpos)), np.zeros(len(qpos), dtype=bool), {}
            )

    env._evaluator = OldEvaluator()
    assert not env.evaluate().scalar_success()


def test_tall_target_with_center_above_rim_can_be_placed_and_contact_penetration_is_supported(
    scene, tmp_path
):
    model, data, evaluator = scene
    model.geom_size[model.geom("target").id, 2] = 0.1
    path = tmp_path / "tall.xml"
    mujoco.mj_saveLastXML(str(path), model)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    evaluator.configure_from_randomization(model, {})
    scene = model, data, evaluator
    result = score(scene, 2, target=[0, 0, 0.1119])
    assert result["target_placed"]
    assert result["settling_deadline_s"] == 5
    assert score(scene, 5)["success"]


def test_configurable_settling_and_invalid_clock_do_not_corrupt_history(scene):
    _, _, evaluator = scene
    evaluator.settling_s = 0.5
    assert not score(scene, 1)["success"]
    assert score(scene, 1.5)["success"]
    evaluator.reset()
    with pytest.raises(ValueError, match="finite"):
        score(scene, np.nan, wrong=[0, 0, 0.05])
    assert not evaluator._ever_failed.any()


def test_per_world_probe_does_not_start_clock_or_leave_failure_history(scene):
    model, data, _ = scene
    evaluators = PerWorldEvaluators(
        model, get_task_spec("grab_specific_object_from_clutter"), 2
    )
    evaluators.per_world_keys(data.qpos[None])
    assert np.isnan(evaluators.evaluators[0]._first_placement_s).all()
    poses = np.repeat(data.qpos[None], 2, axis=0)
    result = evaluators.evaluate_qpos_batch(poses, time_s=[10, 20])
    assert result.metrics["settling_deadline_s"].tolist() == [13, 23]


def test_per_world_time_dispatch_preserves_old_evaluators(scene):
    model, data, _ = scene
    evaluators = PerWorldEvaluators(
        model, get_task_spec("grab_specific_object_from_clutter"), 2
    )

    class OldEvaluator:
        def evaluate_qpos_batch(self, qpos):
            return TaskEvalResult(
                np.ones(len(qpos)), np.zeros(len(qpos), dtype=bool), {}
            )

    evaluators.evaluators = [OldEvaluator(), OldEvaluator()]
    result = evaluators.evaluate_qpos_batch(
        np.repeat(data.qpos[None], 2, axis=0), time_s=[5, 50]
    )
    assert result.reward.tolist() == [1, 1]


@pytest.mark.parametrize(
    "kind,size", [("sphere", ".01"), ("capsule", ".01 .02"), ("cylinder", ".01 .02")]
)
def test_round_collision_support_uses_exact_rotated_extents(
    scene, tmp_path, kind, size
):
    import xml.etree.ElementTree as ET

    model, _, evaluator = scene
    path = tmp_path / "round.xml"
    mujoco.mj_saveLastXML(str(path), model)
    tree = ET.parse(path)
    geom = tree.find(".//geom[@name='target']")
    geom.set("type", kind)
    geom.set("size", size)
    tree.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    evaluator.configure_from_randomization(model, {})
    address = evaluator._addresses[0]
    angle = np.pi / 8
    data.qpos[address + 3 : address + 7] = [np.cos(angle), np.sin(angle), 0, 0]
    extent = (
        0.01
        if kind == "sphere"
        else 0.01 + 0.02 / np.sqrt(2)
        if kind == "capsule"
        else 0.03 / np.sqrt(2)
    )
    data.qpos[address + 2] = 0.012 + extent - 0.000001
    mujoco.mj_forward(model, data)
    result = evaluator.evaluate_model_data(model, data)
    assert result.metrics["object_bottom_box_frame"][0, 0, 2] == pytest.approx(
        0.012 - 0.000001
    )
    assert result.metrics["target_placed"][0]
