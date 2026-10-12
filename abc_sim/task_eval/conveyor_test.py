"""Conveyor losses require a genuine lifted grasp, with independent episode/world latches."""

import copy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_sim.task_eval.conveyor import ConveyorPickObjectsInBinEvaluator
from abc_sim.task_runtime.conveyor_pick import ConveyorPickRuntime
from abc_sim.task_specs import get_task_spec


def scene(*, yaw=0, scale=1, gripper_qpos=False):
    finger_attribute = "" if gripper_qpos else 'mocap="true"'
    model = mujoco.MjModel.from_xml_string(f'''
    <mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>
      <body name="conveyor_belt" euler="0 0 {yaw}">
        <geom name="conveyor_belt_surface" type="box" size="{0.1 * scale} {0.5 * scale} {0.01 * scale}"/>
      </body>
      <body name="conveyor_bin" pos="1 0 0" euler="0 0 {yaw}">
        <geom type="box" size="{0.15 * scale} {0.15 * scale} {0.01 * scale}"/>
        <geom type="box" pos="{-0.16 * scale} 0 {0.08 * scale}" size="{0.01 * scale} {0.16 * scale} {0.08 * scale}"/>
        <geom type="box" pos="{0.16 * scale} 0 {0.08 * scale}" size="{0.01 * scale} {0.16 * scale} {0.08 * scale}"/>
        <geom type="box" pos="0 {-0.16 * scale} {0.08 * scale}" size="{0.16 * scale} {0.01 * scale} {0.08 * scale}"/>
        <geom type="box" pos="0 {0.16 * scale} {0.08 * scale}" size="{0.16 * scale} {0.01 * scale} {0.08 * scale}"/>
      </body>
      {"".join(f'<body name="conveyor_object_{i}" pos="0 {y} {0.03 * scale}"><freejoint name="conveyor_object_{i}_joint"/><geom type="box" size="{0.02 * scale} {0.02 * scale} {0.02 * scale}"/></body>' for i, y in enumerate([-0.2, 0, 0.2], 1))}
      <body name="left_link_left_finger" {finger_attribute} pos="2 2 2">{'<freejoint name="test_left_finger_joint"/>' if gripper_qpos else ""}<geom type="sphere" size="{0.006 * scale}"/></body>
      <body name="left_link_right_finger" {finger_attribute} pos="2 -2 2">{'<freejoint name="test_right_finger_joint"/>' if gripper_qpos else ""}<geom type="sphere" size="{0.006 * scale}"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    evaluator = ConveyorPickObjectsInBinEvaluator(
        model=model, spec=get_task_spec("conveyor_pick")
    )
    mujoco.mj_forward(model, data)
    # Put the initially laid-out objects onto the yawed belt in its own frame.
    belt = evaluator._pick_geometry.belt
    rotation = data.geom_xmat[belt].reshape(3, 3)
    for index, address in enumerate(evaluator._object_qpos_addrs):
        data.qpos[address : address + 3] = rotation @ np.asarray(
            [0, (index - 1) * 0.2 * scale, 0.03 * scale]
        )
    mujoco.mj_forward(model, data)
    return model, data, evaluator


def put(scene, index, position):
    _, data, evaluator = scene
    address = evaluator._object_qpos_addrs[index]
    data.qpos[address : address + 3] = position


def grasp(scene, index, *, both=True):
    model, data, evaluator = scene
    position = data.qpos[
        evaluator._object_qpos_addrs[index] : evaluator._object_qpos_addrs[index] + 3
    ]
    scale = model.geom_size[evaluator._pick_geometry.belt, 2] / 0.01
    data.mocap_pos[:] = [
        position + [-0.024 * scale, 0, 0],
        position + [0.024 * scale, 0, 0] if both else [2, -2, 2],
    ]


def sample(scene):
    model, data, evaluator = scene
    mujoco.mj_forward(model, data)
    return evaluator.evaluate_model_data(model, data).to_info(squeeze=True)


def test_object_falls_off_belt_without_pick_latches_failure_and_cannot_be_rescued():
    setup = scene()
    assert not sample(setup)["ever_failed"]
    put(setup, 0, [0, -0.55, 0.03])
    result = sample(setup)
    assert result["ever_failed"] and result["terminal_failure"]
    assert result["failed_objects"] == ["conveyor_object_1"]
    assert not result["ever_success"]
    for i in range(3):
        put(setup, i, [1, (i - 1) * 0.05, 0.04])
    result = sample(setup)
    assert result["num_objects_in_bin"] == 3
    assert (
        result["ever_failed"] and not result["success"] and not result["ever_success"]
    )


@pytest.mark.parametrize("lifted,both", [(False, True), (True, False)])
def test_pushing_or_one_finger_brushing_cannot_count_as_picked(lifted, both):
    setup = scene()
    put(setup, 0, [0, -0.2, 0.12 if lifted else 0.03])
    grasp(setup, 0, both=both)
    assert not sample(setup)["object_picked_history_mask"][0]
    put(setup, 0, [0, -0.6, 0.12 if lifted else 0.03])
    grasp(setup, 0, both=both)
    assert sample(setup)["ever_failed"]


def test_real_two_finger_lift_is_picked_and_history_survives_release_off_belt():
    setup = scene()
    put(setup, 0, [0, -0.2, 0.12])
    grasp(setup, 0)
    result = sample(setup)
    assert result["object_two_finger_grasp_mask"][0]
    assert result["object_picked_history_mask"][0]
    put(setup, 0, [0.4, -0.6, 0.03])
    setup[1].mocap_pos[:] = [2, 2, 2]
    result = sample(setup)
    assert not result["ever_failed"]
    assert result["object_picked_history_mask"][0]


def test_objects_already_in_container_are_valid_without_pick_history():
    setup = scene()
    for i in range(3):
        put(setup, i, [1, (i - 1) * 0.05, 0.04])
    result = sample(setup)
    assert result["success"] and result["ever_success"]
    assert not result["ever_failed"]
    assert not any(result["object_picked_history_mask"])


@pytest.mark.parametrize("yaw,scale", [(np.pi / 2, 1), (0.61, 1.6)])
def test_scaled_yawed_belt_and_bin_use_actual_geometry(yaw, scale):
    setup = scene(yaw=yaw, scale=scale)
    _model, data, evaluator = setup
    assert not sample(setup)["ever_failed"]
    # A large rotated object may straddle the physical belt end; it remains valid.
    belt = evaluator._pick_geometry.belt
    rotation = data.geom_xmat[belt].reshape(3, 3)
    put(setup, 0, rotation @ np.asarray([0, -0.51 * scale, 0.03 * scale]))
    address = evaluator._object_qpos_addrs[0]
    data.qpos[address + 3 : address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    assert not sample(setup)["ever_failed"]
    put(setup, 0, rotation @ np.asarray([0, -0.56 * scale, 0.03 * scale]))
    assert sample(setup)["ever_failed"]
    evaluator.reset(nworld=1)
    for i in range(3):
        put(
            setup,
            i,
            data.xpos[evaluator._bin_body_id]
            + data.xmat[evaluator._bin_body_id].reshape(3, 3)
            @ np.asarray([0, (i - 1) * 0.05 * scale, 0.04 * scale]),
        )
    assert sample(setup)["success"]


def test_qpos_batch_latches_are_independent_and_reset_and_model_swap_clear_them():
    model, data, evaluator = scene()
    qpos = np.repeat(data.qpos[None], 2, axis=0)
    address = evaluator._object_qpos_addrs[0]
    qpos[0, address + 1] = -0.6
    result = evaluator.evaluate_qpos_batch(qpos)
    np.testing.assert_array_equal(result.metrics["ever_failed"], [True, False])
    qpos[:, address + 1] = -0.2
    result = evaluator.evaluate_qpos_batch(qpos)
    np.testing.assert_array_equal(result.metrics["ever_failed"], [True, False])
    evaluator.reset(nworld=2)
    assert not evaluator.evaluate_qpos_batch(qpos).metrics["ever_failed"].any()
    qpos[0, address + 1] = -0.6
    evaluator.evaluate_qpos_batch(qpos)
    evaluator.configure_from_randomization(copy.deepcopy(model), SimpleNamespace())
    qpos[:, address + 1] = -0.2
    assert not evaluator.evaluate_qpos_batch(qpos).metrics["ever_failed"].any()


def test_runtime_substep_pick_and_miss_histories_reach_evaluator_between_samples(
    monkeypatch,
):
    setup = scene()
    model, data, evaluator = setup
    runtime = ConveyorPickRuntime()
    runtime.bind(model, data)
    evaluator.bind_runtime(runtime)
    put(setup, 0, [0, -0.2, 0.12])
    grasp(setup, 0)
    mujoco.mj_forward(model, data)
    runtime.after_step(model, data)
    put(setup, 0, [0.4, -0.6, 0.03])
    data.mocap_pos[:] = [2, 2, 2]
    assert not sample(setup)["ever_failed"]
    assert sample(setup)["object_picked_history_mask"][0]
    put(setup, 1, [0, -0.6, 0.03])
    mujoco.mj_forward(model, data)
    runtime.after_step(model, data)
    put(setup, 1, [0, 0, 0.03])
    assert sample(setup)["ever_failed"]
    assert "conveyor_object_2" in runtime.debug_state()["missed_objects"]
    monkeypatch.setattr(runtime, "_open_idle_grippers", lambda *a: None)
    data.qpos[:] = model.qpos0
    runtime.after_reset(model, data)
    evaluator.reset(nworld=1)
    assert not sample(setup)["ever_failed"]


def test_runtime_carries_along_rotated_belt_axis():
    model, data, evaluator = scene(yaw=np.pi / 2)
    runtime = ConveyorPickRuntime()
    runtime.bind(model, data)
    address = evaluator._object_qpos_addrs[0]
    before = data.qpos[address : address + 3].copy()
    dof = int(model.jnt_dofadr[model.joint("conveyor_object_1_joint").id])
    runtime._carry_object_on_belt(model, data, address, dof)
    expected = (
        data.geom_xmat[evaluator._pick_geometry.belt].reshape(3, 3)[:, 1]
        * -0.2
        * model.opt.timestep
    )
    np.testing.assert_allclose(
        data.qpos[address : address + 3] - before, expected, atol=1e-15
    )


def test_rotated_object_extents_preserve_real_belt_edge_overlap():
    setup = scene()
    model, data, evaluator = setup
    object_geom = next(iter(evaluator._pick_geometry.object_geoms[0]))
    model.geom_size[object_geom] = [0.06, 0.01, 0.02]
    put(setup, 0, [0, -0.53, 0.03])
    address = evaluator._object_qpos_addrs[0]
    data.qpos[address + 3 : address + 7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    assert not sample(setup)["ever_failed"]  # Long axis straddles the edge.
    data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
    assert sample(setup)["ever_failed"]  # Same center, entire geometry now off.


def test_batch_pick_history_is_not_shared_between_worlds():
    model, data, evaluator = scene(gripper_qpos=True)
    qpos = np.repeat(data.qpos[None], 2, axis=0)
    address = evaluator._object_qpos_addrs[0]
    qpos[0, address : address + 3] = [0, -0.2, 0.12]
    for name, x in (
        ("test_left_finger_joint", -0.024),
        ("test_right_finger_joint", 0.024),
    ):
        finger_address = int(model.joint(name).qposadr[0])
        qpos[0, finger_address : finger_address + 3] = [x, -0.2, 0.12]
    result = evaluator.evaluate_qpos_batch(qpos)
    np.testing.assert_array_equal(
        result.metrics["object_picked_history_mask"][:, 0], [True, False]
    )
    qpos[:, address : address + 3] = [0.4, -0.6, 0.03]
    for name in ("test_left_finger_joint", "test_right_finger_joint"):
        finger_address = int(model.joint(name).qposadr[0])
        qpos[:, finger_address : finger_address + 3] = [2, 2, 2]
    result = evaluator.evaluate_qpos_batch(qpos)
    np.testing.assert_array_equal(result.metrics["ever_failed"], [False, True])


def test_runtime_shortlist_preserves_losses_and_genuine_picks_across_edge_and_drop_poses():
    setup = scene(yaw=0.61, scale=1.4)
    model, data, evaluator = setup
    belt = evaluator._pick_geometry.belt
    rotation = data.geom_xmat[belt].reshape(3, 3)
    for local in (
        [0, -0.28, 0.042],
        [0, -0.72, 0.042],
        [0, -0.9, 0.042],
        [0, 0, -0.04],
        [0, 0, 0.2],
        [0.4, 0, 0.04],
    ):
        put(setup, 0, rotation @ np.asarray(local))
        for held in (False, True):
            if held:
                grasp(setup, 0)
            else:
                data.mocap_pos[:] = [2, 2, 2]
            mujoco.mj_forward(model, data)
            exact = evaluator._sample(data)
            shortlisted = evaluator._sample(data, picked_history=set())
            # These are the only geometry decisions used by the runtime; a
            # fast safe proxy cannot hide a miss or invent a legitimate lift.
            np.testing.assert_array_equal(
                shortlisted["object_on_belt_mask"], exact["object_on_belt_mask"]
            )
            np.testing.assert_array_equal(
                shortlisted["object_picked_now_mask"], exact["object_picked_now_mask"]
            )


def test_runtime_refreshes_scale_proxies_after_reset(monkeypatch):
    setup = scene()
    model, data, evaluator = setup
    runtime = ConveyorPickRuntime()
    runtime.bind(model, data)
    runtime.after_step(model, data)
    old_scorer = runtime._pick_scorer
    geom = next(iter(evaluator._pick_geometry.object_geoms[0]))
    model.geom_size[geom] *= 2
    model.geom_pos[geom] = [0.4, 0, 0]
    # The original zero offset was compiled with BODY-frame shorthand.
    model.geom_sameframe[geom] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
    monkeypatch.setattr(runtime, "_open_idle_grippers", lambda *a: None)
    runtime.after_reset(model, data)
    runtime.after_step(model, data)
    assert runtime._pick_scorer is not old_scorer
    assert "conveyor_object_1" in runtime.debug_state()["missed_objects"]
