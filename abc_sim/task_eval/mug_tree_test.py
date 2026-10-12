"""Real MuJoCo hanging and stacked-mug regressions without external assets."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_sim.batched_env import PerWorldEvaluators
from abc_sim.task_eval.mug_tree import MugOnRackEvaluator
from abc_sim.task_specs import get_task_spec

BASE_XML = '<mujoco><option timestep=".002" gravity="0 0 -9.81"/><asset>\n<mesh name="trunk" vertex="-.01 -.01 -.2  .01 -.01 -.2  .01 .01 -.2  -.01 .01 -.2  -.01 -.01 .2  .01 -.01 .2  .01 .01 .2  -.01 .01 .2"/>\n</asset><worldbody><geom name="table" type="plane" size="1 1 .01"/>\n<body name="mug_tree" pos="0 0 .2"><geom name="trunk" type="mesh" mesh="trunk" group="3"/>\n<geom name="mug_tree_peg_1" type="capsule" fromto="-.06 0 .10 .06 0 .10" size=".006"/></body>\n<body name="mug_1" pos=".04 .055 .2859"><freejoint name="mug_1_jnt"/>\n<geom name="cup" type="cylinder" size=".03 .05" mass=".1"/>\n<geom name="handle_top" type="capsule" fromto="0 -.03 .025 0 -.08 .025" size=".005" mass=".01"/>\n<geom name="handle_bottom" type="capsule" fromto="0 -.03 -.025 0 -.08 -.025" size=".005" mass=".01"/>\n<geom name="handle_outer" type="capsule" fromto="0 -.08 -.025 0 -.08 .025" size=".005" mass=".01"/>\n</body></worldbody></mujoco>'


def scene(count=1, scale=1.0, tree_position=(0, 0, 0.2), tree_yaw=0.0):
    root = ET.fromstring(BASE_XML)
    wb = root.find("worldbody")
    tree = wb.find("body[@name='mug_tree']")
    tree.set("pos", " ".join(map(str, tree_position)))
    tree.set("quat", f"{np.cos(tree_yaw / 2)} 0 0 {np.sin(tree_yaw / 2)}")
    mug = wb.find("body[@name='mug_1']")
    for i in range(2, count + 1):
        other = copy.deepcopy(mug)
        other.set("name", f"mug_{i}")
        other.find("freejoint").set("name", f"mug_{i}_jnt")
        for geom in other.findall("geom"):
            geom.set("name", f"mug_{i}_" + geom.get("name"))
        wb.append(other)
    if scale != 1:
        for body in wb.findall("body"):
            if (
                not body.get("name", "").startswith("mug_")
                or body.get("name") == "mug_tree"
            ):
                continue
            for geom in body.findall("geom"):
                for attr in ("size", "fromto"):
                    if geom.get(attr):
                        geom.set(
                            attr,
                            " ".join(
                                str(float(x) * scale) for x in geom.get(attr).split()
                            ),
                        )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    return model, data, evaluator


def score(case):
    model, data, evaluator = case
    mujoco.mj_forward(model, data)
    return evaluator.evaluate_model_data(model, data).to_info(squeeze=True)


def test_real_settled_hanging_positive_in_live_and_batch():
    model, data, evaluator = scene()
    for _ in range(3000):
        mujoco.mj_step(model, data)
    result = score((model, data, evaluator))
    assert result["success"] and result["mug_threaded_peg"] == [True]
    assert not result["mug_other_contact"][0]
    assert (
        result["mug_peg_support_force_n"][0]
        >= evaluator.support_weight_fraction * evaluator._mug_weights[0]
    )
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_stack_near_tree_no_longer_scores_all_hanging():
    model, data, evaluator = scene(count=2)
    for i, adr in enumerate(evaluator._mug_qpos_addrs):
        data.qpos[adr : adr + 3] = [0.02, 0.08, 0.0499 + i * 0.0999]
    positions = np.stack([data.qpos[a : a + 3] for a in evaluator._mug_qpos_addrs])
    # Both centers satisfy the historical cylinder/height criterion.
    assert np.all(
        np.linalg.norm(positions[:, :2] - evaluator._axis_xy, axis=1)
        < evaluator.hang_radius_m
    )
    assert np.all(
        (positions[:, 2] >= evaluator._hang_z_lo)
        & (positions[:, 2] <= evaluator._hang_z_hi)
    )
    result = score((model, data, evaluator))
    assert result["num_mugs_hanging"] == 0 and not result["ever_success"]
    assert all(result["mug_other_contact"])
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_brushing_peg_without_handle_enclosure_is_not_hanging():
    model, data, evaluator = scene()
    data.qpos[:3] = [0.04, 0.0339, 0.3]
    result = score((model, data, evaluator))
    assert data.ncon > 0
    assert not result["success"] and not result["mug_threaded_peg"][0]


def test_trunk_contact_does_not_count_as_peg_support():
    model, data, evaluator = scene()
    data.qpos[:3] = [0.02, 0.02, 0.2]
    result = score((model, data, evaluator))
    assert data.ncon > 0 and result["mug_other_contact"][0]
    assert not result["success"]


def test_contact_inside_solid_handle_is_not_threading():
    model, data, evaluator = scene()
    # Put the peg's centerline in the upper solid handle bar, not the hole.
    data.qpos[:3] = [0.04, 0.055, 0.275]
    mujoco.mj_forward(model, data)
    peg = next(iter(evaluator._peg_segments))
    assert not evaluator._peg_threaded(data, 0, peg, model)
    assert not score((model, data, evaluator))["success"]


def test_robot_touching_otherwise_hanging_mug_rejects_success():
    root = ET.fromstring(BASE_XML)
    wb = root.find("worldbody")
    finger = ET.SubElement(
        wb, "body", name="left_link_left_finger", mocap="true", pos=".04 .085 .2859"
    )
    ET.SubElement(finger, "geom", name="finger", type="sphere", size=".006")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    result = score((model, data, evaluator))
    assert result["mug_threaded_peg"][0] and result["mug_other_contact"][0]
    assert not result["success"]


@pytest.mark.parametrize("scale", [0.9, 1.0, 1.1])
def test_scaled_mug_can_hang_after_gravity_settles(scale):
    model, data, evaluator = scene(scale=scale)
    # Preserve the handle opening at the same peg for each compiled scale.
    data.qpos[:3] = [0.045, 0.055 * scale, 0.306 - 0.02 * scale - 0.0001]
    for _ in range(3000):
        mujoco.mj_step(model, data)
    assert score((model, data, evaluator))["success"]
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_live_tree_translation_and_yaw_are_followed():
    model, data, evaluator = scene(tree_position=(0.3, -0.2, 0.2), tree_yaw=0.4)
    local = np.array([0.04, 0.055, 0.2859])
    rotation = np.array(
        [[np.cos(0.4), -np.sin(0.4), 0], [np.sin(0.4), np.cos(0.4), 0], [0, 0, 1]]
    )
    data.qpos[:3] = rotation @ local + [0.3, -0.2, 0]
    data.qpos[3:7] = [np.cos(0.2), 0, 0, np.sin(0.2)]
    assert score((model, data, evaluator))["success"]


def test_batched_worlds_restore_independent_static_tree_poses():
    model, data, _ = scene()
    spec = get_task_spec("hang_mug_on_mug_rack")
    evaluators = PerWorldEvaluators(model, spec, 2)
    states = [
        SimpleNamespace(
            object_states={"mug_tree": {"pos": [shift, 0, 0.2], "quat": [1, 0, 0, 0]}}
        )
        for shift in (0.0, 0.3)
    ]
    evaluators.reset(model, states)
    qpos = np.repeat(data.qpos[None], 2, axis=0)
    qpos[1, 0] += 0.3
    result = evaluators.evaluate_qpos_batch(qpos)
    np.testing.assert_array_equal(result.success, [True, True])
    qpos[1, 0] -= 0.3
    np.testing.assert_array_equal(
        evaluators.evaluate_qpos_batch(qpos).success, [True, False]
    )


def test_reset_and_model_reload_rebind_geometry_and_clear_success():
    model, data, evaluator = scene()
    assert score((model, data, evaluator))["ever_success"]
    evaluator.reset(nworld=1)
    data.qpos[:3] = [0.3, 0.3, 0.05]
    assert not score((model, data, evaluator))["ever_success"]
    other_model, other_data, _ = scene(count=2)
    evaluator.configure_from_randomization(other_model, None)
    assert len(evaluator.mug_names) == 2
    other_data.qpos[:3] = [0.3, 0.3, 0.05]
    assert not evaluator.evaluate_model_data(other_model, other_data).scalar_success()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_weight_fraction": 0},
        {"support_weight_fraction": 1.1},
        {"bottom_slice_fraction": 0},
        {"geometry_tolerance_m": 0},
        {"max_penetration_m": 0},
    ],
)
def test_invalid_hanging_configuration_rejected(kwargs):
    model, _, _ = scene()
    with pytest.raises(ValueError):
        MugOnRackEvaluator(
            model=model, spec=get_task_spec("hang_mug_on_mug_rack"), **kwargs
        )


ACTUAL_SETTLED_POSES = {
    "mug_0": [
        0.6800025405706627,
        -0.004242402755389454,
        0.9933282099047096,
        0.7523402319737242,
        -0.6583372659245912,
        0.003540274693750568,
        -0.023742074546042054,
    ],
    "mug_2": [
        0.6828870692310255,
        0.0009889294409580157,
        0.9907866860751182,
        0.7745759050720864,
        -0.6260703219343563,
        0.03719340718621332,
        -0.08176044114760572,
    ],
    "mug_13": [
        0.6871115059308729,
        0.0020453100691285087,
        1.0004155551133667,
        0.7457787193579729,
        -0.6514544301926191,
        0.07934227203498777,
        -0.11456889195378846,
    ],
    "mug_18": [
        0.6949301241076262,
        0.00046735822057206996,
        0.9988002876078315,
        0.8480186188863599,
        -0.4917349183399374,
        0.05649182914221636,
        -0.18939341421351333,
    ],
}
HAS_MUG_ASSETS = all(
    (Path(__file__).parents[1] / "models/assets" / asset).is_file()
    for asset in (
        "task_mug_tree/mug/mug_0/model.xml",
        "i2rt_yam/assets/model2.stl",
        "mug_tree/model_collision_0.obj",
    )
)


def actual_model(variant):
    from abc_sim.randomization.assets.mugs import _build_mug_scene_xml

    xml = _build_mug_scene_xml(
        task_name="mug_tree",
        mug_variant=variant,
        mug_variants=[variant],
        scale_states={},
        base_scene_xml=None,
        base_scene_dir=None,
    )
    root = ET.fromstring(xml)
    for tag in ("actuator", "equality", "keyframe"):
        entry = root.find(tag)
        if entry is not None:
            root.remove(entry)
    wb = root.find("worldbody")
    for child in list(wb):
        if child.get("name") not in ("mug_tree", "mug_1"):
            wb.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@pytest.mark.skipif(not HAS_MUG_ASSETS, reason="Full task-mug and YAM assets required")
@pytest.mark.parametrize("variant", list(ACTUAL_SETTLED_POSES))
def test_actual_collision_meshes_hang_after_gravity_settles(variant):
    model = actual_model(variant)
    data = mujoco.MjData(model)
    data.qpos[:] = ACTUAL_SETTLED_POSES[variant]
    for _ in range(1000):
        mujoco.mj_step(model, data)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    result = evaluator.evaluate_model_data(model, data)
    assert result.scalar_success()
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()
    assert not np.any(result.metrics["mug_other_contact"])


@pytest.mark.skipif(not HAS_MUG_ASSETS, reason="Full task-mug and YAM assets required")
def test_real_mesh_solid_handle_occupancy_is_not_an_empty_opening():
    model = actual_model("mug_1")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    geom = next(
        g
        for g in evaluator._mug_geoms[0]
        if model.mesh(int(model.geom_dataid[g])).name.endswith("collision_mesh_11")
    )
    point = data.geom_xpos[geom].copy()
    body = evaluator._mug_bodies[0]
    rotation = data.xmat[body].reshape(3, 3)
    # Rays starting inside this real convex handle limb report positive exits.
    assert all(
        mujoco.mj_rayMesh(model, data, geom, point, sign * rotation[:, axis]) > 0
        for axis in (1, 2)
        for sign in (-1, 1)
    )
    assert evaluator._inside_mug_solid(data, evaluator._mug_geoms[0], point, model)


def test_reused_qpos_scratch_discards_stale_dynamic_state():
    _model, data, evaluator = scene()
    qpos = data.qpos[None].copy()
    expected = evaluator.evaluate_qpos_batch(qpos).to_info()
    scratch = evaluator._qpos_scratch_data
    scratch.qvel[:] = 100
    scratch.qfrc_applied[:] = 100
    scratch.xfrc_applied[:] = 100
    scratch.qacc_warmstart[:] = 100
    scratch.time = 100
    actual = evaluator.evaluate_qpos_batch(qpos).to_info()
    assert evaluator._qpos_scratch_data is scratch
    assert actual == expected


def test_qpos_scratch_is_replaced_when_static_tree_model_changes():
    model, data, evaluator = scene()
    evaluator.evaluate_qpos_batch(data.qpos[None])
    old_scratch = evaluator._qpos_scratch_data
    evaluator.configure_from_randomization(
        model,
        SimpleNamespace(
            object_states={
                "mug_tree": {"pos": [0.3, 0, 0.2], "quat": [1, 0, 0, 0]},
            }
        ),
    )
    moved_qpos = data.qpos[None].copy()
    moved_qpos[0, 0] += 0.3
    assert evaluator.evaluate_qpos_batch(moved_qpos).scalar_success()
    assert evaluator._qpos_scratch_data is not old_scratch
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_threading_geometry_is_cached_only_within_one_sample(monkeypatch):
    root = ET.fromstring(BASE_XML)
    mug = root.find("worldbody/body[@name='mug_1']")
    top = copy.deepcopy(mug.find("geom[@name='handle_top']"))
    top.set("name", "second_handle_top_collision_piece")
    mug.append(top)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    original = evaluator._peg_threaded
    calls = []

    def track(data, index, peg, model):
        calls.append((index, peg))
        return original(data, index, peg, model)

    monkeypatch.setattr(evaluator, "_peg_threaded", track)
    first = evaluator.evaluate_model_data(model, data)
    assert first.scalar_success()
    assert data.ncon >= 2
    assert len(calls) == len(set(calls)) == 1
    # A changed pose must recompute and reject threading; no cached success
    # can survive into the next evaluation.
    data.qpos[:3] = [0.04, 0.09, 0.2859]
    mujoco.mj_forward(model, data)
    second = evaluator.evaluate_model_data(model, data)
    assert not second.scalar_success()
    assert len(calls) == 2


def test_light_trunk_bracing_does_not_reject_genuine_handle_support():
    root = ET.fromstring(BASE_XML)
    tree = root.find("worldbody/body[@name='mug_tree']")
    ET.SubElement(
        tree,
        "geom",
        name="trunk_brace",
        type="sphere",
        size=".006",
        pos=".04 .0909 .075",
        friction=".02 0 0",
        priority="2",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    live = evaluator.evaluate_model_data(model, data)
    assert live.scalar_success()
    assert live.metrics["mug_threaded_peg"][0, 0]
    assert live.metrics["mug_rack_brace_contact"][0, 0]
    assert not live.metrics["mug_other_contact"][0, 0]
    assert (
        abs(live.metrics["mug_rack_brace_force_n"][0, 0])
        < 0.25 * evaluator._mug_weights[0]
    )
    assert (
        live.metrics["mug_peg_support_force_n"][0, 0]
        >= evaluator.support_weight_fraction * evaluator._mug_weights[0]
    )
    batch = evaluator.evaluate_qpos_batch(data.qpos[None])
    assert batch.scalar_success()
    np.testing.assert_array_equal(batch.metrics["mug_rack_brace_contact"], [[True]])


def test_deep_rack_bracing_still_rejects_hanging():
    root = ET.fromstring(BASE_XML)
    tree = root.find("worldbody/body[@name='mug_tree']")
    ET.SubElement(
        tree,
        "geom",
        name="trunk_brace",
        type="sphere",
        size=".006",
        pos=".04 .087 .075",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    result = evaluator.evaluate_model_data(model, data)
    assert result.metrics["mug_other_contact"][0, 0]
    assert not result.scalar_success()
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_shallow_trunk_bracing_without_threaded_peg_load_is_not_hanging():
    root = ET.fromstring(BASE_XML)
    tree = root.find("worldbody/body[@name='mug_tree']")
    tree.find("geom[@name='mug_tree_peg_1']").set("fromto", "-.06 0 .2 .06 0 .2")
    ET.SubElement(
        tree,
        "geom",
        name="trunk_brace",
        type="sphere",
        size=".006",
        pos=".04 .0909 .075",
        friction=".02 0 0",
        priority="2",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    result = evaluator.evaluate_model_data(model, data)
    assert result.metrics["mug_rack_brace_contact"][0, 0]
    assert not result.metrics["mug_other_contact"][0, 0]
    assert result.metrics["mug_peg_support_force_n"][0, 0] == 0
    assert not result.scalar_success()
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_shallow_tangency_between_independently_hanging_mugs_is_allowed():
    model, data, evaluator = scene(count=2)
    second = evaluator._mug_qpos_addrs[1]
    data.qpos[second : second + 3] = [-0.0199, 0.055, 0.2859]
    result = score((model, data, evaluator))
    assert result["success"]
    assert result["mug_peer_contact"] == [True, True]
    assert result["mug_threaded_peg"] == [True, True]
    assert result["mug_other_contact"] == [False, False]
    assert all(
        force >= evaluator.support_weight_fraction * weight
        for force, weight in zip(
            result["mug_peg_support_force_n"], evaluator._mug_weights
        )
    )
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_touching_an_unsupported_mug_does_not_count_as_independent_hanging():
    model, data, evaluator = scene(count=2)
    second = evaluator._mug_qpos_addrs[1]
    data.qpos[second : second + 3] = [-0.0199, 0.055, 0.2359]
    result = score((model, data, evaluator))
    assert result["mug_peer_contact"] == [True, True]
    assert not result["mug_threaded_peg"][1]
    assert result["mug_other_contact"] == [True, True]
    assert not result["success"]
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_deep_mug_to_mug_penetration_still_rejects_hanging():
    model, data, evaluator = scene(count=2)
    second = evaluator._mug_qpos_addrs[1]
    data.qpos[second : second + 3] = [-0.015, 0.055, 0.2859]
    result = score((model, data, evaluator))
    assert result["mug_peer_contact"] == [True, True]
    assert result["mug_other_contact"] == [True, True]
    assert not result["success"]
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_extra_unthreaded_peg_brush_does_not_earn_support_or_reject_hanging():
    root = ET.fromstring(BASE_XML)
    tree = root.find("worldbody/body[@name='mug_tree']")
    ET.SubElement(
        tree,
        "geom",
        name="mug_tree_peg_2",
        type="capsule",
        size=".006",
        fromto=".04 .0909 .075 .04 .0909 .085",
        friction=".02 0 0",
        priority="2",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    result = evaluator.evaluate_model_data(model, data)
    assert result.scalar_success()
    assert result.metrics["mug_rack_brace_contact"][0, 0]
    assert not result.metrics["mug_other_contact"][0, 0]
    extra = model.geom("mug_tree_peg_2").id
    assert not evaluator._peg_threaded(data, 0, extra, model)
    assert (
        result.metrics["mug_peg_support_force_n"][0, 0]
        >= evaluator.support_weight_fraction * evaluator._mug_weights[0]
    )
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_peg_tip_in_empty_handle_can_hang_beyond_vessel_center_plane():
    model, data, evaluator = scene()
    data.qpos[:3] = [0.067, 0.055, 0.281]
    mujoco.mj_forward(model, data)
    peg = next(iter(evaluator._peg_segments))
    body = evaluator._mug_bodies[0]
    segment = (
        evaluator._peg_segments[peg] @ data.geom_xmat[peg].reshape(3, 3).T
        + data.geom_xpos[peg]
        - data.xpos[body]
    ) @ data.xmat[body].reshape(3, 3)
    center, _ = evaluator._vessel_geometry[0]
    central_t = (center[0] - segment[0, 0]) / (segment[1, 0] - segment[0, 0])
    assert not 0 <= central_t <= 1  # The single-plane test missed the actual tip.
    assert evaluator._peg_threaded(data, 0, peg, model)
    assert evaluator.evaluate_model_data(model, data).scalar_success()
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_asymmetric_handle_opening_is_found_off_vessel_center_plane():
    root = ET.fromstring(BASE_XML)
    mug = root.find("worldbody/body[@name='mug_1']")
    for geom in mug.findall("geom"):
        if not geom.get("name", "").startswith("handle_"):
            continue
        endpoints = np.asarray(list(map(float, geom.get("fromto").split())))
        endpoints[[0, 3]] -= 0.012
        geom.set("fromto", " ".join(map(str, endpoints)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evaluator = MugOnRackEvaluator(
        model=model, spec=get_task_spec("hang_mug_on_mug_rack")
    )
    peg = next(iter(evaluator._peg_segments))
    body = evaluator._mug_bodies[0]
    rotation = data.xmat[body].reshape(3, 3)
    segment = (
        evaluator._peg_segments[peg] @ data.geom_xmat[peg].reshape(3, 3).T
        + data.geom_xpos[peg]
        - data.xpos[body]
    ) @ rotation
    center, _ = evaluator._vessel_geometry[0]
    central_t = (center[0] - segment[0, 0]) / (segment[1, 0] - segment[0, 0])
    point = (
        rotation @ (segment[0] + central_t * (segment[1] - segment[0]))
        + data.xpos[body]
    )
    rays = [
        evaluator._ray_mug(
            data, evaluator._mug_geoms[0], point, sign * rotation[:, axis], model
        )
        for axis in (1, 2)
        for sign in (-1, 1)
    ]
    assert min(rays) < 0  # The vessel-center rays do not enclose this opening.
    assert evaluator._peg_threaded(data, 0, peg, model)
    assert evaluator.evaluate_model_data(model, data).scalar_success()
    assert evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()


def test_open_handle_around_peg_without_load_still_does_not_succeed():
    model, data, evaluator = scene()
    data.qpos[2] = 0.3
    mujoco.mj_forward(model, data)
    peg = next(iter(evaluator._peg_segments))
    assert evaluator._peg_threaded(data, 0, peg, model)
    assert not evaluator.evaluate_model_data(model, data).scalar_success()
    assert not evaluator.evaluate_qpos_batch(data.qpos[None]).scalar_success()
