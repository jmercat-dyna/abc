"""Small damped nuts must lose rotational energy rather than numerically explode."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

SCENE = Path(__file__).parent / "models" / "yam_nuts_bolts_sorting_scene.xml"


def asset_free_nut_model():
    """Use the shipped nut's actual inertia, damping, compiler and collision geoms."""
    scene = ET.parse(SCENE).getroot()
    root = ET.Element("mujoco")
    root.append(copy.deepcopy(scene.find("compiler")))
    option = copy.deepcopy(scene.find("option"))
    option.set("gravity", "0 0 0")
    root.append(option)
    defaults = ET.SubElement(root, "default")
    defaults.append(copy.deepcopy(scene.find("default/default[@class='sort_part']")))
    body = copy.deepcopy(scene.find("worldbody/body[@name='nut_1']"))
    for geom in list(body.findall("geom")):
        if geom.get("type") == "mesh":
            body.remove(geom)
    ET.SubElement(root, "worldbody").append(body)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def rotational_energy(model, data, body=1, angular_addr=3):
    return float(
        np.sum(
            model.body_inertia[body] * data.qvel[angular_addr : angular_addr + 3] ** 2
        )
        / 2
    )


def test_old_midpoint_update_amplifies_normal_nut_rotation_without_external_force():
    if mujoco.__version__ != "3.8.1":
        pytest.skip("Historical MuJoCo 3.8.1 midpoint bug; newer engines may fix it")
    model = asset_free_nut_model()
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    data = mujoco.MjData(model)
    data.qvel[3] = 0.1
    initial_energy = rotational_energy(model, data)
    mujoco.mj_step(model, data)
    assert abs(data.qvel[3]) > 0.19
    assert rotational_energy(model, data) > 3.9 * initial_energy
    for _ in range(39):
        mujoco.mj_step(model, data)
        if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            break
    assert data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0
    assert not np.any(data.xfrc_applied)


@pytest.mark.parametrize("angular_velocity", [(0.1, 0, 0), (0.1, 0.2, 0.3)])
def test_shipped_nut_integrator_dissipates_rotation_with_real_scene_parameters(
    angular_velocity,
):
    model = asset_free_nut_model()
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    np.testing.assert_allclose(model.body_inertia[1], [1e-6, 1e-6, 1.1e-6])
    np.testing.assert_allclose(model.dof_damping[3:6], [0.0015, 0.0015, 0.0015])
    data = mujoco.MjData(model)
    data.qvel[3:6] = angular_velocity
    energy = rotational_energy(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)
        next_energy = rotational_energy(model, data)
        assert next_energy <= energy * (1 + 1e-12)
        energy = next_energy
    assert energy < 1e-20
    assert data.time == pytest.approx(0.2)
    assert data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0


def test_actual_randomized_scene_nut_spin_decays_without_reset():
    assets = SCENE.parent / "assets"
    if not (assets / "i2rt_yam" / "assets" / "model2.stl").is_file():
        pytest.skip(
            "Actual YAM scene assets unavailable; asset-free physics regressions still run"
        )
    import abc_sim

    env = abc_sim.make_env(
        task="sort_nuts_and_bolts", render_cameras=False, camera_backend="mujoco"
    )
    try:
        env.reset(seed=20260511, randomize=True)
        model, data = env.model, env.data
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "nut_5_joint")
        body = int(model.jnt_bodyid[joint])
        addr = int(model.jnt_qposadr[joint])
        angular_addr = int(model.jnt_dofadr[joint]) + 3
        data.qpos[addr : addr + 3] = [0.6, 0, 3]
        data.qvel[angular_addr : angular_addr + 3] = [0.1, 0, 0]
        mujoco.mj_forward(model, data)
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        before = rotational_energy(model, data, body, angular_addr)
        env.advance_physics(steps=100)
        assert rotational_energy(model, data, body, angular_addr) < before * 1e-10
        assert data.time == pytest.approx(0.2)
    finally:
        env.close()
