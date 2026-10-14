"""Captured pawn/tin-floor collisions must stay inside the real scene geometry."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

MODELS = Path(__file__).parent / "models"
SCENES = ("yam_chess_scene.xml", "yam_flexible_chess_scene.xml")

# Original seed 20260511, physics step 56156, immediately before the historical
# FactorizeHessian exception. Only the tin floor and white_pawn_6 are needed.
CAPTURED_QPOS = np.array(
    [
        0.7001755321497082,
        -0.018446278267291834,
        0.8641244749635577,
        0.04810830784686856,
        0.012550402894584409,
        -0.0008229818577349575,
        -0.998762935237464,
        0.7311431451560292,
        0.0856862884814897,
        0.7942761775466104,
        0.9990651863634683,
        -0.003065396471571291,
        -0.03319556509886838,
        -0.027521104606579815,
    ]
)


def pair_model(scene=SCENES[0], *, fixed_floor=False):
    """Real authored pawn mesh and captured geometry; production task options."""
    asset = MODELS / "assets" / "chess" / "white_pawn_6.obj"
    if not asset.is_file():
        pytest.skip("Actual pawn asset unavailable for captured collision regression")
    root = ET.fromstring(
        f"""<mujoco>
          <compiler angle="radian" meshdir="{asset.parent}"/>
          <default><default class="chess_piece">
            <geom type="mesh" condim="6" priority="1" friction="3 .03 .003"
              solref=".004 1" solimp=".998 .998 .001 .5 2" density="0"/>
          </default></default>
          <asset><mesh name="pawn" file="white_pawn_6.obj"
            scale=".029 .029 .029"/></asset>
          <worldbody>
            <body name="tin_box" pos=".6 .46 .757"><freejoint/>
              <geom name="floor" type="box" size=".1825 .132848 .006"
                condim="6" priority="1" friction="2 .1 .01" mass=".06"
                solref=".004 1" solimp=".998 .998 .001 .5 2"/>
            </body>
            <body name="pawn" pos=".7245 .0797 .7969"><freejoint/>
              <geom name="pawn" class="chess_piece" mesh="pawn" mass=".002"
                quat=".707105 .707108 0 0"/>
            </body>
          </worldbody>
        </mujoco>"""
    )
    root.append(copy.deepcopy(ET.parse(MODELS / scene).getroot().find("option")))
    if fixed_floor:
        floor = root.find("worldbody/body[@name='tin_box']")
        floor.remove(floor.find("freejoint"))
        floor.set("pos", "0 0 0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def test_historical_multicontact_generates_impossible_contact_before_solver():
    if mujoco.__version__ != "3.8.1":
        pytest.skip("Historical 3.8.1 collision bug; newer engines may fix it")
    model = pair_model()
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    # The original 220-DOF scene selects sparse storage under auto; explicitly
    # keep that representation after reducing the scene to twelve DOFs.
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    data = mujoco.MjData(model)
    data.qpos[:] = CAPTURED_QPOS
    mujoco.mj_fwdPosition(model, data)
    assert np.max(np.abs(data.xpos)) < 2
    assert data.ncon > 0
    assert np.max(np.abs(data.contact.pos[: data.ncon])) > 1e6
    with pytest.raises(mujoco.FatalError, match="rank-deficient sparse Hessian"):
        mujoco.mj_forward(model, data)


@pytest.mark.parametrize("scene", SCENES)
def test_configured_chess_contact_lies_on_the_captured_objects(scene):
    model = pair_model(scene)
    data = mujoco.MjData(model)
    data.qpos[:] = CAPTURED_QPOS
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    for contact in data.contact[: data.ncon]:
        for geom in contact.geom:
            distance = np.linalg.norm(contact.pos - data.geom_xpos[geom])
            assert distance <= model.geom_rbound[geom] + model.opt.ccd_tolerance
    for _ in range(100):
        before = data.time
        mujoco.mj_step(model, data)
        assert data.time > before
    assert not np.any(data.warning.number[4:7])


def test_configured_single_contact_still_supports_released_pawn_on_floor():
    model = pair_model(fixed_floor=True)
    data = mujoco.MjData(model)
    data.qpos[:] = [0, 0, 0.15, 1, 0, 0, 0]
    for _ in range(2500):
        before = data.time
        mujoco.mj_step(model, data)
        assert data.time > before
    assert not np.any(data.warning.number[4:7])
    assert data.ncon > 0
    assert data.qpos[2] == pytest.approx(model.geom_size[0, 2], abs=1e-4)
    assert np.linalg.norm(data.qvel[:3]) < 1e-3


def test_separated_pawn_has_no_phantom_floor_contact():
    model = pair_model(fixed_floor=True)
    data = mujoco.MjData(model)
    data.qpos[:] = [1, 0, 0.15, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


@pytest.mark.parametrize("scene", SCENES)
def test_chess_collision_option_compiles_without_assets(scene):
    """Both task variants reach the native single-contact path in any install."""
    option = copy.deepcopy(ET.parse(MODELS / scene).getroot().find("option"))
    root = ET.Element("mujoco")
    root.append(option)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
    assert not (model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
