"""Mug mass properties on the real task-mug assets."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from abc_sim.mass_properties import free_object_mass_problems
from abc_sim.randomization.assets import mugs

MUG_ROOT = Path(__file__).parents[2] / "models/assets/task_mug_tree/mug"
HAS_MUG_ASSETS = all(
    path.is_file()
    for path in (
        MUG_ROOT / "mug_1/model.xml",
        MUG_ROOT.parents[1] / "mug_tree/model_collision_0.obj",
    )
)
requires_mug_assets = pytest.mark.skipif(
    not HAS_MUG_ASSETS, reason="Full task-mug assets required"
)
TABLE_TOP_Z = 0.75


def single_mug_model(variant, *, mass_on_first_visual_mesh=False):
    """The mug_tree scene with only the rack and one ``variant`` mug (no arms)."""
    xml = mugs._build_mug_scene_xml(
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
    worldbody = root.find("worldbody")
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") not in ("mug_tree", "mug_1"):
            worldbody.remove(child)
    if mass_on_first_visual_mesh:  # the assignment this module used to make
        visual = [
            geom
            for geom in worldbody.find(".//body[@name='mug_1']").iter("geom")
            if mugs._mug_is_visual_geom(geom) and geom.get("mesh")
        ]
        for index, geom in enumerate(visual):
            geom.set("density", "0")
            if index == 0:
                geom.set("mass", str(mugs._MUG_MASS_KG))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@requires_mug_assets
@pytest.mark.parametrize(
    "variant", mugs._mug_variant_names("mug_tree") if HAS_MUG_ASSETS else []
)
def test_every_mug_variant_has_sane_mass_properties(variant):
    model = single_mug_model(variant)
    mug = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug_1")
    assert model.body_subtreemass[mug] == pytest.approx(mugs._MUG_MASS_KG, rel=1e-6)
    assert free_object_mass_problems(model) == []


@requires_mug_assets
def test_mass_on_mug_1s_first_visual_mesh_is_flagged():
    model = single_mug_model("mug_1", mass_on_first_visual_mesh=True)
    problems = free_object_mass_problems(model)
    assert any("radius of gyration" in problem for problem in problems)


@requires_mug_assets
def test_mug_1_settles_on_the_table():
    """With its mass on a handle fragment, mug_1 was launched off the table."""
    model = single_mug_model("mug_1")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mug = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug_1")
    start = data.xpos[mug].copy()
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.xpos[mug]))
    assert np.linalg.norm(data.xpos[mug][:2] - start[:2]) < 0.02
    assert TABLE_TOP_Z < data.xpos[mug][2] < start[2] + 0.01
    assert np.linalg.norm(data.qvel[:6]) < 0.05


def test_visual_mass_is_spread_by_volume():
    body = ET.Element("body", name="object")
    for name in ("big", "small"):
        ET.SubElement(
            body, "geom", type="mesh", mesh=name, contype="0", conaffinity="0"
        )
    ET.SubElement(body, "geom", type="box", size=".02 .02 .02")
    mugs._mug_assign_visual_mass(body, 0.05, volume=0.04**3 + 0.02**3)
    big, small, collision = body.findall("geom")
    assert big.get("density") == small.get("density")
    assert float(big.get("density")) == pytest.approx(0.05 / (0.04**3 + 0.02**3))
    assert "density" not in collision.attrib
    with pytest.raises(ValueError, match="no volume"):
        mugs._mug_assign_visual_mass(body, 0.05, volume=0.0)
