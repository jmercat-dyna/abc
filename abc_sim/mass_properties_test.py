"""Free-object mass sanity check on synthetic MuJoCo bodies."""

import logging

import mujoco
import numpy as np

from abc_sim.mass_properties import (
    free_object_mass_problems,
    warn_on_free_object_mass_problems,
)


def free_cup(point_mass_offset=None):
    """A 10 cm collision box whose mass is spread over it or on a tiny sphere."""
    if point_mass_offset is None:
        mass_geom = '<geom type="box" size=".05 .05 .05" mass=".05"/>'
    else:
        mass_geom = (
            '<geom type="box" size=".05 .05 .05" mass="0"/>'
            f'<geom type="sphere" size=".002" pos="{point_mass_offset} 0 0" '
            'mass=".05" contype="0" conaffinity="0"/>'
        )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .01"/>'
        f'<body name="cup" pos="0 0 1"><freejoint/>{mass_geom}</body>'
        "</worldbody></mujoco>"
    )


def test_solid_object_passes():
    assert free_object_mass_problems(free_cup()) == []


def test_point_mass_and_off_body_centre_of_mass_are_flagged():
    inside = free_object_mass_problems(free_cup(point_mass_offset=0.0))
    assert len(inside) == 1 and "radius of gyration" in inside[0]
    outside = free_object_mass_problems(free_cup(point_mass_offset=0.2))
    assert any("centre of mass is outside" in problem for problem in outside)


def test_check_leaves_the_callers_state_untouched_and_logs(caplog):
    model = free_cup(point_mass_offset=0.0)
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.3, -0.2, 2.0]
    before = data.qpos.copy()
    with caplog.at_level(logging.WARNING, logger="abc_sim.mass_properties"):
        problems = warn_on_free_object_mass_problems(model, data)
    assert problems and "cup" in caplog.text
    np.testing.assert_array_equal(data.qpos, before)
    assert not data.xpos.any()  # no kinematics were written into the caller's data
