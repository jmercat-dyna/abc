"""Unit tests for the reset-time free-object penetration check."""

import logging

import mujoco

from abc_sim.initial_penetration import (
    free_object_penetrations,
    warn_on_free_object_penetrations,
)

SCENE = """
<mujoco>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.01"/>
    <geom name="wall" type="box" size="0.01 1 1" pos="0.5 0 1"/>
    <body name="sunk" pos="0 0 0.04"><freejoint/>
      <geom name="sunk_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="resting" pos="0.2 0 0.0505"><freejoint/>
      <geom name="resting_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="in_wall" pos="0.45 0 0.3"><freejoint/>
      <geom name="in_wall_geom" type="sphere" size="0.06"/>
    </body>
    <body name="parked" pos="0 0 -10"><freejoint/>
      <geom name="parked_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="parked_twin" pos="0 0.01 -10"><freejoint/>
      <geom name="parked_twin_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_reports_deep_overlaps_only():
    model = mujoco.MjModel.from_xml_string(SCENE)
    found = free_object_penetrations(model)
    pairs = {(item.body, item.other_geom): round(item.depth_m, 3) for item in found}
    # 10 mm into the table, 20 mm into the wall; the resting box (0.5 mm gap)
    # and the overlapping pair parked below the floor are not reported.
    assert pairs == {("sunk", "table"): 0.010, ("in_wall", "wall"): 0.020}
    assert found[0].body == "in_wall"  # deepest first
    assert "20.0 mm inside world (wall)" in found[0].describe()


def test_tolerance_and_caller_state():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    sunk = model.jnt_qposadr[0]
    data.qpos[sunk + 2] = 0.048  # 2 mm deep: under the 5 mm default
    qpos_before = data.qpos.copy()
    bodies = {item.body for item in free_object_penetrations(model, data)}
    assert bodies == {"in_wall"}
    assert {item.body for item in free_object_penetrations(model, data, tolerance=0.001)} == {
        "sunk",
        "in_wall",
    }
    assert (data.qpos == qpos_before).all()
    assert data.ncon == 0  # collision ran on a scratch MjData


def test_warning_is_deduplicated(caplog):
    model = mujoco.MjModel.from_xml_string(SCENE)
    warned: set[tuple[str, str]] = set()
    with caplog.at_level(logging.WARNING, logger="abc_sim.initial_penetration"):
        first = warn_on_free_object_penetrations(model, already_warned=warned)
        second = warn_on_free_object_penetrations(model, already_warned=warned)
    assert len(first) == len(second) == 2
    assert len(caplog.records) == 2
    assert warned == {("sunk", "world"), ("in_wall", "world")}
