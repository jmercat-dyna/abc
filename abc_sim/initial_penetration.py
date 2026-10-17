"""Reset-time check for free objects spawned inside other geometry.

A free object that starts interpenetrating another collision geom (the table, a
container wall, another object) is pushed out by the contact solver on the
first steps; with the stiff ABC contacts that push can launch it metres away.
``free_object_penetrations`` lists such overlaps at the reset pose so a bad
placement is reported at reset instead of surfacing as a vanished object or a
scattered scene mid-episode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# Contacts deeper than this at the reset pose are reported. Resting contacts
# settle well under 1 mm; the spawn overlaps that launched objects were 9-105 mm
# deep, so 5 mm flags real overlaps without flagging contact noise.
PENETRATION_TOLERANCE_M = 0.005
# Free bodies parked below this height are hidden, inactive objects.
PARKED_BELOW_Z_M = -0.25


@dataclass(frozen=True)
class Penetration:
    body: str
    geom: str
    other_body: str
    other_geom: str
    depth_m: float

    def describe(self) -> str:
        return (
            f"{self.body} ({self.geom}) starts {self.depth_m * 1000:.1f} mm inside "
            f"{self.other_body} ({self.other_geom})"
        )


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, index: int, fallback: str) -> str:
    return mujoco.mj_id2name(model, obj, index) or f"{fallback} {index}"


def free_object_penetrations(
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    *,
    tolerance: float = PENETRATION_TOLERANCE_M,
) -> list[Penetration]:
    """Contacts deeper than ``tolerance`` that involve a free-jointed object.

    Collision runs on a scratch ``MjData`` at ``data.qpos`` (``qpos0`` if
    ``data`` is None), so the caller's state is never touched. Free objects
    parked below ``PARKED_BELOW_Z_M`` are ignored. The deepest contact per pair
    of geoms is reported, deepest first.
    """
    scratch = mujoco.MjData(model)
    if data is not None:
        scratch.qpos[:] = data.qpos
        if model.nmocap:
            scratch.mocap_pos[:] = data.mocap_pos
            scratch.mocap_quat[:] = data.mocap_quat
    mujoco.mj_fwdPosition(model, scratch)

    free_roots: set[int] = set()
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body = int(model.jnt_bodyid[joint])
        if scratch.xpos[body][2] < PARKED_BELOW_Z_M:
            continue
        free_roots.add(body)
    if not free_roots:
        return []

    def free_root(geom: int) -> int | None:
        body = int(model.geom_bodyid[geom])
        while body > 0:
            if body in free_roots:
                return body
            body = int(model.body_parentid[body])
        return None

    deepest: dict[tuple[int, int], float] = {}
    for index in range(scratch.ncon):
        contact = scratch.contact[index]
        depth = -float(contact.dist)
        if not np.isfinite(depth) or depth <= tolerance:
            continue
        geom_1, geom_2 = int(contact.geom1), int(contact.geom2)
        root_1, root_2 = free_root(geom_1), free_root(geom_2)
        if root_1 is None and root_2 is None:
            continue
        if root_1 is None:
            geom_1, geom_2 = geom_2, geom_1
        key = (geom_1, geom_2)
        deepest[key] = max(deepest.get(key, 0.0), depth)

    found = [
        Penetration(
            body=_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]), "body"),
            geom=_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom, "geom"),
            other_body=_name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other]), "body"
            ),
            other_geom=_name(model, mujoco.mjtObj.mjOBJ_GEOM, other, "geom"),
            depth_m=depth,
        )
        for (geom, other), depth in deepest.items()
    ]
    return sorted(found, key=lambda item: -item.depth_m)


def warn_on_free_object_penetrations(
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    *,
    tolerance: float = PENETRATION_TOLERANCE_M,
    already_warned: set[tuple[str, str]] | None = None,
) -> list[Penetration]:
    """Log each of :func:`free_object_penetrations` as a warning; return them all.

    With ``already_warned``, a (body, other body) pair is logged only the first
    time it is seen and then added to the set, so a recurring overlap does not
    flood the log on every reset.
    """
    found = free_object_penetrations(model, data, tolerance=tolerance)
    for penetration in found:
        key = (penetration.body, penetration.other_body)
        if already_warned is not None:
            if key in already_warned:
                continue
            already_warned.add(key)
        logger.warning("Free object spawned inside other geometry: %s", penetration.describe())
    return found
