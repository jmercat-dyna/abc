"""Reset-time sanity check of free objects' mass properties.

An object whose mass sits on a small fragment of its geometry (a point-like
inertia, or a centre of mass outside its collision shape) is launched at its
first stiff contact. ``free_object_mass_problems`` flags such objects so a bad
asset is reported at reset instead of surfacing as a vanished object mid-episode.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# A body's smallest radius of gyration, sqrt(I_min / m), below this fraction of
# its smallest collision extent is point-mass-like (sound mugs: ~0.2; a mug with
# its mass on a handle fragment: ~0.03).
MIN_GYRATION_FRACTION = 0.1
# Allowed centre-of-mass excursion beyond the collision bounding box, as a
# fraction of the box extent.
COM_BOX_MARGIN_FRACTION = 0.1


def _subtree_bodies(model: mujoco.MjModel, root: int) -> list[int]:
    bodies = [root]
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.append(body)
    return bodies


def _composite_inertia(
    model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]
) -> tuple[float, np.ndarray]:
    """Total mass and world-frame inertia tensor of ``bodies`` about their CoM."""
    masses = np.asarray([model.body_mass[body] for body in bodies])
    total = float(masses.sum())
    if total <= 0:
        return 0.0, np.zeros((3, 3))
    positions = np.asarray([data.xipos[body] for body in bodies])
    com = (masses[:, None] * positions).sum(axis=0) / total
    tensor = np.zeros((3, 3))
    for body, mass, position in zip(bodies, masses, positions):
        rotation = np.asarray(data.ximat[body]).reshape(3, 3)
        offset = position - com
        tensor += rotation @ np.diag(model.body_inertia[body]) @ rotation.T
        tensor += mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))
    return total, tensor


def _collision_box(
    model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    body_set = set(bodies)
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in body_set]
    colliding = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for geom in colliding or geoms:
        rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
        center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
        half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
        low = np.minimum(low, center - half)
        high = np.maximum(high, center + half)
    return low, high


def free_object_mass_problems(
    model: mujoco.MjModel, data: mujoco.MjData | None = None
) -> list[str]:
    """Describe each free-jointed body with implausible mass properties.

    Kinematics are recomputed on a scratch ``MjData`` (at ``data.qpos`` if
    given, else ``qpos0``), so the caller's state is never touched.
    """
    scratch = mujoco.MjData(model)
    if data is not None:
        scratch.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    problems: list[str] = []
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        root = int(model.jnt_bodyid[joint])
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or f"body {root}"
        )
        bodies = _subtree_bodies(model, root)
        mass, tensor = _composite_inertia(model, scratch, bodies)
        low, high = _collision_box(model, scratch, bodies)
        if mass <= 0 or not np.all(np.isfinite(low)):
            continue
        extent = high - low
        smallest_extent = float(extent.min())
        gyration = float(np.sqrt(max(np.linalg.eigvalsh(tensor).min(), 0.0) / mass))
        if smallest_extent > 0 and gyration < MIN_GYRATION_FRACTION * smallest_extent:
            problems.append(
                f"{name}: radius of gyration {gyration * 100:.2f} cm is under "
                f"{MIN_GYRATION_FRACTION:g} x its smallest extent "
                f"{smallest_extent * 100:.1f} cm"
            )
        com = np.asarray(scratch.subtree_com[root])
        margin = COM_BOX_MARGIN_FRACTION * extent
        if np.any(com < low - margin) or np.any(com > high + margin):
            offset = float(np.linalg.norm(com - (low + high) / 2))
            problems.append(
                f"{name}: centre of mass is outside its collision box "
                f"({offset * 100:.1f} cm from its centre)"
            )
    return problems


def warn_on_free_object_mass_problems(
    model: mujoco.MjModel, data: mujoco.MjData | None = None
) -> list[str]:
    """Log each of :func:`free_object_mass_problems` as a warning; return them."""
    problems = free_object_mass_problems(model, data)
    for problem in problems:
        logger.warning("Implausible free-object mass properties: %s", problem)
    return problems
