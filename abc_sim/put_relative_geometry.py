"""Directional placement geometry shared by sampling and scoring."""

import numpy as np

from abc_sim.put_relative_config import DEFAULT_PUT_RELATIVE_CONFIG, PutRelativeConfig


def relative_placement_geometry(
    mover_xy: np.ndarray,
    reference_xy: np.ndarray,
    direction_xy: np.ndarray,
    *,
    config: PutRelativeConfig = DEFAULT_PUT_RELATIVE_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return along/perpendicular distances and membership in the goal band."""
    axis = np.asarray(direction_xy, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    relative_xy = np.asarray(mover_xy, dtype=np.float32) - np.asarray(
        reference_xy, dtype=np.float32
    )
    along = relative_xy @ axis
    perp = np.abs(relative_xy @ np.asarray([-axis[1], axis[0]], dtype=np.float32))
    in_cone = (
        (along >= config.along_min_m)
        & (along <= config.along_max_m)
        & (perp <= config.perp_max_m)
    )
    return along, perp, in_cone
