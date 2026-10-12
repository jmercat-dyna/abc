"""Directional placement geometry shared by sampling and scoring."""

import numpy as np

ALONG_MIN_M = 0.06
ALONG_MAX_M = 0.30
PERP_MAX_M = 0.08


def relative_placement_geometry(
    mover_xy: np.ndarray,
    reference_xy: np.ndarray,
    direction_xy: np.ndarray,
    *,
    along_min_m: float = ALONG_MIN_M,
    along_max_m: float = ALONG_MAX_M,
    perp_max_m: float = PERP_MAX_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return along/perpendicular distances and membership in the goal band."""
    axis = np.asarray(direction_xy, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    relative_xy = np.asarray(mover_xy, dtype=np.float32) - np.asarray(
        reference_xy, dtype=np.float32
    )
    along = relative_xy @ axis
    perp = np.abs(relative_xy @ np.asarray([-axis[1], axis[0]], dtype=np.float32))
    in_cone = (along >= along_min_m) & (along <= along_max_m) & (perp <= perp_max_m)
    return along, perp, in_cone
