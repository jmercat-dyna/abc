"""Shared, validated scoring configuration for put-relative scene sampling and evaluation."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PutRelativeConfig:
    along_min_m: float = 0.06
    along_max_m: float = 0.30
    perp_max_m: float = 0.08
    placed_z_tolerance_m: float = 0.03

    def __post_init__(self) -> None:
        values = (
            self.along_min_m,
            self.along_max_m,
            self.perp_max_m,
            self.placed_z_tolerance_m,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(
                "put-relative scoring distances must be finite and nonnegative"
            )
        if self.along_min_m >= self.along_max_m:
            raise ValueError(
                "put-relative along_min_m must be smaller than along_max_m"
            )


DEFAULT_PUT_RELATIVE_CONFIG = PutRelativeConfig()
