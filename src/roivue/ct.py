"""CT conventions as optional helpers. The library never applies them on its
own; loaders call them and record that they did (see the examples)."""

from __future__ import annotations

import numpy as np

# Display windows, (low, high) in HU, for Case(display={"window": ...}).
LUNG_WINDOW = (-1350.0, 150.0)
SOFT_TISSUE_WINDOW = (-160.0, 240.0)
BONE_WINDOW = (-450.0, 1050.0)

# Scanners store pixels outside the circular field of view as padding (often
# -2000 or -3024 HU); tissue and air never go below about -1100 HU.
PADDING_BELOW_HU = -1500.0


def unpad(ct: np.ndarray, below: float = PADDING_BELOW_HU, air: float = -1000.0) -> int:
    """Sets scanner padding (< below) to air, in place; returns how many voxels changed."""
    padding = ct < below
    ct[padding] = air
    return int(padding.sum())
