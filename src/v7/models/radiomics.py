"""V7 mask geometry and semantic features, excluding all intensity radiomics."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from v7._core.common.features import (
    FEATURE_NAMES as LEGACY_FEATURE_NAMES, RADIOMICS_NAMES as LEGACY_RADIOMICS_NAMES,
    SEMANTIC_NAMES,
)

RADIOMICS_NAMES = (
    'volume_mm3', 'log1p_volume_mm3', 'soft_surface_mm2', 'equivalent_diameter_mm',
    'radius_gyration_mm', 'sd_x_mm', 'sd_y_mm', 'sd_z_mm',
)
INTENSITY_NAMES = tuple(name for name in LEGACY_RADIOMICS_NAMES if name not in RADIOMICS_NAMES)
FEATURE_NAMES = RADIOMICS_NAMES + SEMANTIC_NAMES
WARMUP_FEATURE_INDICES = tuple(LEGACY_FEATURE_NAMES.index(name) for name in FEATURE_NAMES)


def select_warmup_features(features):
    """Project the known V5 paired cache by feature name before fitting V7 risk."""
    values = np.asarray(features)
    if values.ndim < 1 or values.shape[-1] != len(LEGACY_FEATURE_NAMES):
        raise ValueError('Expected the original 24-column paired warmup feature cache')
    selected = values[..., list(WARMUP_FEATURE_INDICES)].copy()
    if not np.isfinite(selected).all():
        raise ValueError('Nonfinite retained warmup geometry or semantic feature')
    return selected


class ShapeRadiomics3D(nn.Module):
    """The original eight geometric formulas on the 1-mm soft mask.

    The HU argument retains the shared extractor interface but is never read.
    """
    def forward(self, image_hu, logits, valid):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            p = logits.float().sigmoid() * valid.float()
            dims = (1, 2, 3, 4)
            mass = p.sum(dims)
            denom = mass.clamp_min(1.0)
            variances = []
            for axis in (2, 3, 4):
                marginal = p.sum(tuple(d for d in dims if d != axis))
                coordinates = torch.arange(p.shape[axis], device=p.device, dtype=p.dtype)
                centre = (marginal * coordinates).sum(1) / denom
                variances.append((marginal * (coordinates[None] - centre[:, None]).square()).sum(1) / denom)
            padded = F.pad(p, (1, 1, 1, 1, 1, 1))
            surface = sum(
                ((padded.diff(dim=axis).square() + 1e-6).sqrt() - 1e-3).sum(dims)
                for axis in (2, 3, 4)
            )
            values = [mass, torch.log1p(mass), surface,
                      (6.0 * mass.clamp_min(1e-6) / math.pi).pow(1.0 / 3.0),
                      (sum(variances) + 1e-6).sqrt()]
            values += [(v + 1e-6).sqrt() for v in variances]
            return torch.stack(values, dim=1)
