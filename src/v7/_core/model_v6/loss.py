"""V5 hard objectives, with twice the default missed-nodule penalty."""
import math

from v7._core.model_v5.loss import WholeCTCriterionV5, LossAccumulator, TERM_NAMES

LOSS_SCHEMA = 'v6_direct_risk_missing4_v1'


class WholeCTCriterionV6(WholeCTCriterionV5):
    def __init__(self, *, missing_nodule_weight=4., **kwargs):
        if not math.isfinite(missing_nodule_weight) or missing_nodule_weight <= 0:
            raise ValueError('Missing-nodule weight must be finite and positive')
        super().__init__(missing_nodule_weight=missing_nodule_weight, **kwargs)
