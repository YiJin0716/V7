"""Direct malignancy prediction using FasterRisk's continuous stages 1 and 2."""
import torch

from v7._core.common.features import FEATURE_NAMES
from v7._core.common.rashomon import ContinuousRashomonBank

SCHEMA = 'joint_radiomics_semantic_continuous_fasterrisk_v6'


class JointRiskBank(ContinuousRashomonBank):
    """One sparse model pool over concatenated radiomics and CNN semantics.

    All 24 columns compete for the same sparse support. There is no radiomics
    baseline, offset, residual label, integerization, or trainable risk MLP.
    Fitting uses detached training rows; frozen coefficients retain input grads.
    """

    def __init__(self, **kwargs):
        super().__init__(FEATURE_NAMES, **kwargs)

    def get_extra_state(self):
        return dict(**super().get_extra_state(), schema=SCHEMA)

    def set_extra_state(self, state):
        if state.get('schema') != SCHEMA:
            raise ValueError('V6 requires a direct joint-feature risk bank')
        super().set_extra_state(state)

    @torch.no_grad()
    def observe_and_refit(self, features, labels, keys, *, refit=True):
        if features.ndim != 2 or features.shape != (len(labels), len(FEATURE_NAMES)):
            raise ValueError('V6 expects one 24-feature row per nodule label')
        if labels.ndim != 1 or len(keys) != len(labels):
            raise ValueError('V6 feature, label, and key counts differ')
        return super().observe_and_refit(features, labels, keys, refit=refit)
