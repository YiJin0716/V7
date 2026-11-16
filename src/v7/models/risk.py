"""Direct FasterRisk over eight mask-shape and six semantic features."""
import torch
from v7._core.common.rashomon import ContinuousRashomonBank
from v7.models.radiomics import FEATURE_NAMES

SCHEMA = 'shape_semantic_continuous_fasterrisk_v7'


class ShapeSemanticRiskBank(ContinuousRashomonBank):
    def __init__(self, **kwargs):
        super().__init__(FEATURE_NAMES, **kwargs)

    def get_extra_state(self):
        return dict(**super().get_extra_state(), schema=SCHEMA)

    def set_extra_state(self, state):
        if state.get('schema') != SCHEMA:
            raise ValueError('V7 requires the 14-feature shape/semantic risk bank')
        super().set_extra_state(state)

    @torch.no_grad()
    def observe_and_refit(self, features, labels, keys, *, refit=True):
        if features.ndim != 2 or features.shape != (len(labels), len(FEATURE_NAMES)):
            raise ValueError('V7 expects one 14-feature row per nodule label')
        if labels.ndim != 1 or len(keys) != len(labels):
            raise ValueError('V7 feature, label, and key counts differ')
        return super().observe_and_refit(features, labels, keys, refit=refit)
