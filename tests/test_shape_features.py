import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from v7._core.common.features import SoftRadiomics3D, FEATURE_NAMES as OLD_NAMES
from v7.models.radiomics import (
    FEATURE_NAMES, RADIOMICS_NAMES, INTENSITY_NAMES, ShapeRadiomics3D,
    select_warmup_features,
)
from v7.models.joint import ARCHITECTURE, LEGACY_ARCHITECTURE
from v7.models.risk import ShapeSemanticRiskBank
from tests.test_v7 import tiny, TinyAnatomy


class ShapeFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_geometry_matches_original_values_and_mask_gradients_without_hu(self):
        hu = torch.randn(2, 1, 7, 8, 9) * 600
        logits = torch.randn_like(hu, requires_grad=True)
        valid = torch.ones_like(hu); valid[:, :, -2:] = 0
        old = SoftRadiomics3D()(hu, logits, valid)[:, :8]
        new = ShapeRadiomics3D()(hu, logits, valid)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        torch.testing.assert_close(ShapeRadiomics3D()(torch.full_like(hu, float('nan')), logits, valid),
                                   new, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(old.sum(), logits, retain_graph=True)[0],
                                   torch.autograd.grad(new.sum(), logits)[0], rtol=0, atol=0)
        self.assertEqual(len(RADIOMICS_NAMES), 8)
        self.assertEqual(len(INTENSITY_NAMES), 10)
        self.assertFalse(set(FEATURE_NAMES) & set(INTENSITY_NAMES))

    def test_paired_cache_projection_preserves_geometry_and_semantic_order(self):
        original = np.arange(48, dtype=float).reshape(2, 24)
        expected = original[:, [OLD_NAMES.index(n) for n in FEATURE_NAMES]]
        np.testing.assert_array_equal(select_warmup_features(original), expected)
        original[:, 8:18] = np.nan
        np.testing.assert_array_equal(select_warmup_features(original), expected)
        with self.assertRaises(ValueError):select_warmup_features(np.zeros(14))
        original[0, 18] = np.nan
        with self.assertRaises(ValueError):select_warmup_features(original)

    def test_risk_fit_has_only_14_columns_and_retains_both_groups(self):
        rng = np.random.default_rng(17)
        x = np.zeros((300, 14)); x[:, [0, 8]] = rng.normal(size=(300, 2))
        y = (x[:, 0] + x[:, 8] + rng.normal(scale=.4, size=300) > 0).astype(float)
        bank = ShapeSemanticRiskBank(sparsity=2, beam=3, pool_size=5)
        bank.allowed_cases = {'train'}
        keys = [('train', i) for i in range(len(x))]
        bank.observe_and_refit(torch.tensor(x), torch.tensor(y), keys)
        self.assertEqual(bank.weights.shape, (5, 14))
        self.assertEqual(set(torch.nonzero(bank.weights[0]).flatten().tolist()), {0, 8})
        self.assertTrue(all(len(v[0]) == 14 for v in bank.memory.values()))
        restored = ShapeSemanticRiskBank(sparsity=2, beam=3, pool_size=5)
        restored.load_state_dict(copy.deepcopy(bank.state_dict()), strict=True)
        bank.eval(); restored.eval()
        torch.testing.assert_close(bank(torch.tensor(x))[0], restored(torch.tensor(x))[0])
        bank.train()
        with self.assertRaises(ValueError):
            bank.observe_and_refit(torch.zeros(300, 24), torch.tensor(y), keys)
        bad = bank.get_extra_state(); bad['schema'] = 'joint_radiomics_semantic_continuous_fasterrisk_v6'
        with self.assertRaises(ValueError):bank.set_extra_state(bad)

    @staticmethod
    def ready(model):
        semantic_start = len(model.radiomics_names)
        with torch.no_grad():
            model.rashomon.count.fill_(2)
            model.rashomon.weights[0, 0] = .02
            model.rashomon.weights[0, semantic_start] = .5
            model.rashomon.weights[1, 1] = -.02
            model.rashomon.weights[1, semantic_start + 1] = -.5

    def test_full_forward_risk_is_hu_invariant_with_live_semantic_gradients(self):
        model, b = tiny(); self.ready(model); model.eval()
        with torch.no_grad():
            a = model(b['image'], image_hu=b['image_hu'], object_threshold=0.)
            c = model(b['image'], image_hu=torch.full_like(b['image_hu'], 2000), object_threshold=0.)
        self.assertEqual(a.radiomics_features.shape[1], 8)
        self.assertEqual(a.diagnostic_features.shape[1], 14)
        torch.testing.assert_close(a.radiomics_features, c.radiomics_features, rtol=0, atol=0)
        torch.testing.assert_close(a.risk_logit, c.risk_logit, rtol=0, atol=0)
        model.train()
        out = model(b['image'], batch=b, teacher_probability=1., update_bank=False)
        self.assertFalse(out.radiomics_features.requires_grad)
        out.ensemble_logits.sum().backward()
        for module in (model.semantics, model.refiner):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_empty_and_geometry_only_outputs_keep_14_feature_layout(self):
        model, b = tiny(); model.eval()
        with torch.no_grad():
            out = model(b['image'], image_hu=b['image_hu'], object_threshold=1., compute_diagnostics=False)
            self.assertEqual(out.radiomics_features.shape, (0, 8))
            self.assertEqual(out.diagnostic_features.shape, (0, 14))
            out = model(b['image'], image_hu=b['image_hu'], object_threshold=0., geometry_only=True)
            self.assertEqual(out.radiomics_features.shape[1], 8)
            self.assertEqual(out.diagnostic_features.shape[1], 14)

    def test_evaluator_roundtrip_new_and_historical_v7(self):
        from v7.inference.inference import load_joint_model
        config = dict(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
            num_queries=4, detr_hidden_dim=16, detr_coarse_shape=(4,4,4), detr_nheads=4,
            detr_encoder_layers=1, detr_decoder_layers=1, vista_feature_dim=1,
            context_channels=2, refiner_base=2, hard_negatives=1, teacher_jitter=0,
            semantic_roi_size=8, semantic_width=4, semantic_min_dice=0., detector_kind='native_seeded',
            anatomy_dropout=0.)
        for legacy, architecture in [(False, ARCHITECTURE), (True, LEGACY_ARCHITECTURE)]:
            model, b = tiny(legacy_intensity_radiomics=legacy); self.ready(model); model.eval()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'epoch.pt'
                torch.save(dict(model=model.state_dict(), config=config, architecture=architecture, epoch=0), path)
                with patch('v7.inference.inference._uninitialized_segmenter', TinyAnatomy):
                    restored = load_joint_model(path, device='cpu')
                self.assertEqual(len(restored.rashomon.feature_names), 24 if legacy else 14)
                with torch.no_grad():
                    expected = model(b['image'], image_hu=b['image_hu'], object_threshold=0.)
                    actual = restored(b['image'], image_hu=b['image_hu'], object_threshold=0.)
                torch.testing.assert_close(actual.risk_logit, expected.risk_logit, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
