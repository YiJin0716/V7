import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from v7._core.common.features import FEATURE_NAMES
from v7._core.common.rashomon import scan_logits_per_model
from v7._core.common.schedule import TemperatureSchedule
from v7._core.model_v5.loss import WholeCTCriterionV5
from v7._core.model_v5.preparation import SCHEMA
from tests.test_v5 import tiny as tiny_v5
from v7._core.model_v6.initialization import load_warmup_network
from v7._core.model_v6.loss import WholeCTCriterionV6
from v7._core.model_v6.model import WholeCTJointModelV6, ARCHITECTURE
from v7._core.model_v6.risk import JointRiskBank
from tests.model_fixtures import TinySegmenter


def tiny(detector_kind='legacy'):
    old, batch = tiny_v5(detector_kind)
    model = WholeCTJointModelV6(window_size=(8, 8, 8), screen_shape=(8, 8, 8),
        roi_shape=(8, 8, 8), num_queries=4, detr_hidden_dim=16,
        detr_coarse_shape=(4, 4, 4), detr_nheads=4, detr_encoder_layers=1,
        detr_decoder_layers=1, vista_feature_dim=1, context_channels=2,
        refiner_base_channels=2, hard_negatives=1, teacher_jitter=0,
        segmenter_factory=TinySegmenter, semantic_roi_size=8, semantic_width=4,
        semantic_min_dice=0., use_checkpoint=False, detector_kind=detector_kind)
    load_warmup_network(model, dict(model=old.state_dict(),
        audit=dict(complete=True, schema=SCHEMA, smoke_only=False)))
    with torch.no_grad():
        model.rashomon.count.fill_(2)
        model.rashomon.weights[0, 0] = .02
        model.rashomon.weights[0, 18] = .5
        model.rashomon.weights[1, 1] = -.02
        model.rashomon.weights[1, 19] = -.5
    return model, batch


class V6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_geometry_and_features_equal_v5_hard(self):
        for kind in ('legacy', 'native_seeded'):
            old, b = tiny_v5(kind)
            new, _ = tiny(kind)
            for temperature in (1., .1):
                old.set_temperature(temperature); new.set_temperature(temperature)
                for training, teacher in ((False, 0.), (True, 1.), (True, 0.)):
                    old.train(training); new.train(training)
                    torch.manual_seed(123)
                    a = old(b['image'], batch=b, teacher_probability=teacher,
                            object_threshold=0., update_bank=False)
                    torch.manual_seed(123)
                    c = new(b['image'], batch=b, teacher_probability=teacher,
                            object_threshold=0., update_bank=False)
                    for name in ('boxes', 'object_logits', 'coarse_mask_logits',
                        'fine_mask_logits', 'fine_mask_probabilities', 'crop_origins',
                        'semantic_features', 'radiomics_features', 'semantic_probabilities',
                        'fine_soft_dice', 'semantic_supervision_valid', 'refined_target_indices'):
                        torch.testing.assert_close(getattr(a, name), getattr(c, name), rtol=0, atol=0)

    def test_one_direct_head_and_live_semantic_gradient(self):
        model, b = tiny(); model.train()
        out = model(b['image'], batch=b, teacher_probability=1., update_bank=False)
        bank = model.rashomon
        x = ((out.diagnostic_features - bank.mean) / bank.scale).clamp(-10., 10.) * bank.varying
        expected = x @ bank.weights[out.model_indices].T + bank.intercepts[out.model_indices]
        torch.testing.assert_close(out.ensemble_logits, expected)
        torch.testing.assert_close(out.scan_logits, scan_logits_per_model(expected,
                                   out.object_logits[out.refined_query_indices]))
        self.assertEqual(out.diagnostic_features.shape[1], 24)
        self.assertFalse(hasattr(bank, 'radiomics_weights'))
        self.assertFalse(hasattr(out, 'residual_targets'))
        self.assertFalse(out.radiomics_features.requires_grad)
        out.ensemble_logits.sum().backward()
        for module in (model.semantics, model.refiner):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()
                                   if p.grad is not None), 0)

    def test_missing_penalty_doubles_and_survives_empty_detection(self):
        model, b = tiny(); model.train()
        out = model(b['image'], batch=b, teacher_probability=0., object_threshold=1., update_bank=False)
        old = WholeCTCriterionV5()(out, b, synchronize=False)
        new = WholeCTCriterionV6()(out, b, synchronize=False)
        self.assertEqual(len(out.refined_query_indices), 0)
        self.assertGreater(new.mean('missing_nodule'), 0)
        torch.testing.assert_close(new.total - old.total,
                                   old.numerators['missing_nodule'].double() / old.normalizers['missing_nodule'] * 2)
        new.total.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.detr.parameters()
                               if p.grad is not None), 0)
        self.assertEqual(new.counts['risk'], 0)

    def test_warmup_load_is_strict_and_never_reuses_risk(self):
        old, _ = tiny_v5(); new, _ = tiny()
        new.rashomon.count.zero_()
        source = dict(model=old.state_dict(), audit=dict(complete=True, schema=SCHEMA))
        state = load_warmup_network(new, source)
        for key, value in state.items():
            torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(int(new.rashomon.count), 0)
        with self.assertRaisesRegex(ValueError, 'production warmup'):
            load_warmup_network(new, dict(source, epoch=0))
        corrupt = copy.deepcopy(source)
        del corrupt['model']['mask_temperature']
        with self.assertRaisesRegex(ValueError, 'network mismatch'):
            load_warmup_network(new, corrupt)

    def test_inference_is_label_independent_and_state_roundtrips(self):
        model, b = tiny('native_seeded'); model.eval(); model.set_temperature(.1)
        a = model(b['image'], batch=b, object_threshold=0., update_bank=False)
        c = model(b['image'], image_hu=b['image_hu'], object_threshold=0., update_bank=False)
        torch.testing.assert_close(a.risk_logit, c.risk_logit, rtol=0, atol=0)
        stream = io.BytesIO(); torch.save(model.state_dict(), stream); stream.seek(0)
        other, _ = tiny('native_seeded')
        other.load_state_dict(torch.load(stream, weights_only=False), strict=True); other.eval()
        d = other(b['image'], image_hu=b['image_hu'], object_threshold=0., update_bank=False)
        torch.testing.assert_close(c.risk_logit, d.risk_logit, rtol=0, atol=0)
        self.assertAlmostEqual(float(other.mask_temperature), .1)

    def test_continuous_fit_uses_both_feature_groups_and_training_only_cache(self):
        rng = np.random.default_rng(17)
        x = np.zeros((300, 24)); x[:, [0, 18]] = rng.normal(size=(300, 2))
        y = (x[:, 0] + x[:, 18] + rng.normal(scale=.4, size=300) > 0).astype(float)
        bank = JointRiskBank(sparsity=2, beam=3, pool_size=5)
        bank.allowed_cases = {'train'}
        keys = [('train', i) for i in range(len(x))]
        bank.observe_and_refit(torch.tensor(x), torch.tensor(y), keys)
        support = set(torch.nonzero(bank.weights[0]).flatten().tolist())
        self.assertEqual(support, {0, 18})
        self.assertFalse(torch.equal(bank.weights, bank.weights.round()))
        bank.eval()
        logits, _ = bank(torch.tensor(x), all_models=True)
        normalized = ((torch.tensor(x).float() - bank.mean) / bank.scale).clamp(-10., 10.) * bank.varying
        torch.testing.assert_close(logits, normalized @ bank.weights[:int(bank.count)].T + bank.intercepts[:int(bank.count)])
        with self.assertRaisesRegex(RuntimeError, 'Evaluation'):
            bank.observe_and_refit(torch.tensor(x), torch.tensor(y), keys)
        bank.train()
        with self.assertRaisesRegex(ValueError, 'Non-training'):
            bank.observe_and_refit(torch.tensor(x[:1]), torch.tensor(y[:1]), [('test', 1)])

    def test_hard_schedule(self):
        schedule = TemperatureSchedule(start=1., end=.1, epochs=18)
        self.assertEqual(schedule(0), 1.)
        self.assertAlmostEqual(schedule(17), .1)

    def test_shared_evaluator_loads_v6_without_a_baseline(self):
        from v7.inference.inference import load_joint_model
        model, b = tiny('native_seeded'); model.eval(); model.set_temperature(.1)
        config = dict(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
            num_queries=4, detr_hidden_dim=16, detr_coarse_shape=(4,4,4), detr_nheads=4,
            detr_encoder_layers=1, detr_decoder_layers=1, vista_feature_dim=1,
            context_channels=2, refiner_base=2, hard_negatives=1, teacher_jitter=0,
            semantic_roi_size=8, semantic_width=4, semantic_min_dice=0., detector_kind='native_seeded')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'epoch.pt'
            torch.save(dict(model=model.state_dict(), config=config, architecture=ARCHITECTURE, epoch=0), path)
            with patch('v7.inference.inference._uninitialized_segmenter', TinySegmenter):
                restored = load_joint_model(path, device='cpu')
            self.assertTrue(restored.evaluation_metadata['risk_bank_ready'])
            with torch.no_grad():
                expected = model(b['image'], image_hu=b['image_hu'], object_threshold=0.)
                actual = restored(b['image'], image_hu=b['image_hu'], object_threshold=0.)
            torch.testing.assert_close(actual.risk_logit, expected.risk_logit, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
