import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from v7.data.cache import PreparedCTCache, assert_same_batch
from v7.data.anatomy import PREPARED_DIRECTORY
from tests.test_v7 import tiny
from tests import test_shape_features


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shape_only_model_requires_no_hu_and_preserves_outputs(self):
        model, batch = tiny(); test_shape_features.ShapeFeatureTests.ready(model); model.eval()
        with torch.no_grad():
            before = model(batch['image'], image_hu=batch['image_hu'], object_threshold=0.)
            batch.pop('image_hu')
            after = model(batch['image'], object_threshold=0.)
        for name in ['radiomics_features', 'diagnostic_features', 'risk_logit', 'fine_mask_logits']:
            torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
        model.train()
        out = model(batch['image'], batch=batch, update_bank=False)
        out.ensemble_logits.sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.semantics.parameters()))

    def test_semantic_dataset_can_skip_only_the_extra_hu_read(self):
        from v7._core.common.semantic_data import WholeCTLIDCDataset, V2Dataset
        from v7._core.common.features import SEMANTIC_INDICES_V2
        data = WholeCTLIDCDataset.__new__(WholeCTLIDCDataset)
        data.include_image_hu = False
        original = dict(image=torch.rand(1,4,4,4), semantic_histograms=torch.rand(2,7,5),
                        semantic_targets=torch.rand(2,7), semantic_reader_counts=torch.ones(2,7))
        with patch.object(V2Dataset, '__getitem__', return_value=dict(original)), \
             patch('v7._core.common.semantic_data.nib.load', side_effect=AssertionError('HU read')):
            out = data[0]
        self.assertIs(out['image'], original['image'])
        self.assertNotIn('image_hu', out)
        for name in ['semantic_histograms','semantic_targets','semantic_reader_counts']:
            torch.testing.assert_close(out[name], original[name][:,list(SEMANTIC_INDICES_V2)], rtol=0, atol=0)

    def test_cache_exact_roundtrip_missing_required_and_source_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'image.nii.gz'; image.write_bytes(b'original CT')
            prepared = root/PREPARED_DIRECTORY; prepared.mkdir()
            name = 'patient__scan1'
            (prepared/f'{name}.npz').write_bytes(b'original anatomy')
            mask = root/'patient__scan1__ann1_mask.nii.gz'; mask.write_bytes(b'official mask')
            class Dataset(SimpleNamespace):
                def __len__(self):return 1
            dataset = Dataset(include_image_hu=False, target_spacing=1., target_mask_shape=(4,4,4),
                min_votes=2, exclude_indeterminate=True, cases=[dict(patient_id='patient',scan_id='1',image=str(image))],
                anatomy_root=root, official_mask_dir=root, anatomy_metadata=[dict(case_id=name)],
                annotations={('patient','1'):[dict(annotation_ids=(1,), reader_features=np.ones((1,6)))]})
            cache = PreparedCTCache(dataset,root/'cache',required=True)
            batch = dict(case_id=name,image=torch.arange(64,dtype=torch.float32).reshape(1,4,4,4),
                anatomy_packed=torch.ones(4,4,4,dtype=torch.int16), target_mask_crops=[torch.ones(1,2,3,dtype=torch.uint8)],
                semantic_histograms=torch.rand(1,6,5), target_mask_origins=torch.tensor([[1,2,1]]),
                risk_target=torch.tensor(1.), isotropic_shape=(4,4,4), ignored_nodule_ids=[])
            with self.assertRaises(FileNotFoundError):cache.get(0,lambda _:batch)
            self.assertTrue(cache.build(0,lambda _:batch))
            assert_same_batch(batch,cache.get(0,lambda _:self.fail('Cache rebuilt')))
            self.assertFalse(cache.build(0,lambda _:self.fail('Cache rebuilt')))
            cache.validate_all()
            bad=copy.deepcopy(batch);bad['target_mask_origins'][0,0]+=1
            with self.assertRaises(AssertionError):assert_same_batch(batch,bad)
            mask.write_bytes(b'changed official mask')
            with self.assertRaisesRegex(ValueError,'Stale'):cache.get(0,lambda _:batch)

    def test_separable_anatomy_dilation_preserves_values_and_tied_gradients(self):
        from v7.models.joint import group_probabilities
        for values in [torch.rand(1,13,9,10,11), torch.ones(1,13,9,10,11),
                       torch.randint(0,4,(1,13,9,10,11)).float()/4]:
            x = values.requires_grad_()
            old, new = group_probabilities(x), group_probabilities(x, fast=True)
            torch.testing.assert_close(old,new,rtol=0,atol=0)
            weight = torch.rand_like(old)
            a=torch.autograd.grad((old*weight).sum(),x,retain_graph=True)[0]
            b=torch.autograd.grad((new*weight).sum(),x)[0]
            torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-5)

    def test_linear_proposal_scores_preserve_selection_ties_crops_and_empty_case(self):
        from v7._core.model_v5.native_proposals import native_proposals
        p=torch.rand(24,25,26)*.08
        p[1:4,1:5,1:6]=.7; p[9:12,8:11,12:15]=.9; p[18:21,18:21,18:21]=.9
        for value in [p,torch.zeros_like(p)]:
            a=native_proposals(value,(8,8,8),threshold=.1,maximum=2)
            b=native_proposals(value,(8,8,8),threshold=.1,maximum=2,linear_scores=True)
            assert_same_batch(vars(a),vars(b))

    def test_complete_loss_and_parameter_gradients_match_fast_kernels(self):
        from v7.training.loss import WholeCTCriterionV7
        a,batch=tiny(True);b,_=tiny(True)
        a.fast_kernels=False;b.fast_kernels=True
        test_shape_features.ShapeFeatureTests.ready(a)
        b.load_state_dict(a.state_dict())
        criterion=WholeCTCriterionV7()
        outputs=[];losses=[]
        for model in [a,b]:
            model.train();torch.manual_seed(51)
            out=model(batch['image'],batch=batch,teacher_probability=1.,update_bank=False)
            loss=criterion(out,batch,diagnostic_scale=1.)
            loss.total.backward();outputs.append(out);losses.append(loss.total.detach())
        torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
        for name in ['boxes','fine_mask_logits','radiomics_features','semantic_features','risk_logit']:
            torch.testing.assert_close(getattr(outputs[0],name),getattr(outputs[1],name),rtol=0,atol=0)
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-5,msg=name)

    def test_resume_performance_changes_do_not_allow_training_policy_changes(self):
        from v7.training.runtime import validate_resume_arguments
        config=dict(world_size=4,lr_main=1e-4,anatomy_weight=.1,mask_temperature_end=.1,
                    benchmark_steps=None,skip_save=False,wandb_mode='online')
        args=SimpleNamespace(**{k:v for k,v in config.items() if k!='world_size'},
                             data_workers=2,data_cache=Path('/cache'),fast_kernels=True,
                             prefetch_factor=1,allow_world_size_change=True)
        validate_resume_arguments(args,config,8)
        for name,value in [('lr_main',1e-3),('anatomy_weight',1.),('mask_temperature_end',.2)]:
            bad=copy.copy(args);setattr(bad,name,value)
            with self.assertRaises(ValueError):validate_resume_arguments(bad,config,8)
        bad=copy.copy(args);bad.allow_world_size_change=False
        with self.assertRaises(ValueError):validate_resume_arguments(bad,config,8)
        bench=copy.copy(args);bench.benchmark_steps=4;bench.skip_save=True
        with self.assertRaisesRegex(ValueError,'disable W&B'):validate_resume_arguments(bench,config,8)
        bench.wandb_mode='disabled';validate_resume_arguments(bench,config,8)

    def test_resize_preserves_old_rng_streams_and_seeds_distinct_new_ranks(self):
        import random
        from v7.training.runtime import restore_rank_rng
        torch.manual_seed(81);np.random.seed(81);random.seed(81)
        state=dict(torch=torch.get_rng_state(),cuda=torch.arange(16,dtype=torch.uint8),
                   numpy=np.random.get_state(),python=random.getstate())
        expected=(torch.rand(4),np.random.rand(4),random.random())
        current=[state['cuda'].clone()]
        with patch('v7.training.runtime.adapt_cuda_rng_state',side_effect=lambda s,n:s.clone()), \
             patch('torch.cuda.get_rng_state',side_effect=lambda d:current[0].clone()), \
             patch('torch.cuda.set_rng_state',side_effect=lambda s,d:current.__setitem__(0,s.clone())), \
             patch('torch.cuda.manual_seed'),patch('torch.cuda.manual_seed_all'):
            info=restore_rank_rng([state]*4,rank=2,world=8,seed=42,completed_epochs=6,
                                  device='cuda',allow_resize=True)
            self.assertTrue(info['restored'])
            torch.testing.assert_close(torch.rand(4),expected[0],rtol=0,atol=0)
            np.testing.assert_array_equal(np.random.rand(4),expected[1])
            self.assertEqual(random.random(),expected[2])
            streams=[]
            for rank in [4,5,4]:
                info=restore_rank_rng([state]*4,rank=rank,world=8,seed=42,completed_epochs=6,
                                      device='cuda',allow_resize=True)
                self.assertFalse(info['restored']);streams.append(torch.rand(4))
            self.assertFalse(torch.equal(streams[0],streams[1]))
            torch.testing.assert_close(streams[0],streams[2],rtol=0,atol=0)
            with self.assertRaises(ValueError):
                restore_rank_rng([],rank=0,world=8,seed=42,completed_epochs=6,
                                 device='cuda',allow_resize=True)


if __name__ == '__main__':
    unittest.main()
