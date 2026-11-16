import copy
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from tests.test_v5 import tiny as tiny_v5
from v7.data.anatomy import (pack_groups, unpack_groups, pack_prompt_targets,
    LOBES, TOTAL_ROIS, PROMPT_IDS, NODULE_INDEX, READER_UNION_BIT)
from v7._core.common.base_model import DEFAULT_VISTA_SOURCE, VistaNoduleSegmenter
from v7.models.vista import VistaAnatomySegmenter, expand_input, set_vista_trainable
from v7.models.joint import WholeCTJointModelV7
from v7.training.loss import WholeCTCriterionV7, binary_segmentation_losses, patch_objectives, LossAccumulator
from v7.data.sampling import AnatomyOversamplingSampler
from v7.training.vista_warmup import (sample_patch, readiness, summarize_overlap,
    combined_readiness, cycle_case_indices, prefetch_cases,
    validate_resume_configuration, restore_optimizer)
from v7.training.vista_warmup import deployment_crop_center, WINDOW_POLICY
from v7.training.warmup_sampling import POLICY as NODULE_POLICY, visit_plan, coverage, require_coverage
from v7.training.warmup_background import (background_score, BackgroundCropMiner,
    POLICY as BACKGROUND_POLICY, COVERAGE_POLICY as BACKGROUND_COVERAGE_POLICY,
    coverage as background_coverage, load_coverage_visits as load_background_visits,
    require_coverage as require_background_coverage)
from v7.training.warmup_quality import (WHOLE_CT_CASE_IDS, whole_ct_readiness,
    completion_ready, require_whole_ct_checkpoint, full_ct_overlap)
from v7.training.initialization import load_warmup_network
from v7._core.model_v5.preparation import SCHEMA


sys.path.insert(0, str(DEFAULT_VISTA_SOURCE))
from vista3d.modeling.class_head import Class_Mapping_Classify


class TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = None
        self.up_layers_auto = nn.ModuleList([nn.ModuleDict(
            dict(upsample=nn.Identity(), blocks=nn.Identity(), head=nn.Identity()))])
        self.calls = 0

    def is_valid_shape(self, image):
        return True

    def encoder(self, image):
        self.calls += 1
        value = image.expand(-1,48,-1,-1,-1)
        return [value, value]


def tiny_backbone():
    model = nn.Module()
    model.image_encoder = TinyImageEncoder()
    # The real upstream prompt-conditioned head, not a substitute output head.
    model.class_head = Class_Mapping_Classify(512, 48)
    return model


class TinyAnatomy(VistaAnatomySegmenter):
    def __init__(self):
        super().__init__(tiny_backbone())
        self.pyramid_projections = nn.ModuleList([nn.Conv3d(48,1,1)])


class TinySingle(VistaNoduleSegmenter):
    def __init__(self):
        super().__init__(tiny_backbone())
        self.pyramid_projections = nn.ModuleList([nn.Conv3d(48,1,1)])


def tiny(use_checkpoint=False, legacy_intensity_radiomics=False):
    _, batch = tiny_v5('native_seeded')
    model = WholeCTJointModelV7(window_size=(8,8,8), screen_shape=(8,8,8), roi_shape=(8,8,8),
        num_queries=4, detr_hidden_dim=16, detr_coarse_shape=(4,4,4), detr_nheads=4,
        detr_encoder_layers=1, detr_decoder_layers=1, vista_feature_dim=1, context_channels=2,
        refiner_base_channels=2, hard_negatives=1, teacher_jitter=0, segmenter_factory=TinyAnatomy,
        semantic_roi_size=8, semantic_width=4, semantic_min_dice=0., use_checkpoint=use_checkpoint,
        anatomy_dropout=0., legacy_intensity_radiomics=legacy_intensity_radiomics)
    batch['anatomy_packed'] = torch.ones(batch['image'].shape[1:], dtype=torch.int16)
    batch['anatomy_packed'][4:7,4:7,4:7] |= (1<<READER_UNION_BIT)+(1<<NODULE_INDEX)+2
    return model, batch


class V7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_background_ranking_ignores_readers_targets_and_padding(self):
        logits = torch.zeros(1, 13, 1, 1, 5)
        logits[:, NODULE_INDEX] = torch.tensor([100., 80., 60., -2., 2.])
        packed = torch.tensor([[(1<<READER_UNION_BIT)+(1<<NODULE_INDEX),
                                1<<NODULE_INDEX, 0, 0, 0]], dtype=torch.int16).reshape(1,1,1,5)
        valid = torch.tensor([1,1,0,1,1], dtype=torch.bool).reshape(1,1,1,1,5)
        expected = torch.nn.functional.softplus(torch.tensor([-2.,2.])).mean().item()
        self.assertAlmostEqual(background_score(logits, packed, valid), expected)
        self.assertIsNone(background_score(logits, packed, torch.zeros_like(valid)))

    def test_background_selection_preserves_targets_mode_and_training(self):
        from v7._core.common.roi import integer_crop
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.))
                self.modes = []
            def forward_with_features(self, image):
                self.modes.append(self.training)
                return (image*self.scale).expand(-1,13,-1,-1,-1), ()
        image = torch.full((1,256,128,128), -8.)
        image[:,128:] = 2.
        image[:,:,0:8] = 100.  # Annotated foreground must not win the ranking.
        packed = torch.zeros((256,128,128), dtype=torch.int16)
        packed[:,0:8] = (1<<READER_UNION_BIT)+(1<<NODULE_INDEX)
        ct, _, valid = integer_crop(image, (64,64,64))
        target, _, _ = integer_crop(packed[None], (64,64,64))
        initial = (ct[None], target, valid[None], 'context')
        model = Probe().train()
        chosen, details = BackgroundCropMiner(model, image, packed).choose(initial)
        self.assertTrue(details['deployment_selected'])
        self.assertGreater(details['selected_score'], details['initial_score']+1.)
        self.assertTrue(model.training)
        self.assertEqual(model.modes, [False]*5)
        self.assertIsNone(model.scale.grad)
        self.assertTrue(torch.equal(chosen[1], target))
        self.assertTrue(torch.equal(chosen[2], valid[None]))
        self.assertEqual(chosen[3], 'context_deployment_window')
        logits, _ = model.forward_with_features(chosen[0])
        loss = patch_objectives(logits, chosen[1], chosen[2])[-1]
        loss.backward()
        self.assertTrue(torch.isfinite(model.scale.grad))

    def test_cached_background_windows_are_distinct_and_rescored(self):
        from v7._core.common.roi import integer_crop
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.))
                self.modes = []
            def forward_with_features(self, image):
                self.modes.append(self.training)
                return (image*self.scale).expand(-1,13,-1,-1,-1), ()
        image = torch.full((1,256,128,128), -4.)
        image[:,128:] = 2.
        image[:,:,0:8] = 100.
        packed = torch.zeros((256,128,128), dtype=torch.int16)
        packed[:,0:8] = (1<<READER_UNION_BIT)+(1<<NODULE_INDEX)
        ct, _, valid = integer_crop(image, (64,64,64))
        target, _, _ = integer_crop(packed[None], (64,64,64))
        initial = (ct[None], target, valid[None], 'context')
        model = Probe().train()
        miner = BackgroundCropMiner(model, image, packed)
        self.assertEqual(miner.ranked_windows, 3)
        chosen, first = miner.choose(initial)
        self.assertTrue(first['deployment_selected'])
        self.assertEqual(first['selected_origin'], [128,0,0])
        self.assertTrue(torch.equal(chosen[1], target))
        self.assertFalse(torch.is_inference(chosen[0]))
        with torch.no_grad():
            model.scale.fill_(-1.)
        chosen, second = miner.choose(initial)
        self.assertNotEqual(first['proposed_origin'], second['proposed_origin'])
        self.assertIs(chosen, initial)
        self.assertFalse(second['deployment_selected'])
        self.assertTrue(model.training)
        self.assertFalse(any(model.modes))
        self.assertIsNone(model.scale.grad)
        with torch.no_grad():
            model.scale.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            BackgroundCropMiner(model, image, packed)
        self.assertTrue(model.training)

    def test_readiness_preserves_the_legacy_nodule_reference_and_every_organ(self):
        from v7.data.anatomy import PROMPT_NAMES
        args=SimpleNamespace(min_dice=[.8]*12, max_organ_dice_drop=.05,
                             min_nodule_dice=.5, max_nodule_dice_drop=.05)
        ref=dict(dice=dict.fromkeys(PROMPT_NAMES,.9),legacy_nodule_dice=.88)
        ref['dice']['nodule']=.28
        current=dict(dice=dict.fromkeys(PROMPT_NAMES,.9),target_voxels=dict.fromkeys(PROMPT_NAMES,100))
        current['dice']['nodule']=.82
        result=readiness(current,ref,args)
        self.assertEqual(set(result['failures']), {'nodule'})
        self.assertAlmostEqual(result['thresholds']['nodule'], .83)
        current['dice']['nodule']=.84
        self.assertTrue(readiness(current,ref,args)['passed'])
        current['dice']['airway']=.84
        self.assertEqual(set(readiness(current,ref,args)['failures']), {'airway'})
        current['dice']['airway']=.9
        current['target_voxels']['nodule']=0
        self.assertFalse(readiness(current,ref,args)['passed'])

    def test_overlap_report_distinguishes_false_positives_and_false_negatives(self):
        sums=torch.zeros(13,3)
        sums[12]=torch.tensor([10,100,20])
        report=summarize_overlap(sums)
        self.assertAlmostEqual(report['dice']['nodule'], 1/6)
        self.assertAlmostEqual(report['precision']['nodule'], .1)
        self.assertAlmostEqual(report['recall']['nodule'], .5)
        self.assertEqual(report['target_voxels']['airway'], 0)

    def test_warmup_cycle_covers_every_ct_and_resume_does_not_skip_or_repeat(self):
        whole = cycle_case_indices(17, 42, 0, 51)
        for start in (0, 17, 34):
            self.assertEqual(sorted(whole[start:start+17]), list(range(17)))
        resumed = cycle_case_indices(17, 42, 0, 23)+cycle_case_indices(17, 42, 23, 28)
        self.assertEqual(whole, resumed)
        self.assertNotEqual(whole[:17], whole[17:34])

    def test_lr_continuation_retains_adam_history_and_matches_a_manual_continuation(self):
        params = [nn.Parameter(torch.tensor([1.])), nn.Parameter(torch.tensor([2.]))]
        old = torch.optim.AdamW([dict(params=[p], lr=1e-3) for p in params])
        for _ in range(4):
            old.zero_grad(); sum(p.square().sum() for p in params).backward(); old.step()
        state = copy.deepcopy(old.state_dict())
        resumed_params = [nn.Parameter(p.detach().clone()) for p in params]
        resumed = torch.optim.AdamW([dict(params=[p], lr=1e-3) for p in resumed_params])
        rates = [3e-4, 2e-4]
        change = restore_optimizer(resumed, copy.deepcopy(state), learning_rates=rates)
        self.assertEqual(change, dict(previous=[1e-3, 1e-3], current=rates))
        for p, q in zip(params, resumed_params):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(old.state[p][key], resumed.state[q][key])
        for group, rate in zip(old.param_groups, rates):
            group['lr'] = rate
        for optimizer, values in ((old, params), (resumed, resumed_params)):
            optimizer.zero_grad(); sum(p.square().sum() for p in values).backward(); optimizer.step()
        for p, q in zip(params, resumed_params):
            torch.testing.assert_close(p, q)
        for invalid in ([0., 1e-3], [float('nan'), 1e-3], [1e-3]):
            with self.assertRaises(ValueError):
                restore_optimizer(resumed, state, learning_rates=invalid)

    def test_lr_continuation_cannot_change_quality_thresholds_or_training_data(self):
        old = dict(lr_head=1e-5, lr_vista=1e-5, required_passes=3,
                   min_nodule_dice=.5, anatomy_cache_sha256='original')
        new = dict(old, lr_head=3e-6, lr_vista=3e-6,
                   resume_learning_rates=[3e-6, 3e-6], new_wandb_run=True)
        with self.assertRaises(ValueError):
            validate_resume_configuration(old, new)
        validate_resume_configuration(old, new, change_learning_rates=True)
        for key, value in [('required_passes', 1), ('min_nodule_dice', .1),
                           ('anatomy_cache_sha256', 'changed')]:
            with self.assertRaises(ValueError):
                validate_resume_configuration(old, dict(new, **{key:value}), change_learning_rates=True)

    def test_prefetch_preserves_ct_order_and_does_not_hide_read_errors(self):
        from unittest.mock import patch
        from pathlib import Path
        def load(index, root):
            if index == 2:
                raise OSError('unreadable training CT')
            return index
        with patch('v7.training.vista_warmup.load_case', load):
            result = list(prefetch_cases(list(range(4)), Path('.'), [3,0,1], 2))
            self.assertEqual(result, [(3,3),(0,0),(1,1)])
            with self.assertRaisesRegex(OSError, 'unreadable'):
                list(prefetch_cases(list(range(4)), Path('.'), [0,2,1], 2))

    def test_window_upgrade_is_explicit_and_preserves_data_and_original_thresholds(self):
        old = dict(augmentation='crop_center_jitter_preserve_anatomical_orientation',
                   max_nodule_dice_drop=.05, anatomy_cache_sha256='original')
        new = dict(old, augmentation=WINDOW_POLICY, deployment_patch_probability=.5,
                   whole_ct_case_ids=list(WHOLE_CT_CASE_IDS), upgrade_window_training=True)
        with self.assertRaises(ValueError):
            validate_resume_configuration(old, new)
        validate_resume_configuration(old, new, upgrade_window_training=True)
        for key, value in [('max_nodule_dice_drop', .2), ('anatomy_cache_sha256', 'changed'),
                           ('deployment_patch_probability', 0.), ('whole_ct_case_ids', [])]:
            with self.assertRaises(ValueError):
                validate_resume_configuration(old, dict(new, **{key:value}), upgrade_window_training=True)

    def test_nodule_visit_covers_every_retained_object_and_keeps_paired_negatives(self):
        metadata = dict(nodules=[dict(nodule_id=1,retained=True),
                                dict(nodule_id=2,retained=True),
                                dict(nodule_id=3,retained=False)])
        for seed in range(10):
            plan=visit_plan(metadata,np.random.default_rng(seed))
            self.assertEqual(sorted(i for i,p in plan if i is not None),[1,2])
            self.assertEqual(plan[-2:],[(None,True),(None,False)])
            self.assertEqual([p for _,p in plan],[True,False]*3)
        self.assertEqual(visit_plan(dict(nodules=[]),np.random.default_rng(1)),
                         [(None,True),(None,False)])
        # Different intensities let us identify the selected distant lesion.
        image=torch.zeros(1,100,100,100);packed=torch.zeros(100,100,100,dtype=torch.int16)
        nodes=[]
        for i,center in enumerate(([20,20,20],[75,75,75]),1):
            image[(0,*center)]=float(i)
            packed[tuple(center)]=(1<<NODULE_INDEX)|(1<<READER_UNION_BIT)
            nodes.append(dict(nodule_id=i,retained=True,malignant=1,diameter_mm=8,
                              center_mm_grid=center,distances_mm=dict(lung=0 if i==1 else 20)))
        for i in (1,2):
            ct,target,_,_=sample_patch(image,packed,dict(nodules=nodes),np.random.default_rng(42),
                shape=(32,32,32),force_nodule=True,nodule_id=i)
            self.assertEqual(float(ct.max()),float(i))
            self.assertEqual(int(((target>>NODULE_INDEX)&1).sum()),1)
        with self.assertRaises(ValueError):
            sample_patch(image,packed,dict(nodules=nodes),np.random.default_rng(1),
                         force_nodule=True,nodule_id=999)

    def test_nodule_coverage_requires_all_cases_and_all_retained_selections(self):
        expected={'a':[1,2],'b':[]}
        self.assertFalse(coverage(expected,{'a':1,'b':1},{'a':{'1':1}})['passed'])
        self.assertFalse(coverage(expected,{'a':1},{'a':{'1':1,'2':1}})['passed'])
        visits={'a':1,'b':1};selections={'a':{'1':1,'2':1}}
        quality=coverage(expected,visits,selections)
        saved=dict(config=dict(nodule_sampling_policy=NODULE_POLICY,
                               retained_nodule_ids=expected,anatomy_case_ids=['a','b']),
                   epoch=5,nodule_coverage_epoch=5,coverage_case_visits=visits,
                   nodule_selection_counts=selections,nodule_coverage=quality)
        require_coverage(saved)
        saved['nodule_coverage_epoch']=4
        with self.assertRaises(ValueError):require_coverage(saved)
        saved['nodule_coverage_epoch']=5
        del selections['a']['2']
        with self.assertRaises(ValueError):require_coverage(saved)
        with self.assertRaises(ValueError):coverage(expected,visits,{'a':{'3':1}})

    def test_nodule_coverage_upgrade_is_explicit_and_cannot_relax_quality(self):
        old=dict(augmentation=WINDOW_POLICY,max_nodule_dice_drop=.05,anatomy_cache_sha256='original')
        new=dict(old,nodule_sampling_policy=NODULE_POLICY,finish_ct_before_epoch_end=True,
                 retained_nodule_ids={'a':[1,2]},upgrade_nodule_coverage=True)
        with self.assertRaises(ValueError):validate_resume_configuration(old,new)
        validate_resume_configuration(old,new,upgrade_nodule_coverage=True)
        for key,value in [('max_nodule_dice_drop',.2),('anatomy_cache_sha256','changed'),
                          ('finish_ct_before_epoch_end',False),('nodule_sampling_policy','random')]:
            with self.assertRaises(ValueError):
                validate_resume_configuration(old,dict(new,**{key:value}),upgrade_nodule_coverage=True)

    def test_background_mining_upgrade_preserves_data_and_quality(self):
        old=dict(anatomy_cache_sha256='original',max_nodule_dice_drop=.05,
                 min_nodule_dice=.5,anatomy_weight=.1,lr_vista=1e-6)
        baseline=dict(old,background_mining=False,background_mining_schema=None,
                      upgrade_background_mining=False)
        validate_resume_configuration(old,baseline)
        new=dict(old,background_mining=True,background_mining_schema=BACKGROUND_POLICY,
                 upgrade_background_mining=True,background_coverage_policy=BACKGROUND_COVERAGE_POLICY)
        with self.assertRaises(ValueError):validate_resume_configuration(old,new)
        validate_resume_configuration(old,new,upgrade_background_mining=True)
        validate_resume_configuration(new,dict(new,upgrade_background_mining=False))
        for key,value in [('anatomy_cache_sha256','changed'),('max_nodule_dice_drop',.2),
                          ('min_nodule_dice',.2),('anatomy_weight',.8),('lr_vista',1e-4),
                          ('background_mining_schema','random'),('background_mining',False)]:
            with self.assertRaises(ValueError):
                validate_resume_configuration(old,dict(new,**{key:value}),upgrade_background_mining=True)
        with self.assertRaises(ValueError):
            validate_resume_configuration(new,new,upgrade_background_mining=True)

    def test_background_coverage_does_not_reuse_earlier_ct_visits(self):
        old=dict(config=dict(background_mining=True,anatomy_case_ids=['a','b']),
                 epoch=44,complete=True,case_draw_counts={'a':10,'b':12},
                 coverage_case_visits={'a':3,'b':4})
        with self.assertRaises(ValueError):require_background_coverage(old)
        visits=load_background_visits(old,reset=True)
        self.assertEqual(visits,{})
        visits['a']=1
        quality=background_coverage(['a','b'],visits,enabled=True)
        self.assertFalse(quality['passed'])
        self.assertEqual(quality['missing_cases'],1)
        self.assertTrue(background_coverage(['a','b'],{},enabled=False)['passed'])
        with self.assertRaises(ValueError):background_coverage(['a','b'],{'test':1},enabled=True)
        with self.assertRaises(ValueError):background_coverage(['a','b'],{'a':0},enabled=True)

    def test_background_coverage_resume_and_completed_checkpoint(self):
        visits={'a':2}
        saved=dict(config=dict(background_mining=True,anatomy_case_ids=['a','b'],
                               background_coverage_policy=BACKGROUND_COVERAGE_POLICY),
                   epoch=46,background_coverage_epoch=46,background_case_visits=visits,
                   background_coverage=background_coverage(['a','b'],visits,enabled=True))
        restored=load_background_visits(saved)
        self.assertEqual(restored,visits)
        restored['b']=1
        self.assertNotIn('b',saved['background_case_visits'])
        with self.assertRaises(ValueError):require_background_coverage(saved)
        saved['background_case_visits']=restored
        saved['background_coverage']=background_coverage(['a','b'],restored,enabled=True)
        require_background_coverage(saved)
        saved['background_coverage_epoch']=45
        with self.assertRaises(ValueError):load_background_visits(saved)

    def test_background_coverage_upgrade_cannot_change_training_or_quality(self):
        old=dict(background_mining=True,background_mining_schema=BACKGROUND_POLICY,
                 anatomy_cache_sha256='original',max_nodule_dice_drop=.05,
                 min_nodule_dice=.5,anatomy_weight=.1,lr_vista=1e-6)
        new=dict(old,upgrade_background_coverage=True,
                 background_coverage_policy=BACKGROUND_COVERAGE_POLICY)
        with self.assertRaises(ValueError):validate_resume_configuration(old,new)
        validate_resume_configuration(old,new,upgrade_background_coverage=True)
        validate_resume_configuration(new,dict(new,upgrade_background_coverage=False))
        for key,value in [('anatomy_cache_sha256','changed'),('max_nodule_dice_drop',.2),
                          ('min_nodule_dice',.2),('anatomy_weight',.8),('lr_vista',1e-4),
                          ('background_mining_schema','random'),('background_mining',False),
                          ('background_coverage_policy',None)]:
            with self.assertRaises(ValueError):
                validate_resume_configuration(old,dict(new,**{key:value}),upgrade_background_coverage=True)
        with self.assertRaises(ValueError):
            validate_resume_configuration(new,new,upgrade_background_coverage=True)

    def test_deployment_crops_match_inference_grid_keep_targets_and_orientation(self):
        from v7._core.common.base_model import sliding_starts
        from v7._core.common.roi import integer_crop
        packed = torch.zeros(70,55,25,dtype=torch.int16)
        packed[29,27,18] = (1<<NODULE_INDEX)|(1<<READER_UNION_BIT)
        # Encoded coordinates expose an accidental flip or a different crop origin.
        image = torch.arange(packed.numel(),dtype=torch.float32).reshape(1,*packed.shape)
        rng = np.random.default_rng(42); origins = set()
        shape = (32,32,32)
        for _ in range(64):
            center = deployment_crop_center(packed, [30,27,18], shape, rng)
            ct, origin, valid = integer_crop(image, center, shape)
            target, other_origin, _ = integer_crop(packed[None], center, shape)
            self.assertTrue(torch.equal(origin,other_origin))
            self.assertTrue(bool(((target>>NODULE_INDEX)&1).any()))
            origins.add(tuple(origin.tolist()))
            for n,w,s in zip(packed.shape,shape,origin):
                self.assertIn(int(s),sliding_starts(n,w,.25))
            local = tuple(np.array([29,27,18])-origin.numpy())
            self.assertEqual(float(ct[(0,*local)]),float(image[0,29,27,18]))
            self.assertFalse(bool(valid[...,25:].any()))
        self.assertGreater(len(origins),1)
        self.assertIsNone(deployment_crop_center(torch.zeros_like(packed),[0,0,0],shape,rng))

    def test_whole_ct_failure_blocks_completion_and_joint_loading_despite_patch_pass(self):
        from v7.data.anatomy import PROMPT_NAMES
        args=SimpleNamespace(max_nodule_dice_drop=.05,max_organ_dice_drop=.05,
                             min_nodule_dice=.5,smoke=False,min_epochs=5,required_passes=3)
        rows = {name:dict(overlap={p:dict(dice=.9,target=100,predicted=100)
                                  for p in PROMPT_NAMES}) for name in WHOLE_CT_CASE_IDS}
        ref = dict(case_ids=list(WHOLE_CT_CASE_IDS),public=copy.deepcopy(rows),
                   legacy_nodule=copy.deepcopy(rows))
        rows[WHOLE_CT_CASE_IDS[2]]['overlap']['nodule']['dice']=.69
        quality=whole_ct_readiness(rows,ref,args)
        self.assertFalse(quality['passed'])
        self.assertFalse(completion_ready(dict(passed=True),quality,28,3,args))
        self.assertFalse(completion_ready(dict(passed=True),None,28,3,args))
        saved=dict(config=dict(vars(args),whole_ct_case_ids=list(WHOLE_CT_CASE_IDS)),
                   epoch=27,whole_ct_epoch=27,whole_ct_audit=rows,
                   whole_ct_readiness=dict(passed=True),reference=dict(whole_ct=ref))
        with self.assertRaisesRegex(ValueError,'failed whole-CT'):
            require_whole_ct_checkpoint(saved)
        rows[WHOLE_CT_CASE_IDS[2]]['overlap']['nodule']['dice']=.86
        saved['whole_ct_readiness']=whole_ct_readiness(rows,ref,args)
        require_whole_ct_checkpoint(saved)
        self.assertTrue(completion_ready(dict(passed=True),saved['whole_ct_readiness'],28,3,args))
        saved['whole_ct_epoch']=26
        with self.assertRaisesRegex(ValueError,'saved epoch'):
            require_whole_ct_checkpoint(saved)
        rows[WHOLE_CT_CASE_IDS[1]]['overlap']['airway']['dice']=.2
        self.assertFalse(whole_ct_readiness(rows,ref,args)['passed'])
        del rows[WHOLE_CT_CASE_IDS[0]]
        self.assertFalse(whole_ct_readiness(rows,ref,args)['passed'])

    def test_whole_ct_reconstruction_handles_overlap_padding_and_reader_ignore(self):
        class Predictor(nn.Module):
            def __init__(self):
                super().__init__(); self.dummy=nn.Parameter(torch.zeros(1))
            def forward_with_features(self,image):
                return ((image-.5)*20).expand(-1,13,-1,-1,-1), []
        image=torch.zeros(1,170,64,48)
        packed=torch.zeros(170,64,48,dtype=torch.int16)
        image[:,90:110,20:30,20:30]=1
        packed[90:110,20:30,20:30]=(1<<14)-1
        # Reader-only voxels are ignored only for the nodule channel.
        image[:,92:94,32:34,20:22]=1
        packed[92:94,32:34,20:22]=1<<READER_UNION_BIT
        image[:,130:132,40:42,30:32]=1
        result=full_ct_overlap(Predictor(),image,packed)
        self.assertEqual(result['windows'],2)
        self.assertEqual(result['overlap']['nodule']['tp'],2000)
        self.assertEqual(result['overlap']['nodule']['predicted'],2008)
        self.assertEqual(result['overlap']['nodule']['target'],2000)
        self.assertEqual(result['overlap']['airway']['predicted'],2016)
        self.assertEqual(result['nodule_components_without_reader_overlap']['count'],1)
        single=full_ct_overlap(Predictor(),image,packed,nodule_only=True)
        self.assertEqual(single['overlap'],{'nodule':result['overlap']['nodule']})

    def test_primary_probe_cannot_hide_additional_case_failure_or_unvisited_cts(self):
        from v7.data.anatomy import PROMPT_NAMES
        args = SimpleNamespace(min_dice=[.8]*12, max_organ_dice_drop=.05,
                               min_nodule_dice=.5, max_nodule_dice_drop=.05)
        ref = dict(dice=dict.fromkeys(PROMPT_NAMES,.9),legacy_nodule_dice=.88)
        ref['additional'] = dict(dice=dict.fromkeys(PROMPT_NAMES,.9),legacy_nodule_dice=.89)
        good = dict(dice=dict.fromkeys(PROMPT_NAMES,.91),target_voxels=dict.fromkeys(PROMPT_NAMES,100))
        bad = copy.deepcopy(good);bad['dice']['nodule']=.61
        result = combined_readiness(good,bad,ref,args,cohort_covered=True)
        self.assertEqual(set(result['failures']), {'additional/nodule'})
        self.assertAlmostEqual(result['additional_thresholds']['nodule'],.84)
        self.assertFalse(combined_readiness(good,good,ref,args,cohort_covered=False)['passed'])
        self.assertFalse(combined_readiness(good,None,ref,args,cohort_covered=True)['passed'])
        self.assertTrue(combined_readiness(good,good,ref,args,cohort_covered=True)['passed'])

    def test_paired_positive_uses_retained_voxels_and_does_not_invent_labels(self):
        image = torch.zeros(1,40,40,40)
        packed = torch.zeros(40,40,40,dtype=torch.int16)
        packed[35,35,35] = (1<<NODULE_INDEX)|(1<<READER_UNION_BIT)
        # The selected metadata center has no retained consensus in its crop.
        meta = dict(nodules=[dict(retained=True,malignant=0,diameter_mm=3,
                                 center_mm_grid=[0,0,0],distances_mm=dict(lung=0))])
        _, target, _, kind = sample_patch(image,packed,meta,np.random.default_rng(6),
                                          shape=(8,8,8),force_nodule=True)
        self.assertEqual(kind,'nodule')
        self.assertTrue(bool(((target>>NODULE_INDEX)&1).any()))
        packed &= ~(1<<NODULE_INDEX)
        _, target, _, kind = sample_patch(image,packed,meta,np.random.default_rng(6),
                                          shape=(8,8,8),force_nodule=True)
        self.assertEqual(kind,'context')
        self.assertFalse(bool(((target>>NODULE_INDEX)&1).any()))

    def test_full_warmup_trains_encoder_and_all_auto_levels_but_not_point_branch(self):
        backbone=nn.Module()
        backbone.image_encoder=nn.Module()
        backbone.image_encoder.encoder=nn.Linear(2,2)
        backbone.image_encoder.up_layers_auto=nn.ModuleList([nn.Linear(2,2),nn.Linear(2,2)])
        backbone.image_encoder.up_layers=nn.ModuleList([nn.Linear(2,2)])
        backbone.class_head=nn.Linear(2,2)
        backbone.point_head=nn.Linear(2,2)
        segmenter=VistaAnatomySegmenter(backbone)
        set_vista_trainable(segmenter,full_auto=True)
        for name,p in backbone.named_parameters():
            expected=name.startswith(('class_head.','image_encoder.encoder.','image_encoder.up_layers_auto.'))
            self.assertEqual(p.requires_grad,expected,name)
        set_vista_trainable(segmenter,decoder=True)
        self.assertFalse(backbone.image_encoder.encoder.weight.requires_grad)
        self.assertFalse(backbone.image_encoder.up_layers_auto[0].weight.requires_grad)
        self.assertTrue(backbone.image_encoder.up_layers_auto[-1].weight.requires_grad)

    def test_new_channels_preserve_old_conv_and_receive_gradient(self):
        old = nn.Conv3d(3, 8, 3, padding=1)
        new = expand_input(old)
        image, organs = torch.randn(1,3,8,8,8), torch.randn(1,4,8,8,8)
        torch.testing.assert_close(old(image), new(torch.cat((image, organs), 1)), rtol=0, atol=0)
        new(torch.cat((image, organs),1)).square().sum().backward()
        self.assertGreater(float(new.weight.grad[:,3:].abs().sum()), 0)

    def test_added_inputs_preserve_v6_pipeline_before_training(self):
        from tests.test_v6 import tiny as tiny_v6
        old, batch = tiny_v6('native_seeded')
        old.segmenter = TinySingle()
        new, _ = tiny()
        load_warmup_network(new, dict(model=old.state_dict(),
            audit=dict(complete=True, schema=SCHEMA, smoke_only=False)))
        old.eval(); new.eval()
        with torch.no_grad():
            a = old(batch['image'], image_hu=batch['image_hu'], compute_diagnostics=False)
            b = new(batch['image'], image_hu=batch['image_hu'], compute_diagnostics=False)
        for name in ('boxes','object_logits','coarse_mask_logits','fine_mask_logits','semantic_features'):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=1e-5,atol=1e-5)

    def test_native_class_vector_encodes_once_and_matches_individual_prompts(self):
        model = TinyAnatomy().eval()
        image = torch.rand(1,1,8,8,8)
        captured = []
        hook = model.model.class_head.register_forward_pre_hook(lambda m, args: captured.append(args))
        together, _ = model.forward_with_features(image)
        hook.remove()
        self.assertEqual(model.model.image_encoder.calls, 1)
        self.assertEqual(captured[0][1].tolist(), [[i] for i in PROMPT_IDS])
        self.assertFalse(hasattr(model, 'anatomy_head'))
        self.assertEqual(together.shape, (1,13,8,8,8))
        for i, prompt in enumerate(PROMPT_IDS):
            single, _ = model.model.class_head(captured[0][0], torch.tensor([[prompt]]))
            torch.testing.assert_close(together[:,i:i+1], single, rtol=1e-5, atol=1e-5)
        broken = model.state_dict(); broken['label_prompt'] = broken['label_prompt'].flip(0)
        with self.assertRaisesRegex(RuntimeError, 'label_prompt'):
            model.load_state_dict(broken)

    def test_targets_never_change_predictions_and_do_change_auxiliary_loss(self):
        model, batch = tiny(); model.train()
        other = copy.copy(batch); other['anatomy_packed'] = batch['anatomy_packed'] ^ 15
        outputs = []
        for b in (batch, other):
            torch.manual_seed(10)
            outputs.append(model(b['image'], batch=b, geometry_only=True, update_bank=False, teacher_probability=0.))
        for key in ('boxes','object_logits','coarse_mask_logits','fine_mask_logits','anatomy_probabilities'):
            torch.testing.assert_close(getattr(outputs[0],key), getattr(outputs[1],key), rtol=0, atol=0)
        self.assertFalse(torch.equal(outputs[0].vista_auxiliary_losses, outputs[1].vista_auxiliary_losses))
        model.eval()
        with torch.no_grad():
            out = model(batch['image'], image_hu=batch['image_hu'], compute_diagnostics=False)
        self.assertEqual(out.anatomy_probabilities.shape, (1,4,8,8,8))
        self.assertIsNone(out.vista_auxiliary_losses)

    def test_whole_ct_checkpoint_backward_and_loss_reporting(self):
        a, batch = tiny(); b, _ = tiny(True)
        b.load_state_dict(a.state_dict())
        criterion = WholeCTCriterionV7()
        losses = []
        for model in (a,b):
            model.train(); torch.manual_seed(13)
            out = model(batch['image'], batch=batch, update_bank=False, teacher_probability=1.)
            loss = criterion(out, batch, diagnostic_scale=0., synchronize=False)
            self.assertTrue(torch.isfinite(loss.total))
            loss.total.backward(); losses.append(loss)
            grads = model.segmenter.model.class_head.class_embeddings.weight.grad
            self.assertTrue(bool((grads[list(PROMPT_IDS)].abs().sum(1) > 0).all()))
            self.assertGreater(float(model.detr.input_projection[0].weight.grad[:,-4:].abs().sum()), 0)
        torch.testing.assert_close(losses[0].total, losses[1].total)
        torch.testing.assert_close(a.segmenter.model.class_head.class_embeddings.weight.grad,
                                   b.segmenter.model.class_head.class_embeddings.weight.grad)
        accumulator = LossAccumulator(); accumulator.update(losses[0])
        record = accumulator.compute(criterion.loss_weights, diagnostic_scale=0., synchronize=False)
        self.assertEqual(record['loss_counts']['anatomy'], 1)
        self.assertAlmostEqual(record['anatomy'], losses[0].mean('anatomy'))

    def test_annotation_ignore_and_empty_channel_gradients(self):
        logits = torch.zeros(1,13,4,4,4,requires_grad=True)
        packed = torch.zeros(1,4,4,4,dtype=torch.int16)
        packed[:,0] = 1<<READER_UNION_BIT
        loss = patch_objectives(logits, packed)
        loss.sum().backward()
        self.assertEqual(float(logits.grad[:,NODULE_INDEX,0].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:,:NODULE_INDEX].abs().sum()), 0)
        self.assertTrue(torch.isfinite(loss).all())

    def test_native_segmentation_loss_matches_original_lidc_vista_objective(self):
        from monai.losses import DiceCELoss
        torch.manual_seed(19)
        logits=torch.randn(1,13,4,4,4,requires_grad=True)
        target=torch.zeros_like(logits)
        target[:,:,1:3,1:3,1:3]=1
        original=DiceCELoss(include_background=True,sigmoid=True,squared_pred=True,
                            smooth_nr=0,smooth_dr=1e-5,to_onehot_y=False)
        # VISTA exposes each prompt as a separate binary prediction. Passing
        # all prompts as MONAI channels would incorrectly select softmax CE.
        expected=torch.stack([original(logits[:,i:i+1],target[:,i:i+1])
                              for i in range(13)]).mean()
        actual=binary_segmentation_losses(logits,target).mean()
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,logits,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,logits)[0]
        torch.testing.assert_close(ga,ge)

    def test_native_segmentation_padding_cannot_change_loss_or_gradients(self):
        logits=torch.zeros(1,2,4,4,4,requires_grad=True)
        target=torch.zeros_like(logits);target[:,:,1,1,1]=1
        valid=torch.ones_like(target);valid[:,:,-1]=0
        before=binary_segmentation_losses(logits,target,valid)
        changed=logits.detach().clone();changed[:,:,-1]=100
        after=binary_segmentation_losses(changed,target,valid)
        torch.testing.assert_close(before,after)
        before.sum().backward()
        self.assertEqual(float(logits.grad[:,:,-1].abs().sum()),0.)

    def test_overlapping_organ_targets_are_not_erased(self):
        names = {i+1:n for i,n in enumerate(TOTAL_ROIS)}
        ids = {v:k for k,v in names.items()}
        total = np.zeros((12,12,12), dtype=np.uint8)
        total[2:10,2:10,2:10] = ids[LOBES[0]]
        total[5,5,5] = ids['aorta']
        pulmonary = np.zeros_like(total)
        maps = {1:'lung_arteries',2:'lung_veins',3:'lung_airways',4:'lung_airways_wall'}
        groups = unpack_groups(pack_groups(total,pulmonary,names,maps,(1,1,1)))
        self.assertTrue(groups[0,5,5,5] and groups[1,5,5,5] and groups[3,5,5,5])

    def test_prompt_labels_map_anatomy_not_totalsegmentator_numeric_ids(self):
        names = {i+1:n for i,n in enumerate(TOTAL_ROIS)}
        ids = {v:k for k,v in names.items()}
        total = np.zeros((4,4,4),dtype=np.uint8)
        total[0,0,0] = ids[LOBES[1]]
        total[1,0,0] = ids['trachea']
        total[2,0,0] = ids['pulmonary_vein']
        pulmonary = np.zeros_like(total)
        pulmonary[3,0,0] = 1
        pulmonary[3,1,0] = 2
        mapping = {1:'lung_airways',2:'lung_airways_wall',3:'lung_arteries',4:'lung_veins'}
        packed = pack_prompt_targets(total,pulmonary,names,mapping)
        self.assertEqual(packed.dtype, np.uint16)
        self.assertEqual(int(packed[0,0,0]),1<<1)
        self.assertEqual(int(packed[1,0,0]),(1<<5)|(1<<6))
        self.assertEqual(int(packed[2,0,0]),1<<7)
        self.assertEqual(int(packed[3,0,0]),1<<6)
        self.assertEqual(int(packed[3,1,0]),0)  # Wall is not airway lumen.

    def test_sampler_balances_buckets_and_keeps_every_ct(self):
        meta=[]
        for i in range(20):
            meta.append(dict(split='training',nodules=[dict(nodule_id=1,retained=True,
                malignant=i%2,diameter_mm=4 if i==0 else 10,
                distances_mm=dict(lung=1 if i<2 else 30,vessels=1,airway=30,mediastinum=30))]))
        class Dataset:
            anatomy_metadata=meta
            def __len__(self):return len(meta)
        dataset=Dataset()
        one=AnatomyOversamplingSampler(dataset,extra_fraction=2.)
        all_indices=list(one)
        self.assertTrue(set(range(20)) <= set(all_indices))
        count={}
        for row in one.last_draws:
            key=(row['group'],row['malignant']);count[key]=count.get(key,0)+1
        self.assertEqual(set(count.values()), {10})
        ranks=[AnatomyOversamplingSampler(dataset,num_replicas=3,rank=r,extra_fraction=2.) for r in range(3)]
        restored=np.array([list(s) for s in ranks]).T.ravel().tolist()
        self.assertEqual(restored, all_indices)
        one.set_epoch(1); self.assertNotEqual(list(one),all_indices)

    def test_sampled_negative_has_no_reader_lesion_and_keeps_padding_mask(self):
        image=torch.zeros(1,20,20,20)
        packed=torch.ones(20,20,20,dtype=torch.int16)
        packed[8:12,8:12,8:12] |= 1<<READER_UNION_BIT
        rng=np.random.default_rng(7)
        for _ in range(12):
            ct,target,valid,kind=sample_patch(image,packed,dict(nodules=[]),rng,shape=(8,8,8))
            self.assertEqual(ct.shape, (1,1,8,8,8))
            if kind=='anatomical_negative':self.assertFalse(bool(((target>>READER_UNION_BIT)&1).any()))
            self.assertEqual(valid.dtype,torch.bool)

    def test_warmup_crops_preserve_anatomical_orientation(self):
        xyz = torch.meshgrid(*(torch.arange(20) for _ in range(3)), indexing='ij')
        image = (10000+10000*xyz[0]+100*xyz[1]+xyz[2]).float()[None]
        packed = torch.ones((20,20,20), dtype=torch.int16)
        rng = np.random.default_rng(7)
        for _ in range(12):
            ct, _, valid, _ = sample_patch(image, packed, dict(nodules=[]), rng, shape=(8,8,8))
            for axis, increment in enumerate((10000,100,1), start=2):
                lower, upper = [slice(None)]*5, [slice(None)]*5
                lower[axis], upper[axis] = slice(None,-1), slice(1,None)
                usable = valid[tuple(lower)] & valid[tuple(upper)]
                difference = ct[tuple(upper)]-ct[tuple(lower)]
                self.assertTrue(bool((difference[usable] == increment).all()))


if __name__ == '__main__':
    unittest.main()
