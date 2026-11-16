"""Whole-CT warmup regression checks using the deployment reconstruction."""
import itertools
import math
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from v7._core.common.base_model import build_vista3d, sliding_starts
from v7._core.common.coarse_model import _gaussian_weight
from v7.data.anatomy import case_id, PROMPT_IDS, PROMPT_NAMES, NODULE_INDEX, READER_UNION_BIT
from v7.models.vista import VistaAnatomySegmenter

WHOLE_CT_CASE_IDS = ('LIDC-IDRI-0875__scan780', 'LIDC-IDRI-0127__scan1018',
                     'LIDC-IDRI-0865__scan790', 'LIDC-IDRI-0244__scan244')


@torch.inference_mode()
def full_ct_overlap(model, image, packed, *, nodule_only=False):
    shape = tuple(image.shape[1:])
    device = next(model.parameters()).device
    indices = [NODULE_INDEX] if nodule_only else list(range(len(PROMPT_IDS)))
    sums = torch.zeros((len(indices), *shape), device=device)
    weights = torch.zeros(shape, device=device)
    windows = 0
    for start in itertools.product(*(sliding_starts(n, 128, .25) for n in shape)):
        valid = tuple(min(n-s, 128) for n, s in zip(shape, start))
        extent = tuple(slice(s, s+n) for s, n in zip(start, valid))
        patch = image[(..., *extent)][None].to(device)
        patch = F.pad(patch, tuple(v for n in valid[::-1] for v in (0, 128-n)))
        with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=torch.bfloat16):
            logits, _ = model.forward_with_features(patch)
        cropped = logits[(0, indices, *(slice(0, n) for n in valid))].float().sigmoid()
        importance = _gaussian_weight(valid, device, torch.float32).reshape(valid)
        sums[(..., *extent)] += cropped * importance
        weights[extent] += importance
        windows += 1
    if not bool((weights > 0).all()):
        raise RuntimeError('Sliding-window reconstruction missed CT voxels')
    packed = packed.numpy()
    counts, components = {}, None
    for channel, index in enumerate(indices):
        prediction = (sums[channel] / weights.clamp_min(1e-6) >= .5).cpu().numpy()
        truth = ((packed >> index) & 1).astype(bool)
        if index == NODULE_INDEX:
            readers = ((packed >> READER_UNION_BIT) & 1).astype(bool)
            labels, count = ndimage.label(prediction, np.ones((3, 3, 3)))
            sizes = np.bincount(labels.ravel())
            overlaps_reader = np.unique(labels[readers])
            false_ids = np.setdiff1d(np.arange(1, count+1), overlaps_reader)
            false_sizes = sizes[false_ids]
            components = dict(count=int(len(false_ids)), voxels=int(false_sizes.sum()),
                              largest_voxels=sorted(false_sizes.tolist(), reverse=True)[:10])
            prediction &= ~readers | truth
        tp = int((prediction & truth).sum())
        predicted, target = int(prediction.sum()), int(truth.sum())
        counts[PROMPT_NAMES[index]] = dict(tp=tp, predicted=predicted, target=target,
            dice=2*tp/max(predicted+target, 1), precision=tp/max(predicted, 1),
            recall=tp/max(target, 1))
    return dict(shape=shape, windows=windows, overlap=counts,
                nodule_components_without_reader_overlap=components)


def whole_ct_audit(model, cases, root, load_case, *, nodule_only=False):
    model.eval()
    rows = {}
    for case in cases:
        if case['split'] != 'training':
            raise ValueError('Warmup regression checks cannot use held-out CTs')
        image, packed, _ = load_case(case, root)
        rows[case_id(case)] = full_ct_overlap(model, image, packed, nodule_only=nodule_only)
    return rows


def whole_ct_reference(cases, args, load_case):
    """Original public organ and legacy nodule models, measured only once."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model = VistaAnatomySegmenter(build_vista3d(Path(args.public_vista_checkpoint))).cuda()
        public = whole_ct_audit(model, cases, args.anatomy_root, load_case)
        legacy = public
        if args.base_warmup is not None:
            source = torch.load(args.base_warmup, map_location='cpu', mmap=True, weights_only=False)
            state = {k.removeprefix('segmenter.'):v for k,v in source['model'].items()
                     if k.startswith('segmenter.')}
            missing = model.load_state_dict(state, strict=False)
            if missing.unexpected_keys or missing.missing_keys != ['label_prompt']:
                raise ValueError(f'Whole-CT nodule reference mismatch: {missing}')
            legacy = whole_ct_audit(model, cases, args.anatomy_root, load_case, nodule_only=True)
    return dict(case_ids=[case_id(c) for c in cases], public=public, legacy_nodule=legacy,
                inference=dict(window=128, overlap=.25, threshold=.5, blend='gaussian_probability'),
                population='Training CT regression checks; not independent test performance')


def whole_ct_readiness(current, reference, args):
    """Require each CT to preserve reference quality; averages cannot hide a miss.

    The original absolute organ minima remain enforced by both patch groups.
    This additional check applies their existing allowed regression (0.05) to
    each full CT, plus the existing absolute nodule minimum.
    """
    failures, thresholds = {}, {}
    names = tuple(reference['case_ids'])
    if set(current) != set(names) or len(names) != len(set(names)) or not names:
        return dict(passed=False, failures={'cases':'Whole-CT cases are missing or different'}, thresholds={})
    for name in names:
        thresholds[name] = {}
        for prompt in PROMPT_NAMES:
            baseline = (reference['legacy_nodule'] if prompt == 'nodule' else reference['public'])
            old = baseline[name]['overlap'][prompt]
            new = current[name]['overlap'][prompt]
            drop = args.max_nodule_dice_drop if prompt == 'nodule' else args.max_organ_dice_drop
            required = old['dice']-drop
            if prompt == 'nodule':
                required = max(args.min_nodule_dice, required)
            thresholds[name][prompt] = required
            if new['target'] != old['target']:
                failures[f'{name}/{prompt}'] = dict(reason='Reference/target voxel counts differ')
            elif old['target'] > 0:
                if not math.isfinite(new['dice']) or new['dice'] < required:
                    failures[f'{name}/{prompt}'] = dict(dice=new['dice'], required=required)
            elif new['predicted'] > old['predicted']:
                failures[f'{name}/{prompt}'] = dict(reason='More false voxels for an absent target')
    return dict(passed=not failures, failures=failures, thresholds=thresholds)


def completion_ready(patch_quality, whole_quality, epoch, passes, args):
    return bool(not args.smoke and epoch >= args.min_epochs and passes >= args.required_passes
                and patch_quality.get('passed') and whole_quality and whole_quality.get('passed'))


def require_whole_ct_checkpoint(saved):
    """Reject patch-only or stale whole-CT approval at the joint entry point."""
    from types import SimpleNamespace
    config = saved['config']
    if config.get('whole_ct_case_ids') != list(WHOLE_CT_CASE_IDS):
        raise ValueError('Anatomy warmup lacks the fixed whole-CT regression protocol')
    if saved.get('whole_ct_epoch') != saved.get('epoch') or saved.get('epoch') is None:
        raise ValueError('Whole-CT results must be measured on the saved epoch')
    audit, reference = saved.get('whole_ct_audit'), saved.get('reference', {}).get('whole_ct')
    if not audit or not reference or reference.get('case_ids') != list(WHOLE_CT_CASE_IDS):
        raise ValueError('Anatomy warmup lacks whole-CT results/references')
    quality = whole_ct_readiness(audit, reference, SimpleNamespace(**config))
    if not quality['passed'] or quality != saved.get('whole_ct_readiness'):
        raise ValueError('Anatomy warmup failed whole-CT regression checks')

