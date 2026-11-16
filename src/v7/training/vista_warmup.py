"""Warm up VISTA's anatomy outputs while retaining its nodule segmentation."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from torch.utils.checkpoint import checkpoint

from v7._core.common.base_model import build_vista3d, DEFAULT_VISTA_CHECKPOINT, _state_dict, sliding_starts
from v7._core.common.roi import integer_crop
from v7._core.common.train_utils import atomic_save
from v7._core.common.checkpoint_compat import adapt_cuda_rng_state
from v7._core.model_v4.monitor import Monitor
from v7.data.anatomy import (CHANNELS, PROMPT_IDS, PROMPT_NAMES, NODULE_INDEX, READER_UNION_BIT,
                      GROUP_INDICES, PREPARED_DIRECTORY, PROMPT_SCHEMA, case_id)
from v7.data.organ_labels import DEFAULT_ROOT, DEFAULT_MANIFEST
from v7.training.loss import patch_objectives, LOSS_SCHEMA
from v7.models.vista import VistaAnatomySegmenter, set_vista_trainable
from v7.training.warmup_quality import (WHOLE_CT_CASE_IDS, whole_ct_audit, whole_ct_reference,
                             whole_ct_readiness, completion_ready)
from v7.training.warmup_sampling import POLICY as NODULE_POLICY, visit_plan, coverage as nodule_coverage
from v7.training.warmup_background import (POLICY as BACKGROUND_POLICY, BackgroundCropMiner,
    COVERAGE_POLICY as BACKGROUND_COVERAGE_POLICY, coverage as background_coverage,
    load_coverage_visits as load_background_visits)

SCHEMA = 'v7_native_prompt_warmup_v6'
WINDOW_POLICY = 'half_centered_half_deployment_grid_preserve_orientation'
from v7 import paths
DEFAULT_BASE = paths.COMPONENT_WARMUP
DEFAULT_PUBLIC_VISTA = paths.PUBLIC_VISTA


def summarize_overlap(sums):
    """Expose false-positive growth instead of reporting Dice alone."""
    intersection, predicted, target = sums.unbind(-1)
    values = dict(dice=2*intersection/(predicted+target).clamp_min(1),
                  precision=intersection/predicted.clamp_min(1),
                  recall=intersection/target.clamp_min(1),
                  predicted_voxels=predicted, target_voxels=target,
                  true_positive_voxels=intersection)
    return {key:dict(zip(PROMPT_NAMES, value.cpu().tolist())) for key,value in values.items()}


def readiness(current, reference, args):
    thresholds = {name:max(minimum, reference['dice'][name]-args.max_organ_dice_drop)
                  for name,minimum in zip(PROMPT_NAMES[:NODULE_INDEX], args.min_dice)}
    thresholds['nodule'] = max(args.min_nodule_dice,
                               reference['legacy_nodule_dice']-args.max_nodule_dice_drop)
    gaps = {name:current['dice'][name]-threshold for name,threshold in thresholds.items()}
    failures = {name:dict(dice=current['dice'][name], required=threshold,
                          target_voxels=current['target_voxels'][name])
                for name,threshold in thresholds.items()
                if current['target_voxels'][name] <= 0 or gaps[name] < 0}
    return dict(passed=not failures, failures=failures, thresholds=thresholds,
                worst_dice_margin=min(gaps.values()))


def combined_readiness(current, additional, reference, args, *, cohort_covered):
    """Both disjoint training probes and a complete CT cycle must pass."""
    primary = readiness(current, reference, args)
    extra = readiness(additional, reference['additional'], args) if additional is not None else None
    failures = dict(primary['failures'])
    if extra is None:
        failures['additional_audit'] = dict(reason='No additional cases were checked')
    else:
        failures.update({f'additional/{name}':value for name,value in extra['failures'].items()})
    if not cohort_covered:
        failures['training_cohort'] = dict(reason='Not every training CT has been visited')
    return dict(passed=not failures, failures=failures, thresholds=primary['thresholds'],
                additional_thresholds=extra['thresholds'] if extra else None,
                cohort_covered=cohort_covered,
                worst_dice_margin=min(primary['worst_dice_margin'],
                    extra['worst_dice_margin'] if extra else primary['worst_dice_margin']))


def validate_resume_configuration(saved, current, *, change_learning_rates=False,
                                  upgrade_window_training=False, upgrade_nodule_coverage=False,
                                  upgrade_background_mining=False, upgrade_background_coverage=False):
    """An explicit LR continuation must not relax data or quality checks."""
    mutable = {'resume', 'epochs', 'output_dir', 'wandb_name', 'wandb_mode',
               'resume_learning_rates', 'new_wandb_run', 'upgrade_window_training',
               'upgrade_nodule_coverage', 'upgrade_background_mining', 'upgrade_background_coverage'}
    saved = dict(saved)
    saved.setdefault('background_mining', False)
    saved.setdefault('public_only', False)
    saved.setdefault('background_ratio', 1.)
    saved.setdefault('tuning_allow_unready', False)
    if change_learning_rates:
        mutable.update(('lr_head', 'lr_vista'))
    if upgrade_window_training:
        additions = dict(augmentation=WINDOW_POLICY, deployment_patch_probability=.5,
                         whole_ct_case_ids=list(WHOLE_CT_CASE_IDS))
        if any(current.get(k) != v for k, v in additions.items()):
            raise ValueError('Window-training upgrade must use the fixed sampling and whole-CT protocol')
        mutable.update(additions)
    if upgrade_nodule_coverage:
        if (current.get('nodule_sampling_policy') != NODULE_POLICY
                or current.get('finish_ct_before_epoch_end') is not True):
            raise ValueError('Nodule coverage upgrade must finish every complete CT visit')
        mutable.update(('nodule_sampling_policy', 'finish_ct_before_epoch_end',
                        'retained_nodule_ids'))
    if upgrade_background_mining:
        if (saved.get('background_mining') or current.get('background_mining') is not True
                or current.get('background_mining_schema') != BACKGROUND_POLICY
                or current.get('background_coverage_policy') != BACKGROUND_COVERAGE_POLICY):
            raise ValueError('Background mining upgrade must enable the fixed ranked-window policy')
        mutable.update(('background_mining', 'background_mining_schema', 'background_coverage_policy'))
    if upgrade_background_coverage:
        if (not saved.get('background_mining') or saved.get('background_coverage_policy') is not None
                or current.get('background_mining') is not True
                or current.get('background_coverage_policy') != BACKGROUND_COVERAGE_POLICY):
            raise ValueError('Background coverage upgrade requires an older mining checkpoint')
        mutable.add('background_coverage_policy')
    changed = [key for key, value in current.items()
               if key not in mutable and saved.get(key) != value]
    if changed:
        raise ValueError(f'Anatomy warmup resume configuration differs: {changed}')


def restore_optimizer(optimizer, state, *, learning_rates=None):
    """Preserve Adam moments and step counts while explicitly reducing its LR."""
    if learning_rates is not None:
        if (len(learning_rates) != len(state['param_groups'])
                or any(not math.isfinite(rate) or rate <= 0 for rate in learning_rates)):
            raise ValueError('Expected one positive finite learning rate per optimizer group')
    optimizer.load_state_dict(state)
    old_rates = [group['lr'] for group in optimizer.param_groups]
    if learning_rates is None or list(learning_rates) == old_rates:
        return None
    for group, rate in zip(optimizer.param_groups, learning_rates):
        group['lr'] = rate
    return dict(previous=old_rates, current=list(learning_rates))


def cycle_case_indices(count, seed, offset, draws):
    """A deterministic complete-cohort order, resumable between any two CTs."""
    if count < 1 or offset < 0 or draws < 0:
        raise ValueError('Invalid case cycle size or position')
    result = []
    while len(result) < draws:
        cycle, position = divmod(offset, count)
        order = np.random.default_rng(np.random.SeedSequence([seed, 7919, cycle])).permutation(count)
        take = min(draws-len(result), count-position)
        result.extend(order[position:position+take].tolist())
        offset += take
    return result


def prefetch_cases(cases, root, indices, workers):
    """Overlap CPU preprocessing with GPU work without changing sample order."""
    if not workers:
        for index in indices:
            yield index, load_case(cases[index], root)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        remaining, pending = iter(indices), deque()
        for _ in range(workers):
            index = next(remaining, None)
            if index is None:
                break
            pending.append((index, executor.submit(load_case, cases[index], root)))
        while pending:
            index, future = pending.popleft()
            value = future.result()  # A failed read remains a training error.
            following = next(remaining, None)
            if following is not None:
                pending.append((following, executor.submit(load_case, cases[following], root)))
            yield index, value


def load_case(case, root):
    source = nib.as_closest_canonical(nib.load(case['image']))
    spacing = nib.affines.voxel_sizes(source.affine)
    hu = ndimage.zoom(np.asarray(source.dataobj, dtype=np.float32), spacing,
                      order=1, mode='nearest', prefilter=False)
    image = torch.from_numpy(((hu.clip(-1024, 1024)+1024)/2048).astype(np.float32))[None]
    with np.load(root/PREPARED_DIRECTORY/f'{case_id(case)}.npz') as saved:
        values = saved['packed']
        if values.dtype != np.uint16 or values.max() >= 16384:
            raise ValueError('Invalid packed prompt labels')
        packed = torch.from_numpy(values.view(np.int16))
    if tuple(image.shape[1:]) != tuple(packed.shape):
        raise ValueError('Warmup CT/anatomy grids differ')
    metadata = json.loads((root/PREPARED_DIRECTORY/f'{case_id(case)}.json').read_text())
    if (metadata['source'] != case['image'] or metadata['split'] != 'training'
            or metadata['schema'] != PROMPT_SCHEMA or tuple(metadata['prompt_ids']) != PROMPT_IDS):
        raise ValueError('Warmup case provenance mismatch')
    return image, packed, metadata


def deployment_crop_center(packed, center, shape, rng):
    """Sample an actual inference-grid window containing a retained GT voxel.

    A metadata diameter is not a bounding box. Find a real voxel when the
    metadata center falls outside the retained mask instead of relying on it.
    """
    center = np.clip(np.asarray(center, dtype=int), 0, np.asarray(packed.shape)-1)
    if not bool((packed[tuple(center)] >> NODULE_INDEX) & 1):
        voxels = torch.nonzero(((packed >> NODULE_INDEX)&1) != 0).numpy()
        if not len(voxels):
            return None
        center = voxels[np.argmin(((voxels-center)**2).sum(axis=1))]
    starts = []
    for n, width, coordinate in zip(packed.shape, shape, center):
        choices = [s for s in sliding_starts(n, width, .25) if s <= coordinate < min(n, s+width)]
        starts.append(choices[int(rng.integers(len(choices)))])
    return np.asarray(starts)+np.asarray(shape)//2


def sample_patch(image, packed, metadata, rng, shape=(128, 128, 128), *, force_nodule=None,
                 deployment_probability=0., nodule_id=None):
    """Balanced nodule crops, verified lesion-free anatomical negatives, context."""
    nodules = [n for n in metadata['nodules'] if n['retained']]
    if nodule_id is not None:
        if force_nodule is not True:
            raise ValueError('An explicit nodule selection requires a positive crop')
        nodules = [n for n in nodules if n['nodule_id'] == nodule_id]
        if len(nodules) != 1:
            raise ValueError('Explicit nodule selection is missing or ambiguous')
    draw = rng.random()
    if force_nodule is True and nodules:
        draw = 0.
    elif force_nodule is False:
        draw = .5+.5*draw
    if nodules and draw < .5:
        nearby = [n for n in nodules if any(d is not None and d <= 5 for d in n['distances_mm'].values())]
        choices = nearby if nearby and rng.random() < .75 else nodules
        classes = sorted({n['malignant'] for n in choices})
        label = classes[int(rng.integers(len(classes)))]
        choices = [n for n in choices if n['malignant'] == label]
        weights = np.array([4 if n['diameter_mm'] <= 6 and label else
                            2 if n['diameter_mm'] <= 6 else 1 for n in choices], dtype=float)
        nodule = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
        center = np.rint(nodule['center_mm_grid']).astype(int)
        kind = 'nodule'
        if deployment_probability and rng.random() < deployment_probability:
            position = deployment_crop_center(packed, center, shape, rng)
            if position is not None:
                center, kind = position, 'nodule_deployment_window'
        if kind == 'nodule':
            center += rng.integers(-12, 13, size=3)
    else:
        # Coarse coordinate search bounds memory; crop labels remain full resolution.
        coarse = packed[::4, ::4, ::4].numpy()
        group = int(rng.integers(4))
        region = (coarse & sum(1 << i for i in GROUP_INDICES[group])) != 0
        if group == 0:
            region &= ~ndimage.binary_erosion(region)
        locations = np.argwhere(region)
        center = np.array(packed.shape)//2
        kind = 'context'
        for _ in range(32 if draw < .75 else 1):
            if len(locations):
                center = locations[int(rng.integers(len(locations)))]*4
            else:
                center = np.array([rng.integers(n) for n in packed.shape])
            test, _, _ = integer_crop(packed[None], center, shape)
            if not ((test >> READER_UNION_BIT)&1).any():
                kind = 'anatomical_negative'
                break
        # If no lesion-free crop exists, retain its real labels as context;
        # it is never relabelled as a negative.
    ct, origin, valid = integer_crop(image, center, shape)
    target, target_origin, _ = integer_crop(packed[None], center, shape)
    if force_nodule is True and not bool(((target >> NODULE_INDEX)&1).any()):
        # A tiny consensus can disappear after resampling, or a metadata
        # center can miss its remaining voxels. Use actual retained labels;
        # a CT with no such voxels stays context, never a fabricated positive.
        locations = torch.nonzero(((packed >> NODULE_INDEX)&1) != 0, as_tuple=False)
        if len(locations):
            center = locations[int(rng.integers(len(locations)))].numpy()
            ct, origin, valid = integer_crop(image, center, shape)
            target, target_origin, _ = integer_crop(packed[None], center, shape)
            kind = 'nodule'
        else:
            kind = 'context'
    if not torch.equal(origin, target_origin):
        raise AssertionError('CT/label crops differ')
    # Preserve anatomical orientation for the left/right and upper/lower
    # organ prompts. Random crop centers provide positional augmentation.
    return ct[None], target, valid[None], kind


@torch.no_grad()
def audit(model, cases, root, *, seed, patches=13):
    model.eval()
    sums = torch.zeros((len(PROMPT_IDS), 3), device='cuda', dtype=torch.float64)
    rng = np.random.default_rng(seed)
    case_results = {}
    for case in cases:
        image, packed, metadata = load_case(case, root)
        case_sums = torch.zeros_like(sums)
        for index in range(patches):
            # Cycle through classes so small organs are represented in the
            # fixed audit, rather than hidden by a good combined-organ Dice.
            bit = index % len(PROMPT_IDS)
            coarse = packed[::4, ::4, ::4].numpy()
            locations = np.argwhere(((coarse >> bit)&1) != 0)
            if len(locations):
                center = locations[int(rng.integers(len(locations)))]*4
                ct, _, valid = integer_crop(image, center)
                target, _, _ = integer_crop(packed[None], center)
                ct, valid = ct[None], valid[None]
            else:
                ct, target, valid, _ = sample_patch(image, packed, metadata, rng)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model.forward_with_features(ct.cuda())
            target, valid = target.cuda(), valid.cuda()
            truth = torch.stack([((target >> i)&1) for i in range(len(PROMPT_IDS))], dim=1).bool()
            support = valid.expand_as(truth).clone()
            support[:, NODULE_INDEX:NODULE_INDEX+1] &= (
                ~((target >> READER_UNION_BIT)&1).bool() | truth[:, NODULE_INDEX])[:, None]
            prediction = (logits.float().sigmoid() >= .5) & support
            truth &= support
            dims = (0, 2, 3, 4)
            case_sums[:, 0] += (prediction & truth).sum(dims)
            case_sums[:, 1] += prediction.sum(dims)
            case_sums[:, 2] += truth.sum(dims)
        sums += case_sums
        case_results[case_id(case)] = summarize_overlap(case_sums)
    return dict(**summarize_overlap(sums), per_case=case_results,
                prompt_ids=list(PROMPT_IDS),
                cases=len(cases), patches_per_case=patches,
                population='fixed training-cohort patches; not independent test performance')


def additional_reference(cases, args):
    """Measure the original models on the additional cases, preserving RNG."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        probe = VistaAnatomySegmenter(build_vista3d(args.public_vista_checkpoint)).cuda()
        public = audit(probe, cases, args.anatomy_root, seed=args.seed+949, patches=args.audit_patches)
        legacy = public
        if args.base_warmup is not None:
            saved = torch.load(args.base_warmup, map_location='cpu', mmap=True, weights_only=False)
            state = {k.removeprefix('segmenter.'):v for k,v in saved['model'].items() if k.startswith('segmenter.')}
            missing = probe.load_state_dict(state, strict=False)
            if missing.unexpected_keys or missing.missing_keys != ['label_prompt']:
                raise ValueError(f'Additional nodule reference mismatch: {missing}')
            legacy = audit(probe, cases, args.anatomy_root, seed=args.seed+949, patches=args.audit_patches)
        public['legacy_nodule_dice'] = legacy['dice']['nodule']
        public['case_ids'] = [case_id(c) for c in cases]
        public['seed'] = args.seed+949
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anatomy-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--base-warmup', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--public-only', action='store_true',
                        help='Use public VISTA only; no previously LIDC-trained weights or references')
    parser.add_argument('--vista-checkpoint', type=Path, default=DEFAULT_VISTA_CHECKPOINT)
    parser.add_argument('--public-vista-checkpoint', type=Path, default=DEFAULT_PUBLIC_VISTA)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps-per-epoch', type=int, default=512)
    parser.add_argument('--patches-per-case', type=int, default=2)
    parser.add_argument('--preprocess-workers', type=int, default=2)
    parser.add_argument('--audit-cases', type=int, default=8)
    parser.add_argument('--additional-audit-cases', type=int, default=16)
    parser.add_argument('--audit-patches', type=int, default=13)
    parser.add_argument('--min-epochs', type=int, default=5)
    parser.add_argument('--required-passes', type=int, default=3)
    parser.add_argument('--min-dice', type=float, nargs=12,
        default=(.8,.8,.8,.8,.8,.75,.65,.65,.8,.8,.65,.6))
    parser.add_argument('--max-nodule-dice-drop', type=float, default=.05)
    parser.add_argument('--min-nodule-dice', type=float, default=.5)
    parser.add_argument('--max-organ-dice-drop', type=float, default=.05)
    parser.add_argument('--lr-head', type=float, default=1e-5, help='Existing native VISTA class head')
    parser.add_argument('--lr-vista', type=float, default=1e-5)
    parser.add_argument('--trainable-scope', choices=('full_auto', 'last_decoder'), default='full_auto')
    parser.add_argument('--anatomy-weight', type=float, default=.1)
    parser.add_argument('--background-ratio', type=float, choices=(.5, 1., 2.), default=1.,
                        help='Background crops per positive crop; retain every nodule visit')
    parser.add_argument('--tuning-allow-unready', action='store_true',
                        help='Record an exhausted candidate without promoting it to complete.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-learning-rates', type=float, nargs=2, metavar=('HEAD', 'VISTA'),
                        help='With --resume, change learning rates while preserving optimizer state; restart consecutive quality checks')
    parser.add_argument('--new-wandb-run', action='store_true',
                        help='With --resume, use a separate W&B run for a checkpoint branch')
    parser.add_argument('--upgrade-window-training', action='store_true',
                        help='Explicitly resume a v4 checkpoint with deployment-window sampling and whole-CT completion checks')
    parser.add_argument('--upgrade-nodule-coverage', action='store_true',
                        help='Resume v5 (or v4 with window upgrade), visiting all retained nodules plus a priority/background pair per CT')
    parser.add_argument('--background-mining', action='store_true',
                        help='Rank deployment windows once per CT and select distinct difficult background/context crops')
    parser.add_argument('--upgrade-background-mining', action='store_true',
                        help='Explicitly enable background mining when resuming a v6 checkpoint, preserving optimizer and positive sampling')
    parser.add_argument('--upgrade-background-coverage', action='store_true',
                        help='Resume an older mining checkpoint with new mining-only CT coverage counters')
    parser.add_argument('--initialize-from', type=Path,
                        help='Start a new optimization run from native-prompt weights; retain original quality references')
    parser.add_argument('--restore-initial-optimizer', action='store_true',
                        help='With --initialize-from, preserve the optimizer when loss, scope and learning rates match')
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--wandb-mode', choices=('online','offline','disabled'), default='disabled')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-project', default='V7')
    parser.add_argument('--wandb-name')
    args = parser.parse_args()
    if args.public_only:
        args.base_warmup = None
        args.vista_checkpoint = args.public_vista_checkpoint
    if args.resume and args.initialize_from:
        raise ValueError('Use either exact resume or weight initialization, not both')
    if args.new_wandb_run and not args.resume:
        raise ValueError('--new-wandb-run requires --resume')
    if args.upgrade_window_training and not args.resume:
        raise ValueError('--upgrade-window-training requires a v4 --resume checkpoint')
    if args.upgrade_nodule_coverage and not args.resume:
        raise ValueError('--upgrade-nodule-coverage requires --resume')
    if args.upgrade_background_mining and (not args.resume or not args.background_mining):
        raise ValueError('--upgrade-background-mining requires --resume and --background-mining')
    if args.upgrade_background_coverage and (not args.resume or not args.background_mining
                                            or args.upgrade_background_mining):
        raise ValueError('--upgrade-background-coverage requires a mining --resume, without --upgrade-background-mining')
    if args.resume_learning_rates is not None:
        if not args.resume:
            raise ValueError('--resume-learning-rates requires --resume')
        args.lr_head, args.lr_vista = args.resume_learning_rates
    if any(not math.isfinite(rate) or rate <= 0 for rate in (args.lr_head, args.lr_vista)):
        raise ValueError('Warmup learning rates must be positive and finite')
    if args.restore_initial_optimizer and not args.initialize_from:
        raise ValueError('--restore-initial-optimizer requires --initialize-from')
    if min(args.epochs, args.steps_per_epoch, args.patches_per_case, args.audit_cases,
           args.additional_audit_cases, args.audit_patches, args.min_epochs, args.required_passes) < 1:
        raise ValueError('Warmup sizes/epoch counts must be positive')
    if args.patches_per_case < 2 or args.patches_per_case % 2 or args.preprocess_workers < 0:
        raise ValueError('Use pairs of nodule/context patches and a nonnegative worker count')
    if args.patches_per_case != 2:
        raise ValueError('Complete nodule visits use one positive/background pair per selection')
    if args.max_cases and not args.smoke:
        raise ValueError('Limited cohorts are smoke tests, not completed production warmup')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output_dir/'config.json').exists():
        raise ValueError('Output already contains a run; use --resume or a new output directory')
    config = {**vars(args), 'variant':'v7-anatomy-warmup',
              'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'base_warmup_sha256':(hashlib.sha256(args.base_warmup.read_bytes()).hexdigest()
                                   if args.base_warmup is not None else None),
              'vista_initialization':'public_multi_organ',
              'loss_schema':LOSS_SCHEMA,
              'case_sampling':'complete_cohort_cycles_paired_patches',
              'sampling_epoch_offset':0,
              'augmentation':WINDOW_POLICY,
              'deployment_patch_probability':.5,
              'nodule_sampling_policy':NODULE_POLICY,
              'background_mining_schema':BACKGROUND_POLICY if args.background_mining else None,
              'background_coverage_policy':BACKGROUND_COVERAGE_POLICY if args.background_mining else None,
              'finish_ct_before_epoch_end':True,
              'whole_ct_case_ids':list(WHOLE_CT_CASE_IDS),
              'public_vista_sha256':hashlib.sha256(args.public_vista_checkpoint.read_bytes()).hexdigest()}
    config = json.loads(json.dumps(config, default=str))
    plan = json.loads((args.anatomy_root/'cohort.json').read_text())
    from v7.data.cohort import select_training_cases
    cases = select_training_cases(args.manifest, plan)
    if args.max_cases:
        cases = cases[:args.max_cases]
    metadata = [json.loads((args.anatomy_root/PREPARED_DIRECTORY/f'{case_id(c)}.json').read_text()) for c in cases]
    if any(m['schema'] != PROMPT_SCHEMA or tuple(m.get('prompt_ids', ())) != PROMPT_IDS for m in metadata):
        raise ValueError('Warmup requires individual labels for all native prompts')
    config['prompt_ids'] = list(PROMPT_IDS)
    config['anatomy_case_ids'] = [case_id(c) for c in cases]
    config['retained_nodule_ids'] = {case_id(c):[n['nodule_id'] for n in m['nodules'] if n['retained']]
                                   for c,m in zip(cases,metadata)}
    config['anatomy_cache_sha256'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    model = VistaAnatomySegmenter(build_vista3d(args.vista_checkpoint))
    if args.base_warmup is not None:
        saved = torch.load(args.base_warmup, map_location='cpu', mmap=True, weights_only=False)
        if not saved.get('audit', {}).get('complete'):
            raise ValueError('Base networks need a completed warmup')
        network = {k.removeprefix('segmenter.'):v for k,v in saved['model'].items() if k.startswith('segmenter.')}
        missing = model.load_state_dict(network, strict=False)
        if missing.unexpected_keys or any(k != 'label_prompt' for k in missing.missing_keys):
            raise ValueError(f'Anatomy initialization mismatch: {missing}')
        del saved, network
    model.cuda()
    set_vista_trainable(model, decoder=True, full_auto=args.trainable_scope == 'full_auto')
    for p in model.pyramid_projections.parameters():
        p.requires_grad_(False)
    head_parameters = list(model.model.class_head.parameters())
    head_ids = {id(p) for p in head_parameters}
    optimizer = torch.optim.AdamW([
        dict(params=head_parameters, lr=args.lr_head),
        dict(params=[p for p in model.model.parameters() if p.requires_grad and id(p) not in head_ids], lr=args.lr_vista)],
        weight_decay=1e-5)
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    audit_order = np.random.default_rng(args.seed+17).permutation(len(cases))
    audit_cases = [cases[int(i)] for i in audit_order[:args.audit_cases]]
    additional_cases = [cases[int(i)] for i in
        audit_order[args.audit_cases:args.audit_cases+args.additional_audit_cases]]
    if not args.smoke and len(additional_cases) != args.additional_audit_cases:
        raise ValueError('Production warmup needs disjoint primary and additional audit cases')
    config['additional_audit_case_ids'] = [case_id(c) for c in additional_cases]
    by_name = {case_id(c):c for c in cases}
    if not args.smoke and not set(WHOLE_CT_CASE_IDS) <= set(by_name):
        raise ValueError('Production warmup needs its fixed whole-CT training regression cases')
    whole_cases = [by_name[name] for name in WHOLE_CT_CASE_IDS if name in by_name]
    start_epoch, passed, saved_wandb = 0, 0, None
    best_margin = float('-inf')
    case_draw_counts = {}
    case_count_start_epoch = 1
    case_draw_offset = 0
    learning_rate_changes = []
    protocol_changes = []
    coverage_case_visits, nodule_selection_counts = {}, {}
    background_case_visits = {}
    background_coverage_start_epoch = 1
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        saved['config'].setdefault('sampling_epoch_offset',0)
        # The current checkpoint carries source provenance even if the source
        # was an older-format run. Resuming does not reload that source file.
        for key in ('initialize_from', 'initialization_checkpoint_sha256', 'sampling_epoch_offset',
                    'restore_initial_optimizer'):
            if key in saved['config']:
                config[key] = saved['config'][key]
        upgrade = args.upgrade_window_training and saved['schema'] == 'v7_native_prompt_warmup_v4'
        if args.upgrade_window_training and not upgrade:
            raise ValueError('Window-training upgrade requires a v4 source checkpoint')
        upgrade_nodules = (args.upgrade_nodule_coverage and
            (saved['schema'] == 'v7_native_prompt_warmup_v5' or upgrade))
        if args.upgrade_nodule_coverage and not upgrade_nodules:
            raise ValueError('Nodule coverage upgrade requires a v5 checkpoint, or v4 with window upgrade')
        if saved['schema'] != SCHEMA and not upgrade_nodules:
            raise ValueError('Anatomy warmup resume schema differs')
        upgrade_background = (args.upgrade_background_mining and saved['schema'] == SCHEMA
                              and not saved['config'].get('background_mining', False))
        if args.upgrade_background_mining and not upgrade_background:
            raise ValueError('Background mining upgrade requires a v6 source without background mining')
        validate_resume_configuration(saved['config'], config,
                                      change_learning_rates=args.resume_learning_rates is not None,
                                      upgrade_window_training=upgrade,
                                      upgrade_nodule_coverage=upgrade_nodules,
                                      upgrade_background_mining=upgrade_background,
                                      upgrade_background_coverage=args.upgrade_background_coverage)
        background_case_visits = load_background_visits(saved,
            reset=upgrade_background or args.upgrade_background_coverage)
        background_coverage_start_epoch = (saved['epoch']+2
            if upgrade_background or args.upgrade_background_coverage
            else saved.get('background_coverage_start_epoch', 1))
        model.load_state_dict(saved['segmenter'], strict=True)
        lr_change = restore_optimizer(optimizer, saved['optimizer'],
                                      learning_rates=args.resume_learning_rates)
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(adapt_cuda_rng_state(saved['cuda_rng'], torch.cuda.get_rng_state().numel()))
        start_epoch, passed, reference = saved['epoch']+1, saved['passed_epochs'], saved['reference']
        best_margin = saved.get('best_margin', float('-inf'))
        learning_rate_changes = list(saved.get('learning_rate_changes', []))
        protocol_changes = list(saved.get('protocol_changes', []))
        if upgrade_nodules:
            passed, best_margin = 0, float('-inf')
            change = dict(source_checkpoint=str(args.resume), source_schema=saved['schema'],
                          schema=SCHEMA, start_epoch=start_epoch+1,
                          nodule_sampling_policy=NODULE_POLICY,
                          optimizer_state_preserved=True, consecutive_quality_passes_reset=True)
            protocol_changes.append(change)
            print(json.dumps(dict(event='nodule_coverage_upgrade', **change)), flush=True)
        else:
            coverage_case_visits = saved['coverage_case_visits']
            nodule_selection_counts = saved['nodule_selection_counts']
            if nodule_coverage(config['retained_nodule_ids'], coverage_case_visits,
                               nodule_selection_counts) != saved['nodule_coverage']:
                raise ValueError('Saved nodule coverage counts differ')
        if upgrade:
            passed, best_margin = 0, float('-inf')
            change = dict(source_checkpoint=str(args.resume), source_schema=saved['schema'],
                          schema=SCHEMA, start_epoch=start_epoch+1,
                          optimizer_state_preserved=True, consecutive_quality_passes_reset=True)
            protocol_changes.append(change)
            print(json.dumps(dict(event='window_training_upgrade', **change)), flush=True)
        if upgrade_background:
            passed, best_margin = 0, float('-inf')
            change = dict(source_checkpoint=str(args.resume), source_schema=saved['schema'],
                          schema=SCHEMA, start_epoch=start_epoch+1,
                          background_mining_schema=BACKGROUND_POLICY,
                          optimizer_state_preserved=True, positive_sampling_preserved=True,
                          consecutive_quality_passes_reset=True)
            protocol_changes.append(change)
            print(json.dumps(dict(event='background_mining_upgrade', **change)), flush=True)
        if args.upgrade_background_coverage:
            passed, best_margin = 0, float('-inf')
            change = dict(source_checkpoint=str(args.resume), start_epoch=start_epoch+1,
                          background_coverage_policy=BACKGROUND_COVERAGE_POLICY,
                          coverage_counts_reset=True, optimizer_state_preserved=True,
                          positive_sampling_preserved=True, consecutive_quality_passes_reset=True)
            protocol_changes.append(change)
            print(json.dumps(dict(event='background_coverage_upgrade', **change)), flush=True)
        if lr_change is not None:
            passed, best_margin = 0, float('-inf')
            lr_change.update(start_epoch=start_epoch+1, source_checkpoint=str(args.resume))
            learning_rate_changes.append(lr_change)
            print(json.dumps(dict(event='resume_learning_rates_changed', **lr_change,
                                  optimizer_state_preserved=True,
                                  consecutive_quality_passes_reset=True)), flush=True)
        case_draw_counts = saved['case_draw_counts']
        case_count_start_epoch = saved['case_count_start_epoch']
        case_draw_offset = saved['case_draw_offset']
        if case_draw_offset != sum(case_draw_counts.values()) or not set(case_draw_counts) <= set(config['anatomy_case_ids']):
            raise ValueError('Checkpoint CT cycle position/counts differ')
        saved_wandb = None if args.new_wandb_run else saved.get('wandb')
        del saved
    elif args.initialize_from:
        saved = torch.load(args.initialize_from, map_location='cpu', mmap=True, weights_only=False)
        if saved.get('schema') not in ('v7_native_prompt_warmup_v2', 'v7_native_prompt_warmup_v3',
                                      'v7_native_prompt_warmup_v4', 'v7_native_prompt_warmup_v5', SCHEMA):
            raise ValueError('Initialization requires a native-prompt warmup checkpoint')
        source_config = saved['config']
        for key in ('manifest_sha256', 'base_warmup_sha256', 'public_vista_sha256',
                    'vista_initialization', 'prompt_ids', 'anatomy_case_ids', 'anatomy_cache_sha256',
                    'audit_cases', 'audit_patches', 'seed', 'min_dice', 'min_nodule_dice',
                    'max_nodule_dice_drop', 'max_organ_dice_drop', 'min_epochs', 'required_passes'):
            if source_config.get(key) != config.get(key):
                raise ValueError(f'Warmup source provenance differs: {key}')
        if saved.get('smoke_only') and not args.smoke:
            raise ValueError('Cannot use limited smoke weights to initialize production training')
        model.load_state_dict(saved['segmenter'], strict=True)
        if args.restore_initial_optimizer:
            for key in ('loss_schema', 'trainable_scope', 'trainable_parameters',
                        'lr_head', 'lr_vista', 'anatomy_weight'):
                if source_config.get(key) != config.get(key):
                    raise ValueError(f'Cannot restore optimizer after changing {key}')
            optimizer.load_state_dict(saved['optimizer'])
        reference = saved['reference']
        config['sampling_epoch_offset'] = source_config.get('sampling_epoch_offset',0)+saved['epoch']+1
        config['initialization_checkpoint_sha256'] = hashlib.sha256(args.initialize_from.read_bytes()).hexdigest()
        print(json.dumps(dict(event='weights_initialized', source=str(args.initialize_from),
                              source_epoch=saved['epoch']+1, optimizer_reset=not args.restore_initial_optimizer,
                              original_quality_references_retained=True)), flush=True)
        del saved
    else:
        # The nodule-only checkpoint has forgotten most organ predictions.
        # Keep its nodule quality as the reference, then initialize every native
        # VISTA parameter from the released multi-organ network. Pyramid
        # projections remain the reused V5 weights outside model.model.
        legacy_reference = audit(model, audit_cases, args.anatomy_root, seed=args.seed+131, patches=args.audit_patches)
        model.model.load_state_dict(_state_dict(args.public_vista_checkpoint), strict=True)
        reference = audit(model, audit_cases, args.anatomy_root, seed=args.seed+131, patches=args.audit_patches)
        reference['legacy_nodule_dice'] = legacy_reference['dice']['nodule']
    if additional_cases:
        extra = reference.get('additional')
        valid_extra = (extra is not None and extra.get('case_ids') == config['additional_audit_case_ids']
                       and extra.get('seed') == args.seed+949
                       and extra.get('patches_per_case') == args.audit_patches)
        if not valid_extra:
            if args.resume:
                raise ValueError('Exact resume requires the saved additional-case reference')
            reference['additional'] = additional_reference(additional_cases, args)
    if not args.smoke:
        whole_ref = reference.get('whole_ct')
        if whole_ref is None:
            if args.resume and not args.upgrade_window_training:
                raise ValueError('Exact resume requires the saved whole-CT references')
            reference['whole_ct'] = whole_ct_reference(whole_cases, args, load_case)
        elif whole_ref.get('case_ids') != list(WHOLE_CT_CASE_IDS):
            raise ValueError('Whole-CT reference cases differ')
    (args.output_dir/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    monitor = Monitor(args, config, saved_wandb)
    if monitor.run is not None:
        monitor.run.tags = ('v7', 'anatomy-warmup', 'totalsegmentator')
    print(json.dumps(dict(event='reference', audit=reference)), flush=True)
    complete = False
    if start_epoch >= args.epochs:
        raise ValueError('No warmup epochs remain; increase --epochs when resuming')
    for epoch in range(start_epoch, args.epochs):
        start = time.monotonic()
        rng = np.random.default_rng(args.seed+epoch+config['sampling_epoch_offset'])
        total = np.zeros(len(PROMPT_IDS)); counts = {}; steps = 0
        explicit_crops = dict(nonempty=0, empty=0)
        mining_stats = dict(ranked_windows=0, selections=0, deployment_selected=0,
                            scored_pairs=0, initial_bce_sum=0., selected_bce_sum=0., seconds=0.)
        epoch_case_draws = {}
        draws = (args.steps_per_epoch+args.patches_per_case-1)//args.patches_per_case
        indices = cycle_case_indices(len(cases), args.seed, case_draw_offset, draws)
        for index, (image, packed, info) in prefetch_cases(
                cases, args.anatomy_root, indices, args.preprocess_workers):
            name = case_id(cases[index])
            epoch_case_draws[name] = epoch_case_draws.get(name, 0)+1
            case_draw_counts[name] = case_draw_counts.get(name, 0)+1
            case_draw_offset += 1
            model.train()
            miner = None
            if args.background_mining:
                mining_start = time.monotonic()
                miner = BackgroundCropMiner(model, image, packed)
                mining_stats['ranked_windows'] += miner.ranked_windows
                mining_stats['seconds'] += time.monotonic()-mining_start
            plan = visit_plan(info, rng, args.background_ratio)
            for selected_id, positive in plan:
                # Finish the entire CT visit, even across the nominal epoch
                # boundary. Every retained nodule gets an explicit selection;
                # the final pair keeps the original anatomy/size priorities.
                ct, target, valid, kind = sample_patch(image, packed, info, rng,
                    force_nodule=positive, nodule_id=selected_id,
                    deployment_probability=config['deployment_patch_probability'])
                if miner is not None and not positive:
                    mining_start = time.monotonic()
                    (ct, target, valid, kind), selection = miner.choose((ct, target, valid, kind))
                    mining_stats['selections'] += 1
                    mining_stats['deployment_selected'] += int(selection['deployment_selected'])
                    if selection['initial_score'] is not None and selection['selected_score'] is not None:
                        mining_stats['scored_pairs'] += 1
                        mining_stats['initial_bce_sum'] += selection['initial_score']
                        mining_stats['selected_bce_sum'] += selection['selected_score']
                    mining_stats['seconds'] += time.monotonic()-mining_start
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    def objective(value, target=target, valid=valid):
                        logits, _ = model.forward_with_features(value)
                        return patch_objectives(logits, target, valid.cuda())
                    values = checkpoint(objective, ct.cuda(), use_reentrant=False)
                    loss = values[NODULE_INDEX]+args.anatomy_weight*values[:NODULE_INDEX].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite anatomy warmup loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += values.detach().cpu().numpy()
                counts[kind] = counts.get(kind, 0)+1
                if selected_id is not None:
                    selected = nodule_selection_counts.setdefault(name, {})
                    key = str(selected_id)
                    selected[key] = selected.get(key, 0)+1
                    selection_kind = ('nonempty' if
                        bool(((target >> NODULE_INDEX)&1).any()) else 'empty')
                    explicit_crops[selection_kind] += 1
                steps += 1
            coverage_case_visits[name] = coverage_case_visits.get(name, 0)+1
            if miner is not None:
                background_case_visits[name] = background_case_visits.get(name, 0)+1
            if steps >= args.steps_per_epoch:
                break
        current = audit(model, audit_cases, args.anatomy_root, seed=args.seed+131, patches=args.audit_patches)
        additional = (audit(model, additional_cases, args.anatomy_root,
                            seed=args.seed+949, patches=args.audit_patches) if additional_cases else None)
        quality = combined_readiness(current, additional, reference, args,
                                     cohort_covered=len(case_draw_counts) == len(cases))
        ready = quality['passed']
        improved = quality['worst_dice_margin'] > best_margin
        if improved:
            best_margin = quality['worst_dice_margin']
        passed = passed+1 if ready else 0
        whole_audit, whole_quality = None, None
        if not args.smoke and ready:
            # Never reuse an earlier epoch's whole-CT pass for current weights.
            whole_audit = whole_ct_audit(model, whole_cases, args.anatomy_root, load_case)
            whole_quality = whole_ct_readiness(whole_audit, reference['whole_ct'], args)
        coverage = nodule_coverage(config['retained_nodule_ids'], coverage_case_visits,
                                   nodule_selection_counts)
        mining_coverage = background_coverage(config['anatomy_case_ids'], background_case_visits,
                                              enabled=args.background_mining)
        complete = (completion_ready(quality, whole_quality, epoch+1, passed, args)
                    and coverage['passed'] and mining_coverage['passed'])
        record = dict(epoch=epoch+1, losses=dict(zip(PROMPT_NAMES, (total/steps).tolist())),
                      audit=current, additional_audit=additional, readiness=quality,
                      whole_ct_audit=whole_audit, whole_ct_readiness=whole_quality,
                      passed=ready, consecutive_passes=passed, complete=complete,
                      patches=steps, sampling=counts, case_draws=epoch_case_draws,
                      nodule_coverage=coverage,
                      explicit_nodule_crops=explicit_crops,
                      background_mining=mining_stats,
                      background_coverage=mining_coverage,
                      case_coverage=dict(cohort_cases=len(cases),
                          distinct_this_epoch=len(epoch_case_draws),
                          distinct_since_recording=len(case_draw_counts),
                          draws_this_epoch=sum(epoch_case_draws.values()),
                          draws_since_recording=sum(case_draw_counts.values()),
                          counted_from_epoch=case_count_start_epoch),
                      seconds=time.monotonic()-start)
        print(json.dumps(record), flush=True)
        with (args.output_dir/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        monitor.log(dict(warmup_epoch=epoch+1,
            **{f'warmup/loss_{k}':v for k,v in record['losses'].items()},
            **{f'warmup/dice_{k}':v for k,v in current['dice'].items()},
            **{f'warmup/precision_{k}':v for k,v in current['precision'].items()},
            **{f'warmup/recall_{k}':v for k,v in current['recall'].items()},
            **{f'warmup/samples_{k}':v for k,v in counts.items()},
            **{f'warmup/cases_{k}':v for k,v in record['case_coverage'].items()},
            **{f'warmup/nodule_coverage_{k}':v for k,v in coverage.items()},
            **{f'warmup/explicit_nodule_crops_{k}':v for k,v in explicit_crops.items()},
            **{f'warmup/background_{k}':v for k,v in mining_stats.items()},
            **{f'warmup/background_coverage_{k}':v for k,v in mining_coverage.items()},
            **({f'warmup/additional_dice_{k}':v for k,v in additional['dice'].items()} if additional else {}),
            **({f'warmup/additional_{k}_nodule':additional[k]['nodule']
                for k in ('precision', 'recall')} if additional else {}),
            **({f'warmup/whole_ct_nodule_dice_{name}':row['overlap']['nodule']['dice']
                for name,row in whole_audit.items()} if whole_audit else {}),
            **({'warmup/whole_ct_passed':int(whole_quality['passed'])} if whole_quality else {}),
            **{'warmup/passed':int(ready), 'warmup/lr_head':args.lr_head, 'warmup/lr_vista':args.lr_vista}))
        payload = dict(schema=SCHEMA, epoch=epoch, segmenter=model.state_dict(), optimizer=optimizer.state_dict(),
                       complete=complete, smoke_only=args.smoke, config=config, audit=current,
                       additional_audit=additional, reference=reference,
                       whole_ct_audit=whole_audit, whole_ct_readiness=whole_quality,
                       whole_ct_epoch=epoch if whole_audit is not None else None,
                       readiness=quality, best_margin=best_margin,
                       case_draw_counts=case_draw_counts, case_count_start_epoch=case_count_start_epoch,
                       case_draw_offset=case_draw_offset,
                       learning_rate_changes=learning_rate_changes,
                       protocol_changes=protocol_changes,
                       background_mining=mining_stats,
                       background_case_visits=background_case_visits,
                       background_coverage=mining_coverage, background_coverage_epoch=epoch,
                       background_coverage_start_epoch=background_coverage_start_epoch,
                       coverage_case_visits=coverage_case_visits,
                       nodule_selection_counts=nodule_selection_counts,
                       nodule_coverage=coverage, nodule_coverage_epoch=epoch,
                       passed_epochs=passed, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                       wandb=monitor.state_dict())
        atomic_save(payload, args.output_dir/'latest.pt')
        if improved:
            atomic_save(payload, args.output_dir/'best.pt')
        if complete:
            atomic_save(payload, args.output_dir/'complete.pt')
            break
    monitor.finish()
    if not args.smoke and not complete and not args.tuning_allow_unready:
        raise RuntimeError('Anatomy warmup has not met readiness thresholds; latest.pt is resumable, '
                           'but joint training must wait for complete.pt')


if __name__ == '__main__':
    main()
