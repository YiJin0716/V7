"""Reuse completed V5 hard warmup networks, and fit a fresh V7 risk bank."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from v7.models.radiomics import FEATURE_NAMES, SEMANTIC_NAMES, INTENSITY_NAMES, select_warmup_features
from v7._core.common.train_utils import atomic_save
from v7._core.model_v5.model import ARCHITECTURE as V5_ARCHITECTURE
from v7._core.model_v5.preparation import SCHEMA
from v7.data.anatomy import PREPARED_DIRECTORY, PROMPT_IDS
from v7.training.vista_warmup import SCHEMA as ANATOMY_WARMUP_SCHEMA
from v7.training.warmup_quality import require_whole_ct_checkpoint
from v7.training.warmup_sampling import require_coverage
from v7.training.warmup_background import require_coverage as require_background_coverage

NETWORK_CONFIG_KEYS = (
    'manifest', 'vista_checkpoint', 'window_size', 'overlap', 'screen_shape',
    'roi_shape', 'num_queries', 'detr_hidden_dim', 'detr_coarse_shape',
    'detr_nheads', 'detr_encoder_layers', 'detr_decoder_layers', 'context_channels',
    'refiner_base', 'semantic_roi_size', 'semantic_width', 'semantic_min_dice',
    'detector_kind', 'detector_nms_iou', 'native_proposal_threshold',
    'native_confidence_residual_bound', 'inference_object_threshold',
    'minimum_roi_coverage', 'hard_negatives', 'teacher_full_epochs',
    'teacher_zero_epoch', 'teacher_jitter',
)


def load_warmup_network(model, saved):
    """Transfer old inputs exactly and zero-pad only the new anatomy channels."""
    audit = saved.get('audit', {})
    if (not audit.get('complete') or audit.get('schema') != SCHEMA
            or audit.get('smoke_only') or 'optimizer' in saved or 'epoch' in saved):
        raise ValueError('Initialization requires a completed production warmup')
    state = {k:v for k,v in saved['model'].items() if not k.startswith('rashomon.')}
    current = model.state_dict()
    expanded = ('detr.input_projection.0.weight', 'detr.mask_feature_projection.0.weight',
                'context_compressor.weight')
    for key in expanded:
        value = state[key]
        if current[key].shape[1] != value.shape[1]+4:
            raise ValueError(f'Expected exactly four new inputs: {key}')
        padded = torch.zeros_like(current[key], device='cpu')
        padded[:, :value.shape[1]].copy_(value)
        state[key] = padded
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = ('rashomon.', 'segmenter.label_prompt')
    if incompatible.unexpected_keys or any(not k.startswith(allowed_missing) for k in incompatible.missing_keys):
        raise ValueError(f'Warmup network mismatch: {incompatible}')
    model.refiner.use_checkpoint = False
    return state


@torch.no_grad()
def initialize_from_warmup(model, args, rank, world):
    path = args.warmup_checkpoint.resolve()
    config = json.loads((path.parent / 'config.json').read_text())
    if config.get('architecture') != V5_ARCHITECTURE:
        raise ValueError('Expected V5 warmup source configuration')
    if config['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        raise ValueError('Warmup and V7 training manifests differ')
    for key in NETWORK_CONFIG_KEYS:
        actual, expected = getattr(args, key), config[key]
        if isinstance(actual, (tuple, list)):
            actual, expected = list(actual), list(expected)
        if isinstance(actual, Path):
            actual = str(actual)
        if actual != expected:
            raise ValueError(f'Warmup network configuration differs: {key}')
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    network = load_warmup_network(model, saved)
    anatomy = torch.load(args.anatomy_checkpoint, map_location='cpu', weights_only=False)
    smoke = bool(args.preparation_smoke and anatomy.get('smoke_only'))
    if anatomy.get('schema') not in (ANATOMY_WARMUP_SCHEMA, 'v7_native_prompt_warmup_v5') or (not anatomy.get('complete') and not smoke):
        raise ValueError('Complete anatomy warmup before joint training')
    if not smoke and (not anatomy.get('readiness', {}).get('passed')
                      or anatomy.get('additional_audit') is None
                      or not anatomy.get('readiness', {}).get('cohort_covered')):
        raise ValueError('Anatomy warmup requires both probe groups and complete training-cohort coverage')
    if not smoke:
        require_whole_ct_checkpoint(anatomy)
        if anatomy['schema'] == ANATOMY_WARMUP_SCHEMA:
            require_coverage(anatomy)
            require_background_coverage(anatomy)
    if tuple(anatomy['config'].get('prompt_ids', ())) != PROMPT_IDS:
        raise ValueError('Anatomy warmup prompt order differs')
    if anatomy['config'].get('vista_initialization') != 'public_multi_organ':
        raise ValueError('V7 requires native prompt warmup initialized from public multi-organ VISTA')
    if anatomy['config']['manifest_sha256'] != config['manifest_sha256']:
        raise ValueError('Anatomy warmup manifest mismatch')
    public_only = bool(anatomy['config'].get('public_only'))
    if public_only:
        if anatomy['config'].get('base_warmup_sha256') is not None:
            raise ValueError('Public-only anatomy warmup must not reuse an LIDC warmup')
        # In clean tuning the anatomy warmup runs FIRST. Downstream warmup then
        # trains fresh networks against its frozen VISTA. Prove that lineage
        # by checking every native VISTA tensor before keeping its projections.
        for key, value in anatomy['segmenter'].items():
            if key.startswith('model.'):
                torch.testing.assert_close(saved['model']['segmenter.'+key], value, rtol=0, atol=0)
    elif anatomy['config']['base_warmup_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Anatomy warmup used different base network weights')
    anatomy_cases = anatomy['config'].get('anatomy_case_ids')
    if anatomy_cases is None and not smoke:
        raise ValueError('Anatomy warmup checkpoint lacks its exact training cohort')
    if anatomy_cases is not None:
        current_metadata = [json.loads((args.anatomy_root/PREPARED_DIRECTORY/f'{name}.json').read_text())
                            for name in anatomy_cases]
        current_hash = hashlib.sha256(json.dumps(current_metadata, sort_keys=True).encode()).hexdigest()
        if current_hash != anatomy['config']['anatomy_cache_sha256']:
            raise ValueError('Anatomy labels/statistics changed after warmup')
    if public_only:
        expected_cases = {f"{c['patient_id']}__scan{int(c['scan_id'])}"
                          for c in json.loads(args.manifest.read_text())['training']}
        if set(anatomy_cases or ()) != expected_cases:
            raise ValueError('Public-only anatomy warmup must use exactly the tuning training set')
        model.segmenter.model.load_state_dict(
            {k.removeprefix('model.'): v for k, v in anatomy['segmenter'].items() if k.startswith('model.')}, strict=True)
    else:
        model.segmenter.load_state_dict(anatomy['segmenter'], strict=True)
    # V5 stored paired [18 radiomics, 6 predicted semantics] in this cache even
    # though its residual head could only select the final six columns.
    extra = saved['model']['rashomon._extra_state']
    if extra.get('schema') != 'radiomics_offset_semantic_v4':
        raise ValueError('Unexpected warmup feature-cache schema')
    if tuple(extra.get('feature_names', ())) != SEMANTIC_NAMES:
        raise ValueError('Warmup semantic feature order differs')
    allowed = model.rashomon.allowed_cases
    if not allowed <= set(extra['allowed_cases']):
        raise ValueError('V7 training cases were not in the warmup training cohort')
    rows = []
    for key, (features, label) in sorted(extra['memory'].items()):
        if key[0] not in allowed:
            continue
        features = select_warmup_features(features)
        if features.shape != (len(FEATURE_NAMES),):
            raise ValueError(f'Invalid paired warmup features: {key}')
        if label not in (0., 1.):
            raise ValueError(f'Invalid warmup label: {key}')
        rows.append((key, (features, label)))
    local = rows[rank::world]
    features = torch.as_tensor(np.stack([v[0] for _, v in local]) if local else
                               np.empty((0, len(FEATURE_NAMES))), device=model.device)
    labels = torch.tensor([v[1] for _, v in local], device=model.device)
    model.rashomon.train()
    model.rashomon.observe_and_refit(features, labels, [key for key, _ in local])
    if not int(model.rashomon.count):
        raise RuntimeError('Warmup cache needs sufficient training nodules of both classes')
    model.set_temperature(args.mask_temperature_start)
    result = deepcopy(saved['audit'])
    result['smoke_only'] = smoke
    for key in ('baseline_samples', 'semantic_samples'):
        result.pop(key, None)
    result.update(risk_samples=len(model.rashomon.memory), reused_network_tensors=len(network),
                  warmup_checkpoint=str(path),
                  retrained_components=(['vista_encoder', 'vista_auto_decoder', 'vista_class_head']
                      if anatomy['config'].get('trainable_scope') == 'full_auto'
                      else ['vista_last_auto_decoder', 'vista_class_head']),
                  anatomy_checkpoint=str(args.anatomy_checkpoint), anatomy_audit=anatomy['audit'],
                  risk_initialization='fresh continuous FasterRisk on 8 geometry + 6 predicted semantic warmup features',
                  excluded_intensity_features=list(INTENSITY_NAMES),
                  risk_feature_names=list(FEATURE_NAMES), risk_bank_size=int(model.rashomon.count))
    if not Path(result['cache_dir']).is_dir():
        raise ValueError('Warmup semantic anchor cache is missing')
    if rank == 0:
        (args.output_dir / 'warmup.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(event='v7_warmup_reused', checkpoint=str(path),
                              tensors=len(network), samples=len(rows),
                              bank_size=int(model.rashomon.count), fit=model.rashomon.last_fit)), flush=True)
        if not args.skip_save:
            atomic_save(dict(model=model.state_dict(), audit=result), args.output_dir / 'warmup.pt')
    if dist.is_initialized():
        dist.barrier()
    return result
