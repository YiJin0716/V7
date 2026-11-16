"""Reuse completed V5 hard warmup networks, and fit a fresh V6 risk bank."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from v7._core.common.features import FEATURE_NAMES
from v7._core.common.train_utils import atomic_save
from v7._core.model_v5.model import ARCHITECTURE as V5_ARCHITECTURE
from v7._core.model_v5.preparation import SCHEMA

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
    """Strictly load every non-risk tensor; never reuse the V5 residual head."""
    audit = saved.get('audit', {})
    if (not audit.get('complete') or audit.get('schema') != SCHEMA
            or audit.get('smoke_only') or 'optimizer' in saved or 'epoch' in saved):
        raise ValueError('Initialization requires completed production warmup, not a joint epoch')
    state = {k: v for k, v in saved['model'].items() if not k.startswith('rashomon.')}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not k.startswith('rashomon.') for k in incompatible.missing_keys):
        raise ValueError(f'Warmup network mismatch: {incompatible}')
    model.refiner.use_checkpoint = False  # Whole-refiner checkpointing in joint forward.
    return state


@torch.no_grad()
def initialize_from_warmup(model, args, rank, world):
    path = args.warmup_checkpoint.resolve()
    config = json.loads((path.parent / 'config.json').read_text())
    if config.get('architecture') != V5_ARCHITECTURE:
        raise ValueError('Expected V5 warmup source configuration')
    if config['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        raise ValueError('Warmup and V6 training manifests differ')
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
    # V5 stored paired [18 radiomics, 6 predicted semantics] in this cache even
    # though its residual head could only select the final six columns.
    extra = saved['model']['rashomon._extra_state']
    if extra.get('schema') != 'radiomics_offset_semantic_v4':
        raise ValueError('Unexpected warmup feature-cache schema')
    allowed = model.rashomon.allowed_cases
    if not allowed <= set(extra['allowed_cases']):
        raise ValueError('V6 training cases were not in the warmup training cohort')
    rows = [(key, value) for key, value in sorted(extra['memory'].items()) if key[0] in allowed]
    for key, (features, label) in rows:
        if np.asarray(features).shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError(f'Invalid paired warmup features: {key}')
        if label not in (0., 1.):
            raise ValueError(f'Invalid warmup label: {key}')
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
    for key in ('baseline_samples', 'semantic_samples'):
        result.pop(key, None)
    result.update(risk_samples=len(model.rashomon.memory), reused_network_tensors=len(network),
                  warmup_checkpoint=str(path), retrained_components=False,
                  risk_initialization='fresh continuous FasterRisk on paired warmup prediction features',
                  risk_feature_names=list(FEATURE_NAMES), risk_bank_size=int(model.rashomon.count))
    if not Path(result['cache_dir']).is_dir():
        raise ValueError('Warmup semantic anchor cache is missing')
    if rank == 0:
        (args.output_dir / 'warmup.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(event='v6_warmup_reused', checkpoint=str(path),
                              tensors=len(network), samples=len(rows),
                              bank_size=int(model.rashomon.count), fit=model.rashomon.last_fit)), flush=True)
        if not args.skip_save:
            atomic_save(dict(model=model.state_dict(), audit=result), args.output_dir / 'warmup.pt')
    if dist.is_initialized():
        dist.barrier()
    return result
