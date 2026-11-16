"""Performance-only resume options and deterministic additional GPU RNG streams."""
import random
import numpy as np
import torch

from v7._core.common.checkpoint_compat import adapt_cuda_rng_state

PERFORMANCE_ARGUMENTS = {'data_cache', 'data_workers', 'prefetch_factor', 'fast_kernels',
                         'cudnn_benchmark', 'allow_world_size_change'}


def validate_resume_arguments(args, config, world):
    mutable = PERFORMANCE_ARGUMENTS | {
        'output_dir', 'resume', 'warmup_checkpoint', 'epochs', 'stop_after_epoch',
        'save_every', 'wandb_name',
    }
    if args.benchmark_steps is not None and args.skip_save:
        mutable |= {'benchmark_steps', 'skip_save', 'wandb_mode'}
        if args.wandb_mode != 'disabled':
            raise ValueError('Resume benchmarks must disable W&B to protect the production run')
    defaults = {'detector_kind':'legacy','detector_nms_iou':.5,'object_background_weight':1.,
                'native_proposal_threshold':.1,'native_confidence_residual_bound':None,
                'mask_temperature_start':1.,'mask_temperature_end':1.,
                'mask_temperature_epochs':18,'cudnn_benchmark':True,
                'loss_segmentation_weight':1., 'loss_semantic_weight':1., 'loss_malignancy_weight':1.}
    for name, value in vars(args).items():
        if name not in mutable and str(value) != str(config.get(name, defaults.get(name))):
            raise ValueError(f'Resume argument mismatch: {name}')
    if config['world_size'] != world and not args.allow_world_size_change:
        raise ValueError('Changing GPU count requires --allow-world-size-change')


def restore_rank_rng(states, *, rank, world, seed, completed_epochs, device, allow_resize):
    if not states:
        raise ValueError('Checkpoint has no saved random streams')
    if len(states) != world and not allow_resize:
        raise ValueError('Checkpoint RNG rank count differs')
    if rank < len(states):
        rng = states[rank]
        torch.set_rng_state(rng['torch'])
        cuda_rng = adapt_cuda_rng_state(rng['cuda'], torch.cuda.get_rng_state(device).numel())
        torch.cuda.set_rng_state(cuda_rng, device)
        if not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
            raise RuntimeError('CUDA RNG state changed while restoring checkpoint')
        np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])
        return dict(restored=True, saved_rank=rank, saved_bytes=rng['cuda'].numel(),
                    restored_bytes=cuda_rng.numel())
    # Additional ranks have no prior stream. Keep each new stream distinct and
    # reproducible; model/Adam/schedules still load identically on every rank.
    new_seed = (seed + 1000003 * completed_epochs + rank) % (2**32)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed(new_seed)
    np.random.seed(new_seed)
    random.seed(new_seed)
    return dict(restored=False, new_rank=rank, seed=new_seed)
