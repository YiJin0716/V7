#!/usr/bin/env python3
"""DDP training entry point for the coarse-to-fine model v7."""

from __future__ import annotations

import argparse
import hashlib
from itertools import count
import json
import math
import os
from pathlib import Path
from v7 import paths
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from v7.data.policy import POLICY_NAME
from v7.data.dataset import DEFAULT_MANIFEST, WholeCTLIDCDataset, load_cases, whole_ct_collate
from v7.data.sampling import AnatomyOversamplingSampler
from v7.models.vista import set_vista_trainable
from v7.data.organ_labels import DEFAULT_ROOT
from v7.data.anatomy import PROMPT_IDS
from v7.training.loss import WholeCTCriterionV7, LossAccumulator, LOSS_SCHEMA, LOSS_BLOCKS
from v7.models.joint import ARCHITECTURE, WholeCTJointModelV7
from v7._core.common.schedule import TemperatureSchedule, LearningRateSchedule, make_lr_scheduler, assert_learning_rates
from v7._core.model_v5.features import NORMALIZATION, CONVOLUTION
from v7._core.model_v5.warmup import AnchorStream
from v7.training.initialization import initialize_from_warmup
from v7._core.model_v5.preparation import SCHEMA, geometry_audit
from v7.training.monitor import Monitor, epoch_metrics
from v7._core.model_v5.warmup import audit_semantics
from v7._core.model_v5.diagnostics import SemanticAccumulator
from v7.models.radiomics import FEATURE_NAMES, SEMANTIC_NAMES, RADIOMICS_NAMES, INTENSITY_NAMES
from v7.models.risk import SCHEMA as RISK_SCHEMA
from v7.training.runtime import validate_resume_arguments, restore_rank_rng
from v7._core.common.checkpoint_compat import adapt_cuda_rng_state
from v7._core.common.base_model import DEFAULT_VISTA_CHECKPOINT


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--vista-checkpoint", type=Path, default=DEFAULT_VISTA_CHECKPOINT)
    parser.add_argument("--semantic-roi-size", type=int, default=64)
    parser.add_argument("--semantic-width", type=int, default=16)
    parser.add_argument("--inference-object-threshold", type=float, default=.5)
    parser.add_argument('--detector-kind',choices=('legacy','mask_guided','native_seeded'),
                        help='Defaults to the checkpoint variant on resume/restart, otherwise legacy')
    parser.add_argument('--native-proposal-threshold',type=float)
    parser.add_argument('--native-confidence-residual-bound',type=float,
                        help='Optional maximum learned native confidence correction in log odds')
    parser.add_argument('--detector-nms-iou',type=float)
    parser.add_argument('--object-background-weight',type=float)
    parser.add_argument('--warmup-checkpoint', type=Path, default=paths.COMPONENT_WARMUP,
        help='Completed V5 hard warmup; networks reused and direct risk bank refitted')
    parser.add_argument("--amp-dtype", choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument("--sparsity", type=int, default=5)
    parser.add_argument("--coefficient-bound", type=float, default=5.0)
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--gap", type=float, default=0.05)
    parser.add_argument("--sample-fraction", type=float, default=0.3)
    parser.add_argument("--bank-min-samples", type=int, default=32)
    parser.add_argument("--bank-refresh-every", type=int, default=1)
    parser.add_argument("--bank-mode", choices=['sample', 'single', 'all', 'best'], default='sample')
    parser.add_argument("--missing-nodule-weight", type=float, default=4.0)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule-epochs", type=int, default=18)
    parser.add_argument('--mask-temperature-start', type=float,
                        help='First joint epoch temperature; defaults to 1, or the resumed schedule')
    parser.add_argument('--mask-temperature-end', type=float,
                        help='Final temperature; defaults to the start value (constant)')
    parser.add_argument('--mask-temperature-epochs', type=int,
                        help='Cosine temperature horizon, including both endpoints; default 18')
    parser.add_argument("--stop-after-epoch", type=int, help="Save then stop; only for a limited smoke cohort")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18,
                        help='Total epochs; 0 continues a checkpoint until explicitly stopped')
    parser.add_argument("--window-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--screen-shape", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--roi-shape", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--num-queries", type=int, default=24)
    parser.add_argument("--detr-hidden-dim", type=int, default=128)
    parser.add_argument("--detr-coarse-shape", type=int, nargs=3, default=(12, 12, 12))
    parser.add_argument("--detr-nheads", type=int, default=4)
    parser.add_argument("--detr-encoder-layers", type=int, default=2)
    parser.add_argument("--detr-decoder-layers", type=int, default=2)
    parser.add_argument("--context-channels", type=int, default=8)
    parser.add_argument("--refiner-base", type=int, default=8)
    parser.add_argument("--hard-negatives", type=int, default=2)
    parser.add_argument("--teacher-full-epochs", type=int, default=1)
    parser.add_argument("--teacher-zero-epoch", type=int, default=11)
    parser.add_argument("--teacher-jitter", type=int, default=8)
    parser.add_argument("--minimum-roi-coverage", type=float, default=0.8)
    parser.add_argument("--lr-main", type=float, default=1e-4)
    parser.add_argument("--lr-vista", type=float, default=1e-6)
    parser.add_argument("--lr-semantic", type=float, default=3e-5)
    parser.add_argument('--lr-semantic-warmup', type=float, default=3e-4)
    parser.add_argument('--refiner-min-epochs', type=int, default=15)
    parser.add_argument('--refiner-max-epochs', type=int, default=30)
    parser.add_argument('--refiner-probe-nodules', type=int, default=64)
    parser.add_argument('--refiner-min-dice', type=float, default=.5)
    parser.add_argument('--geometry-min-epochs', type=int, default=15)
    parser.add_argument('--geometry-max-epochs', type=int, default=30)
    parser.add_argument('--warmup-ready-epochs', type=int, default=3,
                        help='Consecutive passing audits required to finish each warmup stage')
    parser.add_argument('--geometry-probe-cases', type=int, default=32)
    parser.add_argument('--geometry-min-coverage', type=float, default=.75)
    parser.add_argument('--geometry-min-dice', type=float, default=.25)
    parser.add_argument('--geometry-min-coarse-dice', type=float, default=.1)
    parser.add_argument('--geometry-min-usable', type=float, default=.5)
    parser.add_argument('--semantic-min-dice', type=float, default=.3)
    parser.add_argument('--semantic-min-epochs', type=int, default=20)
    parser.add_argument('--semantic-probe-nodules', type=int, default=128)
    parser.add_argument('--semantic-min-std-ratio', type=float, default=.15)
    parser.add_argument('--semantic-min-correlation', type=float, default=.25)
    parser.add_argument('--semantic-nll-margin', type=float, default=.01)
    parser.add_argument('--semantic-anchor-weight', type=float, default=.5)
    parser.add_argument('--semantic-anchor-batch-size', type=int, default=4)
    parser.add_argument('--semantic-guard-patience', type=int, default=2)
    parser.add_argument('--preparation-smoke', action='store_true',
                        help='Limited joint pipeline test; bypass the post-epoch regression stop only')
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=4.0,
        help=(
            "Initial GradScaler value for optional FP16 runs. BF16 runs do not "
            "use loss scaling."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--cudnn-benchmark', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable cuDNN algorithm autotuning (off for the V7 recovery run)')
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--smoke-curriculum-epoch",
        type=int,
        help="Test one later curriculum phase; allowed only with --epochs 1 and --max-cases",
    )
    parser.add_argument("--save-every", type=int, default=3)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--benchmark-steps", type=int,
                        help="Stop after this many synchronized steps; requires --skip-save")
    parser.add_argument('--warmup-epochs', type=int, default=40,
                        help='Maximum augmented semantic preparation epochs')
    parser.add_argument('--warmup-batch-size', type=int, default=8)
    parser.add_argument('--wandb-mode', choices=['online','offline','disabled'], default='disabled')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-project', default='V7')
    parser.add_argument('--wandb-name')
    parser.add_argument('--anatomy-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--anatomy-checkpoint', type=Path, required=True)
    parser.add_argument('--anatomy-dropout', type=float, default=.2)
    parser.add_argument('--anatomy-weight', type=float, default=.1)
    parser.add_argument('--vista-nodule-weight', type=float, default=1.)
    for block in LOSS_BLOCKS:
        parser.add_argument(f'--loss-{block}-weight', type=float, default=1.,
                            help=f'Multiply the entire {block} block; preserve its internal ratios')
    parser.add_argument('--oversample-fraction', type=float, default=.5)
    parser.add_argument('--near-anatomy-mm', type=float, default=5.)
    parser.add_argument('--data-cache', type=Path, help='Fully prepared, source-validated V7 CPU batches')
    parser.add_argument('--data-workers', type=int, default=2)
    parser.add_argument('--prefetch-factor', type=int, default=1)
    parser.add_argument('--fast-kernels', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow-world-size-change', action='store_true',
                        help='Keep model/Adam/epoch schedules while changing CTs per synchronized update')
    args = parser.parse_args(argv)
    source = args.resume or args.warmup_checkpoint
    saved_config = json.loads((source.parent / 'config.json').read_text())
    for name, fallback, convert in (
        ('detector_kind', 'native_seeded', str),
        ('native_proposal_threshold', .1, float),
        ('detector_nms_iou', .5, float),
        ('object_background_weight', .1, float),
    ):
        if getattr(args, name) is None:
            setattr(args, name, convert(saved_config.get(name, fallback)))
    if args.native_confidence_residual_bound is None:
        args.native_confidence_residual_bound = saved_config.get('native_confidence_residual_bound', 2.)
    temperature_config = saved_config if args.resume else {}
    for name, fallback in (('mask_temperature_start', 1.0),
                           ('mask_temperature_end', .1), ('mask_temperature_epochs', 18)):
        if getattr(args, name) is None:
            setattr(args, name, temperature_config.get(name, fallback))
    return args


from v7._core.common.train_utils import (
    distributed_setup, reduce, gradient_norm, finish_scaled_step, atomic_save,
)


def set_training_phase(model: WholeCTJointModelV7, epoch: int) -> str:
    set_vista_trainable(model.segmenter, decoder=epoch >= 15)
    return "end_to_end_unfrozen" if epoch >= 15 else "joint_anatomy"


def diagnostic_scale(epoch: int) -> float:
    # Official-mask warmup precedes this loop. Both risk objectives participate
    # at their full configured weights from the first joint-training update.
    return 1.0


def epoch_indices(start_epoch: int, epochs: int):
    if start_epoch < 0 or epochs < 0:
        raise ValueError('Epochs must be nonnegative')
    return count(start_epoch) if epochs == 0 else range(start_epoch, epochs)


def main() -> None:
    args = parse_args()
    if args.epochs < 0 or (args.epochs == 0 and args.resume is None):
        raise ValueError('An open-ended run (--epochs 0) requires a checkpoint')
    variant = 'v7'
    if args.data_workers < 0 or args.prefetch_factor < 1:
        raise ValueError('Data workers must be nonnegative and prefetch factor positive')
    if args.preparation_smoke and not (args.max_cases and args.benchmark_steps and args.skip_save):
        raise ValueError('--preparation-smoke requires --max-cases, --benchmark-steps and --skip-save')
    if not (1 <= args.geometry_min_epochs <= args.geometry_max_epochs
            and 1 <= args.semantic_min_epochs <= args.warmup_epochs
            and 1 <= args.refiner_min_epochs <= args.refiner_max_epochs):
        raise ValueError('Preparation epoch bounds are invalid')
    if min(args.semantic_anchor_batch_size,args.geometry_probe_cases,args.semantic_probe_nodules,
           args.semantic_guard_patience,args.warmup_ready_epochs,args.refiner_probe_nodules) < 1:
        raise ValueError('Preparation/probe/anchor sizes and guard patience must be positive')
    if args.warmup_ready_epochs > min(args.geometry_max_epochs,args.refiner_max_epochs,args.warmup_epochs):
        raise ValueError('Warmup epoch limits must allow consecutive passing audits')
    for key in ('geometry_min_coverage','geometry_min_dice','geometry_min_coarse_dice','geometry_min_usable',
                'refiner_min_dice','semantic_min_dice','semantic_min_std_ratio','semantic_min_correlation'):
        if not 0 <= getattr(args,key) <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    if not 0 <= args.inference_object_threshold <= 1:
        raise ValueError('Object threshold must be in [0, 1]')
    if args.epochs and args.smoke_curriculum_epoch is None and args.epochs - args.teacher_zero_epoch + 1 < 3:
        raise ValueError('V7 requires at least three epochs after teacher reaches zero')
    if args.benchmark_steps is not None and (args.benchmark_steps < 1 or not args.skip_save):
        raise ValueError('--benchmark-steps requires a positive value and --skip-save')
    if args.smoke_curriculum_epoch is not None and (
        args.epochs != 1 or args.max_cases is None or args.smoke_curriculum_epoch < 0
    ):
        raise ValueError(
            "--smoke-curriculum-epoch requires --epochs 1, --max-cases, and a nonnegative value"
        )
    if args.stop_after_epoch is not None and (args.max_cases is None or args.stop_after_epoch < 1):
        raise ValueError('--stop-after-epoch requires a limited smoke cohort and positive epoch')
    lr_schedule = LearningRateSchedule(min_ratio=args.lr_min_ratio, epochs=args.lr_schedule_epochs)
    temperature_schedule = TemperatureSchedule(start=args.mask_temperature_start,
        end=args.mask_temperature_end, epochs=args.mask_temperature_epochs)
    base_lrs = dict(main=args.lr_main, vista=args.lr_vista, semantic=args.lr_semantic)
    rank, local_rank, world = distributed_setup()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    manifest_cases = load_cases(args.manifest, "training")
    dataset = WholeCTLIDCDataset(
        manifest_cases,
        target_mask_shape=tuple(args.screen_shape),
        max_cases=args.max_cases,
        anatomy_root=args.anatomy_root,
        cache_root=args.data_cache,
        cache_required=args.data_cache is not None,
    )
    if dataset.cache is not None:
        dataset.cache.validate_all()
    sampler = AnatomyOversamplingSampler(dataset, num_replicas=world, rank=rank,
        seed=args.seed, extra_fraction=args.oversample_fraction, distance_mm=args.near_anatomy_mm)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.data_workers,
        pin_memory=False,
        collate_fn=whole_ct_collate,
        **(dict(prefetch_factor=args.prefetch_factor, persistent_workers=True,
                multiprocessing_context='spawn') if args.data_workers else {}),
    )
    # Only the manifest training cohort is allowed to update the model bank.
    test_cases = load_cases(args.manifest, "testing")
    training_patients = {c['patient_id'] for c in dataset.cases}
    if training_patients & {c['patient_id'] for c in test_cases}:
        raise ValueError('Training/testing patient overlap')
    manifest_payload = json.loads(args.manifest.read_text())
    if manifest_payload.get('tuning_split'):
        from v7.data.split import validate_split
        validate_split(manifest_payload)
    device = torch.device("cuda", local_rank)
    amp_dtype = getattr(torch, args.amp_dtype)
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16; explicitly select another AMP dtype')
    bare_model = WholeCTJointModelV7(
        anatomy_dropout=args.anatomy_dropout,
        fast_kernels=args.fast_kernels,
        window_size=tuple(args.window_size),
        overlap=args.overlap,
        screen_shape=tuple(args.screen_shape),
        roi_shape=tuple(args.roi_shape),
        num_queries=args.num_queries,
        detr_hidden_dim=args.detr_hidden_dim,
        detr_coarse_shape=tuple(args.detr_coarse_shape),
        detr_nheads=args.detr_nheads,
        detr_encoder_layers=args.detr_encoder_layers,
        detr_decoder_layers=args.detr_decoder_layers,
        context_channels=args.context_channels,
        refiner_base_channels=args.refiner_base,
        hard_negatives=args.hard_negatives,
        teacher_full_epochs=args.teacher_full_epochs,
        teacher_zero_epoch=args.teacher_zero_epoch,
        teacher_jitter=args.teacher_jitter,
        minimum_roi_coverage=args.minimum_roi_coverage,
        vista_checkpoint=args.vista_checkpoint,
        semantic_roi_size=args.semantic_roi_size,
        semantic_width=args.semantic_width,
        semantic_min_dice=args.semantic_min_dice,
        inference_object_threshold=args.inference_object_threshold,
        detector_kind=args.detector_kind,
        detector_nms_iou=args.detector_nms_iou,
        native_proposal_threshold=args.native_proposal_threshold,
        native_confidence_residual_bound=args.native_confidence_residual_bound,
        amp_dtype=amp_dtype,
        bank_config=dict(sparsity=args.sparsity, bound=args.coefficient_bound,
                         beam=args.beam, pool_size=args.pool_size, gap=args.gap,
                         fraction=args.sample_fraction, min_samples=args.bank_min_samples,
                         refresh_every=args.bank_refresh_every, mode=args.bank_mode),
    ).to(device)
    # Match the original warmup initialization on resume: the whole refiner
    # call already has activation checkpointing in the joint forward.
    bare_model.refiner.use_checkpoint = False
    bare_model.rashomon.allowed_cases = {f"{c['patient_id']}__scan{int(c['scan_id'])}" for c in dataset.cases}
    criterion = WholeCTCriterionV7(
        segmentation_weight=args.loss_segmentation_weight,
        semantic_weight=args.loss_semantic_weight,
        malignancy_weight=args.loss_malignancy_weight,
        anatomy_weight=args.anatomy_weight, vista_nodule_weight=args.vista_nodule_weight,
        missing_nodule_weight=args.missing_nodule_weight,
        semantic_anchor_weight=args.semantic_anchor_weight,
        object_background_weight=args.object_background_weight,
    ).to(device)

    vista_base_module = getattr(bare_model.segmenter, "model", None)
    vista_parameters = list(vista_base_module.parameters()) if vista_base_module else []
    semantic_parameters = list(bare_model.semantics.parameters())
    special_ids = {id(parameter) for parameter in vista_parameters + semantic_parameters}
    main_parameters = [
        parameter for parameter in bare_model.parameters() if id(parameter) not in special_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_parameters, "lr": args.lr_main, "name": "main"},
            {"params": vista_parameters, "lr": args.lr_vista, "name": "vista"},
            {"params": semantic_parameters, "lr": args.lr_semantic, "name": "semantic"},
        ],
        weight_decay=args.weight_decay,
    )
    if args.amp_init_scale <= 0:
        raise ValueError("--amp-init-scale must be positive")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_dtype == torch.float16,
        init_scale=args.amp_init_scale,
        growth_interval=1000,
    )
    start_epoch = 0
    global_step = 0
    saved_lr_scheduler = None
    resume_summary = None
    saved_wandb = None
    warmup_state = None
    semantic_guard_failures = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("architecture") != ARCHITECTURE:
            raise ValueError(f"Resume architecture mismatch: {saved.get('architecture')}")
        if saved['config'].get('loss_schema') != LOSS_SCHEMA:
            raise ValueError('V7 resume loss schema differs')
        if saved['config'].get('variant') != variant:
            raise ValueError('Cannot resume a different training variant')
        saved_wandb = saved.get('wandb')
        warmup_state = saved.get('warmup')
        if not warmup_state or not warmup_state['complete'] or warmup_state.get('schema') != SCHEMA:
            raise ValueError('Joint resume requires parallel official-mask preparation; use a completed V7 checkpoint')
        if warmup_state.get('smoke_only') and not args.preparation_smoke:
            raise ValueError('Smoke preparation is not a production checkpoint')
        semantic_guard_failures = saved.get('semantic_guard_failures',0)
        saved_policy = saved.get("config", {}).get("malignancy_policy", {}).get("name")
        if saved_policy != POLICY_NAME:
            raise ValueError(f"Resume malignancy policy mismatch: {saved_policy}")
        if saved['lr_schedule'] != lr_schedule.state_dict():
            raise ValueError('Resume LR schedule mismatch')
        if saved['config']['loss_weights'] != criterion.loss_weights:
            raise ValueError('Resume loss weights mismatch')
        validate_resume_arguments(args, saved['config'], world)
        if saved['config']['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise ValueError('Resume manifest contents changed')
        if saved['config']['anatomy_cache_sha256'] != dataset.cache_fingerprint():
            raise ValueError('Resume anatomy labels or sampling statistics changed')
        saved_lr_scheduler = saved['lr_scheduler']
        global_step = saved['global_step']
        bare_model.load_state_dict(saved["model"], strict=True)
        if not math.isclose(float(bare_model.mask_temperature),
                            temperature_schedule(saved['epoch']), rel_tol=1e-6):
            raise ValueError('Saved mask temperature differs from its epoch schedule')
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        allowed = {f"{c['patient_id']}__scan{int(c['scan_id'])}" for c in dataset.cases}
        if bare_model.rashomon.allowed_cases != allowed:
            raise ValueError('Resume training cohort differs from cached model bank')
        if saved['config']['amp_dtype'] != args.amp_dtype:
            raise ValueError('Resume AMP dtype differs from checkpoint')
        rng_states = saved.get('rng_states', [])
        if len(rng_states) != saved['config']['world_size']:
            raise ValueError('Checkpoint random streams do not cover its saved GPU ranks')
        rng_resume = restore_rank_rng(rng_states, rank=rank, world=world, seed=args.seed,
            completed_epochs=start_epoch, device=device, allow_resize=args.allow_world_size_change)
        resume_summary = dict(checkpoint=str(args.resume), completed_epochs=start_epoch,
                              cache_size=len(bare_model.rashomon.memory),
                              bank_size=int(bare_model.rashomon.count),
                              bank_fits=int(bare_model.rashomon.fit_count),
                              optimizer_state_entries=len(optimizer.state),
                              rng=rng_resume, rng_restored=rng_resume['restored'],
                              saved_world_size=saved['config']['world_size'], world_size=world,
                              optimizer_steps=sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s}))
        del saved

    next_lr_epoch = args.smoke_curriculum_epoch if args.smoke_curriculum_epoch is not None else start_epoch
    lr_scheduler = make_lr_scheduler(optimizer, lr_schedule, base_lrs,
                                     next_epoch=next_lr_epoch, saved_state=saved_lr_scheduler)
    run_config = {
        **vars(args),
        "architecture": ARCHITECTURE,
        "variant": variant,
        "preparation_schema": SCHEMA,
        "loss_schema": LOSS_SCHEMA,
        "semantic_normalization": NORMALIZATION,
        "semantic_convolution": CONVOLUTION,
        "ddp_output_handling": "registered WholeCTOutputV7 pytree; supports unused loss branches",
        "diagnostic_schedule": {"kind": "constant", "scale": 1.0, "start_epoch": 1},
        "world_size": world,
        "effective_ct_batch_size": world,
        "performance": dict(prepared_ct_cache=args.data_cache is not None,
                            data_workers=args.data_workers, intensity_volume_loaded=False,
                            fast_kernels=args.fast_kernels),
        "warmup_policy": "V5 hard networks plus supervised anatomy warmup; fresh direct risk bank",
        "anatomy_cache_sha256": dataset.cache_fingerprint(),
        "anatomy_channels": ['lung', 'vessels', 'airway', 'mediastinum'],
        "vista_label_prompt": list(PROMPT_IDS),
        "oversampling": {"base": "every CT once", "extra_fraction": args.oversample_fraction,
                         "near_mm": args.near_anatomy_mm, "balanced_by": "structure and malignancy"},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "lr_schedule": lr_schedule.state_dict(),
        "mask_temperature": args.mask_temperature_start,
        "temperature_schedule": (temperature_schedule.state_dict()
            if args.mask_temperature_start != args.mask_temperature_end else 'constant'),
        "mask_temperature_scope": "final mask probabilities, semantic inputs and radiomics; raw logits for segmentation losses and supervision gate",
        "risk_stages": "single continuous FasterRisk pool on 8 shape radiomics + 6 predicted semantics",
        "risk_feature_schema": RISK_SCHEMA,
        "excluded_intensity_features": INTENSITY_NAMES,
        "risk_offset": None,
        "radiomics_detached": True,
        "runtime": dict(python=sys.version, executable=sys.executable,
                        torch=torch.__version__, cuda=torch.version.cuda,
                        cudnn=torch.backends.cudnn.version(),
                        cudnn_benchmark=torch.backends.cudnn.benchmark,
                        nccl=torch.cuda.nccl.version(),
                        nccl_environment={key: value for key, value in os.environ.items()
                                          if key.startswith('NCCL_')},
                        gpu=torch.cuda.get_device_name(device)),
        "malignancy_policy": dataset.policy_summary,
        "feature_names": FEATURE_NAMES,
        "semantic_names": SEMANTIC_NAMES,
        "radiomics_names": RADIOMICS_NAMES,
        "fasterrisk_version": "0.1.10",
        "bank_fit_source": "latest detached features of training physical nodules; no test data",
        "loss_weights": criterion.loss_weights,
        "loss_blocks": LOSS_BLOCKS,
        "loss_block_multipliers": criterion.block_weights,
        "base_loss_weights": criterion.base_loss_weights,
        "radiomics_definition": "8 fixed 3-D soft-mask geometry descriptors on the 1-mm grid; no intensity radiomics",
        "endpoint": "scan-level reader-derived malignancy surrogate",
        "source_sha256": {
            str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ('_core/common', '_core/model_v4', '_core/model_v5', '_core/model_v6', 'models', 'training', 'data')
            for path in sorted((Path(__file__).parents[1] / directory).glob('*.py'))
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output_dir / "config.json").write_text(
            json.dumps(run_config, default=str, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "event": "start",
                    "architecture": ARCHITECTURE,
                    "manifest_cases": len(manifest_cases),
                    "training_cases": len(dataset),
                    "world_size": world,
                    "runtime": run_config['runtime'],
                    "resume": resume_summary,
                    "malignancy_policy": dataset.policy_summary,
                }
            ),
            flush=True,
        )

    monitor = Monitor(args, run_config, saved_wandb) if rank == 0 else None
    if warmup_state is None:
        warmup_state = initialize_from_warmup(bare_model, args, rank, world)
    anchor_stream = AnchorStream(warmup_state['cache_dir'],args,rank,world)
    if world > 1:
        dist.barrier()
    model: nn.Module = bare_model
    if world > 1:
        model = DistributedDataParallel(bare_model, device_ids=None, output_device=None,
            find_unused_parameters=True, broadcast_buffers=False)

    consecutive_overflows = 0
    for epoch in epoch_indices(start_epoch, args.epochs):
        curriculum_epoch = args.smoke_curriculum_epoch if args.smoke_curriculum_epoch is not None else epoch
        sampler.set_epoch(epoch)
        anchor_stream.set_epoch(epoch)
        model.train()
        bare_model.set_temperature(temperature_schedule(curriculum_epoch))
        phase = set_training_phase(bare_model, curriculum_epoch)
        if phase != 'end_to_end_unfrozen' and vista_base_module is not None:
            vista_base_module.eval()
        diag_scale = diagnostic_scale(curriculum_epoch)
        teacher_p = bare_model.teacher_probability(curriculum_epoch)
        if lr_scheduler.last_epoch != curriculum_epoch:
            raise ValueError('LR scheduler is not at the current curriculum epoch')
        assert_learning_rates(optimizer, base_lrs, lr_schedule(curriculum_epoch))
        epoch_lrs = {g['name']:g['lr'] for g in optimizer.param_groups}
        accumulator = LossAccumulator(device)
        semantic_accumulator = SemanticAccumulator(device)
        sums = torch.zeros(7, device=device, dtype=torch.float64)
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        previous_step_finished = time.monotonic()
        data_wait_seconds = 0.
        for step,batch in enumerate(loader,start=1):
            started = time.monotonic()
            data_seconds = started - previous_step_finished
            data_wait_seconds += data_seconds
            optimizer.zero_grad(set_to_none=True)
            anchor_roi,anchor_histogram = anchor_stream.next(device)
            with torch.autocast('cuda',dtype=amp_dtype):
                output = model(batch['image'],batch=batch,epoch=curriculum_epoch,
                    teacher_probability=teacher_p,compute_diagnostics=diag_scale>0,
                    semantic_anchor_roi=anchor_roi)
                losses = criterion(output,batch,diagnostic_scale=diag_scale,
                                   anchor=(output.anchor_probabilities,anchor_histogram))
            if not torch.isfinite(losses.total):
                raise FloatingPointError(f"Nonfinite loss for {batch['case_id']}")
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            grads = dict(vista=gradient_norm(vista_parameters),detr=gradient_norm(list(bare_model.detr.parameters())),
                refiner=gradient_norm(list(bare_model.refiner.parameters())),semantic=gradient_norm(semantic_parameters),
                vista_class_head=gradient_norm(list(bare_model.segmenter.model.class_head.parameters())))
            total_grad,before_scale,after_scale = finish_scaled_step(optimizer,scaler,bare_model.parameters(),max_norm=5.)
            skipped = after_scale < before_scale
            consecutive_overflows = consecutive_overflows+1 if skipped else 0
            if consecutive_overflows >= 8:
                raise FloatingPointError('Eight consecutive AMP overflows')
            accumulator.update(losses)
            semantic_accumulator.update(output, batch)
            sums += torch.tensor([1, losses.positive_candidates, losses.refined_candidates,
                losses.supervised_candidates, int(losses.risk_valid), output.teacher_forced_count,
                output.fallback_count], device=device, dtype=torch.float64)
            global_step += 1
            if rank == 0:
                # Batch details remain in the local SLURM log, never W&B.
                print(json.dumps(dict(event='step',epoch=epoch+1,step=step,global_step=global_step,
                    temperature=float(bare_model.mask_temperature),
                    case_id=batch['case_id'],phase=phase,total=float(losses.total.detach()),
                    semantic=losses.mean('semantic'),missing_nodule=losses.mean('missing_nodule'),
                    anatomy=losses.mean('anatomy'),vista_nodule=losses.mean('vista_nodule'),
                    nodule=losses.mean('nodule'),risk=losses.mean('risk'),grad_norm=total_grad,
                    gradients=grads,optimizer_step_skipped=skipped,bank_ready=output.bank_ready,
                    bank_size=int(bare_model.rashomon.count),bank_last_fit=bare_model.rashomon.last_fit,
                    teacher_forced=output.teacher_forced_count,fallbacks=output.fallback_count,
                    data_seconds=data_seconds, seconds=time.monotonic()-started)),flush=True)
            if args.smoke_curriculum_epoch is not None and output.bank_ready and losses.supervised_candidates:
                if not np.isfinite(grads['semantic']) or grads['semantic'] <= 0:
                    raise AssertionError('Smoke requires a nonzero finite CNN gradient')
            del output,losses,batch,anchor_roi,anchor_histogram
            previous_step_finished = time.monotonic()
            if args.benchmark_steps is not None and step >= args.benchmark_steps:
                break
        if args.resume and args.benchmark_steps is not None:
            # A checkpoint-resume benchmark never writes an incomplete epoch
            # or logs partial-cohort quality metrics to the production W&B run.
            if world > 1:
                dist.barrier()
            elapsed = time.time()-epoch_start
            if rank == 0:
                result = dict(event='benchmark', epoch=epoch+1, steps=step,
                    ct_samples=step*world, seconds=elapsed, ct_per_hour=step*world*3600/elapsed,
                    data_wait_seconds_rank0=data_wait_seconds,
                    max_gpu_memory_gib=torch.cuda.max_memory_allocated(device)/2**30)
                print(json.dumps(result), flush=True)
                (args.output_dir/'benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
            break
        if diag_scale > 0 and not int(bare_model.rashomon.count):
            raise RuntimeError('No fitted direct joint-feature risk model')
        sums = reduce(sums,world)
        loss_record = accumulator.compute(criterion.loss_weights, diagnostic_scale=diag_scale)
        semantic_audit = audit_semantics(bare_model,args,warmup_state['cache_dir'],
                                        reference=warmup_state['semantic_reference'])
        # Fixed CTs, frozen checkpoint and prediction-only routing make mask
        # quality interpretable alongside changing training loss populations.
        geometry_probe = geometry_audit(bare_model,dataset,args,rank,world)
        semantic_guard_failures = 0 if semantic_audit['passed'] else semantic_guard_failures+1
        # Online moments mix model states; fixed-checkpoint probes decide the
        # regression stop, after the epoch checkpoint has been preserved.
        predicted_semantic_audit = semantic_accumulator.compute(enforce=False)
        peak = torch.tensor(torch.cuda.max_memory_allocated(device),device=device,dtype=torch.float64)
        if world > 1:
            dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        count = float(sums[0])
        lr_scheduler.step()
        assert_learning_rates(optimizer,base_lrs,lr_schedule(curriculum_epoch+1))
        rng = dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),
                   numpy=np.random.get_state(),python=random.getstate())
        rng_states = [None]*world if world>1 else [rng]
        if world>1:
            dist.all_gather_object(rng_states,rng)
        if rank==0:
            record = dict(event='epoch',epoch=epoch+1,phase=phase,global_step=global_step,
                **loss_record, semantic_audit=semantic_audit, predicted_semantic_audit=predicted_semantic_audit,
                geometry_audit=geometry_probe,
                semantic_guard_failures=semantic_guard_failures,
                lr=epoch_lrs,next_lr={g['name']:g['lr'] for g in optimizer.param_groups},
                diagnostic_scale=diag_scale,temperature=float(bare_model.mask_temperature),
                curriculum_epoch=curriculum_epoch,teacher_probability=teacher_p,
                mean_positive_candidates=float(sums[1]/count),mean_refined_candidates=float(sums[2]/count),
                supervised_nodules=int(sums[3]), valid_risk_cases=int(sums[4]),
                semantic_supervised_nodules=loss_record['loss_counts']['semantic'],
                teacher_forced=int(sums[5]), fallbacks=int(sums[6]),training_samples=int(count),seconds=time.time()-epoch_start,
                bank_fits=int(bare_model.rashomon.fit_count),bank_last_fit=bare_model.rashomon.last_fit,
                max_gpu_memory_gib=float(peak)/2**30)
            print(json.dumps(record),flush=True)
            with (args.output_dir/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            metrics = epoch_metrics(record,criterion.loss_weights)
            monitor.log(metrics)
            if not args.skip_save:
                payload = dict(epoch=epoch,global_step=global_step,lr_schedule=lr_schedule.state_dict(),
                    lr_scheduler=lr_scheduler.state_dict(),model=bare_model.state_dict(),optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),config=run_config,architecture=ARCHITECTURE,rng_states=rng_states,
                    warmup=warmup_state,wandb=monitor.state_dict())
                payload['semantic_guard_failures'] = semantic_guard_failures
                atomic_save(payload,args.output_dir/'latest.pt')
                if (epoch+1)%args.save_every==0 or epoch+1==args.epochs:
                    atomic_save(payload,args.output_dir/f'epoch_{epoch+1:03d}.pt')
        if world>1:
            dist.barrier()
        if args.epochs == 0:
            stop = torch.tensor(int((args.output_dir/'STOP_AFTER_EPOCH').exists()) if rank == 0 else 0,
                                device=device)
            if world > 1:
                dist.broadcast(stop, src=0)
            if stop.item():
                if rank == 0:
                    (args.output_dir/'stopped.json').write_text(json.dumps(dict(
                        reason='explicit_stop_request', completed_epoch=epoch+1,
                        checkpoint=str(args.output_dir/'latest.pt'))) + '\n')
                break
        if not args.preparation_smoke and semantic_guard_failures >= args.semantic_guard_patience:
            raise RuntimeError('Semantic regression persisted; checkpoint saved before stopping: '
                               + '; '.join(semantic_audit['failures']))
        if args.benchmark_steps is not None or (args.stop_after_epoch and epoch+1>=args.stop_after_epoch):
            break
    if rank==0:
        monitor.finish()
    if world>1:
        dist.destroy_process_group()


@record
def run_main() -> None:
    """Keep the original worker error even when a library replaces excepthook."""
    try:
        main()
    except SystemExit as exc:
        if exc.code:
            traceback.print_exc(file=sys.__stderr__)
        raise
    except BaseException:
        # Write before W&B/DDP shutdown hooks run; torchrun also records the
        # exception in its per-rank error.json for the parent process.
        traceback.print_exc(file=sys.__stderr__)
        sys.__stderr__.flush()
        raise


if __name__ == '__main__':
    run_main()
