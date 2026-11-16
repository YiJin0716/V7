# Training

V7 trains the anatomy/nodule VISTA model, warms up DETR, the refiner and semantic extractor independently, and then trains the joint model. All targets and risk-bank fitting come from training cases. W&B is disabled by default; enable it with `--wandb-mode online` and your own account settings.

## 1. Prepare anatomy targets

Install TotalSegmentator in a separate environment using its [official instructions](https://github.com/wasserth/TotalSegmentator). Install this repository there with `pip install -e . --no-deps` if its existing dependencies are already present. This step uses the public `total` and `lung_vessels` models. It generates training labels; inference uses VISTA's own organ predictions.

```bash
python -m v7.data.organ_labels --manifest data/fold_0.json --output data/anatomy --plan-only
python -m v7.data.organ_labels --manifest data/fold_0.json --output data/anatomy
python -m v7.data.prepare --root data/anatomy
```

The generation and preparation commands support `--shard` and `--shards` for separate workers. Prepared masks contain 13 VISTA prompts in this order: `28, 29, 30, 31, 32, 57, 132, 119, 115, 6, 125, 11, 23`. The final prompt is the nodule task. The organ loss weight is 0.1 and the nodule loss weight is 1.

## 2. VISTA warmup

Place the public VISTA checkpoint at `checkpoints/vista_public.pt` and run:

```bash
python scripts/warmup_vista.py --config configs/warmup_vista.json
python scripts/export_vista_weights.py \
  --checkpoint outputs/vista_warmup/complete.pt \
  --output checkpoints/vista_nodule.pt
```

This fresh-training recipe starts from public VISTA weights. Completion depends on the readiness checks and training-cohort coverage; reaching the maximum epoch count does not by itself mean that `complete.pt` is available. Inspect the saved audits before extending a run with `--resume`.

## 3. Independent component warmup

```bash
torchrun --standalone --nproc_per_node=3 scripts/warmup_components.py \
  --config configs/warmup_components.json
```

With three GPUs, DETR, the refiner and the semantic extractor run concurrently on separate GPUs. Additional ranks are assigned to DETR. The warmup freezes VISTA and uses official targets for each component. The internal checkpoint architecture is named V5 because V7 reuses these same component networks and expands their anatomy inputs when joint training begins.

Warmup produces `outputs/component_warmup/warmup.pt` and its adjacent `config.json`. Preserve both. Network settings, manifest, and VISTA initialization must agree with joint training. The supplied component and joint configs use matching network settings.

## 4. Joint training

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py --config configs/joint.json
```

Increase `--nproc_per_node` to the number of available GPUs. Each rank processes one CT per synchronized update. The reference run used eight GPUs; changing this changes the effective batch size. The joint stage transfers the completed component networks, adds anatomy channels with initially zero weights, and fits the V7 risk model on training features.

Initial learning rates are `1e-4` for detection/refinement, `3e-5` for semantics and `1e-6` for the trainable VISTA parameters. The cosine learning-rate schedule spans 18 joint epochs. Mask temperature goes from 1 to 0.1 over the same interval. The loss weights and original run metadata are in `configs/reference_run.json`.

Resume a run using the same data and configuration:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --config configs/joint.json --resume outputs/v7/latest.pt
```

The reference checkpoint reused previously completed component warmup and then organ warmup. The fresh recipe above uses the supported public-only initialization path, which trains VISTA first. It preserves the architecture and supplied joint hyperparameters, but does not promise identical weights or metrics to the reference run. The current warmup implementation also includes additional readiness and coverage checks added after that run.
