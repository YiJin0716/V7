# V7: whole-CT lung nodule analysis

Official code repository for our lung nodule analysis project. V7 takes a whole CT and predicts nodule masks, six semantic features, nodule malignancy risk, and scan risk.

The model combines VISTA3D, DETR, a shared mask refiner, a semantic extractor, and a sparse FasterRisk model. VISTA3D predicts organs and nodules through separate class prompts. Four anatomy maps guide detection. Risk uses eight mask-shape features and six predicted semantic features; intensity radiomics are excluded.

**[View the 18 missed malignant nodules →](missing_malignant/README.md)** These nodules occur in nine test CTs. Download an HTML and open it in a browser to inspect the model outputs and reader annotations. No installation is needed to view the files.

## Repository layout

```text
src/v7/models/          V7 architecture and risk model
src/v7/training/        VISTA warmup and joint training
src/v7/data/            CT loading, anatomy labels and sampling
src/v7/inference/       Checkpoint loading and whole-CT inference
src/v7/evaluation/      Detection, segmentation, semantic and risk evaluation
src/v7/visualization/   CT-to-HTML export
src/v7/_core/          Shared components and component warmup
src/vista3d/            Vendored VISTA3D backbone
src/roivue/             Vendored CT viewer
scripts/               Training, inference and evaluation entry points
configs/               Warmup and joint-training configurations
splits/                LIDC-IDRI case split
tests/                 Model, data and evaluation tests
missing_malignant/      Nine self-contained CT HTMLs and an 18-nodule table
docs/                  Data, training, evaluation and checkpoint instructions
```

The internal `_core` modules retain their original version names so existing checkpoints and component warmup remain compatible. They are included in this repository; no earlier project checkout is needed.

## Installation

Use Python 3.11 and a CUDA-enabled PyTorch installation for training or inference.

```bash
git clone https://github.com/YiJin0716/V7.git
cd V7
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The core implementation was checked with PyTorch 2.8.0, MONAI 1.3.2 and FasterRisk 0.1.10. Choose the PyTorch CUDA build appropriate for your GPU. HTML generation additionally requires:

```bash
python -m pip install -e '.[visualization]'
```

## Run

Prepare the files described in [data.md](docs/data.md), and place the trained checkpoint at `checkpoints/v7_epoch18.pt`. Large checkpoints and the complete dataset are not included; see [checkpoints.md](docs/checkpoints.md).

```bash
# Predict one held-out CT; annotations are attached after inference.
python scripts/infer.py --checkpoint checkpoints/v7_epoch18.pt \
  --manifest data/fold_0.json --case LIDC-IDRI-0259__scan259 \
  --output outputs/example

# Create an offline HTML viewer.
python scripts/export_html.py --result-dir outputs/example \
  --output outputs/example.html --model-label V7

# Evaluate the held-out cohort and create plots.
python scripts/evaluate.py --checkpoint checkpoints/v7_epoch18.pt \
  --manifest data/fold_0.json --output outputs/evaluation
python scripts/plot_results.py --output outputs/evaluation

# Joint training after completed warmup.
torchrun --standalone --nproc_per_node=1 scripts/train.py --config configs/joint.json
```

[Training instructions](docs/training.md) cover anatomy preparation and independent component warmup. [Evaluation instructions](docs/evaluation.md) define the matching, labels and output folders. Run commands from the repository root; JSON configurations provide defaults and subsequent CLI options override them.

## Tests

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -t .
```

The tests use small synthetic inputs and require neither CT data nor pretrained weights. Whole-CT inference needs substantially more memory; the release check used an NVIDIA RTX A6000 and 64 GB host memory.

## Reference model

The HTML examples use the original V7, run **13220013, epoch 18**. They do not use V7_tuned. [reference_run.json](configs/reference_run.json) records the checkpoint hash, loss weights and feature order. Malignancy targets are LIDC reader ratings; they are not pathology diagnoses.

## Acknowledgments

This project uses [VISTA3D](https://github.com/Project-MONAI/VISTA), [MONAI](https://github.com/Project-MONAI/MONAI), [FasterRisk](https://github.com/jiachangliu/FasterRisk), [roivue](https://gitlab.oit.duke.edu/lam135/roivue), and [NiiVue](https://github.com/niivue/niivue). The separation of model code, training scripts, evaluation and external weights follows the general organization of research repositories such as [Sybil](https://github.com/reginabarzilaygroup/Sybil) and [DeepLung](https://github.com/uci-cbcl/DeepLung). See [third-party notices](THIRD_PARTY.md).
