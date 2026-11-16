# Data

Use CT volumes in Hounsfield units and reader masks on exactly the same native voxel grid and affine as their CT. V7 converts both to canonical XYZ orientation and resamples to an approximately 1 mm grid. Do not rescale HU values before loading.

Expected layout:

```text
data/
  images/LIDC-IDRI-0259__scan259.nii.gz
  all_ct_annotations.csv
  nodule_iden.csv
  reader_masks/LIDC-IDRI-0259__scan259__ann0_mask.nii.gz
  fold_0.json
  anatomy/                 # generated for training
```

The mask filename is illustrative: the actual annotation index must match the CSV. Scan and annotation indices are project identifiers; when rebuilding from DICOM/XML, preserve or reconstruct the mapping to the source series and reader annotations.

`all_ct_annotations.csv` contains one row per reader annotation, with `patient_id`, `scan_index`, `annotation_index`, `lobulation`, `margin`, `sphericity`, `spiculation`, `subtlety`, `texture`, and `malignancy`. Keep a row with an empty annotation index for scans with no reader nodules. Ratings are integers from 1 to 5.

`nodule_iden.csv` contains the same identifiers and ratings plus `nodule_id`, which groups annotations of the same physical nodule. Annotation indices must join uniquely between the two tables. Nodule IDs are local to each CT. Reader masks are named `{patient_id}__scan{scan_index}__ann{annotation_index}_mask.nii.gz`.

The released split lists 702 training CTs and 178 held-out test CTs. Its 70 `validation` CTs are a subset of training used for monitoring, **not an independent validation set**. Use a separate training-only split for hyperparameter selection. The test patient set is disjoint from training.

```bash
python scripts/prepare_manifest.py --images-dir data/images --output data/fold_0.json
```

This writes absolute image paths for the local installation. Raw CTs, annotation tables and masks are not distributed here. Obtain LIDC-IDRI and its annotations from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/).

To use existing data without copying it, set these variables **before** starting Python:

```bash
export V7_MANIFEST=/path/to/fold_0.json
export V7_ANNOTATIONS=/path/to/all_ct_annotations.csv
export V7_IDENTITIES=/path/to/nodule_iden.csv
export V7_MASKS=/path/to/reader_masks
export V7_ANATOMY_ROOT=/path/to/anatomy
```

Pass matching `--manifest` and `--anatomy-root` options when using the example JSON configurations, which otherwise specify relative defaults. Additional path defaults are listed in `src/v7/paths.py`.
