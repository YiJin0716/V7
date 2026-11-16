"""Portable paths; set these environment variables before importing V7."""
import os
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DATA_ROOT = Path(os.environ.get('V7_DATA_ROOT', 'data')).expanduser().resolve()
WEIGHTS_ROOT = Path(os.environ.get('V7_WEIGHTS_ROOT', 'checkpoints')).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get('V7_OUTPUT_ROOT', 'outputs')).expanduser().resolve()
MANIFEST = Path(os.environ.get('V7_MANIFEST', str(DATA_ROOT/'fold_0.json')))
ANNOTATIONS = Path(os.environ.get('V7_ANNOTATIONS', str(DATA_ROOT/'all_ct_annotations.csv')))
IDENTITIES = Path(os.environ.get('V7_IDENTITIES', str(DATA_ROOT/'nodule_iden.csv')))
MASKS = Path(os.environ.get('V7_MASKS', str(DATA_ROOT/'reader_masks')))
ANATOMY = Path(os.environ.get('V7_ANATOMY_ROOT', str(DATA_ROOT/'anatomy')))
VISTA = Path(os.environ.get('V7_VISTA_CHECKPOINT', str(WEIGHTS_ROOT/'vista_nodule.pt')))
PUBLIC_VISTA = Path(os.environ.get('V7_PUBLIC_VISTA_CHECKPOINT', str(WEIGHTS_ROOT/'vista_public.pt')))
MEDICALNET = Path(os.environ.get('V7_MEDICALNET_CHECKPOINT', str(WEIGHTS_ROOT/'medicalnet_resnet50.pth')))
COMPONENT_WARMUP = Path(os.environ.get('V7_COMPONENT_WARMUP', str(OUTPUT_ROOT/'component_warmup/warmup.pt')))
