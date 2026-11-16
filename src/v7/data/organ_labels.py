"""Generate pretrained TotalSegmentator labels for train/validation only."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

import nibabel as nib
import numpy as np

from v7.data.anatomy import CHANNELS, SCHEMA, TOTAL_ROIS, case_id, pack_groups

from v7 import paths
DEFAULT_ROOT = paths.ANATOMY
DEFAULT_MANIFEST = paths.MANIFEST


def cohort(manifest):
    payload = json.loads(Path(manifest).read_text())
    patients = {s: {c['patient_id'] for c in payload[s]}
                for s in ('training', 'validation', 'testing')}
    for a, b in (('training', 'testing'), ('validation', 'testing')):
        if patients[a] & patients[b]:
            raise ValueError(f'Patient overlap: {a}/{b}')
    # This workspace's validation list is a subset of training, not a holdout.
    # Keep each CT once and explicitly identify overlapping monitoring cases.
    probes = {case_id(c) for c in payload['validation']}
    train_ids = {case_id(c) for c in payload['training']}
    rows = [dict(c, split='training', training_probe=case_id(c) in probes)
            for c in payload['training']]
    rows += [dict(c, split='validation', training_probe=False)
             for c in payload['validation'] if case_id(c) not in train_ids]
    if len({case_id(c) for c in rows}) != len(rows):
        raise ValueError('Duplicate CT in anatomy cohort')
    return rows


def sequential_prediction(self, *args, **kwargs):
    # Official nnU-Net sequential pre/postprocessing avoids cluster IPC failures.
    kwargs.pop('num_processes_preprocessing', None)
    kwargs.pop('num_processes_segmentation_export', None)
    if kwargs.pop('num_parts', 1) != 1 or kwargs.pop('part_id', 0) != 0:
        raise ValueError('Expected one complete CT')
    return self.predict_from_files_sequential(*args, **kwargs)


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save_image(path, data, source):
    tmp = path.with_name(path.name.replace('.nii.gz', '.tmp.nii.gz'))
    nib.save(nib.Nifti1Image(data, source.affine), tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--case')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard')
    rows = cohort(args.manifest)
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(schema=SCHEMA, manifest=str(args.manifest.resolve()), manifest_sha256=digest,
                channels=CHANNELS, lung_dilation_mm=3., total_rois=TOTAL_ROIS, cases=rows,
                source='public pretrained TotalSegmentator; no local fine-tuning',
                tasks=['total', 'lung_vessels'], fast=False)
    plan_path = args.output / 'cohort.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
            raise ValueError('Anatomy cohort/config changed; use a new output directory')
    elif args.plan_only:
        atomic_json(plan_path, plan)
    else:
        raise ValueError('Run --plan-only once before launching workers')
    if args.plan_only:
        print(json.dumps({'cases': len(rows), 'output': str(args.output)})); return
    rows = [c for c in rows if case_id(c) == args.case] if args.case else rows[args.shard::args.shards]
    if not rows:
        raise ValueError('No cases selected')
    os.environ['TOTALSEG_WEIGHTS_PATH'] = str(Path.home() / '.totalsegmentator/nnunet/results')
    os.environ['nnUNet_compile'] = 'false'
    with tempfile.TemporaryDirectory(prefix='v7-totalseg-', dir='/tmp') as config:
        os.environ['TOTALSEG_HOME_DIR'] = config
        Path(config, 'config.json').write_text(json.dumps(dict(totalseg_id='v7_anatomy',
            send_usage_stats=False, prediction_counter=0)))
        from totalsegmentator.python_api import totalsegmentator
        from totalsegmentator.map_to_binary import class_map
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch
        nnUNetPredictor.predict_from_files = sequential_prediction
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('Request a GPU for label generation')
        version = importlib.metadata.version('TotalSegmentator')
        for i, case in enumerate(rows):
            start = time.monotonic()
            directory = args.output / 'native' / case_id(case)
            directory.mkdir(parents=True, exist_ok=True)
            source = nib.load(case['image'])
            provenance = dict(schema=SCHEMA, case_id=case_id(case), split=case['split'],
                source=case['image'], source_bytes=Path(case['image']).stat().st_size,
                source_mtime_ns=Path(case['image']).stat().st_mtime_ns,
                manifest_sha256=digest, version=version, fast=False,
                channels=CHANNELS, lung_dilation_mm=3.,
                shape=source.shape, affine=source.affine.tolist())
            expected = json.loads(json.dumps(provenance))
            marker = directory / 'complete.json'
            if marker.exists():
                saved = json.loads(marker.read_text())
                if any(saved.get(k) != v for k,v in expected.items()):
                    raise ValueError(f'Cached label provenance differs: {directory}')
                for name in ('total', 'lung_vessels', 'groups'):
                    check = nib.load(directory / f'{name}.nii.gz')
                    if check.shape != source.shape or not np.allclose(check.affine, source.affine, atol=1e-4):
                        raise ValueError(f'Cached label geometry differs: {directory}')
                print(f'REUSE {case_id(case)}', flush=True); continue
            print(f'START {i+1}/{len(rows)} {case_id(case)}', flush=True)
            arrays = {}
            for task in ('total', 'lung_vessels'):
                dest = directory / f'{task}.nii.gz'
                # A case is committed only after both predictions and group masks exist.
                result = totalsegmentator(source, output=None, ml=True, task=task,
                    fast=False, device='gpu', nr_thr_resamp=1, nr_thr_saving=1,
                    roi_subset=list(TOTAL_ROIS) if task == 'total' else None,
                    quiet=True, skip_saving=True)
                if result.shape != source.shape or not np.allclose(result.affine, source.affine, atol=1e-4):
                    raise ValueError(f'TotalSegmentator output grid differs: {case_id(case)}')
                data = np.asarray(result.dataobj, dtype=np.uint8)
                if not set(np.unique(data)) <= {0, *class_map[task]}:
                    raise ValueError('Unexpected anatomy classes')
                save_image(dest, data, source)
                arrays[task] = data
                del result
                gc.collect(); torch.cuda.empty_cache()
            packed = pack_groups(arrays['total'], arrays['lung_vessels'], class_map['total'],
                                 class_map['lung_vessels'], nib.affines.voxel_sizes(source.affine))
            save_image(directory / 'groups.nii.gz', packed, source)
            provenance.update(seconds=time.monotonic()-start,
                              labels={task:class_map[task] for task in arrays},
                              voxel_counts={name:int(((packed >> j)&1).sum()) for j,name in enumerate(CHANNELS)})
            atomic_json(marker, provenance)
            print(json.dumps(dict(event='labels_complete', case=case_id(case),
                                  seconds=provenance['seconds'])), flush=True)
            del arrays, packed
            gc.collect()


if __name__ == '__main__':
    main()
