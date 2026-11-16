"""Resample anatomy labels once and measure training-nodule proximity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import nibabel as nib
import numpy as np
from scipy import ndimage

from v7._core.common.base_data import DEFAULT_ANNOTATIONS, DEFAULT_OFFICIAL_MASK_DIR
from v7._core.common.base_data import load_nodule_annotations, _read_official_mask
from v7.data.policy import is_indeterminate_malignancy, binary_malignancy_target
from v7.data.anatomy import (SCHEMA, PROMPT_SCHEMA, PREPARED_DIRECTORY, PROMPT_IDS,
    NODULE_INDEX, READER_UNION_BIT, LOBES, CHANNELS, case_id, select,
    pack_prompt_targets, grouped_targets)
from v7.data.organ_labels import DEFAULT_ROOT, atomic_json


def wait_for_labels(marker, job):
    """Stream committed labels while the corresponding GPU shard is alive."""
    completed_checks = 0
    while not marker.exists():
        try:
            active = subprocess.run(['squeue','--noheader','--jobs',job,'--format=%T'],
                                    capture_output=True, text=True, timeout=30, check=True).stdout.strip()
            if active:
                completed_checks = 0
            else:
                states = subprocess.run(['sacct','--noheader','--allocations','--jobs',job,'--format=State'],
                    capture_output=True, text=True, timeout=30, check=True).stdout.split()
                if states and any(s not in ('COMPLETED', 'RUNNING', 'PENDING', 'COMPLETING') for s in states):
                    raise RuntimeError(f'Label job {job} terminated with {states}; missing {marker.parent.name}')
                completed_checks += 1
                if completed_checks >= 3:
                    raise RuntimeError(f'Label job {job} is no longer active; missing {marker}')
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            # A scheduler observation failure is not proof that a GPU job died.
            print(f'Scheduler status unavailable for {job}: {exc}', flush=True)
        print(f'WAIT {marker.parent.name} label_job={job}', flush=True)
        time.sleep(20)


def prepare_case(case, root, annotations, mask_dir, *, overwrite=False):
    name = case_id(case)
    native_dir = root/'native'/name
    info = json.loads((native_dir/'complete.json').read_text())
    output = root/PREPARED_DIRECTORY/name
    annotation_hash = hashlib.sha256(DEFAULT_ANNOTATIONS.read_bytes()).hexdigest()
    if not overwrite and output.with_suffix('.json').exists() and output.with_suffix('.npz').exists():
        cached = json.loads(output.with_suffix('.json').read_text())
        if (cached['schema'] == PROMPT_SCHEMA and cached['manifest_sha256'] == info['manifest_sha256']
                and cached['source_bytes'] == info['source_bytes']
                and cached['source_mtime_ns'] == info['source_mtime_ns']
                and cached['annotations_sha256'] == annotation_hash
                and cached['labels_sha256'] == hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest()):
            return cached
        raise ValueError(f'Prepared cache changed: {name}; inspect then use --overwrite')
    source = nib.as_closest_canonical(nib.load(case['image']))
    total = nib.as_closest_canonical(nib.load(native_dir/'total.nii.gz'))
    pulmonary = nib.as_closest_canonical(nib.load(native_dir/'lung_vessels.nii.gz'))
    for label in (total, pulmonary):
        if source.shape != label.shape or not np.allclose(source.affine, label.affine):
            raise ValueError(f'Anatomy geometry mismatch: {name}')
    spacing = nib.affines.voxel_sizes(source.affine)
    packed_native = pack_prompt_targets(np.asarray(total.dataobj), np.asarray(pulmonary.dataobj),
                                       info['labels']['total'], info['labels']['lung_vessels'])
    packed = ndimage.zoom(packed_native, spacing, order=0, mode='nearest', prefilter=False)
    lung = ndimage.binary_fill_holes((packed_native & 31) != 0)
    surface = lung & ~ndimage.binary_erosion(lung)
    key = case['patient_id'], str(int(case['scan_id']))
    nodules = []
    union_all = np.zeros(source.shape, dtype=bool)
    # Compact native unions are enough to query the distance transforms below.
    compact = []
    for row in annotations.get(key, []):
        votes = np.zeros(source.shape, dtype=np.uint8)
        for aid in row['annotation_ids']:
            path = mask_dir/f"{case['patient_id']}__scan{int(case['scan_id'])}__ann{aid}_mask.nii.gz"
            votes += _read_official_mask(path, source)
        union = votes > 0
        union_all |= union
        pos = np.argwhere(union)
        if not len(pos):
            continue
        retained = not is_indeterminate_malignancy(float(row['malignancy']))
        consensus = votes >= min(2, len(row['annotation_ids']))
        if retained:
            packed |= ndimage.zoom(consensus.astype(np.uint16), spacing, order=0,
                                   mode='nearest', prefilter=False) << NODULE_INDEX
        lo, hi = pos.min(0), pos.max(0)+1
        extent = tuple(slice(a,b) for a,b in zip(lo,hi))
        compact.append((extent, union[extent].copy()))
        volume = float(consensus.sum()*np.prod(spacing))
        nodules.append(dict(nodule_id=int(row['nodule_id']), retained=retained,
            malignancy_mean=float(row['malignancy']),
            malignant=int(binary_malignancy_target(float(row['malignancy']))) if retained else None,
            diameter_mm=float((6*volume/np.pi)**(1/3)),
            center_mm_grid=((pos.min(0)+pos.max(0))*.5*spacing).tolist(),
            reader_count=len(row['annotation_ids']), distances_mm={}))
    packed |= ndimage.zoom(union_all.astype(np.uint16), spacing, order=0,
                           mode='nearest', prefilter=False) << READER_UNION_BIT
    # One full distance map at a time bounds CPU memory. Distances are minimum
    # native voxel-center distances from any reader's lesion to each structure.
    groups = grouped_targets(packed_native)
    for index, group in enumerate(CHANNELS):
        mask = surface if group == 'lung' else groups[index]
        if not mask.any():
            for nodule in nodules:
                nodule['distances_mm'][group] = None
            continue
        distances = ndimage.distance_transform_edt(~mask, sampling=spacing)
        for nodule, (extent, union) in zip(nodules, compact):
            nodule['distances_mm'][group] = float(distances[extent][union].min())
        del distances
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, packed=packed)
    tmp.replace(output.with_suffix('.npz'))
    metadata = dict(schema=PROMPT_SCHEMA, native_schema=SCHEMA, prompt_ids=PROMPT_IDS,
        case_id=name, source=case['image'],
        source_bytes=info['source_bytes'], source_mtime_ns=info['source_mtime_ns'],
        manifest_sha256=info['manifest_sha256'], label_version=info['version'],
        split=case['split'], training_probe=case.get('training_probe', False),
        shape=list(packed.shape), spacing_mm=1., native_spacing_mm=spacing.tolist(),
        channels=CHANNELS, nodules=nodules,
        distance_definition='minimum native voxel-center distance from any-reader union; lung uses filled-lobe surface',
        labels_sha256=hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest(),
        annotations_sha256=annotation_hash)
    atomic_json(output.with_suffix('.json'), metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--case')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--wait-for-label-job', help='Stream labels from this matching Slurm array job')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard')
    plan = json.loads((args.root/'cohort.json').read_text())
    candidates = plan['cases'][:args.limit] if args.limit else plan['cases']
    rows = [c for c in candidates if case_id(c) == args.case] if args.case else candidates[args.shard::args.shards]
    if not rows:
        raise ValueError('No cases selected')
    annotations = load_nodule_annotations(DEFAULT_ANNOTATIONS)
    for case in rows:
        if args.wait_for_label_job:
            job = f'{args.wait_for_label_job}_{args.shard}'
            wait_for_labels(args.root/'native'/case_id(case)/'complete.json', job)
        start = time.monotonic()
        result = prepare_case(case, args.root, annotations, DEFAULT_OFFICIAL_MASK_DIR, overwrite=args.overwrite)
        print(json.dumps(dict(event='prepared',case=result['case_id'],
            nodules=len(result['nodules']),seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
