"""Create an immutable patient split; preserve the original test partition exactly."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from v7 import paths
ROOT = paths.PROJECT_ROOT
RESERVED = ('LIDC-IDRI-0875', 'LIDC-IDRI-0127', 'LIDC-IDRI-0865', 'LIDC-IDRI-0244')


def case_id(case):
    return f"{case['patient_id']}__scan{int(case['scan_id'])}"


def validate_split(payload, original=None):
    groups = {}
    for split in ('training', 'validation', 'testing'):
        rows = payload[split]
        if not rows or len({case_id(c) for c in rows}) != len(rows):
            raise ValueError(f'Empty or duplicate {split} cases')
        groups[split] = {c['patient_id'] for c in rows}
    for a, b in [('training', 'validation'), ('training', 'testing'), ('validation', 'testing')]:
        if groups[a] & groups[b]:
            raise ValueError(f'Patient leakage: {a}/{b}')
    if original is not None:
        if payload['testing'] != original['testing']:
            raise ValueError('The original test partition was modified')
        expected = {case_id(c): c for c in original['training']}
        actual = {case_id(c): c for s in ('training', 'validation') for c in payload[s]}
        if expected != actual:
            raise ValueError('Training + validation must equal original training exactly')
    return {s: dict(cts=len(payload[s]), patients=len(p)) for s, p in groups.items()}


def make_split(original, patient_labels, *, seed=20261102, fraction=.2, reserved=RESERVED):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must lie in (0, 1)')
    patients = {c['patient_id'] for c in original['training']}
    if patients & {c['patient_id'] for c in original['testing']}:
        raise ValueError('Original training/testing patients overlap')
    strata = defaultdict(list)
    for patient in sorted(patients-set(reserved)):
        strata[patient_labels[patient]].append(patient)
    rng, validation = random.Random(seed), set()
    for label in sorted(strata):
        pool = strata[label]
        rng.shuffle(pool)
        validation.update(pool[:max(1, round(len(pool)*fraction))])
    result = dict(description='V7 tuning: patient-disjoint train/validation; original testing unchanged.',
                  training=[c for c in original['training'] if c['patient_id'] not in validation],
                  validation=[c for c in original['training'] if c['patient_id'] in validation],
                  testing=original['testing'], tuning_split=dict(seed=seed, validation_fraction=fraction,
                    stratification='Patient has a nodule with mean reader malignancy > 3',
                    reserved_training_patients=sorted(set(reserved)&patients),
                    reservation_reason='Existing V7 whole-CT warmup regression cases stay in training'))
    result['tuning_split']['counts'] = validate_split(result, original)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=paths.MANIFEST)
    parser.add_argument('--anatomy-root', type=Path, default=paths.ANATOMY)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.original.read_text())
    labels = defaultdict(int)
    for c in original['training']:
        metadata = json.loads((args.anatomy_root/'prepared_prompts'/f'{case_id(c)}.json').read_text())
        labels[c['patient_id']] |= any(n['malignancy_mean'] > 3 for n in metadata['nodules'])
    result = make_split(original, labels)
    result['tuning_split'].update(original_manifest=str(args.original.resolve()),
        original_manifest_sha256=hashlib.sha256(args.original.read_bytes()).hexdigest())
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError('Refusing to change an existing tuning split')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['tuning_split'], indent=2))


if __name__ == '__main__':
    main()
