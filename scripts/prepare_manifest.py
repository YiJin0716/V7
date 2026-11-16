"""Attach local NIfTI paths to the released case split."""
import argparse
import json
from pathlib import Path


def build_manifest(splits, images_dir):
    images_dir = Path(images_dir).expanduser().resolve()
    manifest = {}
    for split in ('training', 'validation', 'testing'):
        rows = []
        for key in splits[split]:
            patient, scan = key.rsplit('__scan', 1)
            image = images_dir / (key + '.nii.gz')
            if not image.is_file():
                raise FileNotFoundError(image)
            rows.append(dict(patient_id=patient, scan_id=str(int(scan)), image=str(image)))
        if len(set(splits[split])) != len(rows):
            raise ValueError(f'Duplicate CT in {split}')
        manifest[split] = rows
    train = {r['patient_id'] for r in manifest['training']}
    test = {r['patient_id'] for r in manifest['testing']}
    if train & test:
        raise ValueError('Training/test patient overlap')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, default=Path('splits/lidc_fold0.json'))
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/fold_0.json'))
    args = parser.parse_args()
    manifest = build_manifest(json.loads(args.splits.read_text()), args.images_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2)+'\n')
    print({k: len(v) for k, v in manifest.items()})


if __name__ == '__main__':
    main()
