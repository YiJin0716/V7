"""Exact, source-validated CPU batches for V7; no intensity volume is cached."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

SCHEMA = 'v7_prepared_whole_ct_v1'


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_value,
                                    allow_nan=False).encode()).hexdigest()


def _stat(path):
    path = Path(path)
    s = path.stat()
    return [str(path.resolve()), s.st_size, s.st_mtime_ns]


def assert_same_batch(expected, actual, path='batch'):
    """Compare every tensor, label, crop origin and metadata field exactly."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or expected.dtype != actual.dtype:
            raise AssertionError(f'{path}: tensor dtype/type differs')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f'{path}: keys differ')
        for key in expected:
            assert_same_batch(expected[key], actual[key], f'{path}/{key}')
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(f'{path}: sequence differs')
        for i, (a, b) in enumerate(zip(expected, actual)):
            assert_same_batch(a, b, f'{path}/{i}')
    elif expected != actual:
        raise AssertionError(f'{path}: value differs')


class PreparedCTCache:
    def __init__(self, dataset, root, *, required=False):
        if dataset.include_image_hu:
            raise ValueError('The V7 prepared cache excludes intensity volumes')
        self.dataset, self.root, self.required = dataset, Path(root), bool(required)
        base = Path(__file__).parents[1]
        sources = ['_core/common/base_data.py', '_core/common/coarse_data.py',
                   '_core/common/semantic_data.py', 'data/dataset.py', 'data/cache.py',
                   'data/policy.py']
        self.identity = dict(schema=SCHEMA, target_spacing=dataset.target_spacing,
            target_mask_shape=dataset.target_mask_shape, min_votes=dataset.min_votes,
            exclude_indeterminate=dataset.exclude_indeterminate,
            source_sha256={name:hashlib.sha256((base/name).read_bytes()).hexdigest()
                           for name in sources})

    def path(self, index):
        return self.root / (self.dataset.anatomy_metadata[index]['case_id']+'.pt')

    def signature(self, index):
        from v7.data.anatomy import PREPARED_DIRECTORY
        case = self.dataset.cases[index]
        key = (case['patient_id'], str(int(float(case['scan_id']))))
        annotations = self.dataset.annotations.get(key, [])
        files = [Path(case['image']), self.dataset.anatomy_root/PREPARED_DIRECTORY/
                 (self.dataset.anatomy_metadata[index]['case_id']+'.npz')]
        files.extend(self.dataset.official_mask_dir/
                     f'{key[0]}__scan{key[1]}__ann{a}_mask.nii.gz'
                     for n in annotations for a in n['annotation_ids'])
        return _digest(dict(identity=self.identity, case=case, annotations=annotations,
                            anatomy=self.dataset.anatomy_metadata[index],
                            files=[_stat(p) for p in files]))

    def read(self, index, signature=None):
        payload = torch.load(self.path(index), map_location='cpu', weights_only=False, mmap=True)
        if payload.get('schema') != SCHEMA or payload.get('signature') != (signature or self.signature(index)):
            raise ValueError(f'Stale prepared CT cache: {self.path(index)}')
        batch = payload['batch']
        if 'image_hu' in batch or batch['case_id'] != self.dataset.anatomy_metadata[index]['case_id']:
            raise ValueError('Prepared CT identity/feature policy differs')
        return batch

    def build(self, index, loader):
        signature = self.signature(index)
        if self.path(index).exists():
            self.read(index, signature)
            return False
        batch = loader(index)
        if self.signature(index) != signature:
            raise RuntimeError('Source changed while preparing a CT')
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.preparing-', suffix='.pt', dir=self.root)
        os.close(fd)
        try:
            torch.save(dict(schema=SCHEMA, signature=signature, batch=batch), temporary)
            saved = torch.load(temporary, map_location='cpu', weights_only=False, mmap=True)
            assert_same_batch(batch, saved['batch'])
            del saved
            os.replace(temporary, self.path(index))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return True

    def get(self, index, loader):
        if not self.path(index).exists():
            if self.required:
                raise FileNotFoundError(f'Prepare CT cache before training: {self.path(index)}')
            self.build(index, loader)
        return self.read(index)

    def validate_all(self):
        for i in range(len(self.dataset)):
            self.read(i)

