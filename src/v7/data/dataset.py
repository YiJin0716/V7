"""Training-only anatomy cache and whole-CT loading on the established 1-mm grid."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from v7._core.common.semantic_data import WholeCTLIDCDataset as V6Dataset
from v7._core.common.semantic_data import DEFAULT_MANIFEST, load_cases, whole_ct_collate
from v7.data.anatomy import CHANNELS, SCHEMA, PROMPT_SCHEMA, PREPARED_DIRECTORY, PROMPT_IDS, case_id
from v7.data.organ_labels import DEFAULT_ROOT


class WholeCTLIDCDataset(V6Dataset):
    def __init__(self, *args, anatomy_root=DEFAULT_ROOT, cache_root=None,
                 cache_required=False, include_image_hu=False, **kwargs):
        super().__init__(*args, include_image_hu=include_image_hu, **kwargs)
        self.anatomy_root = Path(anatomy_root)
        self.anatomy_metadata = []
        plan = json.loads((self.anatomy_root/'cohort.json').read_text())
        if plan['schema'] != SCHEMA or tuple(plan['channels']) != CHANNELS:
            raise ValueError('Anatomy label schema/channel order mismatch')
        allowed = {case_id(c):c for c in plan['cases'] if c['split'] == 'training'}
        for case in self.cases:
            name = case_id(case)
            if name not in allowed or allowed[name]['image'] != case['image']:
                raise ValueError(f'Case is not in the anatomy training cohort: {name}')
            path = self.anatomy_root/PREPARED_DIRECTORY/name
            metadata = json.loads(path.with_suffix('.json').read_text())
            if (metadata['schema'] != PROMPT_SCHEMA or metadata['source'] != case['image']
                    or tuple(metadata.get('prompt_ids', ())) != PROMPT_IDS):
                raise ValueError(f'Anatomy cache mismatch: {name}')
            if metadata['manifest_sha256'] != plan['manifest_sha256']:
                raise ValueError(f'Anatomy manifest mismatch: {name}')
            if not path.with_suffix('.npz').is_file():
                raise FileNotFoundError(path.with_suffix('.npz'))
            self.anatomy_metadata.append(metadata)
        self.cache = None
        if cache_root is not None:
            from v7.data.cache import PreparedCTCache
            self.cache = PreparedCTCache(self, cache_root, required=cache_required)
        elif cache_required:
            raise ValueError('A prepared CT cache directory is required')

    def __getitem__(self, index):
        if self.cache is not None:
            return self.cache.get(index, self.load_uncached)
        return self.load_uncached(index)

    def load_uncached(self, index):
        batch = super().__getitem__(index)
        metadata = self.anatomy_metadata[index]
        with np.load(self.anatomy_root/PREPARED_DIRECTORY/f"{batch['case_id']}.npz") as saved:
            packed = saved['packed']
        if packed.dtype != np.uint16 or tuple(packed.shape) != batch['isotropic_shape'] or packed.max() >= 16384:
            raise ValueError(f"CT/anatomy grid mismatch: {batch['case_id']}")
        source = Path(self.cases[index]['image']).stat()
        if (source.st_size, source.st_mtime_ns) != (metadata['source_bytes'], metadata['source_mtime_ns']):
            raise ValueError(f"CT source changed: {batch['case_id']}")
        # Torch bitwise operations support signed int16 on both CPU and CUDA.
        batch['anatomy_packed'] = torch.from_numpy(packed.view(np.int16))
        batch['anatomy_metadata'] = metadata
        return batch

    def cache_fingerprint(self):
        """Checkpoint identity includes labels and sampling statistics, not just their path."""
        digest = hashlib.sha256()
        for metadata in self.anatomy_metadata:
            digest.update(json.dumps(metadata, sort_keys=True).encode())
        return digest.hexdigest()
