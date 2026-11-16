"""Prepare exact V7 batches once, on CPUs, before GPU training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import torch

from v7.data.dataset import WholeCTLIDCDataset, load_cases
from v7.data.cache import SCHEMA

_DATASET = None


def _initialize(config, directory):
    global _DATASET
    torch.set_num_threads(1)
    _DATASET = WholeCTLIDCDataset(load_cases(config['manifest'], 'training'),
        target_mask_shape=tuple(config['screen_shape']), anatomy_root=config['anatomy_root'],
        cache_root=directory)


def _prepare(index):
    start = time.monotonic()
    created = _DATASET.cache.build(index, _DATASET.load_uncached)
    return dict(case=_DATASET.anatomy_metadata[index]['case_id'], created=created,
                seconds=time.monotonic()-start, bytes=_DATASET.cache.path(index).stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--max-cases', type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    config = json.loads(args.config.read_text())
    _initialize(config, args.output_dir)
    count = min(len(_DATASET), args.max_cases or len(_DATASET))
    start, rows = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=_initialize, initargs=(config, args.output_dir)) as pool:
        for row in pool.map(_prepare, range(count), chunksize=1):
            rows.append(row)
            print(json.dumps(dict(completed=len(rows), total=count, **row)), flush=True)
    if count == len(_DATASET):
        _DATASET.cache.validate_all()
    summary = dict(schema=SCHEMA, complete=count == len(_DATASET), cases=count,
                   verified_exact_roundtrip=True, seconds=time.monotonic()-start,
                   bytes=sum(r['bytes'] for r in rows), identity=_DATASET.cache.identity)
    (args.output_dir/'complete.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
