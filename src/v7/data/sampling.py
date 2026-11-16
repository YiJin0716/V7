"""Oversample difficult nodules without excluding any ordinary training CT."""
import math
import numpy as np
from torch.utils.data import Sampler
from v7.data.anatomy import CHANNELS


def hard_buckets(metadata, distance_mm=5.):
    """Equal structure/benign-malignant buckets; small nodules get more draws."""
    buckets = {}
    for index, case in enumerate(metadata):
        if case['split'] != 'training':
            raise ValueError('Sampling statistics must be training-only')
        for nodule in case['nodules']:
            if not nodule['retained']:
                continue
            label = int(nodule['malignant'])
            for group in CHANNELS:
                distance = nodule['distances_mm'].get(group)
                if distance is not None and 0 <= distance <= distance_mm:
                    small = nodule['diameter_mm'] <= 6.
                    weight = 4. if small and label else 2. if small else 1.
                    buckets.setdefault((group, label), []).append((index, nodule['nodule_id'], weight))
    return buckets


class AnatomyOversamplingSampler(Sampler):
    """Each epoch visits every CT once, then adds balanced hard-nodule CT draws.

    A selected nodule repeats its entire CT with all original targets intact.
    DDP ranks shard a single deterministic global list and run equal step counts.
    """
    def __init__(self, dataset, *, num_replicas=1, rank=0, seed=42,
                 extra_fraction=.5, distance_mm=5., **unused):
        if not 0 <= rank < num_replicas or extra_fraction < 0 or distance_mm < 0:
            raise ValueError('Invalid anatomy sampler parameters')
        self.dataset, self.rank, self.world = dataset, rank, num_replicas
        self.seed, self.epoch = seed, 0
        self.extra = math.ceil(len(dataset)*extra_fraction)
        self.buckets = hard_buckets(dataset.anatomy_metadata, distance_mm)
        if self.extra and not self.buckets:
            raise ValueError('No nearby training nodules available for requested oversampling')
        self.num_samples = math.ceil((len(dataset)+self.extra)/self.world)
        self.last_draws = []

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        indices = list(rng.permutation(len(self.dataset)))
        keys = sorted(self.buckets)
        self.last_draws = []
        # Cycle through shuffled buckets: vessels cannot dominate merely
        # because they touch more nodules than the other structures.
        for j in range(self.extra):
            if j % len(keys) == 0:
                order = rng.permutation(len(keys))
            key = keys[int(order[j % len(keys)])]
            entries = self.buckets[key]
            weights = np.array([entry[2] for entry in entries], dtype=float)
            index, nodule, _ = entries[int(rng.choice(len(entries), p=weights/weights.sum()))]
            indices.append(index)
            self.last_draws.append(dict(case=index, nodule=nodule, group=key[0], malignant=key[1]))
        rng.shuffle(indices)
        total = self.num_samples*self.world
        indices += (indices*((total-len(indices))//len(indices)+1))[:total-len(indices)]
        return iter(indices[self.rank:total:self.world])
