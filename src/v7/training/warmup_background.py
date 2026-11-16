"""Select informative background crops without changing any supervision."""
import itertools
import torch
import torch.nn.functional as F
from v7._core.common.base_model import sliding_starts
from v7._core.common.roi import integer_crop
from v7.data.anatomy import NODULE_INDEX, READER_UNION_BIT

POLICY = 'ranked_deployment_background_v1'
COVERAGE_POLICY = 'all_training_cases_with_background_mining_v1'


def coverage(case_ids, visits, *, enabled):
    """Count complete visits under mining, separately from earlier CT visits."""
    expected = set(case_ids)
    if not expected or len(expected) != len(case_ids):
        raise ValueError('Background coverage needs a nonempty unique training cohort')
    if set(visits)-expected or any(type(n) is not int or n < 1 for n in visits.values()):
        raise ValueError('Invalid background-mining CT visit counts')
    missing = len(expected)-len(visits)
    return dict(required=bool(enabled), passed=not enabled or missing == 0,
                visited_cases=len(visits), total_cases=len(expected), missing_cases=missing)


def load_coverage_visits(saved, *, reset=False):
    """Restore only recorded mining visits; never infer them from old CT counts."""
    config = saved['config']
    if reset or not config.get('background_mining', False):
        return {}
    if config.get('background_coverage_policy') != COVERAGE_POLICY:
        raise ValueError('Background-mining resume requires --upgrade-background-coverage')
    visits = saved.get('background_case_visits', {})
    result = coverage(config['anatomy_case_ids'], visits, enabled=True)
    if (saved.get('background_coverage_epoch') != saved['epoch']
            or result != saved.get('background_coverage')):
        raise ValueError('Background-mining coverage must describe the saved epoch')
    return dict(visits)


def require_coverage(saved):
    if not saved['config'].get('background_mining', False):
        return
    visits = load_coverage_visits(saved)
    if not coverage(saved['config']['anatomy_case_ids'], visits, enabled=True)['passed']:
        raise ValueError('Warmup has not visited every training CT with background mining')


def background_score(logits, packed, valid):
    """Mean background BCE; every reader-marked voxel is excluded from ranking."""
    packed = packed.to(logits.device)
    readers = ((packed >> READER_UNION_BIT) & 1).bool()[:, None]
    retained = ((packed >> NODULE_INDEX) & 1).bool()[:, None]
    support = valid.to(logits.device).bool() & ~readers & ~retained
    count = support.sum()
    if not count:
        return None
    values = logits[:, NODULE_INDEX:NODULE_INDEX+1].float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError('Nonfinite nodule logits while ranking background crops')
    return float(F.softplus(values)[support].mean())


def deployment_crop(image, packed, origin):
    center = tuple(s+64 for s in origin)
    ct, actual_origin, valid = integer_crop(image, center)
    target, target_origin, _ = integer_crop(packed[None], center)
    if actual_origin.tolist() != list(origin) or not torch.equal(actual_origin, target_origin):
        raise AssertionError('Background candidate CT and labels differ')
    kind = ('context_deployment_window' if ((target >> READER_UNION_BIT) & 1).any()
            else 'anatomical_negative_deployment_window')
    return ct[None], target, valid[None], kind


class BackgroundCropMiner:
    """Rank deployment windows once per CT, then offer distinct difficult windows.

    Ranking uses the weights at construction. Each proposed window and the
    original anatomical crop are rescored with the current weights at selection.
    No labels, model parameters, gradient buffers or train/eval modes are changed.
    """
    def __init__(self, model, image, packed):
        self.model, self.image, self.packed = model, image, packed
        self.device = next(model.parameters()).device
        windows = itertools.product(*(sliding_starts(n, 128, .25) for n in packed.shape))
        self.remaining = []
        self.ranked_windows = 0
        training = model.training
        try:
            model.eval()
            with torch.no_grad():
                for origin in windows:
                    score = self._score(deployment_crop(image, packed, origin))
                    self.ranked_windows += 1
                    if score is not None:
                        self.remaining.append((score, origin))
        finally:
            model.train(training)
        self.remaining.sort(key=lambda row: row[0])

    def _score(self, crop):
        ct, target, valid, _ = crop
        with torch.autocast(self.device.type, enabled=self.device.type == 'cuda', dtype=torch.bfloat16):
            logits, _ = self.model.forward_with_features(ct.to(self.device))
        return background_score(logits, target, valid)

    def choose(self, initial):
        if not self.remaining:
            return initial, dict(candidates=1, deployment_selected=False,
                initial_score=None, selected_score=None, selected_origin=None)
        _, origin = self.remaining.pop()
        alternative = deployment_crop(self.image, self.packed, origin)
        training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                initial_score = self._score(initial)
                candidate_score = self._score(alternative)
        finally:
            self.model.train(training)
        select = candidate_score is not None and (initial_score is None or candidate_score > initial_score)
        return (alternative if select else initial), dict(candidates=2,
            deployment_selected=select, initial_score=initial_score,
            selected_score=candidate_score if select else initial_score,
            selected_origin=list(origin) if select else None, proposed_origin=list(origin))
