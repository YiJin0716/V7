"""Retain V6 objectives and add native-resolution VISTA supervision."""
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from v7._core.model_v6.loss import WholeCTCriterionV6
from v7._core.model_v5.loss import LossAccumulator as V5Accumulator
from v7.data.anatomy import PROMPT_IDS, NODULE_INDEX, READER_UNION_BIT

LOSS_SCHEMA = 'v7_native_prompt_supervision_v3'
AUXILIARY = ('anatomy', 'vista_nodule')
LOSS_BLOCKS = {
    'segmentation': ('object', 'box_l1', 'box_giou', 'coarse_dice', 'coarse_focal',
                     'fine_dice', 'fine_focal_positive', 'fine_focal_negative',
                     'missing_nodule', 'anatomy', 'vista_nodule'),
    'semantic': ('semantic', 'semantic_anchor'),
    'malignancy': ('nodule', 'risk'),
}


def binary_segmentation_losses(logits, target, valid=None):
    """Voxel-mean BCE plus squared-denominator Dice for each native prompt.

    This follows the successful LIDC VISTA fine-tuning objective. Averaging
    positive and negative BCE separately can amplify each tiny-nodule voxel
    thousands of times and favors oversegmentation.

    Empty-target patches use background BCE alone. They still train the model
    to reject false positives instead of contributing a constant Dice penalty.
    """
    logits, target = logits.float(), target.float()
    valid = torch.ones_like(target) if valid is None else valid.float().expand_as(target)
    dims = (0, 2, 3, 4)
    positive = target * valid
    npos = positive.sum(dims)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    terms = (bce*valid).sum(dims)/valid.sum(dims).clamp_min(1)
    probability = logits.sigmoid()
    dice = 1 - 2*(probability*positive).sum(dims)/((probability.square()*valid).sum(dims)+npos+1e-5)
    return terms + torch.where(npos > 0, dice, torch.zeros_like(dice))


def patch_objectives(logits, packed, valid=None):
    """One loss per native prompt; the last prompt is nodule class 23."""
    if logits.shape[1] != len(PROMPT_IDS):
        raise ValueError('V7 supervision requires all 13 native VISTA predictions')
    packed = packed.to(device=logits.device, dtype=torch.int16)
    anatomy = torch.stack([((packed >> i)&1) for i in range(NODULE_INDEX)], dim=1)
    nodule = ((packed >> NODULE_INDEX)&1)[:, None]
    allowed = (~((packed >> READER_UNION_BIT)&1).bool() | nodule[:, 0].bool())[:, None]
    if valid is not None:
        allowed = allowed & valid.bool()
    organs = binary_segmentation_losses(logits[:, :NODULE_INDEX], anatomy, valid)
    nodules = binary_segmentation_losses(logits[:, NODULE_INDEX:NODULE_INDEX+1], nodule, allowed)
    return torch.cat((organs, nodules))


class WholeCTCriterionV7(WholeCTCriterionV6):
    def __init__(self, *, anatomy_weight=.1, vista_nodule_weight=1.,
                 segmentation_weight=1., semantic_weight=1., malignancy_weight=1., **kwargs):
        super().__init__(**kwargs)
        if any(not math.isfinite(v) or v <= 0 for v in (anatomy_weight, vista_nodule_weight)):
            raise ValueError('Anatomy and nodule retention supervision must be positive')
        self.loss_weights.update(anatomy=float(anatomy_weight), vista_nodule=float(vista_nodule_weight))
        self.base_loss_weights = dict(self.loss_weights)
        self.block_weights = dict(segmentation=segmentation_weight, semantic=semantic_weight,
                                  malignancy=malignancy_weight)
        if any(not math.isfinite(v) or v <= 0 for v in self.block_weights.values()):
            raise ValueError('Loss block multipliers must be finite and positive')
        if set(self.loss_weights) != {term for terms in LOSS_BLOCKS.values() for term in terms}:
            raise ValueError('Every V7 loss must belong to exactly one block')
        for block, terms in LOSS_BLOCKS.items():
            for term in terms:
                self.loss_weights[term] *= float(self.block_weights[block])

    def forward(self, output, batch, **kwargs):
        losses = super().forward(output, batch, **kwargs)
        if output.vista_auxiliary_losses is None:
            raise ValueError('V7 training requires anatomy labels; missing labels are never zero loss')
        values = dict(anatomy=output.vista_auxiliary_losses[:NODULE_INDEX].mean(),
                      vista_nodule=output.vista_auxiliary_losses[NODULE_INDEX])
        # Each whole CT is one supervised sample for these two objectives.
        # Every training rank has labels, so DDP's ordinary mean is correct.
        for name, value in values.items():
            losses.total = losses.total + self.loss_weights[name]*value
            losses.numerators[name] = value
            losses.counts[name] = 1
            losses.normalizers[name] = 1.
        return losses


class LossAccumulator(V5Accumulator):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.auxiliary = torch.zeros((2, 2), device=device, dtype=torch.float64)

    def update(self, losses):
        super().update(losses)
        for i, name in enumerate(AUXILIARY):
            self.auxiliary[i, 0] += losses.numerators[name].detach().double()
            self.auxiliary[i, 1] += losses.counts[name]

    def compute(self, weights, *, synchronize=True, process_group=None, **kwargs):
        result = super().compute(weights, synchronize=synchronize, process_group=process_group, **kwargs)
        values = self.auxiliary.clone()
        if synchronize and dist.is_initialized():
            dist.all_reduce(values, group=process_group)
        for i, name in enumerate(AUXILIARY):
            count = int(values[i, 1])
            result[name] = float(values[i, 0]/count) if count else None
            for key, value in (('loss_counts', count), ('loss_normalizers', float(count)),
                               ('loss_numerators', float(values[i, 0]))):
                result[key][name] = value
            for key in ('total', 'active_total'):
                if not count:
                    result[key] = None
                elif result[key] is not None:
                    result[key] += weights[name]*result[name]
        return result
