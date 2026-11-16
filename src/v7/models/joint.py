"""Anatomy-conditioned V7 with shape-only radiomics and semantic risk inputs."""
from dataclasses import dataclass
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils._pytree import register_dataclass

from v7._core.common.base_model import sliding_starts
from v7._core.common.coarse_model import _gaussian_weight
from v7._core.model_v6.model import WholeCTJointModelV6, WholeCTOutputV6
from v7._core.model_v5.native_proposals import native_proposals
from v7.models.vista import VistaAnatomySegmenter, expand_input
from v7.training.loss import patch_objectives
from v7.data.anatomy import PROMPT_IDS, NODULE_INDEX, GROUP_INDICES
from v7.models.radiomics import RADIOMICS_NAMES, ShapeRadiomics3D
from v7.models.risk import ShapeSemanticRiskBank

LEGACY_ARCHITECTURE = 'whole_ct_native_prompt_anatomy_direct_fasterrisk_v7'
ARCHITECTURE = 'whole_ct_native_prompt_anatomy_shape_fasterrisk_v7'


def group_probabilities(probability, *, fast=False):
    """Combine independent predictions; preserve soft probabilities for DETR."""
    if probability.shape[1] != len(PROMPT_IDS):
        raise ValueError('Expected 13 VISTA prompt predictions')
    groups = [probability[:, list(indices)].amax(1, keepdim=True) for indices in GROUP_INDICES]
    # Windows use the 1-mm CT grid: a seven-voxel kernel expands lung context
    # by 3 mm without masking, suppressing or selecting any nodule candidate.
    if fast:
        # A cubic maximum is separable. Z, then Y, then X preserves the
        # original lexicographic winner on tied finite probabilities.
        for kernel, padding in [((1,1,7),(0,0,3)), ((1,7,1),(0,3,0)), ((7,1,1),(3,0,0))]:
            groups[0] = F.max_pool3d(groups[0], kernel, stride=1, padding=padding)
    else:
        groups[0] = F.max_pool3d(groups[0], 7, stride=1, padding=3)
    return torch.cat(groups, dim=1)


@dataclass
class WholeCTOutputV7(WholeCTOutputV6):
    anatomy_probabilities: torch.Tensor | None = None
    vista_auxiliary_losses: torch.Tensor | None = None
    vista_prompt_probabilities: torch.Tensor | None = None


register_dataclass(WholeCTOutputV7)


class WholeCTJointModelV7(WholeCTJointModelV6):
    def __init__(self, *, anatomy_dropout=.2, bank_config=None, fast_kernels=True,
                 legacy_intensity_radiomics=False, **kwargs):
        kwargs.setdefault('detector_kind', 'native_seeded')
        super().__init__(bank_config=bank_config, **kwargs)
        # The legacy path is used only by the evaluator for old checkpoints.
        # New V7 training has no intensity-radiomics option.
        if not legacy_intensity_radiomics:
            self.requires_image_hu = False
            self.radiomics_names = RADIOMICS_NAMES
            self.radiomics = ShapeRadiomics3D()
            self.rashomon = ShapeSemanticRiskBank(**(bank_config or {}))
        if not 0 <= anatomy_dropout < 1:
            raise ValueError('Anatomy dropout must be in [0,1)')
        self.anatomy_dropout = float(anatomy_dropout)
        self.fast_kernels = bool(fast_kernels)
        if not isinstance(self.segmenter, VistaAnatomySegmenter):
            self.segmenter = VistaAnatomySegmenter.from_nodule(self.segmenter)
        self.detr.input_projection[0] = expand_input(self.detr.input_projection[0])
        self.detr.mask_feature_projection[0] = expand_input(self.detr.mask_feature_projection[0])
        self.context_compressor = expand_input(self.context_compressor)
        self._anatomy_targets = None
        self._anatomy_output = None

    def forward(self, image, *, batch=None, **kwargs):
        # This context carries targets only to the loss calculation below.
        # Neither proposals nor any input channel can read these labels.
        self._anatomy_targets = batch.get('anatomy_packed') if self.training and batch is not None else None
        try:
            base = super().forward(image, batch=batch, **kwargs)
            probabilities, auxiliary, prompts = self._anatomy_output
            return WholeCTOutputV7(**vars(base), anatomy_probabilities=probabilities,
                                  vista_auxiliary_losses=auxiliary, vista_prompt_probabilities=prompts)
        finally:
            self._anatomy_targets = None
            self._anatomy_output = None

    def discovery(self, image, *, targets=None):
        """One VISTA pass/window; only predicted anatomy enters DETR."""
        if image.ndim != 4 or image.shape[0] != 1 or image.device.type != 'cpu':
            raise ValueError('Expected CPU CT [1,X,Y,Z]')
        shape, device = tuple(image.shape[-3:]), self.device
        if targets is not None and (tuple(targets.shape) != shape or targets.device.type != 'cpu'):
            raise ValueError('Anatomy targets must share the CPU 1-mm CT grid')
        probability_sum = torch.zeros((1, len(PROMPT_IDS), *self.screen_shape), device=device)
        group_sum = torch.zeros((1, 4, *self.screen_shape), device=device)
        feature_sum = torch.zeros((1, self.vista_feature_dim, *self.screen_shape), device=device)
        weight_sum = torch.zeros((1, 1, *self.screen_shape), device=device)
        native_sum = torch.zeros(shape, device=device)
        native_weight = torch.zeros_like(native_sum)
        auxiliary_sum = torch.zeros(len(PROMPT_IDS), device=device)
        axes = [sliding_starts(n, w, self.overlap) for n, w in zip(shape, self.window_size)]
        windows = 0
        for start in itertools.product(*axes):
            valid = tuple(min(n-s, w) for n, s, w in zip(shape, start, self.window_size))
            lower = tuple(int(np.floor(s*t/n)) for s, t, n in zip(start, self.screen_shape, shape))
            upper = tuple(max(lo+1, int(np.ceil((s+v)*t/n)))
                          for s, v, t, n, lo in zip(start, valid, self.screen_shape, shape, lower))
            block = tuple(hi-lo for hi, lo in zip(upper, lower))
            extent = tuple(slice(s, s+n) for s, n in zip(start, valid))
            patch = self._window(image, start)[None].to(device)
            # Keep packed labels on CPU between window recomputations.
            target = targets[extent][None] if targets is not None else None

            def summarize(value, packed, valid=valid, block=block):
                logits, pyramid = self.segmenter.forward_with_features(value)
                cropped = logits[(..., *(slice(0, n) for n in valid))]
                probability = cropped.float().sigmoid()
                native = probability[:, NODULE_INDEX:NODULE_INDEX+1]
                low = F.interpolate(probability, size=block,
                                    mode='trilinear', align_corners=False)
                groups = F.interpolate(group_probabilities(probability, fast=self.fast_kernels), size=block,
                                       mode='trilinear', align_corners=False)
                maps = []
                for feature in pyramid:
                    size = tuple(max(1, int(np.ceil(v*f/p))) for v, f, p in
                                 zip(valid, feature.shape[-3:], value.shape[-3:]))
                    maps.append(F.interpolate(feature[(..., *(slice(0, n) for n in size))].float(),
                                              size=block, mode='trilinear', align_corners=False))
                loss = patch_objectives(cropped, packed) if packed is not None else low.new_zeros(len(PROMPT_IDS))
                return native.detach(), low, groups, torch.cat(maps, dim=1), loss

            with torch.autocast(device.type, enabled=device.type == 'cuda', dtype=self.amp_dtype):
                if self.training and self.use_checkpoint:
                    native, low, groups, features, loss = checkpoint(summarize, patch, target, use_reentrant=False)
                else:
                    native, low, groups, features, loss = summarize(patch, target)
            dest = (..., *(slice(lo, hi) for lo, hi in zip(lower, upper)))
            weight = _gaussian_weight(block, device, torch.float32)
            probability_sum[dest] += low*weight
            group_sum[dest] += groups*weight
            feature_sum[dest] += features*weight
            weight_sum[dest] += weight
            importance = _gaussian_weight(valid, device, torch.float32).reshape(valid)
            native_sum[extent] += native[0, 0]*importance
            native_weight[extent] += importance
            auxiliary_sum = auxiliary_sum + loss
            windows += 1
        probability = probability_sum/weight_sum.clamp_min(1e-6)
        features = feature_sum/weight_sum.clamp_min(1e-6)
        ct = F.interpolate(image[None].to(device), size=self.screen_shape, mode='trilinear', align_corners=False)
        anatomy = group_sum/weight_sum.clamp_min(1e-6)
        # Hole filling is prediction-only, on DETR's context grid. The outer
        # lung boundary remains soft and retains its gradients.
        from scipy.ndimage import binary_fill_holes
        lung = anatomy[0, 0].detach().cpu().numpy() >= .5
        holes = torch.as_tensor(binary_fill_holes(lung) & ~lung, device=device)[None,None]
        anatomy = torch.cat((torch.where(holes, torch.ones_like(anatomy[:,:1]), anatomy[:,:1]), anatomy[:,1:]), 1)
        auxiliary_input = anatomy
        if self.training and self.anatomy_dropout:
            keep = torch.rand((1, 4, 1, 1, 1), device=device) >= self.anatomy_dropout
            auxiliary_input = auxiliary_input*keep  # Keep retained channels as probabilities.
        nodule = probability[:, NODULE_INDEX:NODULE_INDEX+1]
        screening = torch.cat((ct, nodule, features, auxiliary_input), dim=1)
        native = native_sum/native_weight.clamp_min(1e-6)
        losses = auxiliary_sum/windows if targets is not None else None
        return screening, nodule, native, windows, anatomy, losses, probability

    def detection_inputs(self, image):
        screening, probability, native, windows, anatomy, loss, prompts = self.discovery(image, targets=self._anatomy_targets)
        self._anatomy_output = anatomy, loss, prompts
        proposals = (native_proposals(native, self.screen_shape, threshold=self.native_proposal_threshold,
                                     maximum=self.num_queries, linear_scores=self.fast_kernels)
                     if self.detector_kind == 'native_seeded' else None)
        return screening, probability, (native >= .5).cpu(), windows, proposals

    def discover(self, image):
        screening, probability, native, windows, _, _, _ = self.discovery(image)
        return screening, probability, (native >= .5).cpu(), windows
