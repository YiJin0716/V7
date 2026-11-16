"""Native VISTA class prompts, sharing one image encoding and original class head."""
import torch
from torch import nn
from v7._core.common.base_model import VistaNoduleSegmenter
from v7.data.anatomy import PROMPT_IDS


class VistaAnatomySegmenter(VistaNoduleSegmenter):
    def __init__(self, model, prompt_class=23):
        super().__init__(model, prompt_class)
        if prompt_class != 23:
            raise ValueError('V7 nodule prompt must be class 23')
        self.register_buffer('label_prompt', torch.tensor(PROMPT_IDS, dtype=torch.long)[:, None])

    @classmethod
    def from_nodule(cls, old):
        result = cls(old.model, old.prompt_class)
        missing = result.load_state_dict(old.state_dict(), strict=False)
        if missing.unexpected_keys or any(k != 'label_prompt' for k in missing.missing_keys):
            raise ValueError(f'VISTA transfer mismatch: {missing}')
        return result

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        value = state_dict.get(prefix+'label_prompt')
        expected = torch.tensor(PROMPT_IDS, dtype=torch.long)[:, None]
        if value is not None and not torch.equal(value.cpu(), expected):
            error_msgs.append(f'{prefix}label_prompt differs from the V7 class order')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward_with_features(self, image):
        if image.shape[0] != 1:
            raise ValueError('VISTA windows have batch size one')
        encoder = self.model.image_encoder
        if encoder.preprocess is not None:
            image = encoder.preprocess(image)
        if not encoder.is_valid_shape(image):
            raise ValueError(f'Invalid VISTA window shape {tuple(image.shape)}')
        skip = encoder.encoder(image)
        skip.reverse()
        decoded = skip.pop(0)
        pyramid = []
        for index, level in enumerate(encoder.up_layers_auto):
            decoded = level['blocks'](level['upsample'](decoded) + skip[index])
            pyramid.append(self.pyramid_projections[index](decoded))
        final_auto = level['head'](decoded)
        masks, _ = self.model.class_head(final_auto, self.label_prompt)
        # Upstream class_head returns [prompts, images, X,Y,Z]. No argmax,
        # hard posttransform, or newly initialized segmentation head is used.
        if masks.shape[:2] != (len(PROMPT_IDS), 1) or masks.ndim != 5:
            raise ValueError(f'Unexpected native VISTA class output: {tuple(masks.shape)}')
        return masks.transpose(0, 1), tuple(pyramid)


def expand_input(layer, extra=4):
    """Preserve the learned function exactly; new anatomy weights start at zero."""
    if not isinstance(layer, nn.Conv3d) or layer.groups != 1:
        raise ValueError('Expected an ungrouped Conv3d input layer')
    result = nn.Conv3d(layer.in_channels + extra, layer.out_channels, layer.kernel_size,
                      stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                      bias=layer.bias is not None, padding_mode=layer.padding_mode,
                      device=layer.weight.device, dtype=layer.weight.dtype)
    with torch.no_grad():
        result.weight.zero_()
        result.weight[:, :layer.in_channels].copy_(layer.weight)
        if layer.bias is not None:
            result.bias.copy_(layer.bias)
    return result


def set_vista_trainable(segmenter, *, decoder=False, full_auto=False):
    """Select native VISTA training scope; the unused point branch stays frozen."""
    for parameter in segmenter.model.parameters():
        parameter.requires_grad_(False)
    for parameter in segmenter.model.class_head.parameters():
        parameter.requires_grad_(True)
    if full_auto:
        for module in (segmenter.model.image_encoder.encoder,
                       segmenter.model.image_encoder.up_layers_auto):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif decoder:
        for parameter in segmenter.model.image_encoder.up_layers_auto[-1].parameters():
            parameter.requires_grad_(True)
