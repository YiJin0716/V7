"""Native VISTA class prompts and the four derived anatomy inputs to DETR."""
import numpy as np
from scipy import ndimage

CHANNELS = ('lung', 'vessels', 'airway', 'mediastinum')
SCHEMA = 'v7_totalsegmentator_four_groups_v1'
PROMPT_SCHEMA = 'v7_vista_native_prompts_v2'
PREPARED_DIRECTORY = 'prepared_prompts'
PROMPT_IDS = (28, 29, 30, 31, 32, 57, 132, 119, 115, 6, 125, 11, 23)
PROMPT_NAMES = ('left_upper_lobe', 'left_lower_lobe', 'right_upper_lobe',
                'right_middle_lobe', 'right_lower_lobe', 'trachea', 'airway',
                'pulmonary_vein', 'heart', 'aorta', 'superior_vena_cava', 'esophagus', 'nodule')
NODULE_INDEX = 12
READER_UNION_BIT = 13
GROUP_INDICES = ((0,1,2,3,4), (7,9,10), (5,6), (8,9,10,11))
LOBES = ('lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
         'lung_middle_lobe_right', 'lung_lower_lobe_right')
VESSELS = ('aorta', 'pulmonary_vein', 'superior_vena_cava')
MEDIASTINUM = ('heart', 'aorta', 'superior_vena_cava', 'esophagus')
TOTAL_ROIS = tuple(dict.fromkeys((*LOBES, *VESSELS, 'trachea', *MEDIASTINUM)))


def select(labels, mapping, names):
    lookup = {name: int(key) for key, name in mapping.items()}
    missing = set(names) - lookup.keys()
    if missing:
        raise ValueError(f'Missing TotalSegmentator classes: {sorted(missing)}')
    return np.isin(labels, [lookup[name] for name in names])


def pack_groups(total, pulmonary, total_map, pulmonary_map, spacing, dilation_mm=3.):
    """Pack four overlapping binary targets into bits 0..3, never argmax them.

    Lung holes are filled before physical dilation so nodules are not cut out.
    The lung is context, never a candidate exclusion mask.
    """
    if total.shape != pulmonary.shape or len(total.shape) != 3:
        raise ValueError('Anatomy label grids differ')
    if len(spacing) != 3 or min(spacing) <= 0 or dilation_mm < 0:
        raise ValueError('Invalid physical spacing/dilation')
    lung = ndimage.binary_fill_holes(select(total, total_map, LOBES))
    if not lung.any():
        raise ValueError('Empty lung pseudo-label; inspect this CT')
    if dilation_mm:
        lung = ndimage.distance_transform_edt(~lung, sampling=spacing) <= dilation_mm
    packed = lung.astype(np.uint8)
    vessels = select(total, total_map, VESSELS) | select(
        pulmonary, pulmonary_map, ('lung_arteries', 'lung_veins'))
    packed |= vessels.astype(np.uint8) << 1
    airway = select(total, total_map, ('trachea',)) | select(
        pulmonary, pulmonary_map, ('lung_airways', 'lung_airways_wall'))
    packed |= airway.astype(np.uint8) << 2
    packed |= select(total, total_map, MEDIASTINUM).astype(np.uint8) << 3
    return packed


def unpack_groups(packed):
    return ((np.asarray(packed)[None] >> np.arange(4, dtype=np.uint8)[:, None, None, None]) & 1)


def pack_prompt_targets(total, pulmonary, total_map, pulmonary_map):
    """Bits 0..11 follow the VISTA class vector; keep overlapping prompts.

    TotalSegmentator IDs belong to a different label space. In particular,
    native VISTA 119 maps to TotalSegmentator's pulmonary_vein, not its arteries.
    VISTA 132 receives the trachea + lung airway lumen, without airway walls.
    """
    if total.shape != pulmonary.shape:
        raise ValueError('Anatomy grids differ')
    names = (*LOBES, 'trachea', None, 'pulmonary_vein', 'heart', 'aorta',
             'superior_vena_cava', 'esophagus')
    packed = np.zeros(total.shape, dtype=np.uint16)
    for i, name in enumerate(names):
        mask = (select(total, total_map, ('trachea',)) |
                select(pulmonary, pulmonary_map, ('lung_airways',))) if name is None else select(total, total_map, (name,))
        packed |= mask.astype(np.uint16) << i
    return packed


def grouped_targets(packed):
    """Logical organ unions, for sampling only; individual bits remain targets."""
    return np.stack([np.bitwise_and(packed, sum(1 << i for i in group)) != 0
                     for group in GROUP_INDICES])


def case_id(case):
    return f"{case['patient_id']}__scan{int(case['scan_id'])}"
