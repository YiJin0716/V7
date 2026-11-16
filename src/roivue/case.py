"""The hand-off format: one 3D image and named boolean masks on its voxel grid,
plus an optional region label map, provenance (meta) and display hints (window,
hidden, colors, radiological). Construction validates the geometry and raises
GeometryError listing every problem at once.
"""

from __future__ import annotations

import dataclasses
import pathlib
import warnings
from collections.abc import Mapping
from typing import Any

import nibabel as nib
import numpy as np

from .records import Record

AFFINE_TOL_MM = 1e-3
DISPLAY_KEYS = {"window", "hidden", "colors", "radiological"}


class GeometryError(ValueError):
    """Image, masks and regions do not share one voxel grid."""


def read_volume(obj: Any, what: str) -> tuple[np.ndarray, np.ndarray | None]:
    """ndarray, (ndarray, affine), nibabel image or NIfTI path -> (3D array,
    affine or None)."""
    if isinstance(obj, (str, pathlib.Path)):
        path = pathlib.Path(obj).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{what}: no such file {path}")
        obj = nib.load(str(path))
    if isinstance(obj, tuple) and len(obj) == 2:
        arr, aff = obj[0], np.asarray(obj[1], float)
    elif hasattr(obj, "affine") and hasattr(obj, "dataobj"):
        arr, aff = np.asanyarray(obj.dataobj), np.asarray(obj.affine, float)
    else:
        arr, aff = obj, None
    arr = np.asarray(arr)
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{what}: expected a 3D volume, got shape {arr.shape}")
    if aff is not None and not _valid_affine(aff):
        raise GeometryError(f"{what}: affine must be a finite, invertible 4x4, got\n{aff}")
    return arr, aff


def as_mask(arr: np.ndarray, what: str) -> np.ndarray:
    """Bool mask from bool/int (nonzero = inside) or 0/1 float data."""
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return arr != 0
    if np.issubdtype(arr.dtype, np.floating) and ((arr == 0) | (arr == 1)).all():
        return arr == 1
    raise ValueError(
        f"{what}: values other than 0/1 (a probability map?); threshold it first, "
        f"e.g. mask > 0.5 (dtype {arr.dtype})"
    )


def _valid_affine(aff) -> bool:
    return aff.shape == (4, 4) and np.isfinite(aff).all() and abs(np.linalg.det(aff[:3, :3])) > 1e-9


@dataclasses.dataclass
class Case:
    image: np.ndarray  # 3D; voxel order (x, y, z)
    affine: np.ndarray  # 4x4 voxel -> world (mm)
    masks: dict[str, np.ndarray]  # name -> bool array on the image grid
    regions: np.ndarray | None = None  # integer label map on the image grid
    meta: dict = dataclasses.field(default_factory=dict)
    display: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_file(cls, image: Any, masks: Any = None, **kw: Any) -> Case:
        """From a NIfTI or MetaImage file; see roivue.io.from_file."""
        from .io import from_file

        return from_file(image, masks, **kw)

    @classmethod
    def from_nifti(cls, image: Any, masks: Any = None, **kw: Any) -> Case:
        """From a NIfTI file (.nii, .nii.gz); see roivue.io.from_file."""
        from .io import from_nifti

        return from_nifti(image, masks, **kw)

    @classmethod
    def from_mha(cls, image: Any, masks: Any = None, **kw: Any) -> Case:
        """From a MetaImage file (.mha, .mhd); needs roivue[mha]. See roivue.io."""
        from .io import from_mha

        return from_mha(image, masks, **kw)

    @classmethod
    def from_arrays(
        cls,
        image: Any,
        masks: Mapping[str, Any] | None = None,
        *,
        labels: Any = None,
        label_names: Mapping[int, str] | None = None,
        affine: Any = None,
        regions: Any = None,
        meta: dict | None = None,
        display: dict | None = None,
    ) -> Case:
        """Builds a Case from arrays, nibabel images or paths (see read_volume).

        masks    {name: mask}.
        labels   an integer label map, split into one mask per value, named by
                 label_names ({value: name}) or "label <value>".
        affine   required for a bare image array; conflicting with the image's own raises.
        regions  an integer region map; without it each mask is one region.
        """
        img, img_aff = read_volume(image, "image")
        if affine is not None:
            _, affine = read_volume((img, affine), "affine")
            if img_aff is not None and np.abs(img_aff - affine).max() > AFFINE_TOL_MM:
                raise GeometryError("affine= conflicts with the image's own affine; pass one")
            img_aff = affine
        if img_aff is None:
            raise GeometryError("image has no affine: pass affine= (voxel to RAS mm)")
        if np.issubdtype(img.dtype, np.floating) and not np.isfinite(img).all():
            raise ValueError("image contains NaN or inf; replace them before building a Case")

        if masks is not None and not isinstance(masks, Mapping):
            raise TypeError("masks must be {name: mask}; pass a label map as labels=")
        out, affines = {}, {}
        for name, raw in (masks or {}).items():
            arr, affines[str(name)] = read_volume(raw, f"mask {name!r}")
            out[str(name)] = as_mask(arr, f"mask {name!r}") if arr.shape == img.shape else arr
        if labels is not None:
            arr, affines["labels"] = read_volume(labels, "labels")
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"labels must be an integer label map, got {arr.dtype}")
            for v in np.unique(arr[arr != 0]):
                name = (label_names or {}).get(int(v), f"label {int(v)}")
                if name in out:
                    raise ValueError(f"label name {name!r} is also a mask name")
                out[name], affines[name] = arr == v, affines["labels"]
        reg = None
        if regions is not None:
            reg, affines["regions"] = read_volume(regions, "regions")
        return cls(img, img_aff, out, reg, dict(meta or {}), dict(display or {})).validate(affines)

    def validate(self, affines: Mapping[str, np.ndarray | None] | None = None) -> Case:
        """Checks the Case; `affines` are the affines inputs carried, to compare
        with the image's. Raises GeometryError (grid) or ValueError (hints)."""
        img, aff = self.image, np.asarray(self.affine, float)
        if img.ndim != 3:
            raise ValueError(f"image must be 3D, got shape {img.shape}")
        if not _valid_affine(aff):
            raise GeometryError(f"affine must be a finite, invertible 4x4, got\n{aff}")
        problems = []
        items: list[tuple[str, str, np.ndarray, type]] = [
            (f"mask {n!r}", n, m, bool) for n, m in self.masks.items()
        ]
        if self.regions is not None:
            items.append(("regions", "regions", self.regions, np.integer))
        for label, key, arr, dtype in items:
            other = (affines or {}).get(key)
            if arr.shape != img.shape:
                hint = (
                    " Same sizes in a different order: axes look transposed."
                    if sorted(arr.shape) == sorted(img.shape)
                    else ""
                )
                problems.append(
                    f"{label}: shape {arr.shape} != image {img.shape}.{hint} "
                    "Masks must be on the image's voxel grid; resample first."
                )
            elif other is not None and np.abs(other - aff).max() > AFFINE_TOL_MM:
                problems.append(
                    f"{label}: affine differs from the image's by up to "
                    f"{np.abs(other - aff).max():.3g} mm (tolerance {AFFINE_TOL_MM} mm)."
                    f"\n  image:\n{aff}\n  {label}:\n{other}"
                )
            elif not (arr.dtype == dtype if dtype is bool else np.issubdtype(arr.dtype, dtype)):
                problems.append(
                    f"{label}: dtype {arr.dtype}, must be {'bool' if dtype is bool else 'integer'}"
                )
            elif dtype is bool and not arr.any():
                warnings.warn(f"{label} is empty", stacklevel=2)
        if problems:
            raise GeometryError(
                f"{len(problems)} input(s) not on the image grid "
                f"(image shape {img.shape}):\n- " + "\n- ".join(problems)
            )
        bad = []
        unknown = set(self.display) - DISPLAY_KEYS
        if unknown:
            bad.append(f"unknown keys {sorted(unknown)}; allowed {sorted(DISPLAY_KEYS)}")
        for key in ("hidden", "colors"):
            missing = set(self.display.get(key, ())) - set(self.masks)
            if missing:
                bad.append(f"{key} names not in masks: {sorted(missing)}")
        if bad:
            raise ValueError("invalid display hints:\n- " + "\n- ".join(bad))
        self.affine = aff
        return self

    def facts(self) -> Record:
        """Shape, dtype, spacing, mask sizes, and meta: what the reprs show."""
        return Record(
            shape=self.image.shape,
            dtype=str(self.image.dtype),
            spacing_mm=tuple(
                round(float(x), 3) for x in np.linalg.norm(self.affine[:3, :3], axis=0)
            ),
            mask_voxels={n: int(m.sum()) for n, m in self.masks.items()},
            **({"regions": len(np.unique(self.regions)) - 1} if self.regions is not None else {}),
            **self.meta,
        )

    def __repr__(self) -> str:
        return repr(self.facts())

    def _repr_html_(self) -> str:
        return self.facts()._repr_html_()
