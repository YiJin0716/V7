"""NIfTI (nibabel) and MetaImage (SimpleITK, extra `roivue[mha]`) files to a Case.
dtype="auto" keeps integer data with an integer scaling as integers (CT stays
exact int16) and reads the rest as float32; meta records what was read.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any, cast

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image

from .case import Case

NIFTI = (".nii", ".nii.gz")
METAIMAGE = (".mha", ".mhd")
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
DTYPES = ("auto", np.int16, np.int32, np.float32, np.float64)


def _kind(path: pathlib.Path) -> str:
    name = path.name.lower()
    if name.endswith(NIFTI):
        return "NIfTI"
    if name.endswith(METAIMAGE):
        return "MetaImage"
    raise ValueError(
        f"{path.name}: not NIfTI ({', '.join(NIFTI)}) or MetaImage ({', '.join(METAIMAGE)})"
    )


def _check_dtype(dtype: Any) -> None:
    if not (isinstance(dtype, str) and dtype == "auto") and np.dtype(dtype) not in {
        np.dtype(d) for d in DTYPES[1:]
    }:
        raise ValueError(
            f'dtype must be "auto", np.int16, np.int32, np.float32 or np.float64, got {dtype!r}'
        )


def _cast(arr: np.ndarray, dtype: Any) -> np.ndarray:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer) and np.issubdtype(arr.dtype, np.floating):
        info = np.iinfo(dt)
        return np.clip(np.round(arr), info.min, info.max).astype(dt)
    return arr.astype(dt, copy=False)


def _smallest_int(lo: int, hi: int, raw: np.dtype) -> np.dtype:
    for dt in (raw, np.dtype(np.int16), np.dtype(np.int32)):
        if np.issubdtype(dt, np.integer) and np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max:
            return dt
    return np.dtype(np.int64)


def _read_nifti(path: pathlib.Path, dtype: Any) -> tuple[np.ndarray, np.ndarray]:
    img = cast(Nifti1Image, nib.load(str(path)))
    affine = np.asarray(img.affine, float)
    if isinstance(dtype, str):
        proxy = img.dataobj
        raw = np.asarray(proxy.get_unscaled())  # type: ignore[attr-defined]
        slope = float(proxy.slope) if np.isfinite(proxy.slope) else 1.0  # type: ignore[attr-defined]
        inter = float(proxy.inter) if np.isfinite(proxy.inter) else 0.0  # type: ignore[attr-defined]
        if np.issubdtype(raw.dtype, np.integer) and slope == 1 and inter == int(inter):
            lo, hi = int(raw.min()) + int(inter), int(raw.max()) + int(inter)
            dt = _smallest_int(lo, hi, raw.dtype)
            arr = raw.astype(dt)
            if inter:
                arr += dt.type(inter)
            return arr, affine
        return img.get_fdata(dtype=np.float32), affine
    dt = np.dtype(dtype)
    arr = img.get_fdata(dtype=np.float64 if dt == np.float64 else np.float32)
    return _cast(arr, dt), affine


def _read_metaimage(path: pathlib.Path, dtype: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError("reading .mha/.mhd needs SimpleITK: install roivue[mha]") from e
    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(
            f"{path.name}: expected a 3D scalar image, got "
            f"{img.GetDimension()}D with {img.GetNumberOfComponentsPerPixel()} "
            "components"
        )
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)  # (z, y, x) -> (x, y, z)
    lps = np.eye(4)
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    lps[:3, :3] = direction @ np.diag(img.GetSpacing())
    lps[:3, 3] = img.GetOrigin()
    affine = LPS_TO_RAS @ lps  # ITK physical space is LPS; NIfTI/NiiVue world is RAS
    if isinstance(dtype, str):
        if np.issubdtype(arr.dtype, np.integer):
            return np.ascontiguousarray(arr), affine
        return arr.astype(np.float32), affine
    return _cast(arr, np.dtype(dtype)), affine


def read_image(path: str | pathlib.Path, dtype: Any = "auto") -> tuple[np.ndarray, np.ndarray]:
    """(array in (x, y, z), RAS affine in mm)."""
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such file {path}")
    _check_dtype(dtype)
    if _kind(path) == "NIfTI":
        return _read_nifti(path, dtype)
    return _read_metaimage(path, dtype)


def _read_mask(obj: Any) -> Any:
    """Files become (array, affine) so Case checks their grid; others pass through."""
    if isinstance(obj, (str, pathlib.Path)):
        arr, affine = read_image(obj)
        return arr, affine
    return obj


def from_file(
    image: str | pathlib.Path,
    masks: Mapping[str, Any] | None = None,
    *,
    labels: Any = None,
    label_names: Mapping[int, str] | None = None,
    regions: Any = None,
    meta: dict | None = None,
    display: dict | None = None,
    dtype: Any = "auto",
) -> Case:
    """A Case from an image file (NIfTI or MetaImage). masks: {name: file or array};
    labels: a label-map file or array; regions: a region-map file or array (see
    Case.from_arrays). meta= is merged over what the reader records."""
    path = pathlib.Path(image).expanduser()
    arr, affine = read_image(path, dtype)
    if masks is not None and not isinstance(masks, Mapping):
        raise TypeError("masks must be {name: mask}; pass a label map as labels=")
    recorded = {
        "source_file": path.name,
        "format": _kind(path),
        "image_dtype": str(arr.dtype),
        "dtype_from": "auto" if isinstance(dtype, str) else "argument",
    }
    read = lambda x: _read_mask(x) if x is not None else None
    return Case.from_arrays(
        arr,
        {name: _read_mask(m) for name, m in (masks or {}).items()},
        labels=read(labels),
        label_names=label_names,
        affine=affine,
        regions=read(regions),
        meta={**recorded, **(meta or {})},
        display=display,
    )


def from_nifti(image: str | pathlib.Path, masks: Any = None, **kw: Any) -> Case:
    """from_file for a NIfTI image (.nii, .nii.gz)."""
    if _kind(pathlib.Path(image)) != "NIfTI":
        raise ValueError(f"{pathlib.Path(image).name}: not a NIfTI file; use from_file")
    return from_file(image, masks, **kw)


def from_mha(image: str | pathlib.Path, masks: Any = None, **kw: Any) -> Case:
    """from_file for a MetaImage (.mha, or .mhd with its data file); needs roivue[mha]."""
    if _kind(pathlib.Path(image)) != "MetaImage":
        raise ValueError(f"{pathlib.Path(image).name}: not a MetaImage file; use from_file")
    return from_file(image, masks, **kw)
