"""NiiVue viewers for a Case: load() finds the regions and applies display
choices; View.scan() and View.region(i) build the widgets from Scenes, the
render description export.py shares.
"""

from __future__ import annotations

import dataclasses
import gzip
import io
import itertools
import pathlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import ipyniivue
import ipywidgets as W
import nibabel as nib
import numpy as np
from ipyniivue import Mesh, MultiplanarType, NiiVue, ShowRender, SliceType, Volume
from scipy import ndimage
from skimage import measure

from .case import Case, GeometryError  # noqa: F401  (re-exported for callers)
from .records import Record, Records

# NiiVue colormap -> CSS colour for swatches and surfaces (in assignment order).
PALETTE = {
    "red": "#e33",
    "blue": "#39f",
    "green": "#3c3",
    "violet": "#b5f",
    "cool": "#3ee",
    "warm": "#fa3",
}

TILES = (("axial", "coronal"), ("sagittal", "render"))

_FULL = W.Layout(width="100%")
_SLICE = {
    "axial": SliceType.AXIAL,
    "coronal": SliceType.CORONAL,
    "sagittal": SliceType.SAGITTAL,
    "render": SliceType.RENDER,
}


@dataclasses.dataclass
class Layer:
    """One mask as shown in a scene."""

    name: str
    cmap: str
    on: bool
    overlay: bytes  # uint8 NIfTI (.nii.gz)
    mesh: bytes | None = None  # OBJ text; region scenes only


@dataclasses.dataclass
class Scene:
    """Everything a renderer needs for one view, no widgets: built by
    View.scan_scene / region_scene, used by the widgets here and by export."""

    kind: str  # "scan" | "region"
    title: str
    image: bytes  # NIfTI (.nii.gz), window in its header
    affine: np.ndarray
    shape: tuple[int, ...]
    layers: list[Layer]
    center: np.ndarray | None = None  # voxel the crosshair starts on (region)
    index: int | None = None
    step: int = 1  # scan scenes: downsampling factor per axis
    info: Record = dataclasses.field(default_factory=Record)  # shown by the "info" button


@dataclasses.dataclass
class View:
    """Display state for one Case. Build with load()."""

    image: np.ndarray
    affine: np.ndarray
    masks: dict[str, np.ndarray]
    colors: dict[str, str]
    hidden: frozenset[str]
    window: tuple[float, float]
    radiological: bool
    labels: np.ndarray | None  # the case's region label map; None: each mask is a region
    scan_max_voxels: int | None = None  # see load()
    meta: dict = dataclasses.field(default_factory=dict)
    settings: dict = dataclasses.field(default_factory=dict)  # load-time choices, for "info"
    regions: Records = dataclasses.field(default_factory=Records)
    _bytes: dict = dataclasses.field(default_factory=dict, repr=False)

    def _region(self, i: int) -> dict:
        if not self.regions:
            raise ValueError("no regions to show: every mask is empty")
        if not -len(self.regions) <= i < len(self.regions):
            raise IndexError(
                f"region {i} out of range; this case has "
                f"{len(self.regions)} (0 to {len(self.regions) - 1})"
            )
        return self.regions[i]

    def scan_scene(self, step: int | None = None) -> Scene:
        """The whole volume with every mask as an overlay, downsampled by `step`
        per axis (image: mean of each step^3 block; masks: set if any voxel of
        the block is, so small regions stay visible). None: the smallest step
        that keeps the view within scan_max_voxels (see load()); 1: full
        resolution. Close-ups are never downsampled."""
        auto = step is None
        s = _auto_step(self.image.shape, self.scan_max_voxels) if step is None else int(step)
        if s < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        a = _scaled(self.affine, s)
        layers = [
            Layer(
                n,
                self.colors[n],
                n not in self.hidden,
                self._nifti(f"{n}@{s}", _block(self.masks[n], s, any_=True), a),
            )
            for n in self.masks
        ]
        img = _block(self.image, s)
        title = "whole scan" + (f" (1/{s} resolution)" if s > 1 else "")
        scene = Scene(
            "scan", title, self._nifti(f"image@{s}", img, a), a, img.shape, layers, step=s
        )
        scene.info = self._info(
            scene,
            step=s,  # image: block mean; masks: block any
            step_from="auto" if auto else "argument",
            source_shape=self.image.shape,
            source_spacing_mm=_spacing(self.affine),
        )
        return scene

    def _inside(self, r: dict) -> np.ndarray:
        if self.labels is None:
            return self.masks[r["region"]]
        return self.labels == r["region"]

    def region_scene(self, i: int = 0, min_mm: float = 60.0) -> Scene:
        """Region i cropped (at least min_mm, else 1.5x its extent): each mask's
        part of this region as overlay + surface; masks missing it are left out."""
        r = self._region(i)
        sl, a = _crop(self._inside(r), self.affine, min_mm)
        inside = self._inside(r)[sl]
        layers = []
        for n, m in self.masks.items():
            mk = m[sl] & inside
            if mk.any():
                layers.append(
                    Layer(
                        n,
                        self.colors[n],
                        n not in self.hidden,
                        _nifti(mk.astype(np.uint8), a, (0, 1)),
                        _surface(mk, a),
                    )
                )
        scene = Scene(
            "region",
            f"region #{r['index']}",
            _nifti(self.image[sl], a, self.window),
            a,
            inside.shape,
            layers,
            np.argwhere(inside).mean(axis=0),
            r["index"],
        )
        scene.info = self._info(
            scene,
            region=r["index"],
            voxels=r["voxels"],
            volume_ml=r["volume_ml"],
            crop_vox=tuple((int(b.start), int(b.stop)) for b in sl),
            crop_min_mm=min_mm,  # else 1.5x the region's extent
            masks_absent=tuple(n for n in self.masks if n not in {L.name for L in layers}),
            surface_smoothing_vox=SURFACE_SIGMA_VOX,
        )
        return scene

    def _info(self, s: Scene, **extra) -> Record:
        """Render metadata for one scene, shown by its "info" button."""
        mb = (len(s.image) + sum(len(L.overlay) + len(L.mesh or b"") for L in s.layers)) / 1e6
        return Record(
            view=s.title,
            shape=tuple(int(x) for x in s.shape),
            spacing_mm=_spacing(s.affine),
            **extra,
            **self.settings,
            masks=tuple(L.name for L in s.layers),
            hidden_at_start=tuple(L.name for L in s.layers if not L.on),
            sent_mb=round(mb, 2),
            ipyniivue=ipyniivue.__version__,
            **{f"meta.{k}": v for k, v in self.meta.items()},
        )

    def scan(
        self,
        height: int = 450,
        opacity: float = 0.5,
        crosshair_gap: float = 1.0,
        step: int | None = None,
    ):
        """Whole volume in one row (axial, coronal, sagittal, 3D). Bar below:
        toggles, region picker, center, opacity, crosshair gap, info. step:
        see scan_scene (None = the View's scan_max_voxels rule)."""
        s = self.scan_scene(step)
        nv = _viewer(s, "row", True, height, opacity, self.radiological)
        extra = []
        if self.regions:
            pick = W.Dropdown(
                options=[(f"region #{r['index']}", r["index"]) for r in self.regions],
                layout=W.Layout(width="130px"),
            )
            go = lambda: _center_on(
                [nv],
                s.affine,
                s.shape,
                (np.asarray(self.regions[pick.value]["centroid_vox"]) - (s.step - 1) / 2) / s.step,
            )
            pick.observe(lambda _: go(), names="value")
            extra = [pick, _center_button(go)]
        extra += _gap_slider([nv], s.affine, s.shape, crosshair_gap)
        btn, panel = _info_button(s.info)
        bar = _menu(
            s.title,
            s.layers,
            [[nv.volumes[k + 1]] for k in range(len(s.layers))],
            opacity,
            [*extra, btn],
        )
        return W.VBox([nv, bar, panel], layout=_FULL)

    def region(
        self,
        i: int = 0,
        tile_height: int = 300,
        opacity: float = 0.5,
        crosshair_gap: float = 1.0,
        min_mm: float = 60.0,
    ):
        """Linked 4x2 close-up of region i:

            raw axial    | axial + masks    | coronal + masks | raw coronal
            raw sagittal | sagittal + masks | 3D + masks      | 3D surfaces only

        Bar: toggles, center, opacity, crosshair gap, info.
        """
        s = self.region_scene(i, min_mm)
        meshes = [
            Mesh(
                data=L.mesh,
                name=f"{L.name.replace(' ', '_')}.obj",
                rgba255=[*_rgb(L.cmap), 255],
                opacity=opacity,
            )
            for L in s.layers
        ]

        def tile(kind, with_masks):
            return _viewer(s, kind, with_masks, tile_height, opacity, self.radiological)

        cells, annotated = [], []
        for left, right in TILES:
            if right == "render":
                outer = tile("render", False)
                outer.volumes[0].opacity = 0.0
                outer.load_meshes(meshes)
            else:
                outer = tile(right, False)
            row = [tile(left, False), tile(left, True), tile(right, True), outer]
            annotated += row[1:3]
            cells += row
        _link(cells)
        go = lambda: _center_on(cells, s.affine, s.shape, s.center)
        go()
        targets = [
            [nv.volumes[j + 1] for nv in annotated] + [meshes[j]] for j in range(len(s.layers))
        ]
        extra = [_center_button(go), *_gap_slider(cells, s.affine, s.shape, crosshair_gap)]
        head = [
            W.HTML(f"<div style='text-align:center'><b>{t}</b></div>")
            for t in ("raw", "with masks", "with masks", "raw  ·  bottom: surfaces only")
        ]
        grid = W.GridBox(
            head + cells,
            layout=W.Layout(width="100%", grid_template_columns="repeat(4, 1fr)", grid_gap="4px"),
        )
        btn, panel = _info_button(s.info)
        bar = _menu(s.title, s.layers, targets, opacity, [*extra, btn])
        return W.VBox([grid, bar, panel], layout=_FULL)

    def _nifti(self, key, arr, aff):
        """Full-size volumes are serialized once and cached across scan() calls."""
        if key not in self._bytes:
            is_image = key.startswith("image@")
            cal = self.window if is_image else (0, 1)
            data = arr if is_image else arr.astype(np.uint8)
            self._bytes[key] = _nifti(data, aff, cal)
        return self._bytes[key]


def load(
    case: Any,
    masks: Any = None,
    *,
    window: Any = None,
    hidden: Sequence[str] | None = None,
    colors: Mapping[str, str] | None = None,
    radiological: bool | None = None,
    image_dtype: Any = "auto",
    scan_max_voxels: int | None = 40_000_000,
    **case_kw,
) -> View:
    """A View from a Case, an image file plus masks (Case.from_file keywords), or an
    array plus masks (Case.from_arrays keywords). window/hidden/colors/radiological
    override case.display. scan_max_voxels caps the whole-scan view (None: never
    downsample); close-ups are always full resolution. The "info" panel lists
    every choice made.
    """
    if isinstance(case, (str, pathlib.Path)):
        case = Case.from_file(case, masks, **case_kw)
    elif not isinstance(case, Case):
        case = Case.from_arrays(case, masks, **case_kw)
    elif masks is not None or case_kw:
        raise TypeError(
            "masks= and Case.from_arrays keywords only apply when passing arrays, not a Case"
        )
    d = case.display
    window_from = "argument" if window is not None else ("case" if "window" in d else "default")
    window = d.get("window", "auto") if window is None else window
    hidden = list(d.get("hidden", ())) if hidden is None else list(hidden)
    colors = dict(d.get("colors", {})) if colors is None else dict(colors)
    radiological = bool(d.get("radiological", True)) if radiological is None else radiological

    unknown = set(hidden) - set(case.masks)
    if unknown:
        raise KeyError(
            f"hidden= names not in masks: {sorted(unknown)}; masks are {list(case.masks)}"
        )
    cmaps = dict(zip(case.masks, itertools.cycle(PALETTE)))
    for n, c in colors.items():
        if n not in case.masks:
            raise KeyError(f"colors= name {n!r} not in masks {list(case.masks)}")
        if c not in PALETTE:
            raise ValueError(f"colors[{n!r}]={c!r}; choose from {list(PALETTE)}")
        cmaps[n] = c

    labels = case.regions  # None: each mask is a region
    img = _cast_image(case.image, image_dtype)
    win = _window(img, window)
    settings = {
        "window": win,
        "window_from": window_from,
        "radiological": radiological,
        "image_dtype": str(img.dtype),
        "source_dtype": str(case.image.dtype),
        "regions_from": "case" if case.regions is not None else "masks",
        "scan_max_voxels": scan_max_voxels,
    }
    return View(
        image=img,
        affine=case.affine,
        masks=case.masks,
        colors=cmaps,
        hidden=frozenset(hidden),
        window=win,
        radiological=radiological,
        labels=labels,
        scan_max_voxels=scan_max_voxels,
        meta=dict(case.meta),
        settings=settings,
        regions=_regions(case.masks, labels, case.affine),
    )


def _window(image: np.ndarray, window: Any) -> tuple[float, float]:
    """ "auto" = 0.5-99.5 percentile of the image; else (low, high)."""
    if isinstance(window, str) and window != "auto":
        raise ValueError(f"window {window!r} unknown; use 'auto' or (low, high)")
    if window is None or window == "auto":
        sample = image[::4, ::4, ::2].astype(np.float32)
        lo, hi = np.percentile(sample[np.isfinite(sample)], (0.5, 99.5))
        return float(lo), float(hi if hi > lo else lo + 1)
    lo, hi = map(float, window)
    if not lo < hi:
        raise ValueError(f"window must be (low, high) with low < high, got {window}")
    return lo, hi


def _cast_image(img: np.ndarray, image_dtype: Any) -> np.ndarray:
    """See load(image_dtype=)."""
    if image_dtype == "auto":
        keep = img.dtype == bool or (
            np.issubdtype(img.dtype, np.integer) and img.dtype.itemsize <= 2
        )
        return img if keep else img.astype(np.float32, copy=False)
    dt = np.dtype(image_dtype)
    if np.issubdtype(dt, np.integer):
        lost = np.abs(img - np.round(img)).max() if np.issubdtype(img.dtype, np.floating) else 0
        info = np.iinfo(dt)
        if lost > 1e-3:
            warnings.warn(
                f"casting image to {dt}: values rounded (largest change {lost:.3g})", stacklevel=2
            )
        if img.min() < info.min or img.max() > info.max:
            warnings.warn(
                f"casting image to {dt}: values outside {info.min}..{info.max} are clipped",
                stacklevel=2,
            )
        return np.clip(np.round(img), info.min, info.max).astype(dt)
    return img.astype(dt)


def _auto_step(shape, max_voxels: int | None) -> int:
    """Smallest whole step per axis that brings the grid within max_voxels."""
    step = 1
    while max_voxels is not None and np.prod(np.asarray(shape) // step) > max_voxels:
        step += 1
    return step


def _spacing(affine) -> tuple[float, ...]:
    return tuple(round(float(x), 3) for x in np.linalg.norm(affine[:3, :3], axis=0))


def _info_button(info: Record):
    """A toggle that shows the scene's render metadata."""
    panel = W.HTML(
        info._repr_html_(),
        layout=W.Layout(display="none", width="100%", padding="4px 8px"),
    )
    btn = W.ToggleButton(
        description="info",
        icon="info-circle",
        layout=W.Layout(width="80px"),
        tooltip="rendering metadata: grids, downsampling, window, sizes",
    )
    btn.observe(
        lambda ch: setattr(panel.layout, "display", "block" if ch["new"] else "none"), names="value"
    )
    return btn, panel


def _block(arr: np.ndarray, s: int, any_: bool = False) -> np.ndarray:
    """Downsample by s per axis over s^3 blocks: mean (image, same dtype) or
    any (mask). Trailing voxels that do not fill a block are dropped."""
    if s == 1:
        return arr
    n = [d // s for d in arr.shape]
    v = arr[: n[0] * s, : n[1] * s, : n[2] * s].reshape(n[0], s, n[1], s, n[2], s)
    if any_:
        return v.any(axis=(1, 3, 5))
    out = v.mean(axis=(1, 3, 5), dtype=np.float32)
    return np.round(out).astype(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else out


def _scaled(affine: np.ndarray, s: int) -> np.ndarray:
    """Affine of a block-downsampled grid: voxel i covers original voxels
    s*i .. s*i + s-1, so its centre sits (s-1)/2 original voxels further on."""
    if s == 1:
        return affine
    a = affine.copy()
    a[:3, 3] += affine[:3, :3] @ np.full(3, (s - 1) / 2)
    a[:3, :3] *= s
    return a


def _regions(masks: Mapping[str, np.ndarray], labels: np.ndarray | None, affine) -> Records:
    """One row per region, in the order given: label values ascending, else one per mask
    (a mask's voxels are its region, connected or not). Works inside each region's
    bounding box so many small regions stay fast."""
    found: list[tuple[Any, Any, np.ndarray | None]]
    if labels is None:
        found = [(name, _box(m), m) for name, m in masks.items()]
    else:
        found = [(v, box, None) for v, box in enumerate(ndimage.find_objects(labels), 1)]
    ml = abs(np.linalg.det(affine[:3, :3])) / 1000.0
    rows = Records()
    for key, box, mask in found:
        if box is None:
            continue  # empty mask, or a label value not in the map
        inside = mask[box] if mask is not None else np.asarray(labels)[box] == key
        vox = np.argwhere(inside)
        centroid = vox.mean(0) + [b.start for b in box]
        rows.append(
            {
                "index": len(rows),
                "region": key,
                "voxels": len(vox),
                "volume_ml": round(len(vox) * ml, 4),
                "centroid_vox": tuple(round(float(x), 2) for x in centroid),
                "centroid_mm": tuple(round(float(x), 2) for x in (affine @ [*centroid, 1])[:3]),
                "masks": tuple(n for n, m in masks.items() if (m[box] & inside).any()),
            }
        )
    return rows


def _box(mask: np.ndarray) -> tuple[slice, ...] | None:
    return ndimage.find_objects(mask.astype(np.uint8))[0] if mask.any() else None


def _nifti(arr: np.ndarray, affine: np.ndarray, cal: tuple[float, float]) -> bytes:
    """Gzipped NIfTI bytes; the display window goes in the header's cal_min/max."""
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    elif not (np.issubdtype(arr.dtype, np.integer) and arr.dtype.itemsize <= 2):
        arr = arr.astype(np.float32, copy=False)
    img = nib.Nifti1Image(np.ascontiguousarray(arr), affine)
    img.header["cal_min"], img.header["cal_max"] = cal
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", compresslevel=1) as gz:  # stream: no raw copy
        holder = nib.FileHolder(fileobj=gz)
        img.to_file_map({"header": holder, "image": holder})
    return bio.getvalue()


def _viewer(
    s: Scene, kind: str, with_masks: bool, height: int, opacity: float, radiological: bool
) -> NiiVue:
    nv = NiiVue(height=height)
    nv.layout = W.Layout(width="100%")
    vols = [Volume(data=s.image, name="image.nii.gz", colormap="gray")]
    if with_masks:
        vols += [
            Volume(data=L.overlay, name=f"mask{k}.nii.gz", colormap=L.cmap, opacity=opacity)
            for k, L in enumerate(s.layers)
        ]
    nv.load_volumes(vols)
    if kind == "row":
        nv.set_slice_type(SliceType.MULTIPLANAR)
        nv.opts.multiplanar_layout = MultiplanarType.ROW
        nv.opts.multiplanar_show_render = ShowRender.ALWAYS
    else:
        nv.set_slice_type(_SLICE[kind])
    nv.opts.is_orient_cube = kind in ("row", "render")
    nv.opts.is_radiological_convention = radiological
    nv.opts.is_colorbar = False
    nv.opts.back_color = (0, 0, 0, 1)
    return nv


def _crop(inside: np.ndarray, affine: np.ndarray, min_mm: float):
    """Slicer + shifted affine for a cube around `inside`: at least min_mm,
    else 1.5x the region's extent, clipped to the volume."""
    vox = np.argwhere(inside)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    extent = ((vox.max(0) - vox.min(0) + 1) * spacing).max()
    half = np.ceil(max(min_mm, 1.5 * extent) / spacing / 2).astype(int)
    c = np.round(vox.mean(0)).astype(int)
    lo = np.maximum(c - half, 0)
    hi = np.minimum(c + half, inside.shape)
    a = affine.copy()
    a[:3, 3] += affine[:3, :3] @ lo
    return tuple(slice(x, y) for x, y in zip(lo, hi, strict=True)), a


SURFACE_SIGMA_VOX = 0.5  # Gaussian sigma before meshing: removes voxel steps, keeps the shape


def _surface(mask: np.ndarray, affine: np.ndarray) -> bytes:
    """Smoothed, outward-facing surface of a mask in world mm, as OBJ text."""
    padded = np.pad(mask, 1).astype(np.float32)
    smooth = ndimage.gaussian_filter(padded, SURFACE_SIGMA_VOX)
    field = smooth if smooth.max() > 0.5 else padded  # tiny masks: skip smoothing
    verts, faces, _, _ = measure.marching_cubes(field, 0.5)
    world = (verts - 1) @ affine[:3, :3].T + affine[:3, 3]
    v0, v1, v2 = (world[faces[:, k]] for k in range(3))
    if np.einsum("ij,ij->", v0, np.cross(v1, v2)) < 0:  # inward -> looks hollow
        faces = faces[:, ::-1]
    obj = "".join(f"v {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in world)
    obj += "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)
    return obj.encode()


def _rgb(cmap: str) -> list[int]:
    return [int(PALETTE[cmap][k] * 2, 16) for k in (1, 2, 3)]


def gap_params(affine, shape, stub_frac: float = 0.08) -> tuple[float, float]:
    """(radius_mm, reach_mm). NiiVue measures the crosshair gap in crosshair
    radii (half the smallest voxel); slider value f sets gap = f * reach / radius,
    so f = 1 leaves stubs of stub_frac of the half-width at the edges."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    half = 0.5 * (np.asarray(shape) * spacing).max()
    return float(0.5 * spacing.min()), float((1 - stub_frac) * half)


def _link(nvs):
    """Clicking, scrolling, panning or rotating in any viewer moves all."""
    for nv in nvs:
        nv.broadcast_to([o for o in nvs if o is not nv], {"2d": True, "3d": True})


def _frac(affine, shape, vox):
    """Voxel index -> NiiVue crosshair fraction.

    NiiVue reorients a volume to RAS by permuting and flipping voxel axes
    (each array axis goes to the world axis its affine column mostly points
    along; nothing is resampled) and measures the crosshair as (i + 0.5) / dim
    along those reoriented axes. Shear (gantry tilt) does not change which
    voxel is which, so this is exact for tilted scans too."""
    frac = [0.5, 0.5, 0.5]
    for j in range(3):
        col = affine[:3, j]
        r = int(np.argmax(np.abs(col)))
        f = (vox[j] + 0.5) / shape[j]
        frac[r] = float(f if col[r] > 0 else 1.0 - f)
    return frac


def _center_on(nvs, affine, shape, vox):
    """Scene traits are observed by their NiiVue and pushed to the browser on
    assignment, so plain assignment is enough."""
    pos = _frac(affine, shape, vox)
    for nv in nvs:
        nv.scene.crosshair_pos = pos
        nv.scene.pan2d_xyzmm = [0.0, 0.0, 0.0, 1.0]


def _center_button(on_click):
    b = W.Button(description="center", icon="crosshairs", layout=W.Layout(width="100px"))
    b.on_click(lambda _: on_click())
    return b


CROSSHAIR_NUDGE = 1e-5  # fraction of the volume; far below one voxel


def _redraw_crosshair(nv, sign: int) -> None:
    """NiiVue 0.5x caches the crosshair geometry and rebuilds it only when the
    crosshair moves, so an options change (the gap) is invisible until then.
    No public call forces the rebuild; moving the crosshair by far less than a
    voxel, alternating direction so it never drifts, is the smallest one."""
    pos = list(nv.scene.crosshair_pos)
    pos[0] += sign * CROSSHAIR_NUDGE
    nv.scene.crosshair_pos = pos


def _gap_slider(nvs, affine, shape, value):
    """0 = crosshair all the way through; 1 = short stubs at the image edges."""
    radius, reach = gap_params(affine, shape)
    slider = W.FloatSlider(
        value=value,
        min=0.0,
        max=1.0,
        step=0.02,
        description="crosshair gap",
        readout_format=".2f",
        style={"description_width": "initial"},
        layout=W.Layout(width="300px"),
    )
    sign = [1]

    def apply(_=None):
        sign[0] = -sign[0]
        for nv in nvs:
            nv.opts.crosshair_gap = slider.value * reach / radius
            _redraw_crosshair(nv, sign[0])

    slider.observe(apply, names="value")
    apply()
    return [slider, W.HTML("<span style='opacity:.7'>&nbsp;through&nbsp;&#8596;&nbsp;edge</span>")]


def _menu(title, layers, targets, opacity, extra=()):
    """Bar: a checkbox per Layer, all/none, opacity, `extra`.
    targets[k] = the Volumes/Meshes layer k controls."""
    boxes = [W.Checkbox(value=L.on, indent=False, layout=W.Layout(width="20px")) for L in layers]
    slider = W.FloatSlider(
        value=opacity,
        min=0.05,
        max=1.0,
        step=0.05,
        description="opacity",
        readout_format=".2f",
        layout=W.Layout(width="260px"),
    )

    def apply(_=None):
        for cb, objs in zip(boxes, targets, strict=True):
            for o in objs:
                if isinstance(o, Mesh):
                    o.visible, o.opacity = cb.value, slider.value
                else:
                    o.opacity = slider.value if cb.value else 0.0

    for w in (*boxes, slider):
        w.observe(apply, names="value")
    apply()

    def set_all(on):
        for cb in boxes:
            cb.value = on

    b_all = W.Button(description="all", layout=W.Layout(width="50px"))
    b_none = W.Button(description="none", layout=W.Layout(width="60px"))
    b_all.on_click(lambda _: set_all(True))
    b_none.on_click(lambda _: set_all(False))
    items = [
        W.HBox(
            [cb, W.HTML(f"<span style='color:{PALETTE[L.cmap]}'>&#9632;</span> {L.name}")],
            layout=W.Layout(margin="0 14px 0 0"),
        )
        for cb, L in zip(boxes, layers, strict=True)
    ]
    return W.HBox(
        [W.HTML(f"<b>{title}</b>&nbsp;&nbsp;&nbsp;"), *items, b_all, b_none, slider, *extra],
        layout=W.Layout(
            width="100%",
            flex_flow="row wrap",
            align_items="center",
            padding="6px",
            border_top="1px solid #888",
        ),
    )
