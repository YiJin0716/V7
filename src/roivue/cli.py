"""The `roivue` command: image and masks in, one self-contained HTML page out.

roivue scan.nii.gz -m tumour=tumour.nii.gz --open
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import webbrowser
from typing import Any

from . import ct, export
from . import view as cv
from .case import GeometryError

SHOW_REGIONS = 10  # regions listed in the terminal; the page has all (or --max-regions)
WINDOWS = {"lung": ct.LUNG_WINDOW, "soft-tissue": ct.SOFT_TISSUE_WINDOW, "bone": ct.BONE_WINDOW}


def _pairs(values: list[str] | None, what: str) -> list[tuple[str, str]]:
    out = []
    for v in values or []:
        key, sep, value = v.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"{what} must look like KEY=VALUE, got {v!r}")
        out.append((key, value))
    return out


def _window(value: str) -> Any:
    if value == "auto" or value in WINDOWS:
        return WINDOWS.get(value, "auto")
    low, sep, high = value.partition(",")
    try:
        if sep:
            return (float(low), float(high))
    except ValueError:
        pass
    raise ValueError(f"--window must be auto, {', '.join(WINDOWS)}, or LOW,HIGH; got {value!r}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roivue",
        description="Write one self-contained HTML page showing an image, its masks, and a "
        "linked close-up of every region.",
    )
    p.add_argument(
        "image", type=pathlib.Path, help="NIfTI (.nii, .nii.gz) or MetaImage (.mha, .mhd)"
    )
    masks = p.add_argument_group("masks (one of)")
    masks.add_argument(
        "-m",
        "--mask",
        action="append",
        metavar="NAME=PATH",
        help="a mask file; repeat for several (names label the toggles)",
    )
    masks.add_argument(
        "--labels",
        type=pathlib.Path,
        metavar="PATH",
        help="one integer label map, split into a mask per value",
    )
    p.add_argument(
        "--label-names", nargs="+", metavar="VALUE=NAME", help="names for --labels values"
    )
    p.add_argument(
        "--regions",
        type=pathlib.Path,
        metavar="PATH",
        help="a region label map (default: each mask is one region)",
    )
    p.add_argument(
        "-w",
        "--window",
        default="auto",
        help=f"display window: auto (default), {', '.join(WINDOWS)}, or LOW,HIGH",
    )
    p.add_argument(
        "--hidden", nargs="+", metavar="NAME", default=[], help="masks that start switched off"
    )
    p.add_argument(
        "--meta",
        action="append",
        metavar="KEY=VALUE",
        help="metadata shown on the page; repeat for several",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "int16", "int32", "float32", "float64"],
        help="how to read the image (default: auto)",
    )
    p.add_argument(
        "--max-regions",
        type=int,
        metavar="N",
        help="embed only the first N regions (smaller file)",
    )
    p.add_argument(
        "--scan-max-voxels",
        default="40000000",
        metavar="N|none",
        help="downsample the whole-scan view above N voxels (default 40000000)",
    )
    p.add_argument("--title", help="page title (default: the image file name)")
    p.add_argument(
        "--niivue-js",
        metavar="PATH|URL",
        help="NiiVue bundle to embed (default: the version ipyniivue uses, from npm)",
    )
    p.add_argument("-o", "--out", type=pathlib.Path, help="output file (default: IMAGE_NAME.html)")
    p.add_argument("--open", action="store_true", help="open the page in the default browser")
    return p


def run(args: argparse.Namespace) -> pathlib.Path:
    """Builds the view and writes the page; returns its path."""
    if args.mask and args.labels:
        raise ValueError("use --mask or --labels, not both")
    if args.label_names and not args.labels:
        raise ValueError("--label-names needs --labels")
    masks = dict(_pairs(args.mask, "--mask"))
    label_names = {int(k): v for k, v in _pairs(args.label_names, "--label-names")} or None
    meta = dict(_pairs(args.meta, "--meta"))
    max_voxels = None if args.scan_max_voxels == "none" else int(args.scan_max_voxels)
    name = args.image.name.removesuffix(".gz").rsplit(".", 1)[0]

    view = cv.load(
        args.image,
        masks,
        window=_window(args.window),
        hidden=args.hidden,
        scan_max_voxels=max_voxels,
        labels=args.labels,
        regions=args.regions,
        label_names=label_names,
        meta=meta,
        dtype=args.dtype,
    )
    print(view.regions[:SHOW_REGIONS])
    if len(view.regions) > SHOW_REGIONS:
        print(f"... {len(view.regions) - SHOW_REGIONS} more regions (all are in the page)")
    out = args.out or pathlib.Path(f"{name}.html")
    path = export.write_html(
        view,
        out,
        title=args.title or args.image.name,
        regions=args.max_regions,
        niivue_js=args.niivue_js,
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if args.open:
        webbrowser.open(path.resolve().as_uri())
    return path


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        run(args)
    except (GeometryError, ValueError, KeyError, FileNotFoundError, ImportError) as e:
        p.exit(2, f"roivue: error: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
