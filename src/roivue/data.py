"""Public sample data for the examples and tests, downloaded once and checked (pooch).

path = data.fetch("ct_electrodes")   # cached in ~/.cache/roivue, or $ROIVUE_DATA
"""

from __future__ import annotations

import pathlib

import pooch

_NIIVUE = "https://raw.githubusercontent.com/niivue/niivue-demo-images/main"
_VTK = "https://raw.githubusercontent.com/pyvista/vtk-data/master/Data"

ABOUT = {
    "ct_electrodes": "Head CT with implanted depth electrodes (NIfTI); Seg3DData, MIT licence, "
    "SCI Institute, University of Utah (2015), via the NiiVue demo images.",
    "fullhead": "Head MRI as MetaImage (.mhd + raw data, LAS orientation); the VTK 'FullHead' "
    "example from pyvista/vtk-data (Apache-2.0).",
}

_FILES = pooch.create(
    path=pooch.os_cache("roivue"),
    base_url="",
    env="ROIVUE_DATA",
    registry={
        "CT_Electrodes.nii.gz": "0566942708ff451458d0fdb1f61c4611187bb5cc0efa516c8500f1ff3c142e57",
        "FullHead.mhd": "96925bf993fe7ac317abeee6fdd0ba37cb78f0b3d445bc1a3d69028d51874760",
        "FullHead.raw.gz": "086c1627b9a2a457c6eaae29a04d7ccd868ebc935c96bdf55d0d1fe8a3f01d12",
    },
    urls={
        "CT_Electrodes.nii.gz": f"{_NIIVUE}/CT_Electrodes.nii.gz",
        "FullHead.mhd": f"{_VTK}/FullHead.mhd",
        "FullHead.raw.gz": f"{_VTK}/FullHead.raw.gz",
    },
)


def fetch(name: str) -> pathlib.Path:
    """The main file of a sample dataset (see ABOUT), downloading it on first use."""
    if name == "ct_electrodes":
        return pathlib.Path(_FILES.fetch("CT_Electrodes.nii.gz", progressbar=False))
    if name == "fullhead":  # the header names its data file FullHead.raw, next to it
        _FILES.fetch("FullHead.raw.gz", processor=pooch.Decompress(name="FullHead.raw"))
        return pathlib.Path(_FILES.fetch("FullHead.mhd"))
    raise KeyError(f"unknown dataset {name!r}; available: {sorted(ABOUT)}")
