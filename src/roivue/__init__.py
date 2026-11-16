"""roivue: regions of interest across segmentation masks, viewed in NiiVue.

case (the validated hand-off: image, affine, masks, meta, display hints) ->
view (Jupyter widgets) and export (one HTML file); io reads files into a Case.
Self-contained (relative imports only), so it can be vendored by copying.
"""

from .case import Case, GeometryError

try:  # from the git tag via hatch-vcs; a vendored copy has no package metadata
    from importlib.metadata import version as _version

    __version__ = _version("roivue")
except Exception:
    __version__ = "vendored"

__all__ = ["Case", "GeometryError", "__version__"]
