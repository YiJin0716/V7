# Third-party code and assets

- `src/vista3d/`: selected source from [Project-MONAI/VISTA](https://github.com/Project-MONAI/VISTA), commit `d4a8fe0dbf5cb4b76c531fccca8e29c1e6f6ee45`. Apache 2.0; see `LICENSES/VISTA-Apache-2.0.txt`. Utility imports were moved under `vista3d._vendor_utils` to make the package self-contained. Model weights are not included and have separate terms.
- `src/roivue/`: source from [roivue](https://gitlab.oit.duke.edu/lam135/roivue), commit `14cdebad5f7d6bfb2ee6e42bcf74f63186561b30`. Zero-Clause BSD; see `LICENSES/roivue-0BSD.txt`.
- `missing_malignant/*.html`: includes NiiVue 0.65.0 JavaScript, embedded by roivue. NiiVue is MIT-licensed; see `LICENSES/NiiVue-MIT.txt` and [the source repository](https://github.com/niivue/niivue).
- The CT examples and reader annotations come from [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/). Case IDs are the dataset's de-identified identifiers.

Python dependencies retain their own licenses. These notices describe the included third-party material and do not assign a license to the project's original code.
