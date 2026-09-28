# Third-party licenses and provenance — Omni-iEEG Planner RC20

Prepared 2026-09-28 for the planned **source-only** release, version `1.0-RC20`.
Python: 3.12.14 on Windows. This file is an inventory and attribution record,
not a replacement for upstream licenses and not an application license grant.
The original application code is licensed under the [MIT License](LICENSE).
Third-party portions retain their own terms; see the
[audit](docs/LICENSE_AUDIT_RC20.md) and [NOTICE](NOTICE).

## Scope and method

The two tables describe the installed environments represented by
`requirements-lock.txt` and `requirements-postop-ants-lock.txt`. They include
transitive packages, not only modules directly imported by the application.
Package installer `pip` and the isolated audit tooling are excluded.

`pip-licenses 5.5.5` was run from a separate audit environment with `--python`
pointing to each application environment. The unmodified command outputs are
[main environment](docs/licenses/pip-licenses-main.md) and
[ANTs environment](docs/licenses/pip-licenses-ants.md). The tables below use
installed metadata, prefer SPDX expressions when supplied, and clarify selected
entries using installed license texts. A generic `BSD License` entry remains
generic; it is not a determination of the exact BSD variant or every bundled
component's terms. Project links come from installed package metadata.

These tables can support Supplementary Table S1 when the environment names,
versions, direct/transitive distinction and the qualifications below are retained.
They are not a clearance certificate or a complete binary redistribution notice.

## Main application environment — 38 packages

| Package | Version | Declared license / reviewed clarification | Use in this environment | Upstream |
| --- | --- | --- | --- | --- |
| annotated-doc | 0.0.5 | MIT | Transitive dependency | [Project](https://github.com/fastapi/annotated-doc) |
| cachebox | 5.2.3 | MIT | Transitive dependency | [Project](https://github.com/awolverp/cachebox) |
| colorama | 0.4.6 | BSD License | Transitive dependency | [Project](https://github.com/tartley/colorama) |
| contourpy | 1.4.0 | BSD-3-Clause | Transitive dependency | [Project](https://github.com/contourpy/contourpy) |
| cycler | 0.12.1 | BSD License | Transitive dependency | [Project](https://matplotlib.org/cycler/) |
| dcm2niix | 1.0.20260724 | BSD-2-Clause core; additional component notices | DICOM conversion | [Project](https://github.com/rordenlab/dcm2niix) |
| deepdiff | 9.1.0 | MIT License | Transitive dependency | [Project](https://zepworks.com/deepdiff/) |
| dipy | 1.12.1 | BSD License | Diffusion modelling and tracking | [Project](https://dipy.org) |
| et_xmlfile | 2.0.0 | MIT | Transitive dependency | [Project](https://foss.heptapod.net/openpyxl/et_xmlfile) |
| fonttools | 4.65.0 | MIT | Transitive dependency | [Project](http://github.com/fonttools/fonttools) |
| h5py | 3.16.0 | BSD-3-Clause | Transitive dependency | [Project](https://www.h5py.org/) |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause wrapper; FFmpeg separately GPL-3.0-or-later here | Locate external video encoder | [Project](https://github.com/imageio/imageio-ffmpeg) |
| kiwisolver | 1.5.1 | BSD License | Transitive dependency | [Project](https://github.com/nucleic/kiwi) |
| markdown-it-py | 4.2.0 | MIT License | Transitive dependency | [Project](https://github.com/executablebooks/markdown-it-py) |
| matplotlib | 3.11.2 | Matplotlib license (PSF-based); additional bundled notices | Colour maps | [Project](https://matplotlib.org) |
| mdurl | 0.1.2 | MIT License | Transitive dependency | [Project](https://github.com/executablebooks/mdurl) |
| nibabel | 5.4.2 | MIT | NIfTI and FreeSurfer file I/O | [Project](https://nipy.org/nibabel) |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Numerical arrays | [Project](https://numpy.org) |
| openpyxl | 3.1.5 | MIT | Excel result input | [Project](https://foss.heptapod.net/openpyxl/openpyxl) |
| orderly-set | 5.5.0 | MIT License | Transitive dependency | [Project](https://github.com/seperman/orderly-set) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | Transitive dependency | [Project](https://github.com/pypa/packaging) |
| pillow | 12.3.0 | MIT-CMU | Transitive dependency | [Project](https://python-pillow.github.io) |
| pydicom | 3.0.2 | MIT License | DICOM input | [Project](https://github.com/pydicom/pydicom) |
| Pygments | 2.21.0 | BSD-2-Clause | Transitive dependency | [Project](https://pygments.org) |
| pyparsing | 3.3.2 | MIT | Transitive dependency | [Project](https://github.com/pyparsing/pyparsing/) |
| PySide6_Essentials | 6.11.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | Qt user interface | [Project](https://pyside.org) |
| python-dateutil | 2.9.0.post0 | Apache-2.0 / BSD-3-Clause; see per-contribution terms | Transitive dependency | [Project](https://github.com/dateutil/dateutil) |
| rich | 15.0.0 | MIT | Transitive dependency | [Project](https://github.com/Textualize/rich) |
| scipy | 1.18.1 | BSD License | Scientific and spatial operations | [Project](https://scipy.org/) |
| shellingham | 1.5.4 | ISC License | Transitive dependency | [Project](https://github.com/sarugaku/shellingham) |
| shiboken6 | 6.11.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | Transitive dependency | [Project](https://pyside.org) |
| simpleitk | 2.5.6 | Apache-2.0 | Image registration and resampling | [Project](https://github.com/SimpleITK/SimpleITK) |
| six | 1.17.0 | MIT | Transitive dependency | [Project](https://github.com/benjaminp/six) |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | Transitive dependency | [Project](https://tqdm.github.io) |
| trx-python | 0.6 | BSD License | Transitive dependency | [Project](https://github.com/tee-ar-ex/trx-python) |
| typer | 0.27.2 | MIT | Transitive dependency | [Project](https://github.com/fastapi/typer) |
| typing_extensions | 4.16.0 | PSF-2.0 | Transitive dependency | [Project](https://github.com/python/typing_extensions) |
| vtk | 9.7.0 | BSD License | 3D rendering | [Project](https://vtk.org) |

## Optional ANTs environment — 33 packages

| Package | Version | Declared license / reviewed clarification | Use in this environment | Upstream |
| --- | --- | --- | --- | --- |
| antspyx | 0.6.2 | Apache-2.0 | Optional ANTs registration worker | [Project](https://github.com/antsx/antspy) |
| certifi | 2026.7.22 | MPL-2.0 | Transitive dependency of ANTs environment | [Project](https://github.com/certifi/python-certifi) |
| charset-normalizer | 3.5.1 | MIT | Transitive dependency of ANTs environment | [Project](https://charset-normalizer.readthedocs.io/) |
| cloudpickle | 3.1.2 | BSD-3-Clause | Transitive dependency of ANTs environment | [Project](https://github.com/cloudpipe/cloudpickle) |
| contourpy | 1.4.0 | BSD-3-Clause | Transitive dependency of ANTs environment | [Project](https://github.com/contourpy/contourpy) |
| cycler | 0.12.1 | BSD License | Transitive dependency of ANTs environment | [Project](https://matplotlib.org/cycler/) |
| fonttools | 4.66.0 | MIT | Transitive dependency of ANTs environment | [Project](http://github.com/fonttools/fonttools) |
| formulaic | 1.2.2 | MIT | Transitive dependency of ANTs environment | [Project](https://github.com/matthewwardrop/formulaic) |
| idna | 3.20 | BSD-3-Clause | Transitive dependency of ANTs environment | [Project](https://github.com/kjd/idna) |
| interface_meta | 2.0.1 | MIT | Transitive dependency of ANTs environment | [Project](https://github.com/matthewwardrop/interface_meta) |
| joblib | 1.6.0 | BSD-3-Clause | Transitive dependency of ANTs environment | [Project](https://joblib.readthedocs.io) |
| kiwisolver | 1.5.1 | BSD License | Transitive dependency of ANTs environment | [Project](https://github.com/nucleic/kiwi) |
| matplotlib | 3.11.2 | Matplotlib license (PSF-based); additional bundled notices | Transitive dependency of ANTs environment | [Project](https://matplotlib.org) |
| narwhals | 2.26.0 | MIT | Transitive dependency of ANTs environment | [Project](https://github.com/narwhals-dev/narwhals) |
| numpy | 2.3.5 | BSD License | Transitive dependency of ANTs environment | [Project](https://numpy.org) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | Transitive dependency of ANTs environment | [Project](https://github.com/pypa/packaging) |
| pandas | 3.0.6 | BSD License | Transitive dependency of ANTs environment | [Project](https://pandas.pydata.org) |
| patsy | 1.0.3 | 2-clause BSD | Transitive dependency of ANTs environment | [Project](https://github.com/pydata/patsy) |
| pillow | 12.3.0 | MIT-CMU | Transitive dependency of ANTs environment | [Project](https://python-pillow.github.io) |
| pyparsing | 3.3.3 | MIT | Transitive dependency of ANTs environment | [Project](https://github.com/pyparsing/pyparsing/) |
| python-dateutil | 2.9.0.post0 | Apache-2.0 / BSD-3-Clause; see per-contribution terms | Transitive dependency of ANTs environment | [Project](https://github.com/dateutil/dateutil) |
| PyYAML | 6.0.3 | MIT | Transitive dependency of ANTs environment | [Project](https://github.com/yaml/pyyaml) |
| requests | 2.34.2 | Apache-2.0 | Transitive dependency of ANTs environment | [Project](https://github.com/psf/requests) |
| scikit-learn | 1.9.1 | BSD-3-Clause | Transitive dependency of ANTs environment | [Project](https://scikit-learn.org) |
| scipy | 1.15.3 | BSD License | Transitive dependency of ANTs environment | [Project](https://scipy.org/) |
| six | 1.17.0 | MIT | Transitive dependency of ANTs environment | [Project](https://github.com/benjaminp/six) |
| statsmodels | 0.15.0 | BSD-3-Clause | Transitive dependency of ANTs environment | [Project](https://www.statsmodels.org) |
| threadpoolctl | 3.7.0 | BSD-3-Clause | Transitive dependency of ANTs environment | [Project](https://github.com/joblib/threadpoolctl) |
| typing_extensions | 4.16.0 | PSF-2.0 | Transitive dependency of ANTs environment | [Project](https://github.com/python/typing_extensions) |
| tzdata | 2026.4 | Apache-2.0 | Transitive dependency of ANTs environment | [Project](https://github.com/python/tzdata) |
| urllib3 | 2.8.0 | MIT | Transitive dependency of ANTs environment | [Project](https://urllib3.readthedocs.io) |
| webcolors | 25.10.0 | BSD-3-Clause | Transitive dependency of ANTs environment | [Project](https://github.com/ubernostrum/webcolors) |
| wrapt | 2.4.1 | BSD-2-Clause | Transitive dependency of ANTs environment | [Project](https://github.com/GrahamDumpleton/wrapt) |

## License details that the package table alone does not settle

- **Qt:** The application imports PySide6, not PyQt. Installed
  PySide6-Essentials and shiboken6 declare alternative LGPL/GPL licenses (`OR`).
  That expression is not an instruction to apply all alternatives simultaneously.
  Use under LGPL requires compliance with its terms. The application uses QtCore,
  QtGui, QtWidgets and, in GUI checks, QtTest. If Qt libraries or an executable
  bundle are redistributed, provide the required notices, license texts,
  corresponding-source arrangements and replacement/relinking rights as applicable.
  [Qt for Python](https://doc.qt.io/qtforpython-6/),
  [Qt licensing](https://doc.qt.io/qt-6/licensing.html).
- **FFmpeg:** imageio-ffmpeg's Python wrapper is BSD-2-Clause. The executable
  found in this Windows environment identifies itself as
  `7.1-essentials_build-www.gyan.dev`, **GPL-3.0-or-later**, built with
  `--enable-gpl --enable-version3` and libx264. The application launches it as a
  separate process and supplies raw RGB frames over standard input
  (`brain_viewer/result_export.py`). The proposed source release does not bundle
  the executable or the virtual environment. Installing the wheel can obtain
  that executable separately. Shipping an EXE, installer or environment archive
  requires a new review of FFmpeg's source and notice obligations; a BSD label
  for the wrapper does not cover the executable.
  [FFmpeg licensing](https://ffmpeg.org/legal.html),
  [GNU FAQ on aggregates and communication](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation).
- **dcm2niix:** The installed license grants BSD-2-Clause terms for the core and
  lists separately licensed components. The full installed notice remains
  visible in the raw main-environment table. A bundled converter would require
  the notices and conditions for its actual build.
- **MPL and other licenses:** tqdm declares MPL-2.0 AND MIT; certifi in the ANTs
  environment declares MPL-2.0. Matplotlib has its own PSF-based license.
  SimpleITK and ANTsPy use Apache-2.0. These packages should not all be labelled
  MIT/BSD or described as imposing no conditions.
- Wheel files can include native libraries, fonts, data and additional license
  texts that are not fully represented by one metadata field. The proposed
  source release distributes requirements, not copies of those wheels.

## Code and design provenance

| Source | Relationship to RC20 | Distribution treatment |
| --- | --- | --- |
| Brainstorm `figure_topo.m`, `figure_3d.m`, `panel_ieeg.m`, `bst_colormaps.m` | Locally saved reference source, with upstream GPLv3 and copyright headers | Stored under excluded `private_reports`; not included in the proposed source release |
| Brainstorm `bst_shepards.m`, `panel_surface.m`, `bst_get.m`, ECoG tutorial | Documented consultation for visualization and distance/neighbor settings | Keep design references; `result_surface.py` uses a different inverse-square weighting and missing-value policy; see the audit for evidence and limitations |
| Old Brainstorm-format reader | `brain_viewer/brainstorm_import.py` reads MAT files with SciPy; referenced by `tests/test_multimodal.py`, not by the current application entry path | Disclosed legacy code; format reading is not importing the Brainstorm software. This file has not been removed in the license-preparation change |
| Slicer / MNE documentation | Coordinate-system and interface references in design documents | No import of these packages or identified copied implementation in the reviewed application; this is not a whole-upstream similarity audit |
| MMVT / IntrAnat / CAT12 | No corresponding import or provenance keyword found in app/test source | No positive evidence of inclusion in this review; absence of keywords is not proof of absence of adaptation |
| FreeSurferColorLUT | Selected identifiers/names embedded in `anatomy.py` and anatomical identifiers in `segmentation.py` | Modified data extracts; preserve NOTICE and LICENSES/FreeSurfer-LICENSE.txt |

The review found no positive evidence of direct Brainstorm source copying or a
line-for-line MATLAB port in the compared current display/interpolation code.
It does not establish a clean-room development history or exclude every form
of adaptation. The detailed comparison and its limits are in
[LICENSE_AUDIT_RC20.md](docs/LICENSE_AUDIT_RC20.md).

References:
[Brainstorm](https://github.com/brainstorm-tools/brainstorm3),
[interpolation source](https://github.com/brainstorm-tools/brainstorm3/blob/master/toolbox/math/bst_shepards.m),
[ECoG/sEEG tutorial](https://neuroimage.usc.edu/brainstorm/Tutorials/ECoG#Interpolate_on_the_anatomy).

## Data, label tables and atlases

No MNI/ICBM152 volume or complete atlas was found in the proposed `brain_viewer`
and `tests` source trees. The built-in demo is generated synthetically. This
statement does not cover all data inside dependency packages or private input
folders. DIPY's internal direction-sampling data, for example, belong to that
dependency rather than a patient atlas bundled with this application.

The thalamic mapping contains 29 names corresponding to 58 left/right entries
in the official FreeSurfer lookup table. It and the preset anatomical identifiers
are modified extracts, not independently invented identifiers. See
[NOTICE](NOTICE) and the unmodified
[FreeSurfer Software License Agreement](LICENSES/FreeSurfer-LICENSE.txt).
Patient-specific FreeSurfer outputs, MRI, CT, analysis Excel files and saved
cases are user-supplied inputs and are not licensed or redistributed by this
inventory. Any later distribution of datasets or images needs its own source,
permission and attribution record.

## Reproduction of the inventory

The project-specific PowerShell commands and an explanation of `pip freeze`,
`pip-licenses`, `grep` and `rg` are in
[GITHUB_ZENODO_RC20.md](docs/GITHUB_ZENODO_RC20.md).
Do not overwrite this manually annotated file with a fresh unannotated command
output. Regenerate raw tables separately and update this record for the actual
release version.
