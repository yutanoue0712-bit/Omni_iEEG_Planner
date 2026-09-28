"""Patient-specific FreeSurfer volume labels; labels are never interpolated linearly."""
from pathlib import Path
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .imaging import InputError, _is_regular

# Integer IDs from FreeSurferColorLUT.txt (Iglesias thalamic atlas).
# https://github.com/freesurfer/freesurfer/blob/dev/distribution/FreeSurferColorLUT.txt
# Selected/reformatted label names: see NOTICE and LICENSES/FreeSurfer-LICENSE.txt.
THALAMIC_NAMES = {3: "AV", 4: "CeM", 5: "CL", 6: "CM", 8: "LD", 9: "LGN", 10: "LP",
    11: "L-Sg", 12: "MDl", 13: "MDm", 15: "MGN", 16: "MV(Re)", 17: "Pc", 18: "Pf",
    19: "Pt", 20: "PuA", 21: "PuI", 22: "PuL", 23: "PuM", 25: "R", 26: "VA",
    27: "VAmc", 28: "VLa", 29: "VLp", 30: "VM", 33: "VPL", 34: "PaV", 35: "PuMm", 36: "PuMl"}


def nucleus_color(label):
    import colorsys
    # Same nucleus has the same colour in both hemispheres.
    return colorsys.hsv_to_rgb(((int(label) % 100) * .61803398875) % 1, .58, .96)


def prepare_nuclei_display(scene):
    if scene.nuclei is None:
        return
    image = nib.Nifti1Image(scene.nuclei, scene.nuclei_affine)
    output = resample_from_to(image, (scene.data.shape, scene.affine), order=0)
    scene.nuclei_display = np.asarray(output.dataobj, dtype=np.uint16)


def attach_nuclei(scene, fs_dir):
    paths = [p for p in (fs_dir / "mri").glob("ThalamicNuclei*.mgz")
             if _is_regular(p) and "FSvoxelSpace" not in p.name]
    preferred = [p for p in paths if p.name in ("ThalamicNuclei.mgz", "ThalamicNuclei.v12.T1.mgz")]
    choices = preferred or paths
    if not choices:
        return
    if len(choices) != 1:
        raise InputError("視床核の処理結果が複数あり、使用するデータを特定できません。")
    image = nib.load(str(choices[0]))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.isfinite(data).all() or not np.equal(data, np.rint(data)).all():
        raise InputError("視床核のラベル画像が不正です。")
    labels = np.unique(data[data > 0]).astype(int)
    if not len(labels) or not all(8100 < n < 8300 for n in labels):
        raise InputError("対応していない視床核のラベル番号です。")
    scene.nuclei = data.astype(np.uint16)
    scene.nuclei_affine = image.affine.copy()
    scene.nuclei_names = {int(n): ("Left-" if n < 8200 else "Right-") + THALAMIC_NAMES.get(int(n) % 100, str(n)) for n in labels}
    scene.nuclei_source = "FreeSurfer / " + choices[0].name
    prepare_nuclei_display(scene)
    mask = scene.nuclei_display > 0
    overlap = float(np.mean(np.isin(scene.label_volume[mask], [10, 49]))) if mask.any() else 0.
    if overlap < .5:
        raise InputError("視床核の位置がMRIの視床ラベルと十分に重なりません。")
    scene.quality["thalamic_nuclei"] = {"labels": len(labels), "native_spacing_mm": [float(v) for v in image.header.get_zooms()],
        "overlap_with_aseg_thalamus_fraction": overlap, "display_spacing_mm": scene.spacing.tolist()}


def freesurfer_names(folder: Path) -> dict[int, str]:
    names = {0: "Unknown", 2: "Left-Cerebral-White-Matter", 41: "Right-Cerebral-White-Matter"}
    # aseg.stats already includes FreeSurfer's names for the subcortical integer IDs.
    statistics = folder / "stats" / "aseg.stats"
    if _is_regular(statistics):
        for line in statistics.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 5 and fields[0].isdigit() and fields[1].isdigit():
                names[int(fields[1])] = fields[4]
    # The aparc volume uses 1000 + left annotation index, 2000 + right index.
    for hemi, offset in (("lh", 1000), ("rh", 2000)):
        path = folder / "label" / f"{hemi}.aparc.annot"
        if _is_regular(path):
            _, _, labels = nib.freesurfer.read_annot(str(path))
            for index, name in enumerate(labels):
                names[offset + index] = f"ctx-{hemi}-{name.decode('utf-8', errors='replace')}"
    return names


def attach_anatomy(scene, fs_dir: Path):
    path = fs_dir / "mri" / "aparc+aseg.mgz"
    if not _is_regular(path):
        return
    labels = nib.load(str(path))
    orig = nib.load(str(fs_dir / "mri" / "orig.mgz"))
    if labels.shape != orig.shape or not np.allclose(labels.affine, orig.affine, atol=.002):
        raise InputError("解剖ラベルとFreeSurfer MRIの座標が一致しません。")
    output = resample_from_to(labels, (scene.data.shape, scene.affine), order=0)
    scene.label_volume = np.asarray(output.dataobj, dtype=np.uint16)
    scene.label_names = freesurfer_names(fs_dir)
    scene.label_source = "FreeSurfer aparc+aseg / Desikan–Killiany"
    attach_nuclei(scene, fs_dir)
