"""Local image I/O. Geometry is always expressed in scanner RAS millimetres."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import stat
from typing import Callable
from uuid import uuid4

import nibabel as nib
import numpy as np
from .analysis_results import AnalysisResult, validate_result, result_record
from nibabel.processing import resample_from_to, resample_to_output


class InputError(ValueError):
    """A user-readable error that does not include identifying source paths."""


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class Contact:
    name: str
    group: str
    position: np.ndarray
    color: tuple[float, float, float]
    source_position: np.ndarray | None = None
    kind: str = "SEEG"
    ct_position: np.ndarray | None = None
    provenance: str = "unspecified"
    status: str = "unreviewed"
    evidence: dict = field(default_factory=dict)
    uid: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class MRILayer:
    uid: str
    name: str
    sequence: str
    data: np.ndarray
    valid: np.ndarray
    raw: np.ndarray
    raw_affine: np.ndarray
    to_reference: np.ndarray
    quality: dict = field(default_factory=dict)
    window: float = 1000.
    level: float = 500.
    # Optional pull transform in reference LPS mm -> rigid-aligned image LPS mm.
    # to_reference remains the native-to-reference RIGID component, never a warp.
    deformation: dict | None = None


@dataclass
class Segment:
    uid: str
    name: str
    preset: str
    mask: np.ndarray
    color: tuple[float, float, float]
    visible_3d: bool = True
    visible_2d: bool = True
    opacity: float = .75
    provenance: dict = field(default_factory=dict)
    edits: list[dict] = field(default_factory=list)
    revision: int = 0


@dataclass
class Scene:
    data: np.ndarray
    affine: np.ndarray
    source_affine: np.ndarray
    source_shape: tuple[int, int, int]
    surfaces: dict[str, Mesh]
    kind: str = "local_mri"
    quality: dict = field(default_factory=dict)
    view_state: dict = field(default_factory=dict)
    raw_mri: np.ndarray | None = None
    extra_mris: list[MRILayer] = field(default_factory=list)
    segmentations: list[Segment] = field(default_factory=list)
    results: list[AnalysisResult] = field(default_factory=list)
    diffusions: list = field(default_factory=list)
    tracts: list = field(default_factory=list)
    # Uncommitted extraction preview; deliberately excluded from saved scenes.
    segmentation_preview: Segment | None = field(default=None, repr=False)
    label_volume: np.ndarray | None = None
    label_names: dict[int, str] = field(default_factory=dict)
    label_source: str = ""
    ct: np.ndarray | None = None
    ct_valid: np.ndarray | None = None
    ct_to_mri: np.ndarray | None = None
    ct_quality: dict = field(default_factory=dict)
    contacts: list[Contact] = field(default_factory=list)
    electrode_quality: dict = field(default_factory=dict)
    raw_ct: np.ndarray | None = None
    raw_ct_affine: np.ndarray | None = None
    nuclei: np.ndarray | None = None
    nuclei_affine: np.ndarray | None = None
    nuclei_names: dict[int, str] = field(default_factory=dict)
    nuclei_source: str = ""
    nuclei_display: np.ndarray | None = None

    @property
    def spacing(self) -> np.ndarray:
        return np.diag(self.affine)[:3]

    def world(self, ijk) -> np.ndarray:
        return nib.affines.apply_affine(self.affine, ijk)

    def index(self, world) -> np.ndarray:
        return nib.affines.apply_affine(np.linalg.inv(self.affine), world)

    def native_index(self, world) -> np.ndarray:
        return nib.affines.apply_affine(np.linalg.inv(self.source_affine), world)

    def initial_index(self) -> np.ndarray:
        cortical = [m.vertices for k, m in self.surfaces.items() if k.startswith("pial_")]
        if cortical:
            bounds = np.concatenate(cortical)
            center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
            return np.clip(np.rint(self.index(center)).astype(int), 0, np.array(self.data.shape) - 1)
        return np.array(self.data.shape, dtype=int) // 2

    def contrast_limits(self) -> tuple[float, float]:
        sample = self.data[::3, ::3, ::3]
        nonzero = sample[sample > 0]
        if nonzero.size < 20:
            nonzero = sample[np.isfinite(sample)]
        high = float(np.percentile(nonzero, 99.4)) if nonzero.size else 1.0
        return 0.0, max(high, 1.0)

    def annotation(self, ijk) -> tuple[int, str]:
        from .i18n import tr
        nucleus, name = self.nucleus_at(self.world(ijk))
        if nucleus:
            return nucleus, name
        if self.label_volume is None:
            return 0, tr("解剖ラベル未読込")
        index = np.asarray(ijk, dtype=int)
        if np.any(index < 0) or np.any(index >= self.data.shape):
            return 0, tr("撮像範囲外")
        label = int(self.label_volume[tuple(index)])
        return label, self.label_names.get(label, f"Label {label}") if label else tr("背景・ラベルなし")

    def nucleus_at(self, world):
        if self.nuclei is None:
            return 0, ""
        index = np.rint(nib.affines.apply_affine(np.linalg.inv(self.nuclei_affine), world)).astype(int)
        if np.any(index < 0) or np.any(index >= self.nuclei.shape):
            return 0, ""
        label = int(self.nuclei[tuple(index)])
        return label, self.nuclei_names.get(label, f"Thalamus {label}") if label else ""

    def contact_positions(self, source=False) -> np.ndarray:
        return np.array([c.source_position if source and c.source_position is not None else c.position
                         for c in self.contacts], dtype=float).reshape(-1, 3)


def _is_regular(path: Path) -> bool:
    try:
        info = path.lstat()
        return stat.S_ISREG(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 1024)
    except OSError:
        return False


def discover_inputs(root: Path) -> tuple[Path, Path]:
    if not root.is_dir():
        raise InputError("MRIの入力フォルダがありません。MRIを開く操作で指定してください。")
    fs_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith("_fs")]
    mris: list[Path] = []
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name.endswith(("_fs", "_bs")):
            continue
        for sidecar in folder.rglob("*.json"):
            if not _is_regular(sidecar):
                continue
            try:
                if json.loads(sidecar.read_text(encoding="utf-8-sig")).get("Modality") == "MR":
                    for extension in (".nii.gz", ".nii"):
                        candidate = sidecar.with_suffix(extension)
                        if _is_regular(candidate):
                            mris.append(candidate)
            except (ValueError, OSError, UnicodeError):
                continue
    mris = list(dict.fromkeys(mris))
    if len(mris) != 1 or len(fs_dirs) != 1:
        raise InputError("入力候補を1組に特定できません。MRIを開く操作でMRIとFreeSurferフォルダを指定してください。")
    return mris[0], fs_dirs[0]


def fs_to_scanner(orig: nib.MGHImage) -> np.ndarray:
    """FreeSurfer surface tkregister RAS -> scanner RAS; never use MNI here."""
    return orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())


def _check_surface_volume(metadata: dict, orig: nib.MGHImage) -> None:
    # FreeSurfer commonly writes "1  # volume info valid" on this line.
    if not metadata or str(metadata.get("valid", "0")).strip().split(maxsplit=1)[:1] != ["1"]:
        raise InputError("脳表の座標情報が不足しています。対応するFreeSurfer出力を確認してください。")
    volume = np.asarray(metadata["volume"], dtype=float)
    spacing = np.asarray(metadata["voxelsize"], dtype=float)
    matrix = np.column_stack([metadata[axis] for axis in ("xras", "yras", "zras")]) @ np.diag(spacing)
    affine = np.eye(4)
    affine[:3, :3] = matrix
    affine[:3, 3] = np.asarray(metadata["cras"]) - matrix @ (volume / 2)
    if not np.array_equal(volume, orig.shape[:3]) or not np.allclose(affine, orig.affine, atol=0.002):
        raise InputError("脳表とFreeSurfer MRIの座標情報が一致しません。重ね合わせを中止しました。")


def validate_scene(scene: Scene) -> None:
    from .diffusion import validate_diffusion
    validate_diffusion(scene)
    if scene.data.ndim != 3 or min(scene.data.shape) < 2 or scene.data.size > 300_000_000:
        raise InputError("対応していないMRIサイズです。")
    if not np.isfinite(scene.data).all():
        raise InputError("MRIに有限でない値があります。")
    for affine in (scene.affine, scene.source_affine):
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
            raise InputError("MRIの座標変換が不正です。")
        if not np.allclose(affine[3], [0, 0, 0, 1]):
            raise InputError("MRIの座標変換が不正です。")
    if np.any(scene.spacing <= 0) or not np.allclose(scene.affine[:3, :3], np.diag(scene.spacing)):
        raise InputError("表示用MRIがRAS方向に整列していません。")
    if len(scene.source_shape) != 3 or min(scene.source_shape) < 2:
        raise InputError("元MRIのサイズ情報が不正です。")
    if scene.raw_mri is not None and (scene.raw_mri.shape != scene.source_shape or not np.isfinite(scene.raw_mri).all()):
        raise InputError('基準MRIの元画像が不正です。')
    if len({layer.uid for layer in scene.extra_mris}) != len(scene.extra_mris):
        raise InputError('追加MRIの識別子が重複しています。')
    if len({segment.uid for segment in scene.segmentations}) != len(scene.segmentations):
        raise InputError('領域の識別子が重複しています。')
    if len({result.uid for result in scene.results}) != len(scene.results):
        raise InputError('解析結果の識別子が重複しています。')
    for result in scene.results:
        validate_result(result)
    for segment in scene.segmentations:
        if (segment.mask.shape != scene.data.shape or segment.mask.dtype != np.bool_
                or not segment.uid.startswith('seg_') or not segment.uid[4:].isalnum()
                or not isinstance(segment.name,str) or not segment.name.strip()
                or len(segment.color)!=3 or not np.isfinite(segment.color).all()
                or np.any(np.asarray(segment.color)<0) or np.any(np.asarray(segment.color)>1)
                or not np.isfinite(segment.opacity) or not 0<=segment.opacity<=1):
            raise InputError('セグメンテーションの画像・表示設定が不正です。')
    for layer in scene.extra_mris:
        if layer.deformation is not None:
            if layer.deformation.get('type')=='DisplacementField':
                from .postop_deformation import validate_field
                validate_field(layer.deformation)
            else:
                from .postop_registration import read_deformation
                read_deformation(layer.deformation)
        if (layer.data.shape != scene.data.shape or layer.valid.shape != scene.data.shape
                or not np.isfinite(layer.data).all() or layer.raw.ndim != 3 or min(layer.raw.shape) < 2
                or not np.isfinite(layer.raw).all() or not layer.uid.startswith('mr_')
                or not layer.uid[3:].isalnum() or not np.isfinite([layer.window, layer.level]).all() or layer.window <= 0):
            raise InputError('追加MRIの画像または表示設定が不正です。')
        for matrix in (layer.raw_affine, layer.to_reference):
            if (matrix.shape != (4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0,0,0,1])
                    or abs(np.linalg.det(matrix[:3,:3])) < 1e-8):
                raise InputError('追加MRIの座標変換が不正です。')
        if not np.allclose(layer.to_reference[:3,:3].T @ layer.to_reference[:3,:3], np.eye(3), atol=1e-4) or np.linalg.det(layer.to_reference[:3,:3]) < .99:
            raise InputError('追加MRIの変換が剛体変換ではありません。')
    if not all(key in scene.surfaces for key in ("pial_lh", "pial_rh")):
        raise InputError("左右の脳表が揃っていません。")
    for key, mesh in scene.surfaces.items():
        if key not in ("pial_lh", "pial_rh", "white_lh", "white_rh"):
            raise InputError("未対応の脳表の種類です。")
        if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3 or not np.isfinite(mesh.vertices).all():
            raise InputError("脳表の頂点データが不正です。")
        if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3 or mesh.faces.dtype.kind not in "iu":
            raise InputError("脳表の面データが不正です。")
        if not len(mesh.faces) or mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
            raise InputError("脳表の頂点番号が不正です。")
    if scene.label_volume is not None:
        if scene.label_volume.shape != scene.data.shape or scene.label_volume.dtype.kind not in "iu":
            raise InputError("解剖ラベルの格子がMRIと一致しません。")
    if scene.ct is not None:
        if scene.ct.shape != scene.data.shape or not np.isfinite(scene.ct).all():
            raise InputError("CTの表示格子が不正です。")
        if scene.ct_valid is None or scene.ct_valid.shape != scene.data.shape:
            raise InputError("CTの撮像範囲の情報がありません。")
        matrix = scene.ct_to_mri
        if matrix is None or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise InputError("CTの座標変換が不正です。")
        if not np.allclose(matrix[3], [0, 0, 0, 1]) or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4) or np.linalg.det(matrix[:3, :3]) < .99:
            raise InputError("CTの座標変換が剛体変換ではありません。")
    if len(scene.contacts) > 10000:
        raise InputError("電極数が対応範囲を超えています。")
    if len({c.uid for c in scene.contacts}) != len(scene.contacts) or len({c.name for c in scene.contacts}) != len(scene.contacts):
        raise InputError("コンタクトの名前または識別子が重複しています。")
    for contact in scene.contacts:
        for point in (contact.position, contact.source_position, contact.ct_position):
            if point is not None and (np.asarray(point).shape != (3,) or not np.isfinite(point).all()):
                raise InputError("電極座標が不正です。")
        if len(contact.color) != 3 or not np.isfinite(contact.color).all():
            raise InputError("電極の表示色が不正です。")
        if contact.status not in ("unreviewed", "uncertain", "reviewed"):
            raise InputError("コンタクトの確認状態が不正です。")
        if contact.provenance == "native_ct_only":
            if contact.ct_position is None or scene.raw_ct is None or scene.ct_to_mri is None:
                raise InputError("CT由来コンタクトの元画像・座標がありません。")
            if not np.allclose(contact.position, nib.affines.apply_affine(scene.ct_to_mri,contact.ct_position), atol=1e-4, rtol=0):
                raise InputError("コンタクトのCT座標とMRI座標が一致しません。")
    for data, affine in ((scene.raw_ct, scene.raw_ct_affine), (scene.nuclei, scene.nuclei_affine)):
        if data is not None:
            if data.ndim != 3 or not np.isfinite(data).all() or affine is None or affine.shape != (4, 4) or not np.isfinite(affine).all():
                raise InputError("元CTまたは視床核の画像・座標が不正です。")
            if abs(np.linalg.det(affine[:3, :3])) < 1e-8 or not np.allclose(affine[3], [0, 0, 0, 1]):
                raise InputError("元CTまたは視床核の座標変換が不正です。")
    if scene.nuclei is not None and (scene.nuclei.dtype.kind not in "iu" or scene.nuclei_display is None or scene.nuclei_display.shape != scene.data.shape):
        raise InputError("視床核の表示格子が不正です。")


def alignment_quality(source: nib.spatialimages.SpatialImage, orig: nib.MGHImage) -> dict:
    """Compare same-acquisition volumes on a coarse physical grid before overlay."""
    affine = source.affine.copy()
    affine[:3, :3] *= 3
    shape = tuple(int(x) for x in (np.array(source.shape[:3]) + 2) // 3)
    original = np.asarray(source.dataobj, dtype=np.float32)[::3, ::3, ::3]
    fs = resample_from_to(orig, (shape, affine), order=1).get_fdata(dtype=np.float32)
    mask = (original > max(0, np.percentile(original, 35))) & (fs > max(0, np.percentile(fs, 35)))
    if mask.sum() < 1000 or np.std(original[mask]) == 0 or np.std(fs[mask]) == 0:
        raise InputError("MRIと脳表の位置の対応を確認できませんでした。")
    corr = float(np.corrcoef(original[mask], fs[mask])[0, 1])
    if not np.isfinite(corr) or corr < 0.80:
        raise InputError("MRIとFreeSurferの位置・画像の対応が不十分です。この試作では同じ撮像に由来するMRIを指定してください。")
    return {"same_acquisition_correlation": round(corr, 6), "comparison_voxels": int(mask.sum())}


def validate_freesurfer_folder(folder: Path) -> Path:
    """Check the selected folder before loading any image or altering patient work."""
    if folder is None or not (Path(folder) / "mri" / "orig.mgz").is_file():
        raise InputError("FreeSurferのmri/orig.mgzが見つかりません。MRIの撮像フォルダではなく、同じ患者の処理結果でmriとsurfを含むフォルダ（通常は末尾が_fs）を選択してください。")
    folder = Path(folder)
    for hemi in ("lh", "rh"):
        if not any(_is_regular(folder / "surf" / name) for name in (f"{hemi}.pial", f"{hemi}.pial.T1")):
            raise InputError("FreeSurferの左右の脳表が揃っていません。選択したフォルダのsurf内にlh.pialとrh.pial、または各pial.T1が必要です。")
    return folder


def load_mri(mri_path: Path, fs_dir: Path, progress: Callable[[str], None] = lambda _: None) -> Scene:
    fs_dir = validate_freesurfer_folder(fs_dir)
    progress("MRIと脳表の座標を確認しています…")
    from .volume_io import read_volume
    source = read_volume(mri_path,modality='MR')
    if len(source.shape) != 3 or source.header.get_xyzt_units()[0] != "mm":
        raise InputError("この試作は、距離単位がmmの3D NIfTI MRIに対応しています。")
    orig = nib.load(str(fs_dir / "mri" / "orig.mgz"))
    quality = alignment_quality(source, orig)
    transform = fs_to_scanner(orig)
    surfaces = {}
    for kind in ("pial", "white"):
        for hemi in ("lh", "rh"):
            progress("FreeSurferの脳表を読み込んでいます…")
            paths = [fs_dir / "surf" / f"{hemi}.{kind}"]
            if kind == "pial":
                paths.append(fs_dir / "surf" / f"{hemi}.pial.T1")
            path = next((p for p in paths if _is_regular(p)), None)
            if path is None:
                if kind == "white":
                    continue
                raise InputError("左右のpial脳表が見つかりません。FreeSurfer出力を確認してください。")
            vertices, faces, metadata = nib.freesurfer.read_geometry(str(path), read_metadata=True)
            _check_surface_volume(metadata, orig)
            vertices = nib.affines.apply_affine(transform, vertices)
            surfaces[f"{kind}_{hemi}"] = Mesh(vertices.astype(np.float32), faces.astype(np.int32))
    progress("傾きを補正した3方向のMRI断面を準備しています…")
    ras = resample_to_output(source, voxel_sizes=(1.0, 1.0, 1.0), order=1)
    data = ras.get_fdata(dtype=np.float32)
    scene = Scene(data, ras.affine, source.affine.copy(), tuple(int(v) for v in source.shape), surfaces,
                  quality=quality, raw_mri=source.get_fdata(dtype=np.float32))
    validate_scene(scene)
    cortex = np.concatenate([v.vertices for k, v in surfaces.items() if k.startswith("pial")])
    native = scene.native_index(cortex)
    quality["surface_inside_source_fraction"] = round(float(np.mean(np.all((native >= -0.5) & (native <= np.array(source.shape) - 0.5), axis=1))), 6)
    if quality["surface_inside_source_fraction"] < 0.98:
        raise InputError("脳表がMRIの撮像範囲から外れています。重ね合わせを中止しました。")
    from .anatomy import attach_anatomy
    progress("FreeSurferの解剖ラベルを読み込んでいます…")
    attach_anatomy(scene, fs_dir)
    return scene


def plane_axes(axis: int) -> tuple[int, int]:
    """Horizontal/vertical dimensions in axial(2), coronal(1), sagittal(0)."""
    return {2: (0, 1), 1: (0, 2), 0: (1, 2)}[axis]


def plane_pixels(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(data, index, axis=axis).T[::-1, ::-1]


def display_position(shape, axis: int, ijk) -> np.ndarray:
    h, v = plane_axes(axis)
    return np.array([shape[h] - 1 - ijk[h], shape[v] - 1 - ijk[v]], dtype=float)


def index_from_display(shape, axis: int, uv, current) -> np.ndarray:
    h, v = plane_axes(axis)
    result = np.array(current, dtype=int).copy()
    result[h] = int(np.clip(np.rint(shape[h] - 1 - uv[0]), 0, shape[h] - 1))
    result[v] = int(np.clip(np.rint(shape[v] - 1 - uv[1]), 0, shape[v] - 1))
    return result


def _contained(folder: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise InputError("保存データの参照先が不正です。")
    resolved = (folder / relative).resolve()
    if not resolved.is_relative_to(folder.resolve()):
        raise InputError("保存データが作業フォルダの外を参照しています。")
    return resolved


def save_scene(scene: Scene, parent: Path, state: dict, asset_root=None) -> Path:
    """New self-contained display bundle; manifest is written last."""
    validate_scene(scene)
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / (datetime.now().strftime("view_%Y%m%d_%H%M%S_") + uuid4().hex[:6])
    folder.mkdir()
    volume = nib.Nifti1Image(scene.data, scene.affine)
    volume.header.set_xyzt_units("mm")
    volume.set_sform(scene.affine, code=1)
    volume.set_qform(scene.affine, code=1)
    from .local_assets import write_asset
    def save_volume(image, path):
        write_asset(path, (np.asanyarray(image.dataobj), image.affine), lambda p: nib.save(image, str(p)), asset_root)
    save_volume(volume, folder / "volume.nii.gz")
    meshes = {}
    for key, mesh in scene.surfaces.items():
        filename = key + ".npz"
        write_asset(folder / filename, (mesh.vertices, mesh.faces),
                    lambda p: np.savez_compressed(p, vertices=mesh.vertices, faces=mesh.faces), asset_root)
        meshes[key] = filename
    manifest = {
        "schema": ("cortex-viewer/11" if any(layer.deformation and layer.deformation.get('type')=='DisplacementField' for layer in scene.extra_mris)
                   else "cortex-viewer/10" if any(layer.deformation is not None for layer in scene.extra_mris) else "cortex-viewer/9"),
        "coordinate_system": "scanner_RAS_mm",
        "volume": "volume.nii.gz", "volume_role": "resampled_display_MRI",
        "source_affine": scene.source_affine.tolist(), "source_shape": list(scene.source_shape),
        "surfaces": meshes, "kind": scene.kind, "quality": scene.quality, "view_state": state,
    }
    for name, data in (("labels", scene.label_volume), ("ct", scene.ct), ("ct_valid", scene.ct_valid)):
        if data is not None:
            image = nib.Nifti1Image(data.astype(np.uint8) if name == "ct_valid" else data, scene.affine)
            image.header.set_xyzt_units("mm")
            save_volume(image, folder / (name + ".nii.gz"))
            manifest[name] = name + ".nii.gz"
    manifest["label_names"] = scene.label_names
    manifest["label_source"] = scene.label_source
    for name, data, affine in (("raw_ct", scene.raw_ct, scene.raw_ct_affine), ("nuclei", scene.nuclei, scene.nuclei_affine),
                               ('raw_mri', scene.raw_mri, scene.source_affine)):
        if data is not None:
            image = nib.Nifti1Image(data, affine)
            image.header.set_xyzt_units("mm")
            save_volume(image, folder / (name + ".nii.gz"))
            manifest[name] = name + ".nii.gz"
    manifest['extra_mris'] = []
    from .postop_deformation import save_deformation
    for layer in scene.extra_mris:
        item = {'id': layer.uid, 'name': layer.name, 'sequence': layer.sequence,
                'to_reference_ras_mm': layer.to_reference.tolist(), 'quality': layer.quality,
                'window': layer.window, 'level': layer.level,
                'deformation': save_deformation(layer.deformation,folder,layer.uid,asset_root)}
        for key, data, affine in (('data', layer.data, scene.affine), ('valid', layer.valid.astype(np.uint8), scene.affine),
                                  ('raw', layer.raw, layer.raw_affine)):
            filename = layer.uid + '_' + key + '.nii.gz'
            image = nib.Nifti1Image(data, affine)
            image.header.set_xyzt_units('mm')
            save_volume(image, folder / filename)
            item[key] = filename
        manifest['extra_mris'].append(item)
    manifest['segmentations'] = []
    for segment in scene.segmentations:
        filename = segment.uid + '_mask.nii.gz'
        image = nib.Nifti1Image(segment.mask.astype(np.uint8), scene.affine)
        image.header.set_xyzt_units('mm')
        save_volume(image, folder / filename)
        manifest['segmentations'].append({'id':segment.uid,'name':segment.name,'preset':segment.preset,
            'mask':filename,'color':[float(v) for v in segment.color], 'visible_3d':segment.visible_3d,
            'visible_2d':segment.visible_2d,'opacity':float(segment.opacity),'provenance':segment.provenance,
            'edits':segment.edits,'revision':segment.revision})
    manifest['results'] = []
    for result in scene.results:
        filename = result.uid + '.npz'
        write_asset(folder / filename, (result.values, result.times),
                    lambda p: np.savez_compressed(p, values=result.values, times=result.times), asset_root)
        manifest['results'].append({**result_record(result), 'data':filename})
    from .diffusion_storage import save_diffusion
    save_diffusion(scene, folder, manifest, asset_root)
    manifest["nuclei_names"] = scene.nuclei_names
    manifest["nuclei_source"] = scene.nuclei_source
    manifest["ct_to_mri_ras_mm"] = scene.ct_to_mri.tolist() if scene.ct_to_mri is not None else None
    manifest["ct_quality"] = scene.ct_quality
    manifest["electrode_quality"] = scene.electrode_quality
    manifest["contacts"] = [{"name": c.name, "group": c.group, "position_ras_mm": c.position.tolist(),
        "source_position_ras_mm": c.source_position.tolist() if c.source_position is not None else None,
        "color": [float(v) for v in c.color], "kind": c.kind,
        "ct_ras_mm":c.ct_position.tolist() if c.ct_position is not None else None,
        "provenance":c.provenance,"status":c.status,"evidence":c.evidence,"uid":c.uid} for c in scene.contacts]
    (folder / "scene.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return folder


def load_scene(manifest_path: Path) -> Scene:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    if obj.get("schema") not in tuple(f'cortex-viewer/{i}' for i in range(1,12)) or obj.get("coordinate_system") != "scanner_RAS_mm":
        raise InputError("対応していない保存形式です。")
    folder = manifest_path.parent
    volume = nib.load(str(_contained(folder, obj["volume"])))
    surfaces = {}
    for key, relative in obj["surfaces"].items():
        with np.load(_contained(folder, relative), allow_pickle=False) as mesh:
            surfaces[key] = Mesh(mesh["vertices"].copy(), mesh["faces"].copy())
    scene = Scene(volume.get_fdata(dtype=np.float32), volume.affine,
                  np.array(obj["source_affine"], dtype=float), tuple(obj["source_shape"]),
                  surfaces, kind=obj.get("kind", "local_mri"), quality=obj.get("quality", {}),
                  view_state=obj.get("view_state", {}))
    for field, attribute, dtype in (("labels", "label_volume", np.uint16), ("ct", "ct", np.float32), ("ct_valid", "ct_valid", bool)):
        if field in obj:
            image = nib.load(str(_contained(folder, obj[field])))
            if not np.allclose(image.affine, scene.affine, atol=1e-4):
                raise InputError("追加画像の座標がMRIと一致しません。")
            setattr(scene, attribute, np.asarray(image.dataobj).astype(dtype))
    scene.label_names = {int(k): str(v) for k, v in obj.get("label_names", {}).items()}
    if 'raw_mri' in obj:
        image = nib.load(str(_contained(folder, obj['raw_mri'])))
        if not np.allclose(image.affine, scene.source_affine, atol=1e-4):
            raise InputError('基準MRIの元画像の座標が一致しません。')
        scene.raw_mri = image.get_fdata(dtype=np.float32)
    from .postop_deformation import load_deformation
    for item in obj.get('extra_mris', []):
        images = {key: nib.load(str(_contained(folder, item[key]))) for key in ('data', 'valid', 'raw')}
        if any(not np.allclose(images[key].affine, scene.affine, atol=1e-4) for key in ('data', 'valid')):
            raise InputError('追加MRIの表示格子の座標が一致しません。')
        scene.extra_mris.append(MRILayer(item['id'], item['name'], item['sequence'], images['data'].get_fdata(dtype=np.float32),
            np.asarray(images['valid'].dataobj).astype(bool), images['raw'].get_fdata(dtype=np.float32), images['raw'].affine,
            np.asarray(item['to_reference_ras_mm'], dtype=float), item.get('quality', {}), float(item['window']), float(item['level']),
            load_deformation(item.get('deformation'),folder)))
    scene.label_source = obj.get("label_source", "")
    for item in obj.get('segmentations', []):
        image = nib.load(str(_contained(folder, item['mask'])))
        mask = np.asanyarray(image.dataobj)
        if (not np.allclose(image.affine, scene.affine, atol=1e-4) or mask.shape != scene.data.shape
                or not np.isin(mask,[0,1]).all()):
            raise InputError('領域マスクの座標または画素値が不正です。')
        scene.segmentations.append(Segment(item['id'],item['name'],item['preset'],mask.astype(bool),tuple(item['color']),
            bool(item.get('visible_3d',True)),bool(item.get('visible_2d',True)),float(item.get('opacity',.75)),
            item.get('provenance',{}),item.get('edits',[]),int(item.get('revision',0))))
    for name in ("raw_ct", "nuclei"):
        if name in obj:
            image = nib.load(str(_contained(folder, obj[name])))
            setattr(scene, name, np.asarray(image.dataobj).astype(np.uint16 if name == "nuclei" else np.float32))
            setattr(scene, name + "_affine", image.affine.copy())
    scene.nuclei_names = {int(k): str(v) for k, v in obj.get("nuclei_names", {}).items()}
    scene.nuclei_source = obj.get("nuclei_source", "")
    if scene.nuclei is not None:
        from .anatomy import prepare_nuclei_display
        prepare_nuclei_display(scene)
    if obj.get("ct_to_mri_ras_mm") is not None:
        scene.ct_to_mri = np.asarray(obj["ct_to_mri_ras_mm"], dtype=float)
    scene.ct_quality = obj.get("ct_quality", {})
    scene.electrode_quality = obj.get("electrode_quality", {})
    records=obj.get('contacts',[])
    if scene.electrode_quality.get('source','').startswith('Brainstorm'):
        records=[]
        scene.electrode_quality={'source':'none','legacy_contacts_ignored':True}
    scene.contacts = [Contact(str(c["name"]), str(c["group"]), np.array(c["position_ras_mm"], dtype=float),
        tuple(c["color"]), np.array(c["source_position_ras_mm"], dtype=float) if c.get("source_position_ras_mm") is not None else None,
        c.get("kind", "SEEG"),np.array(c['ct_ras_mm'],dtype=float) if c.get('ct_ras_mm') is not None else None,
        c.get('provenance','unspecified'),c.get('status','unreviewed'),c.get('evidence',{}),c.get('uid',uuid4().hex)) for c in records]
    for item in obj.get('results',[]):
        with np.load(_contained(folder,item['data']),allow_pickle=False) as data:
            kwargs={key:value for key,value in item.items() if key!='data'}
            scene.results.append(AnalysisResult(**kwargs,values=data['values'].copy(),times=data['times'].copy()))
    from .diffusion_storage import load_diffusion
    load_diffusion(scene, folder, obj)
    validate_scene(scene)
    return scene


def make_demo() -> Scene:
    """Entirely synthetic phantom: no patient geometry or pixel intensities."""
    shape = (120, 140, 120)
    spacing = np.array([1.5, 1.5, 1.5])
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = -np.array(shape) * spacing / 2
    x, y, z = np.meshgrid(*[(np.arange(n) - n / 2) * s for n, s in zip(shape, spacing)], indexing="ij")
    r = np.sqrt((x / 73) ** 2 + (y / 88) ** 2 + (z / 75) ** 2)
    data = np.zeros(shape, dtype=np.float32)
    data[r < 1] = 300
    data[r < .9] = 100
    data[r < .84] = 630
    data[r < .69] = 940
    ventricles = ((x / 16) ** 2 + (y / 37) ** 2 + (z / 13) ** 2 < 1)
    data[ventricles] = 70
    data += np.where(data > 350, 65 * np.sin(y / 5) * np.cos(z / 7), 0).astype(np.float32)
    surfaces = {}
    nlat, nlon = 60, 96
    theta = np.linspace(.001, np.pi - .001, nlat)
    phi = np.arange(nlon) * (2 * np.pi / nlon)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    for hemi, sign in (("lh", -1), ("rh", 1)):
        for kind, scale in (("pial", 1), ("white", .88)):
            ripple = 1 + .025 * np.sin(12 * p) * np.sin(14 * t)
            vertices = np.column_stack([
                (sign * 28 + 26 * scale * np.sin(t) * np.cos(p) * ripple).ravel(),
                (70 * scale * np.sin(t) * np.sin(p) * ripple).ravel(),
                (58 * scale * np.cos(t) * ripple).ravel(),
            ]).astype(np.float32)
            faces = []
            for a in range(nlat - 1):
                for b in range(nlon):
                    c = (b + 1) % nlon
                    faces.extend(((a * nlon + b, a * nlon + c, (a + 1) * nlon + b),
                                  (a * nlon + c, (a + 1) * nlon + c, (a + 1) * nlon + b)))
            surfaces[f"{kind}_{hemi}"] = Mesh(vertices, np.array(faces, dtype=np.int32))
    scene = Scene(data, affine, affine.copy(), shape, surfaces, kind="synthetic")
    labels = np.zeros(shape, dtype=np.uint16)
    labels[(r < .84) & (x < 0)] = 1001
    labels[(r < .84) & (x >= 0)] = 2001
    labels[ventricles] = 4
    scene.label_volume = labels
    scene.label_names = {0: "Background", 1001: "Synthetic left region", 2001: "Synthetic right region", 4: "Synthetic ventricle"}
    scene.label_source = "Synthetic phantom"
    scene.ct = np.full(shape, -1000., dtype=np.float32)
    scene.ct[r < 1] = 30
    scene.ct[(r > .88) & (r < .96)] = 1200
    scene.ct_valid = np.ones(shape, dtype=bool)
    scene.ct_to_mri = np.eye(4)
    scene.ct_quality = {"method": "synthetic identity", "review_status": "synthetic"}
    for sign, group, color in ((-1, "Demo-L", (.95, .66, .25)), (1, "Demo-R", (.25, .8, .92))):
        for index in range(1, 9):
            point = np.array([sign * (6 + index * 5), 9., 15.])
            scene.contacts.append(Contact(f"{group}-{index:02d}", group, point, color,
                ct_position=point.copy(), provenance="native_ct_only", evidence={"supported": True, "peak_hu": 3000, "origin":"synthetic"}))
            metal = ((x-point[0])**2+(y-point[1])**2+(z-point[2])**2) < 3
            scene.ct[metal] = 3000
    scene.raw_ct = scene.ct.copy()
    scene.raw_ct_affine = affine.copy()
    scene.electrode_quality = {"source":"native_ct_only", "groups":{g:{"expected_count":8,"status":"unreviewed"} for g in ("Demo-L","Demo-R")}}
    scene.ct_quality["automatic_ct_to_mri_ras_mm"] = np.eye(4).tolist()
    scene.nuclei = np.zeros(shape, dtype=np.uint16)
    scene.nuclei[((x+12)/8)**2 + ((y+3)/12)**2 + (z/9)**2 < 1] = 8103
    scene.nuclei[((x-12)/8)**2 + ((y+3)/12)**2 + (z/9)**2 < 1] = 8203
    scene.nuclei_affine = affine.copy()
    scene.nuclei_names = {8103: "Synthetic Left-AV", 8203: "Synthetic Right-AV"}
    scene.nuclei_source = "Synthetic phantom"
    scene.nuclei_display = scene.nuclei.copy()
    labels[scene.nuclei == 8103] = 10
    labels[scene.nuclei == 8203] = 49
    callosum = (np.abs(x)<4) & ((y/40)**2+((z-8)/23)**2<1) & ((y/34)**2+((z-8)/16)**2>1) & (z>10)
    labels[callosum] = 251
    scene.label_names.update({10:'Synthetic left thalamus',49:'Synthetic right thalamus',251:'Synthetic callosum'})
    validate_scene(scene)
    return scene
