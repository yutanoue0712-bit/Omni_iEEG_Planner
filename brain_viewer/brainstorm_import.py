"""Read-only import of Brainstorm contacts with explicit MRI and CT provenance."""
from __future__ import annotations

import colorsys
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.ndimage import map_coordinates

from .imaging import Contact, InputError, _is_regular


def transforms(metadata) -> dict[str, np.ndarray]:
    array = np.asarray(metadata.get("InitTransf", []), dtype=object)
    if not array.size:
        return {}
    return {str(row[0]): np.asarray(row[1], dtype=float) for row in array.reshape(-1, 2)
            if np.asarray(row[1]).shape == (4, 4)}


def scs_m_to_ras_mm(metadata) -> np.ndarray:
    rotation = np.asarray(metadata.get("SCS", {}).get("R", []), dtype=float)
    translation = np.asarray(metadata.get("SCS", {}).get("T", []), dtype=float).reshape(-1)
    spacing = np.asarray(metadata.get("Voxsize", []), dtype=float).reshape(-1)
    affine = transforms(metadata).get("vox2ras")
    if rotation.shape != (3, 3) or translation.shape != (3,) or spacing.shape != (3,) or affine is None:
        raise InputError("Brainstormの電極に対応するMRI座標情報が不足しています。")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.any(spacing <= 0):
        raise InputError("BrainstormのMRI座標変換が不正です。")
    mri_to_scs_mm = np.eye(4)
    mri_to_scs_mm[:3, :3] = rotation
    mri_to_scs_mm[:3, 3] = translation
    one_to_zero = np.eye(4)
    one_to_zero[:3, 3] = -1
    # SCS m -> SCS mm -> MRI mm -> one-based voxels -> zero-based voxels -> scanner RAS mm.
    return affine @ one_to_zero @ np.diag([*(1/spacing), 1.]) @ np.linalg.inv(mri_to_scs_mm) @ np.diag([1000., 1000., 1000., 1.])


@dataclass
class ElectrodeInput:
    contacts: list[Contact]
    mri_path: Path
    mri_metadata: dict
    source_matrix: np.ndarray
    ct_data: np.ndarray | None = None
    ct_affine: np.ndarray | None = None
    ct_seed: np.ndarray | None = None


def read_contacts(channel_path: Path, mri_path: Path) -> ElectrodeInput:
    metadata = loadmat(mri_path, variable_names=["SCS", "Voxsize", "InitTransf"], simplify_cells=True)
    transform = scs_m_to_ras_mm(metadata)
    obj = loadmat(channel_path, variable_names=["Channel", "IntraElectrodes"], simplify_cells=True)
    channels = obj.get("Channel", [])
    channels = [channels] if isinstance(channels, dict) else channels
    definitions = obj.get("IntraElectrodes", [])
    definitions = [definitions] if isinstance(definitions, dict) else definitions
    supplied_colors = {str(e["Name"]): np.asarray(e.get("Color", [])) for e in definitions if isinstance(e, dict)}
    selected = []
    for item in channels:
        if not isinstance(item, dict) or item.get("Type") not in ("SEEG", "ECOG", "ECoG"):
            continue
        name = str(item.get("Name", ""))
        if re.search(r"\d+\s*[-–]\s*\S+\d+$", name):
            continue  # Bipolar channel midpoints are not physical contact centers.
        point = np.asarray(item.get("Loc", []), dtype=float)
        if point.size != 3 or not np.isfinite(point).all() or np.linalg.norm(point) == 0:
            continue
        group = str(item.get("Group", "")) or re.sub(r"\d+$", "", name)
        selected.append((name, group, point.reshape(3), str(item["Type"])))
    groups = list(dict.fromkeys(item[1] for item in selected))
    contacts = []
    for name, group, point, kind in selected:
        color = supplied_colors.get(group, np.array([]))
        if color.shape != (3,) or not np.isfinite(color).all():
            color = np.asarray(colorsys.hsv_to_rgb((groups.index(group) * .6180339887) % 1, .65, .98))
        elif float(np.max(color)) > 1:
            color = color.astype(float) / 255
        position = nib.affines.apply_affine(transform, point)
        contacts.append(Contact(name, group, position, tuple(float(v) for v in np.clip(color, 0, 1)), position.copy(), kind))
    if len({c.name for c in contacts}) != len(contacts):
        raise InputError("電極名が重複しているため、読み込みを中止しました。")
    return ElectrodeInput(contacts, mri_path, metadata, transform)


def discover_electrodes(root: Path) -> ElectrodeInput | None:
    roots = [p for p in root.iterdir() if p.is_dir() and p.name.endswith("_bs")]
    found = {}
    for bs in roots:
        mris, channels, cts = [], [], []
        for path in bs.rglob("*.mat"):
            if not _is_regular(path):
                continue
            try:
                variables = {name: (shape, kind) for name, shape, kind in whosmat(path)}
            except (OSError, ValueError):
                continue
            if "Channel" in variables:
                channels.append(path)
            if "Cube" not in variables or variables.get("Labels", ((1, 1), ""))[0] != (0, 0):
                continue
            subject = path.parent / "brainstormsubject.mat"
            if variables["Cube"][1] == "uint8" and subject.is_file():
                obj = loadmat(subject, variable_names=["Anatomy", "UseDefaultAnat"], simplify_cells=True)
                name = Path(str(obj.get("Anatomy", "")).replace("\\", "/")).name
                if name == path.name and not obj.get("UseDefaultAnat", 0):
                    mris.append(path)
            elif variables["Cube"][1] in ("int16", "uint16"):
                cts.append(path)
        for channel in channels:
            # Brainstorm stores anatomy and study data under matching subject directories.
            matching = [m for m in mris if m.parent.name in {parent.name for parent in channel.parents}]
            if len(matching) != 1:
                continue
            source = read_contacts(channel, matching[0])
            if not source.contacts:
                continue
            signature = hashlib.sha256(source.source_matrix.tobytes())
            for contact in source.contacts:
                signature.update(contact.name.encode("utf-8"))
                signature.update(contact.position.tobytes())
            if signature.hexdigest() in found:
                continue
            reference_candidates = {}
            for ct in cts:
                if ct.parent != matching[0].parent:
                    continue
                obj = loadmat(ct, variable_names=["Cube", "InitTransf"], simplify_cells=True)
                trans = transforms(obj)
                if "vox2ras" not in trans:
                    continue
                key = hashlib.sha256(obj["Cube"].tobytes() + trans["vox2ras"].tobytes()).hexdigest()
                reference_candidates[key] = (obj["Cube"], trans)
            if len(reference_candidates) == 1:
                source.ct_data, trans = next(iter(reference_candidates.values()))
                source.ct_affine = trans["vox2ras"]
                if "reg" in trans:
                    seed = trans["reg"].copy()
                    seed[:3, 3] *= 1000
                    if np.allclose(seed[:3, :3].T @ seed[:3, :3], np.eye(3), atol=1e-4):
                        source.ct_seed = seed
            found[signature.hexdigest()] = source
    if len(found) > 1:
        raise InputError("異なる電極データが複数あります。読み込む電極ファイルを指定してください。")
    return next(iter(found.values())) if found else None


def check_reference_mri(source: ElectrodeInput, scene) -> float:
    cube = loadmat(source.mri_path, variable_names=["Cube"], simplify_cells=True)["Cube"]
    affine = transforms(source.mri_metadata)["vox2ras"]
    grid = np.mgrid[0:cube.shape[0]:4, 0:cube.shape[1]:4, 0:cube.shape[2]:4].reshape(3, -1).T
    indices = nib.affines.apply_affine(np.linalg.inv(scene.affine) @ affine, grid)
    measured = map_coordinates(scene.data, indices.T, order=1, mode="constant")
    expected = cube[::4, ::4, ::4].ravel()
    mask = (measured > 30) & (expected > 30)
    correlation = float(np.corrcoef(measured[mask], expected[mask])[0, 1])
    if not np.isfinite(correlation) or correlation < .80:
        raise InputError("電極に対応するBrainstorm MRIと現在のMRIが一致しません。")
    return correlation
