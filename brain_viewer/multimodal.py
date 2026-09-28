"""Local MRI / CT scene preparation; no Brainstorm inputs are used."""
from dataclasses import replace
import json
from pathlib import Path

from .imaging import InputError, discover_inputs, load_mri, validate_scene
from .registration import cached_registration
from .electrode_localization import cached_detection, attach_detection


def discover_ct(root: Path) -> Path | None:
    paths = []
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name.endswith(("_bs", "_fs")):
            continue
        for path in folder.rglob("*.json"):
            try:
                if json.loads(path.read_text(encoding="utf-8-sig")).get("Modality") == "CT":
                    for suffix in (".nii.gz", ".nii"):
                        image = path.with_suffix(suffix)
                        if image.is_file(): paths.append(image)
            except (OSError, ValueError, UnicodeError):
                continue
    if len(paths)>1:
        raise InputError("術後CTが複数あります。CTの読み込みから指定してください。")
    return paths[0] if paths else None


def attach_ct(scene, path, cache, initial=None, progress=lambda _:None):
    aligned,valid,matrix,quality,raw,raw_affine=cached_registration(scene,path,initial,cache,progress)
    result=replace(scene,ct=aligned,ct_valid=valid,ct_to_mri=matrix,ct_quality=quality,
                   raw_ct=raw,raw_ct_affine=raw_affine,contacts=[],electrode_quality={})
    return result,raw,raw_affine


def create_ct_contacts(scene, cache, progress=lambda _:None):
    if scene.raw_ct is None:
        raise InputError("元CTがありません。CTを読み込んでください。")
    result=cached_detection(scene.raw_ct,scene.raw_ct_affine,cache,progress=progress)
    return attach_detection(scene,result)


def load_workspace(root: Path, cache: Path, progress=lambda _:None):
    mri_path,fs_dir=discover_inputs(root)
    scene=load_mri(mri_path,fs_dir,progress)
    try:
        ct_path=discover_ct(root)
    except InputError:
        # Keep MRI available so the user can select a CT explicitly from the GUI.
        scene.quality['ct_input_selection_required']=True
        ct_path=None
    if ct_path:
        scene,_,_=attach_ct(scene,ct_path,cache,None,progress)
        scene=create_ct_contacts(scene,cache.parent/'electrodes',progress)
    validate_scene(scene)
    return scene
