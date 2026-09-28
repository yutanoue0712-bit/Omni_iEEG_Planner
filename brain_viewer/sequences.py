"""Additional scalar images registered to the patient's reference MRI."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from .imaging import InputError, MRILayer, validate_scene
from .registration import to_sitk, sampling_transform, ct_to_mri_matrix, _resolution, _powell
from .volume_io import read_volume, dicom_series, validate_volume, check_ct_candidate
from .dicom_input import scan_directories

MRI_REGISTRATION_VERSION = 'mri-rigid-mi-v1'
PET_REGISTRATION_VERSION = 'pet-rigid-mi-v1'
POSTOP_RIGID_VERSION = 'postop-rigid-mi-1.5mm-v1'
SEQUENCES = (('reference', 'MRI（基準・FreeSurfer対応）'), ('T1', 'MRI：T1'), ('T1ce', 'MRI：造影T1'), ('T2', 'MRI：T2'),
             ('FLAIR', 'MRI：FLAIR'), ('DWI', 'MRI：DWI（3D）'), ('SWI', 'MRI：SWI'),
             ('MRI', 'MRI：その他'), ('DTI', 'DTI：拡散MRI・線維'), ('CTA', 'CTA：動脈'), ('CTV', 'CTV：静脈'),
             ('CT-bone', 'CT：骨条件'), ('CT', 'CT：電極留置後'), ('PET', 'PET'))


def source_modality(sequence):
    return 'PT' if sequence == 'PET' else 'CT' if sequence in ('CT','CT-bone','CTA','CTV') else 'MR'


def add_image(scene, source, sequence, name, cache, progress=lambda _: None):
    if sequence not in ('CTA','CTV','CT-bone'):
        return add_mri(scene,source,sequence,name,cache,progress)
    from .registration import cached_registration
    aligned,valid,matrix,quality,raw,affine = cached_registration(scene,source,None,Path(cache),progress)
    width,level=(2000.,500.) if sequence=='CT-bone' else (700.,250.)
    layer = MRILayer('mr_'+uuid4().hex,name,sequence,aligned,valid,raw,affine.copy(),matrix,
                     dict(quality,modality='CT',sequence=sequence,value_units='HU'),width,level)
    result = replace(scene,extra_mris=[*scene.extra_mris,layer])
    validate_scene(result)
    return result


class ImageSources(list):
    def __init__(self):
        super().__init__()
        self.rejected = []
        self.excluded_localizers = 0


def find_image_sources(folder, modality, progress=lambda _: None, *, dicom_only=False):
    """Inspect a chosen scan folder. Never search Brainstorm/FreeSurfer trees."""
    folder = Path(folder)
    if folder.name.lower().endswith('_fs'):
        raise InputError('ここではMRIまたはCTの撮像フォルダを選択してください。FreeSurferフォルダは、基準MRIを選んだ後の2番目の画面で指定します。')
    if folder.name.lower().endswith('_bs'):
        raise InputError('MRIまたはCTの撮像フォルダを指定してください。')
    progress('フォルダ内の画像ファイルを確認しています…')
    entries = scan_directories(folder)
    sources = ImageSources()
    # Inspect headers only; preserve distinct valid stacks for explicit selection.
    series_list = dicom_series(folder, entries=entries, progress=progress)
    sources.excluded_localizers = series_list.excluded_localizers
    sources.rejected.extend(('DICOM', reason) for reason in series_list.issues)
    for series in series_list:
        if series['modality'] == modality:
            description = series.get('description', '')[:100]
            if modality=='PT' and series.get('pet_frame'):
                description += f" / frame {series['pet_frame']} ms"
            sources.append((f"DICOM / {description} / {series['count']} slices", series))
    if dicom_only:
        if not sources:
            raise InputError('このフォルダに読み込める元DICOMがありません。「フォルダを開く」から撮像フォルダを選び直してください。')
        return sources
    paths = sorted(directory/name for directory, names in entries for name in names
                   if name.lower().endswith(('.nii', '.nii.gz')))
    progress('NIfTIの座標情報を確認しています…')
    for path in paths:
        sidecar = path.with_name(path.name.removesuffix('.gz').removesuffix('.nii') + '.json')
        try:
            recorded = json.loads(sidecar.read_text(encoding='utf-8-sig')).get('Modality') if sidecar.exists() else None
        except (OSError, ValueError):
            recorded = None
        if recorded and recorded != modality:
            continue
        label = str(path.relative_to(folder))
        try:
            image = validate_volume(nib.load(str(path)))
            if modality == 'CT': check_ct_candidate(image)
            spacing = ' x '.join(f'{v:.2f}' for v in np.linalg.norm(image.affine[:3,:3], axis=0))
            sources.append((f'NIfTI / {label} / {spacing} mm', path))
        except InputError as exc:
            sources.rejected.append((label, str(exc)))
        except (OSError, ValueError, EOFError, nib.filebasedimages.ImageFileError):
            sources.rejected.append((label, 'NIfTIのヘッダーを読み込めません。元データを確認してください。'))
    if not sources:
        if sources.rejected:
            raise InputError('読み込める3D画像がありません。\n' + '\n'.join(dict.fromkeys(reason for _, reason in sources.rejected)))
        raise InputError('選択した種類のNIfTI/DICOM画像が見つかりません。')
    return sources


def intensity_range(data):
    sample = data[::2, ::2, ::2]
    sample = sample[np.isfinite(sample) & (sample != 0)]
    if sample.size < 20 or np.ptp(sample) < 1e-6:
        raise InputError('画像値に十分な変化がありません。')
    low, high = np.percentile(sample, [.5, 99.5])
    return float(min(0., low)), float(max(high, low + .001))


def register_mri(scene, data, affine, progress=lambda _: None, *, modality='MR', final_spacing_mm=None):
    """Six-DOF MI, native header plus geometry starts; no atlas or CT-derived initial pose."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    lo, hi = intensity_range(scene.data)
    moving_lo, moving_hi = intensity_range(data)
    fixed = to_sitk(np.clip((scene.data-lo)/(hi-lo), 0, 1).astype(np.float32), scene.affine)
    moving = to_sitk(np.clip((data-moving_lo)/(moving_hi-moving_lo), 0, 1).astype(np.float32), affine)
    center = fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(fixed.GetSize())-1)/2))
    pet=modality=='PT'; label='PET' if pet else 'MRI'
    coarse_fixed, coarse_moving = (_resolution(fixed,4.,1.5),_resolution(moving,4.,1.5)) if pet else (
        _resolution(fixed, 3., 1.2), _resolution(moving, 2., 1.))
    geometry = sitk.CenteredTransformInitializer(coarse_fixed, coarse_moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    seeds = [sampling_transform(np.eye(4), center), geometry]
    for angle in (-20., 20.):
        seed = sitk.Euler3DTransform(geometry)
        seed.SetRotation(np.deg2rad(angle), 0., 0.)
        seeds.append(seed)
    candidates = []
    for index, seed in enumerate(seeds):
        progress(f'{label}位置合わせ {index+1}/{len(seeds)}')
        try:
            matrix, value = _powell(coarse_fixed, coarse_moving, seed, iterations=35)
            if np.isfinite(value): candidates.append((value, matrix))
        except RuntimeError:
            continue
    if len(candidates) < 2:
        raise InputError('画像の位置合わせに失敗しました。撮像範囲と基準MRIを確認してください。')
    candidates.sort(key=lambda item: item[0])
    fine_spacing=2.5 if pet else 1.5
    fine_fixed, fine_moving = _resolution(fixed, fine_spacing, .5), _resolution(moving, fine_spacing, .5)
    refined = []
    for _, matrix in candidates[:2]:
        progress('PETの位置合わせを精密調整しています…' if pet else '追加MRIの位置合わせを精密調整しています…')
        try:
            matrix, value = _powell(fine_fixed, fine_moving, sampling_transform(matrix, center), step=.8, iterations=25, tolerance=.002)
            refined.append((value, matrix))
        except RuntimeError:
            continue
    if not refined:
        raise InputError('画像の位置合わせに失敗しました。撮像範囲と基準MRIを確認してください。')
    refined.sort(key=lambda item: item[0])
    # PET does not gain spatial information by optimizing on the full 1 mm MRI grid.
    final_fixed,final_moving=(_resolution(fixed,2.,.4),_resolution(moving,2.,.4)) if pet else (fixed,moving)
    if not pet and final_spacing_mm is not None:
        final_fixed,final_moving=_resolution(fixed,final_spacing_mm,.4),_resolution(moving,final_spacing_mm,.4)
    matrix, metric = _powell(final_fixed, final_moving, sampling_transform(refined[0][1], center), step=.3, iterations=18, tolerance=.001)
    if not np.isfinite(matrix).all() or not np.isfinite(metric):
        raise InputError('画像の位置合わせに失敗しました。')
    cortex = np.concatenate([mesh.vertices for key, mesh in scene.surfaces.items() if key.startswith('pial')])
    native = nib.affines.apply_affine(np.linalg.inv(matrix @ affine), cortex)
    inside = float(np.mean(np.all((native >= -.5) & (native < np.array(data.shape)-.5), axis=1)))
    if inside < .5:
        raise InputError('追加画像が脳の範囲を十分に含みません。撮像範囲を確認してください。')
    return matrix, {'version': PET_REGISTRATION_VERSION if pet else POSTOP_RIGID_VERSION if final_spacing_mm==1.5 else MRI_REGISTRATION_VERSION,
                    'method': '6DOF rigid / deterministic Mattes MI / Powell',
                    'final_sampling_mm':2. if pet else final_spacing_mm,
                    'final_negative_mi': metric, 'successful_starts': len(candidates),
                    'cortex_inside_fraction': inside, 'landmark_accuracy_mm': None,
                    'review_status': 'automatic_alignment_requires_visual_review'}


def resample_mri(data, affine, scene, matrix):
    reference = to_sitk(np.zeros(scene.data.shape, np.uint8), scene.affine)
    transform = sampling_transform(matrix)
    output = sitk.Resample(to_sitk(np.asarray(data, np.float32), affine), reference, transform, sitk.sitkLinear, 0., sitk.sitkFloat32)
    mask = sitk.Resample(to_sitk(np.ones(data.shape, np.uint8), affine), reference, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(output).transpose(2,1,0).copy(), sitk.GetArrayFromImage(mask).transpose(2,1,0).astype(bool)


def add_mri(scene, source, sequence, name, cache, progress=lambda _: None):
    pet=sequence=='PET'
    postop=sequence.startswith('postop-')
    image = read_volume(source, modality='PT' if pet else 'MR')
    raw = image.get_fdata(dtype=np.float32)
    low, high = intensity_range(raw)
    if not np.isfinite(raw).all(): raise InputError('画像に有限でない値があります。')
    digest = hashlib.sha256((PET_REGISTRATION_VERSION if pet else POSTOP_RIGID_VERSION if postop else MRI_REGISTRATION_VERSION).encode())
    for array in (scene.data, scene.affine, raw, image.affine):
        digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / (('pet_' if pet else 'mri_') + digest.hexdigest() + '.json')
    cached = None
    if path.exists():
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
            matrix = np.asarray(item['to_reference'], dtype=float)
            if matrix.shape == (4,4) and np.isfinite(matrix).all(): cached = (matrix, item['quality'])
        except (ValueError, KeyError, OSError): pass
    matrix, quality = cached if cached else (register_mri(scene,raw,image.affine,progress,modality='PT') if pet else
                                            register_mri(scene,raw,image.affine,progress,final_spacing_mm=1.5) if postop else
                                            register_mri(scene,raw,image.affine,progress))
    if cached is None:
        from .patients import atomic_json
        atomic_json(path, {'to_reference': matrix.tolist(), 'quality': quality})
    progress('追加画像を基準MRIの断面に重ねています…')
    aligned, valid = resample_mri(raw, image.affine, scene, matrix)
    quality = dict(quality, input_kind=image.extra.get('input_kind', 'nifti'))
    if pet:
        quality.update(modality='PT',sequence='PET',value_units=image.extra.get('value_units',''),
                       suv_type=image.extra.get('suv_type',''),value_scaling=image.extra.get('value_scaling','input image values'),
                       suv_conversion=False,display_auto_range=[low,high])
    layer = MRILayer('mr_' + uuid4().hex, name, sequence, aligned, valid, raw, image.affine.copy(), matrix,
                     quality, high-low, (high+low)/2)
    result = replace(scene, extra_mris=[*scene.extra_mris, layer])
    validate_scene(result)
    return result
