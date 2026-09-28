"""Rigid CT-to-MRI registration. Every public matrix maps CT RAS mm to MRI RAS mm."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .imaging import InputError, Scene

RAS_TO_LPS = np.diag([-1., -1., 1., 1.])
REGISTRATION_VERSION = "rigid-mi-multistart-v3"


def validate_registration_metric(value):
    if not np.isfinite(value) or value >= -1e-6:
        raise InputError('CTとMRIの画像情報から位置合わせを評価できません。入力画像の種類と濃度を確認してください。')


def to_sitk(data: np.ndarray, affine: np.ndarray):
    lps = RAS_TO_LPS @ affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    direction = lps[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise InputError("画像の座標にせん断があり、この位置合わせでは扱えません。")
    image = sitk.GetImageFromArray(np.ascontiguousarray(data.transpose(2, 1, 0)))
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in lps[:3, 3]))
    image.SetDirection(tuple(float(v) for v in direction.ravel()))
    return image


def sampling_transform(ct_to_mri: np.ndarray, center_lps=None):
    """ITK resampling needs the inverse: output MRI LPS -> input CT LPS."""
    matrix = np.linalg.inv(RAS_TO_LPS @ ct_to_mri @ RAS_TO_LPS)
    center = np.zeros(3) if center_lps is None else np.asarray(center_lps, dtype=float)
    result = sitk.Euler3DTransform()
    result.SetCenter(tuple(center))
    result.SetMatrix(tuple(matrix[:3, :3].ravel()), tolerance=1e-5)
    result.SetTranslation(tuple(matrix[:3, 3] - center + matrix[:3, :3] @ center))
    return result


def ct_to_mri_matrix(transform) -> np.ndarray:
    origin = np.array(transform.TransformPoint((0., 0., 0.)))
    basis = np.column_stack([np.array(transform.TransformPoint(tuple(e))) - origin for e in np.eye(3)])
    matrix = np.eye(4)
    matrix[:3, :3] = basis
    matrix[:3, 3] = origin
    return RAS_TO_LPS @ np.linalg.inv(matrix) @ RAS_TO_LPS


def resample_ct(ct_data, ct_affine, scene, ct_to_mri):
    image = to_sitk(np.asarray(ct_data, dtype=np.float32), ct_affine)
    reference = to_sitk(np.zeros(scene.data.shape, dtype=np.uint8), scene.affine)
    transform = sampling_transform(ct_to_mri)
    output = sitk.Resample(image, reference, transform, sitk.sitkLinear, -1024., sitk.sitkFloat32)
    # Field-of-view mask distinguishes missing CT from measured air.
    mask = to_sitk(np.ones(ct_data.shape, dtype=np.uint8), ct_affine)
    valid = sitk.Resample(mask, reference, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return (sitk.GetArrayFromImage(output).transpose(2, 1, 0).copy(),
            sitk.GetArrayFromImage(valid).transpose(2, 1, 0).astype(bool))


def _resolution(image, spacing, sigma=0):
    blurred = sitk.SmoothingRecursiveGaussian(image, sigma) if sigma else image
    size = [int(np.ceil((n-1)*s/spacing))+1 for n, s in zip(image.GetSize(), image.GetSpacing())]
    return sitk.Resample(blurred, size, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(),
                         [spacing]*3, image.GetDirection(), 0., sitk.sitkFloat32)


def _powell(fixed, moving, transform, step=2., iterations=35, tolerance=.005):
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(48)
    # All voxels of a physical grid: reproducible, without random sample changes.
    registration.SetMetricSamplingStrategy(registration.NONE)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsPowell(numberOfIterations=iterations, maximumLineIterations=25,
        stepLength=step, stepTolerance=tolerance, valueTolerance=1e-7)
    registration.SetOptimizerScales([90., 90., 90., 1., 1., 1.])
    registration.SetInitialTransform(transform, inPlace=False)
    result = registration.Execute(fixed, moving)
    return ct_to_mri_matrix(result), float(registration.GetMetricValue())


def register_ct(scene: Scene, ct_data, ct_affine, initial=None, progress=lambda _: None):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    cortex = np.concatenate([mesh.vertices for key, mesh in scene.surfaces.items() if key.startswith("pial")])
    lower = np.clip(np.floor(scene.index(cortex.min(axis=0) - 25)).astype(int), 0, np.array(scene.data.shape) - 2)
    upper = np.clip(np.ceil(scene.index(cortex.max(axis=0) + 25)).astype(int), lower + 2, scene.data.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    affine = scene.affine.copy()
    affine[:3, 3] = scene.world(lower)
    high = max(scene.contrast_limits()[1], 1.)
    fixed = to_sitk(np.clip(scene.data[crop] / high, 0, 1).astype(np.float32), affine)
    # Suppress extreme metal intensities during optimization, retain original HU for display.
    moving = to_sitk(((np.clip(ct_data, -1000, 1600) + 1000) / 2600).astype(np.float32), ct_affine)
    center = fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(fixed.GetSize()) - 1) / 2))
    coarse_fixed = _resolution(fixed, 3., 1.4)
    coarse_moving = _resolution(moving, 2., 1.)
    geometry = sitk.CenteredTransformInitializer(coarse_fixed, coarse_moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    evaluate = sitk.ImageRegistrationMethod()
    evaluate.SetMetricAsMattesMutualInformation(64)
    evaluate.SetInterpolator(sitk.sitkLinear)
    evaluate.SetInitialTransform(geometry)
    initial_metric = float(evaluate.MetricEvaluate(fixed, moving))
    rotations = [(a, 0, 0) for a in (-40, -25, -10, 5, 20, 35)]
    rotations += [(0, a, b) for a, b in ((-15, 0), (15, 0), (0, -15), (0, 15))]
    candidates = []
    seeds = []
    for angles in rotations:
        transform = sitk.Euler3DTransform(geometry)
        transform.SetRotation(*np.deg2rad(angles))
        seeds.append(transform)
    if initial is not None:
        seeds.append(sampling_transform(initial, center))
    for number, transform in enumerate(seeds):
        progress(f"CT回転探索 {number+1}/{len(seeds)}")
        try:
            matrix, value = _powell(coarse_fixed, coarse_moving, transform)
            candidates.append((value, matrix, number))
        except RuntimeError:
            continue
    if len(candidates) < 2:
        raise InputError("CTの回転探索に失敗しました。")
    candidates.sort(key=lambda item: item[0])
    fine_fixed = _resolution(fixed, 1.5, .5)
    fine_moving = _resolution(moving, 1., .45)
    refined = []
    for rank, (_, matrix, seed) in enumerate(candidates[:3]):
        progress(f"CT精密調整 {rank+1}/3")
        matrix, value = _powell(fine_fixed, fine_moving, sampling_transform(matrix, center),
                                 step=.8, iterations=25, tolerance=.002)
        refined.append((value, matrix, seed))
    refined.sort(key=lambda item: item[0])
    progress("CTの最終1 mm格子での調整")
    matrix, _ = _powell(fixed, moving, sampling_transform(refined[0][1], center),
                          step=.3, iterations=18, tolerance=.001)
    evaluate.SetInitialTransform(sampling_transform(matrix, center))
    final_metric = float(evaluate.MetricEvaluate(fixed, moving))
    validate_registration_metric(final_metric)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4):
        raise InputError("CTの位置合わせに失敗しました。")
    if final_metric > initial_metric + .015:
        raise InputError("CTの位置合わせで画像の対応が改善しませんでした。")
    native = nib.affines.apply_affine(np.linalg.inv(matrix @ ct_affine), cortex)
    inside = float(np.mean(np.all((native >= -.5) & (native < np.array(ct_data.shape) - .5), axis=1)))
    if inside < .85:
        raise InputError("位置合わせ後のCTが脳表の範囲を十分に含みません。")
    # Variability describes convergence only; it is not a landmark error estimate.
    points = cortex[::20]
    spread = [np.linalg.norm(nib.affines.apply_affine(m @ np.linalg.inv(refined[0][1]), points)-points, axis=1)
              for _, m, _ in refined[1:]]
    quality = {"method": "6DOF rigid / multistart deterministic MI / Powell", "version": REGISTRATION_VERSION,
               "initialization": "image_geometry_rotation_search", "initial_negative_mi": initial_metric,
               "final_negative_mi": final_metric, "cortex_inside_ct_fraction": inside,
               "successful_starts": len(candidates), "requested_starts": len(seeds),
               "coarse_candidates": [{"seed": n, "negative_mi": v, "matrix": m.tolist()} for v, m, n in candidates],
               "refined_seed_spread_p95_mm": float(np.percentile(np.concatenate(spread), 95)),
               "automatic_ct_to_mri_ras_mm": matrix.tolist(),
               "landmark_accuracy_mm": None,
               "review_status": "automatic_alignment_requires_visual_review"}
    return matrix, quality


def cached_registration(scene: Scene, ct_path: Path, initial, cache: Path, progress=lambda _: None):
    from .volume_io import read_volume, validate_ct_intensities, check_ct_candidate
    tick = time.perf_counter()
    progress('CTの画像と座標を読み込んでいます…')
    ct = read_volume(ct_path,modality='CT')
    if len(ct.shape) != 3 or ct.header.get_xyzt_units()[0] != "mm":
        raise InputError("CTはmm単位の3D NIfTIを指定してください。")
    check_ct_candidate(ct)
    data = ct.get_fdata(dtype=np.float32)
    validate_ct_intensities(data)
    timings = {'read_seconds': time.perf_counter()-tick}
    tick = time.perf_counter()
    progress('保存済みのCT位置合わせを確認しています…')
    fingerprint = hashlib.sha256(REGISTRATION_VERSION.encode())
    for array in (scene.affine, scene.data, ct.affine, data, initial if initial is not None else np.eye(4)):
        fingerprint.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    key = fingerprint.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / f"{key}.json"
    cached = None
    if manifest.is_file():
        try:
            obj = json.loads(manifest.read_text(encoding="utf-8"))
            matrix = np.asarray(obj["ct_to_mri_ras_mm"], dtype=float)
            validate_registration_metric(float(obj['quality']['final_negative_mi']))
            if (matrix.shape == (4, 4) and np.isfinite(matrix).all()
                    and np.allclose(matrix[3], [0,0,0,1])
                    and np.allclose(matrix[:3,:3].T @ matrix[:3,:3], np.eye(3), atol=1e-4)
                    and np.linalg.det(matrix[:3,:3]) > .99
                    and obj['quality'].get('version') == REGISTRATION_VERSION):
                cached = (matrix, obj["quality"])
        except (ValueError, KeyError, OSError):
            pass
    timings['cache_check_seconds'] = time.perf_counter()-tick
    tick = time.perf_counter()
    if cached is None:
        matrix, quality = register_ct(scene, data, ct.affine, initial, progress)
        obj = {"ct_to_mri_ras_mm": matrix.tolist(), "quality": quality}
        temporary = manifest.with_suffix(".partial")
        temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(manifest)
    else:
        progress("保存済みのCT位置合わせを読み込んでいます…")
        matrix, quality = cached
    quality=dict(quality)
    timings['registration_seconds'] = time.perf_counter()-tick
    quality['registration_cache_hit'] = cached is not None
    quality['input_kind']=ct.extra.get('input_kind','nifti')
    for field in ('dicom_slice_count', 'dicom_excluded_localizers', 'dicom_order'):
        if field in ct.extra: quality[field] = ct.extra[field]
    quality['raw_spacing_mm']=np.linalg.norm(ct.affine[:3,:3],axis=0).tolist()
    progress("CTをMRIの表示格子に重ねています…")
    tick = time.perf_counter()
    aligned, valid = resample_ct(data, ct.affine, scene, matrix)
    timings['resample_seconds'] = time.perf_counter()-tick
    quality['import_timings'] = timings
    return aligned, valid, matrix, quality, data, ct.affine


def match_existing_ct(raw_data, raw_affine, reference_data, reference_affine, initial, cache: Path,
                      progress=lambda _: None):
    """Identify the same CT in an existing workspace before transferring its contacts.

    Returns raw-CT RAS -> reference-CT RAS. It does not use contacts as fitting targets.
    """
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    fingerprint = hashlib.sha256(b"ct-reference-correlation-v1")
    for array in (raw_data, raw_affine, reference_data, reference_affine, initial):
        fingerprint.update(np.ascontiguousarray(array).tobytes())
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / ("ct_reference_" + fingerprint.hexdigest() + ".json")
    if path.is_file():
        obj = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(obj["raw_to_reference_ras_mm"], dtype=float), obj["quality"]
    progress("既存のCTと元の術後CTの対応を確認しています…")
    # Brainstorm's CT import shifts HU by +1024; correlation is insensitive to that offset.
    fixed_data = np.clip(reference_data.astype(np.float32) - 1024, -1000, 2000)
    moving_data = np.clip(raw_data, -1000, 2000).astype(np.float32)
    fixed = to_sitk(fixed_data, reference_affine)
    moving = to_sitk(moving_data, raw_affine)
    center = fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(fixed.GetSize())-1)/2))
    transform = sampling_transform(initial, center)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsCorrelation()
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(.2, 20260914)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=.8, minStep=.0005,
        numberOfIterations=200, relaxationFactor=.6, gradientMagnitudeTolerance=1e-7)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=False)
    result = registration.Execute(fixed, moving)
    matrix = ct_to_mri_matrix(result)
    from scipy.ndimage import map_coordinates
    indices = np.mgrid[0:fixed_data.shape[0]:3, 0:fixed_data.shape[1]:3, 0:fixed_data.shape[2]:3].reshape(3, -1).T
    target = fixed_data[::3, ::3, ::3].ravel()
    sample = nib.affines.apply_affine(np.linalg.inv(matrix @ raw_affine) @ reference_affine, indices)
    sampled = map_coordinates(moving_data, sample.T, order=1, mode="constant", cval=-1000)
    mask = (target > -700) & np.all((sample > 1) & (sample < np.array(raw_data.shape)-2), axis=1)
    corr = float(np.corrcoef(sampled[mask], target[mask])[0, 1])
    if not np.isfinite(corr) or corr < .90:
        raise InputError("元のCTとBrainstorm内のCTが同じ撮像かを確認できません。電極の追従変換を中止しました。")
    quality = {"same_ct_correlation": corr, "compared_voxels": int(mask.sum()),
               "method": "rigid correlation; images only; contacts not used to fit"}
    obj = {"raw_to_reference_ras_mm": matrix.tolist(), "quality": quality}
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    return matrix, quality
