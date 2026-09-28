"""Patient-space postoperative MRI registration. No atlas or missing-tissue synthesis.

The stored BSpline is a *pull* mapping in LPS mm: reference -> rigid postoperative
space. Sampling the original image composes rigid inverse AFTER this BSpline.
This is deliberately not represented as a forward 4x4 coordinate transform.
"""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import nibabel as nib
import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from .imaging import InputError
from .i18n import tr
from .registration import to_sitk, sampling_transform, _resolution
from .sequences import intensity_range

VERSION = 'postop-masked-bspline-v1'
DEFORMATION_SPACE = 'reference_LPS_mm_to_rigid_postop_LPS_mm'
BRAIN_LABELS = (2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,
                41,42,43,44,46,47,49,50,51,52,53,54,58,60,77,78,79,85,192,251,252,253,254,255)


def deformation_record(transform):
    return {'type':'BSpline', 'order':3, 'space':DEFORMATION_SPACE,
            'fixed_parameters':list(transform.GetFixedParameters()),
            'parameters':list(transform.GetParameters())}


def read_deformation(record):
    if isinstance(record,dict) and record.get('type')=='DisplacementField':
        from .postop_deformation import field_image
        return sitk.DisplacementFieldTransform(field_image(record))
    try:
        if (record['type'] != 'BSpline' or record['order'] != 3
                or record['space'] != DEFORMATION_SPACE): raise ValueError()
        fixed = np.asarray(record['fixed_parameters'],float)
        parameters = np.asarray(record['parameters'],float)
        if (fixed.shape != (18,) or parameters.ndim != 1 or not 192 <= len(parameters) <= 100_000
                or not np.isfinite(fixed).all() or not np.isfinite(parameters).all()
                or np.any(fixed[:3] < 4) or np.any(fixed[:3] > 32)
                or not np.allclose(fixed[:3], np.rint(fixed[:3]))
                or int(np.prod(fixed[:3]))*3 != len(parameters)
                or np.any(fixed[6:9] <= 0) or np.any(fixed[6:9] > 1000)
                or np.max(np.abs(parameters)) > 100): raise ValueError()
        direction=fixed[9:].reshape(3,3)
        if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5): raise ValueError()
        transform = sitk.BSplineTransform(3,3)
        transform.SetFixedParameters(fixed.tolist())
        transform.SetParameters(parameters.tolist())
        return transform
    except (KeyError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        raise InputError('術後MRIの変形情報が不正です。') from exc


def pull_transform(layer):
    rigid = sampling_transform(layer.to_reference)
    if layer.deformation is None: return rigid
    # CompositeTransform evaluates the back transform first.
    return sitk.CompositeTransform([rigid, read_deformation(layer.deformation)])


def resample_postop(scene, layer, deformation):
    candidate = replace(layer,deformation=deformation)
    transform = pull_transform(candidate)
    reference = to_sitk(np.zeros(scene.data.shape,np.uint8),scene.affine)
    output = sitk.Resample(to_sitk(layer.raw,layer.raw_affine),reference,transform,
                           sitk.sitkLinear,0.,sitk.sitkFloat32)
    valid = sitk.Resample(to_sitk(np.ones(layer.raw.shape,np.uint8),layer.raw_affine),reference,
                          transform,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    return (sitk.GetArrayFromImage(output).transpose(2,1,0).copy(),
            sitk.GetArrayFromImage(valid).transpose(2,1,0).astype(bool))


def mask_record(segment):
    if segment is None: return None
    return {'uid':segment.uid,'revision':segment.revision,'voxels':int(np.count_nonzero(segment.mask)),
            'sha256':hashlib.sha256(memoryview(np.ascontiguousarray(segment.mask)).cast('B')).hexdigest()}


def reference_brain_mask(scene, residual=None):
    if residual is not None:
        if residual.mask.shape != scene.data.shape: raise InputError('残存脳の領域を確認してください。')
        brain=residual.mask.copy()
    else:
        labels=scene.label_volume
        if labels is None: raise InputError('残存脳として使用する領域を選択してください。')
        brain=np.isin(labels,BRAIN_LABELS) | ((labels>=1000)&(labels<5000))
        # Include the internal unlabelled spaces; do not include scalp/skull labels.
        brain=ndimage.binary_fill_holes(brain)
    if np.count_nonzero(brain)<1000: raise InputError('残存脳の領域が小さすぎます。広い範囲の残存脳を指定してください。')
    return brain


def _mask_image(mask, scene, reference):
    return sitk.Resample(to_sitk(mask.astype(np.uint8),scene.affine),reference,
                         sitk.Transform(3,sitk.sitkIdentity),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)


def exclusion_array(scene, reference, exclusion, margin_mm):
    excluded = np.zeros(reference.GetSize()[::-1],bool)
    if exclusion is not None:
        if exclusion.mask.shape != scene.data.shape or not exclusion.mask.any():
            raise InputError('除外範囲が空です。摘出腔などを複数の断面で囲んでください。')
        excluded=sitk.GetArrayFromImage(_mask_image(exclusion.mask,scene,reference)).astype(bool)
        if not excluded.any(): raise InputError('除外範囲が小さすぎます。複数の断面で囲んでください。')
        if margin_mm>0:
            excluded=ndimage.distance_transform_edt(~excluded,sampling=reference.GetSpacing()[::-1])<=margin_mm
    return excluded


def metric_masks(scene, layer, reference, exclusion, residual, margin_mm):
    brain = sitk.GetArrayFromImage(_mask_image(reference_brain_mask(scene,residual),scene,reference)).astype(bool)
    valid = sitk.GetArrayFromImage(_mask_image(layer.valid,scene,reference)).astype(bool)
    excluded = exclusion_array(scene,reference,exclusion,margin_mm)
    # Both masks live in the common rigid grid, but the moving mask follows T(x).
    # Thus the cavity is excluded even when the estimated deformation moves it.
    fixed_mask=brain & valid & ~excluded
    moving_mask=valid & ~excluded
    count=int(fixed_mask.sum())
    if count<1000 or count<.15*int(brain.sum()):
        raise InputError('位置合わせに使える残存脳が不足しています。除外範囲と撮像範囲を確認してください。')
    images=[]
    for array in (fixed_mask,moving_mask):
        image=sitk.GetImageFromArray(array.astype(np.uint8)); image.CopyInformation(reference); images.append(image)
    return *images, count


def exclusion_preview(scene, exclusion, margin_mm, spacing_mm=2.):
    from .imaging import Segment
    reference=_resolution(to_sitk(scene.data,scene.affine),spacing_mm,0.)
    excluded=exclusion_array(scene,reference,exclusion,margin_mm)
    image=sitk.GetImageFromArray(excluded.astype(np.uint8)); image.CopyInformation(reference)
    display=to_sitk(np.zeros(scene.data.shape,np.uint8),scene.affine)
    resampled=sitk.Resample(image,display,sitk.Transform(3,sitk.sitkIdentity),sitk.sitkNearestNeighbor)
    mask=sitk.GetArrayFromImage(resampled).transpose(2,1,0).astype(bool)
    return Segment('seg_postoppreview',tr('除外範囲（余白込み）'),'manual',mask,(1.,.3,.55),
                   visible_3d=False,visible_2d=True,opacity=.6,provenance={'role':'postop_exclusion_preview'})


def _normalized(data, affine):
    lo,hi=intensity_range(data)
    return to_sitk(np.clip((data-lo)/(hi-lo),0,1).astype(np.float32),affine)


def deformation_qc(transform, reference):
    field=sitk.TransformToDisplacementField(transform,sitk.sitkVectorFloat64,
        reference.GetSize(),reference.GetOrigin(),reference.GetSpacing(),reference.GetDirection())
    vectors=sitk.GetArrayViewFromImage(field)
    lengths=np.linalg.norm(vectors,axis=-1)
    # ITK's Jacobian filter differentiates along image axes (spacing only).
    # Express vectors in that basis first, including RAS grids with -L/-P axes.
    axes=np.asarray(reference.GetDirection()).reshape(3,3)
    axis_field=sitk.GetImageFromArray(vectors @ axes,isVector=True)
    axis_field.SetSpacing(reference.GetSpacing())
    jacobian=sitk.DisplacementFieldJacobianDeterminant(axis_field)
    jac=sitk.GetArrayViewFromImage(jacobian)
    return {'sampling_spacing_mm':list(reference.GetSpacing()),
            'displacement_max_mm':float(np.max(lengths)),
            'displacement_p95_mm':float(np.percentile(lengths,95)),
            'jacobian_min':float(np.min(jac)), 'jacobian_max':float(np.max(jac)),
            'folding_voxels':int(np.count_nonzero(jac<=0)),
            'finite':bool(np.isfinite(lengths).all() and np.isfinite(jac).all()),
            'scope':'full working grid; numerical screen, not anatomical accuracy'}


def correct_postop(scene, layer, exclusion=None, residual=None, *, margin_mm=3.,
                   max_shift_mm=8., grid_mm=40., no_cavity=False, iterations=60, progress=lambda _:None):
    if layer.deformation is not None: raise InputError('剛体合わせの術後MRIを選択してください。')
    if exclusion is None and not no_cavity: raise InputError('除外範囲を指定するか、切除なしを選択してください。')
    if (not np.isfinite([margin_mm,max_shift_mm,grid_mm]).all() or not 0<=margin_mm<=10
            or not 1<=max_shift_mm<=15 or not 25<=grid_mm<=80 or not 1<=iterations<=120):
        raise InputError('術後MRIの補正条件が不正です。')
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    progress('残存脳と除外範囲を準備しています…')
    # Optimization uses 2 mm images; the final output samples the original once.
    fixed=_resolution(_normalized(scene.data,scene.affine),2.,.6)
    moving=sitk.Resample(_normalized(layer.data,scene.affine),fixed,sitk.Transform(3,sitk.sitkIdentity),
                         sitk.sitkLinear,0.,sitk.sitkFloat32)
    fixed_mask,moving_mask,count=metric_masks(scene,layer,fixed,exclusion,residual,margin_mm)
    mesh=np.maximum(1,np.rint((np.array(fixed.GetSize())-1)*fixed.GetSpacing()/grid_mm).astype(int))
    transform=sitk.BSplineTransformInitializer(fixed,mesh.tolist(),order=3)
    method=sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(40)
    method.SetMetricFixedMask(fixed_mask); method.SetMetricMovingMask(moving_mask)
    method.SetInterpolator(sitk.sitkLinear)
    fraction=min(1.,35000/count)
    method.SetMetricSamplingStrategy(method.RANDOM if fraction<1 else method.NONE)
    if fraction<1: method.SetMetricSamplingPercentage(fraction,seed=31415)
    method.SetShrinkFactorsPerLevel([2,1]); method.SetSmoothingSigmasPerLevel([1.,0.])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Convex BSpline weights: this component bound also bounds displacement norm.
    bound=float(max_shift_mm/np.sqrt(3))
    method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,numberOfIterations=iterations,
        maximumNumberOfCorrections=5,maximumNumberOfFunctionEvaluations=iterations*8,
        costFunctionConvergenceFactor=1e7,lowerBound=-bound,upperBound=bound)
    method.SetInitialTransform(transform,inPlace=True)
    last=[None]
    def report():
        key=(method.GetCurrentLevel(),method.GetOptimizerIteration())
        if key!=last[0] and (key[1]%5==0):
            progress(f'術後MRIの変形補正 {key[0]+1}/2・{key[1]}/{iterations}')
            last[0]=key
    method.AddCommand(sitk.sitkIterationEvent,report)
    try: method.Execute(fixed,moving)
    except RuntimeError as exc:
        raise InputError('変形補正を計算できませんでした。剛体合わせと残存脳・除外範囲を確認してください。') from exc
    progress('変形量と折り返しを確認しています…')
    qc=deformation_qc(transform,fixed)
    if (not np.isfinite(method.GetMetricValue()) or not qc['finite'] or qc['folding_voxels'] or qc['jacobian_min']<.2
            or qc['jacobian_max']>5 or qc['displacement_max_mm']>max_shift_mm+.01):
        raise InputError('変形が大きい、または折り返しがあります。補正結果を採用せず、剛体合わせを保持しました。')
    record=deformation_record(transform)
    progress('元の術後MRIから補正画像を作成しています…')
    data,valid=resample_postop(scene,layer,record)
    quality={'version':VERSION,'method':'rigid + masked bounded cubic BSpline / Mattes MI',
             'postop_role':'deformed','rigid_source_uid':layer.uid,'review_status':'candidate_requires_visual_review',
             'landmark_accuracy_mm':None,'created':datetime.now(timezone.utc).isoformat(),
             'exclusion':mask_record(exclusion),'residual':mask_record(residual),
             'residual_source':'manual_region' if residual else 'reference_FreeSurfer_brain_labels',
             'exclusion_margin_mm':float(margin_mm),'no_cavity_declared':bool(no_cavity and exclusion is None),
             'max_shift_mm':float(max_shift_mm),'grid_spacing_requested_mm':float(grid_mm),
             'working_spacing_mm':2.,'mesh_size':mesh.tolist(),'metric_voxels':count,
             'sampling_fraction':fraction,'seed':31415,'iterations_per_level':iterations,
             'final_valid_metric_points':int(method.GetMetricNumberOfValidPoints()),
             'final_negative_mi':float(method.GetMetricValue()),'stop_condition':method.GetOptimizerStopConditionDescription(),
             'numerical_qc':qc,'base_registration':dict(layer.quality),
             'transform_direction':DEFORMATION_SPACE,'missing_tissue_reconstructed':False}
    return replace(layer,uid='mr_'+uuid4().hex,name=layer.name+' · '+tr('補正候補'),
                   data=data,valid=valid,deformation=record,quality=quality)


def export_postop(scene, layer, parent):
    """Create a new bundle directory. Never overwrite imported images or old exports."""
    folder=Path(parent)/('postop_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
    folder.mkdir(parents=True,exist_ok=False)
    for name,data in (('postop_in_preop',layer.data),('valid',layer.valid.astype(np.uint8))):
        image=nib.Nifti1Image(data,scene.affine); image.header.set_xyzt_units('mm')
        image.set_sform(scene.affine,code=1); image.set_qform(scene.affine,code=1)
        nib.save(image,str(folder/(name+'.nii.gz')))
    dense=layer.deformation is not None and layer.deformation.get('type')=='DisplacementField'
    transform_file='reference_to_native_postop_LPS.h5' if dense else 'reference_to_native_postop_LPS.tfm'
    sitk.WriteTransform(pull_transform(layer),str(folder/transform_file))
    from .postop_deformation import save_deformation
    deformation=save_deformation(layer.deformation,folder,'postop')
    record={'schema':'omni-ieeg-postop/2' if dense else 'omni-ieeg-postop/1','layer_id':layer.uid,'quality':layer.quality,
        'reference_affine_ras_mm':scene.affine.tolist(),'native_affine_ras_mm':layer.raw_affine.tolist(),
        'rigid_native_to_reference_ras_mm':layer.to_reference.tolist(),'deformation':deformation,
        'transform_file':transform_file,
        'transform_file_direction':'reference physical LPS mm -> native postoperative physical LPS mm (resampling pull)',
        'point_forward_inverse_available':layer.deformation is None,
        'anatomy_and_contacts':'remain in preoperative reference; not deformed'}
    (folder/'registration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return folder
