"""Patient-native DTI fitting and ROI tractography, powered by DIPY."""
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import nibabel as nib
import numpy as np
from scipy import ndimage
from .imaging import InputError, MRILayer
from .diffusion_io import B0_THRESHOLD, inspect_diffusion, prepare_diffusion

VERSION = 'dti-wls-eudx-1'


@dataclass
class DiffusionModel:
    uid: str
    name: str
    affine: np.ndarray
    to_reference: np.ndarray
    fa: np.ndarray
    directions: np.ndarray
    mask: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    layer_ids: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)


@dataclass
class TractBundle:
    uid: str
    name: str
    diffusion_uid: str
    points: np.ndarray
    offsets: np.ndarray
    settings: dict = field(default_factory=dict)
    visible: bool = True
    opacity: float = .9
    visible_2d: bool = True
    slab_mm: float = 2.
    excluded: np.ndarray | None = None
    edits: list = field(default_factory=list)
    edit_revision: int = 0

    @property
    def total_count(self): return len(self.offsets)-1

    def active_ids(self):
        return np.arange(self.total_count) if self.excluded is None else np.flatnonzero(~self.excluded)

    @property
    def count(self): return self.total_count-(int(np.count_nonzero(self.excluded)) if self.excluded is not None else 0)

    def streamlines(self):
        return (self.points[self.offsets[i]:self.offsets[i+1]] for i in self.active_ids())


def validate_diffusion(scene):
    models = scene.diffusions
    if len({m.uid for m in models}) != len(models):
        raise InputError('DTIの識別子が重複しています。')
    for m in models:
        if (not m.uid.startswith('dti_') or not m.uid[4:].isalnum() or not m.name.strip()
                or m.fa.ndim != 3 or m.directions.shape != m.fa.shape+(3,) or m.mask.shape != m.fa.shape
                or m.mask.dtype != bool or not np.isfinite(m.fa).all() or np.any((m.fa<0)|(m.fa>1))
                or not np.isfinite(m.directions).all() or not np.isfinite(m.bvals).all()
                or m.bvecs.shape != (len(m.bvals),3) or not np.isfinite(m.bvecs).all()):
            raise InputError('DTIの計算済みデータが不正です。')
        for a in (m.affine,m.to_reference):
            if (a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1])
                    or abs(np.linalg.det(a[:3,:3]))<1e-8):
                raise InputError('DTIの座標変換が不正です。')
        if not np.allclose(m.to_reference[:3,:3].T@m.to_reference[:3,:3],np.eye(3),atol=1e-4) or np.linalg.det(m.to_reference[:3,:3])<.99:
            raise InputError('DTIの位置合わせが剛体変換ではありません。')
        if not set(m.layer_ids.values()).issubset({layer.uid for layer in scene.extra_mris}):
            raise InputError('DTIの表示画像が不足しています。')
    if len({b.uid for b in scene.tracts}) != len(scene.tracts):
        raise InputError('線維束の識別子が重複しています。')
    for b in scene.tracts:
        if (not b.uid.startswith('tract_') or not b.uid[6:].isalnum()
                or b.diffusion_uid not in {m.uid for m in models}
                or b.points.ndim!=2 or b.points.shape[1]!=3 or not np.isfinite(b.points).all()
                or b.offsets.ndim!=1 or b.offsets.dtype.kind not in 'iu' or len(b.offsets)<1
                or b.offsets[0]!=0 or b.offsets[-1]!=len(b.points) or np.any(np.diff(b.offsets)<2)
                or not np.isfinite(b.opacity) or not 0<=b.opacity<=1
                or not np.isfinite(b.slab_mm) or not .5<=b.slab_mm<=20):
            raise InputError('線維束の座標または表示設定が不正です。')
        if b.excluded is not None and (b.excluded.shape!=(b.total_count,) or b.excluded.dtype!=bool):
            raise InputError('線維の除外情報が不正です。')
        for edit in b.edits:
            if (not isinstance(edit,dict) or edit.get('action') not in ('exclude','restore')
                    or not isinstance(edit.get('ids'),list)
                    or any(type(i)!=int or not 0<=i<b.total_count for i in edit['ids'])):
                raise InputError('線維の除外履歴が不正です。')


def motion_correct(data, affine, bvals, bvecs, progress=lambda _: None):
    """Rigid native-volume -> b0 mapping and corresponding gradient rotation."""
    import SimpleITK as sitk
    from .registration import to_sitk, ct_to_mri_matrix
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    reference = data[..., bvals<=B0_THRESHOLD].mean(axis=-1)
    fixed = to_sitk(reference.astype(np.float32), affine)
    output = np.empty_like(data); directions = bvecs.copy(); motions = []
    basis = affine[:3,:3]/np.linalg.norm(affine[:3,:3],axis=0)
    b0_indices = np.flatnonzero(bvals<=B0_THRESHOLD)
    for i in range(data.shape[-1]):
        progress(f'DTIの体動を補正しています… {i+1}/{data.shape[-1]}')
        if len(b0_indices)==1 and i==b0_indices[0]:
            output[...,i]=data[...,i]; matrix=np.eye(4)
        else:
            moving=to_sitk(data[...,i],affine)
            reg=sitk.ImageRegistrationMethod()
            reg.SetMetricAsMattesMutualInformation(32)
            reg.SetMetricSamplingStrategy(reg.REGULAR); reg.SetMetricSamplingPercentage(.25, seed=42)
            reg.SetInterpolator(sitk.sitkLinear)
            reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.,minStep=.01,numberOfIterations=70,
                                                        gradientMagnitudeTolerance=1e-6)
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetShrinkFactorsPerLevel([2,1]); reg.SetSmoothingSigmasPerLevel([1,0])
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            seed=sitk.Euler3DTransform()
            seed.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(fixed.GetSize())-1)/2)))
            reg.SetInitialTransform(seed,inPlace=False)
            try: transform=reg.Execute(fixed,moving)
            except RuntimeError:
                raise InputError('DTIの体動補正に失敗しました。撮像を確認してください。')
            matrix=ct_to_mri_matrix(transform)
            angle=np.rad2deg(np.arccos(np.clip((np.trace(matrix[:3,:3])-1)/2,-1,1)))
            center=nib.affines.apply_affine(affine,(np.array(data.shape[:3])-1)/2)
            displacement=float(np.linalg.norm(nib.affines.apply_affine(matrix,center)-center))
            if not np.isfinite(reg.GetMetricValue()) or angle>20 or displacement>25:
                raise InputError('DTIの体動補正量が大きすぎます。撮像・補正結果を確認してください。')
            sampled=sitk.Resample(moving,fixed,transform,sitk.sitkLinear,0.,sitk.sitkFloat32)
            output[...,i]=sitk.GetArrayFromImage(sampled).transpose(2,1,0)
            directions[i]=basis.T @ matrix[:3,:3] @ basis @ bvecs[i]
        motions.append(matrix.tolist())
    return output,directions,motions


def fit_tensor(data, bvals, bvecs, mask, progress=lambda _: None):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    gtab=gradient_table(bvals,bvecs=bvecs,b0_threshold=B0_THRESHOLD)
    model=TensorModel(gtab,fit_method='WLS')
    fa=np.zeros(mask.shape,np.float32); md=fa.copy(); direction=np.zeros(mask.shape+(3,),np.float32)
    indices=np.flatnonzero(mask)
    raw=data.reshape(-1,data.shape[-1])
    for start in range(0,len(indices),8192):
        progress('DTIのFA・MDと方向を計算しています…')
        ids=indices[start:start+8192]
        fit=model.fit(np.maximum(raw[ids],.001))
        fa.ravel()[ids]=np.clip(np.nan_to_num(fit.fa),0,1)
        md.ravel()[ids]=np.maximum(np.nan_to_num(fit.md),0)
        direction.reshape(-1,3)[ids]=fit.evecs[...,0]
    return fa,md,direction


def add_diffusion(scene, source, name, cache, progress=lambda _: None, *, correct_motion=True):
    import dipy
    from dipy.segment.mask import median_otsu
    from .sequences import add_mri,resample_mri
    cache=Path(cache)
    path=prepare_diffusion(source,cache/'conversion',progress)
    image,bvals,bvecs=inspect_diffusion(path)
    progress('DTIの4D画像を読み込んでいます…')
    data=image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or np.any(data<0):
        raise InputError('DTIの画像値が不正です。非負の拡散強調画像を指定してください。')
    if np.any(np.max(data,axis=(0,1,2))<=0):
        raise InputError('DTIに信号のないボリュームがあります。撮像が揃っているか確認してください。')
    digest=hashlib.sha256((VERSION+str(correct_motion)+dipy.__version__).encode())
    for a in (data,image.affine,bvals,bvecs): digest.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    fingerprint=digest.hexdigest()
    cache.mkdir(parents=True,exist_ok=True)
    fitted=cache/(fingerprint+'.npz'); motion_path=cache/(fingerprint+'.json')
    if fitted.is_file() and motion_path.is_file():
        progress('計算済みDTIを再利用しています…')
        with np.load(fitted,allow_pickle=False) as f:
            b0,fa,md,direction,mask,rotated=[f[k].copy() for k in ('b0','fa','md','directions','mask','bvecs')]
        motion=json.loads(motion_path.read_text(encoding='utf-8'))
    else:
        motions=[]
        if correct_motion: data,rotated,motions=motion_correct(data,image.affine,bvals,bvecs,progress)
        else: rotated=bvecs
        b0=data[...,bvals<=B0_THRESHOLD].mean(axis=-1)
        _,mask=median_otsu(b0,median_radius=2,numpass=2,dilate=1,autocrop=False)
        mask &= b0 > np.percentile(b0[b0>0],99)*.08
        if mask.sum()<20: raise InputError('DTIの脳領域を抽出できません。b0画像を確認してください。')
        fa,md,direction=fit_tensor(data,bvals,rotated,mask,progress)
        motion={'corrected':correct_motion,'volume_to_b0_ras_mm':motions}
        temporary=cache/(fingerprint+'.partial.npz')
        np.savez_compressed(temporary,b0=b0,fa=fa,md=md,directions=direction,mask=mask,bvecs=rotated)
        temporary.replace(fitted)
        from .patients import atomic_json
        atomic_json(motion_path,motion)
    del data
    progress('DTIのb0を基準T1へ位置合わせしています…')
    b0_image=nib.Nifti1Image(b0,image.affine); b0_image.header.set_xyzt_units('mm')
    b0_image.set_qform(image.affine,1); b0_image.set_sform(image.affine,1)
    result=add_mri(scene,b0_image,'DTI-b0',name+' · b0',cache/'registration',progress)
    b0_layer=result.extra_mris[-1]
    uid='dti_'+uuid4().hex
    # Use anatomical support in the reference space without changing native DWI directions.
    if scene.label_volume is not None:
        mapping=np.linalg.inv(scene.affine) @ b0_layer.to_reference @ image.affine
        anatomy=ndimage.affine_transform((scene.label_volume>0).astype(np.uint8),mapping[:3,:3],mapping[:3,3],
                                        output_shape=fa.shape,order=0,prefilter=False)
        mask=mask & ndimage.binary_dilation(anatomy.astype(bool),iterations=1)
    layers=[b0_layer]; layer_ids={'b0':b0_layer.uid}
    for key,raw,width in [('fa',fa,1.),('md',md,.003)]:
        aligned,valid=resample_mri(raw,image.affine,scene,b0_layer.to_reference)
        supported,_=resample_mri(mask.astype(np.float32),image.affine,scene,b0_layer.to_reference)
        valid &= supported>.5
        layer=MRILayer('mr_'+uuid4().hex,name+' · '+key.upper(),'DTI-'+key.upper(),aligned,valid,
            raw,image.affine.copy(),b0_layer.to_reference.copy(),
            dict(b0_layer.quality,diffusion_uid=uid,scalar=key,unit='mm²/s' if key=='md' else 'unitless'),width,width/2)
        layers.append(layer); layer_ids[key]=layer.uid
    quality={'version':VERSION,'library':'DIPY '+dipy.__version__,'fit':'WLS single tensor',
        'fingerprint':fingerprint,'input_kind':'dicom' if isinstance(source,dict) else 'nifti_fsl_bvec',
        'gradient_frame':'native_voxel_axes; FSL x handedness corrected',
        'b0_threshold':B0_THRESHOLD,'motion':motion,'distortion_correction':'not_applied',
        'registration':b0_layer.quality,'review_status':'requires_visual_review'}
    model=DiffusionModel(uid,name,image.affine.copy(),b0_layer.to_reference.copy(),fa,direction,
                         mask,bvals,rotated,layer_ids,quality)
    return replace(scene,extra_mris=[*scene.extra_mris,*layers],diffusions=[*scene.diffusions,model])


def roi_native_mask(scene, model, roi):
    affine=model.to_reference @ model.affine
    if roi.get('kind')=='brain':return model.mask.copy()
    if roi.get('kind')=='sphere':
        center=np.asarray(roi['center'],float); radius=float(roi['radius'])
        if center.shape!=(3,) or not np.isfinite(center).all() or not 1<=radius<=50:
            raise InputError('ROIの位置・半径が不正です。')
        grid=np.indices(model.fa.shape,dtype=np.float32).reshape(3,-1).T
        return (np.linalg.norm(nib.affines.apply_affine(affine,grid)-center,axis=1)<=radius).reshape(model.fa.shape)
    if roi.get('kind')=='segment':
        segment=next((s for s in scene.segmentations if s.uid==roi.get('uid')),None)
        if segment is None or not segment.mask.any(): raise InputError('ROIに使う領域がありません。セグメンテーションで作成してください。')
        mapping=np.linalg.inv(scene.affine) @ affine
        return ndimage.affine_transform(segment.mask.astype(np.uint8),mapping[:3,:3],mapping[:3,3],
                   output_shape=model.fa.shape,order=0,prefilter=False).astype(bool)
    raise InputError('ROIを設定してください。')


def roi_contains_streamline(scene, roi, points):
    if roi['kind']=='brain':return True
    if roi['kind']=='sphere':
        center=np.asarray(roi['center'])
        delta=np.diff(points,axis=0)
        t=np.clip(np.sum((center-points[:-1])*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-12),0,1)
        return bool(np.any(np.linalg.norm(points[:-1]+t[:,None]*delta-center,axis=1)<=roi['radius']))
    segment=next(s for s in scene.segmentations if s.uid==roi['uid'])
    # Sample every half reference voxel, so small ROIs cannot be skipped by a tracking step.
    from dipy.tracking.utils import subsegment
    sampled=next(subsegment([points],max_segment_length=float(scene.spacing.min()/2)))
    indices=scene.index(sampled).T
    values=ndimage.map_coordinates(segment.mask.astype(np.uint8),indices,order=0,mode='constant',cval=0,prefilter=False)
    return bool(values.any())


def roi_record(scene,roi):
    result=dict(roi)
    if roi['kind']=='segment':
        segment=next(s for s in scene.segmentations if s.uid==roi['uid'])
        result.update(name=segment.name,revision=segment.revision,
                      mask_sha256=hashlib.sha256(segment.mask.tobytes()).hexdigest())
    return result


def track_roi(scene, model, roi1, roi2=None, *, name='DTI', fa_threshold=.2, max_angle=35.,
              step_size=.5, min_length=10., max_length=200., max_seeds=3000, progress=lambda _:None):
    from dipy.data import default_sphere
    from dipy.direction import PeaksAndMetrics
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking.tracker import eudx_tracking
    from scipy.spatial import cKDTree
    if (not np.isfinite([fa_threshold,max_angle,step_size,min_length,max_length,max_seeds]).all()
            or not .01<=fa_threshold<1 or not 5<=max_angle<=80 or not .2<=step_size<=2
            or not 0<=min_length<max_length<=500 or not 1<=max_seeds<=20000):
        raise InputError('線維追跡の条件が不正です。')
    seed_mask=roi_native_mask(scene,model,roi1) & model.mask & (model.fa>fa_threshold)
    if roi2 is not None: roi_native_mask(scene,model,roi2)
    indices=np.argwhere(seed_mask)
    if not len(indices): raise InputError('ROI内にFAしきい値を満たすDTI画素がありません。ROI位置・半径・FA値を確認してください。')
    available=len(indices)
    if available>max_seeds:
        indices=indices[np.sort(np.random.default_rng(42).choice(available,int(max_seeds),replace=False))]
    seeds=nib.affines.apply_affine(model.affine,indices)
    sphere=default_sphere
    tree=cKDTree(np.r_[sphere.vertices,-sphere.vertices])
    _,nearest=tree.query(model.directions.reshape(-1,3))
    pam=PeaksAndMetrics()
    pam.peak_indices=(nearest%len(sphere.vertices)).astype(np.int32).reshape(model.fa.shape+(1,))
    pam.peak_values=np.where(model.mask,model.fa,0)[...,None].astype(float)
    stopping=ThresholdStoppingCriterion(np.where(model.mask,model.fa,0).astype(float),fa_threshold)
    progress('ROIから線維を追跡しています…')
    generator=eudx_tracking(seeds,stopping,model.affine,pam=pam,sphere=sphere,max_cross=1,
        min_len=max(2,int(min_length)),max_len=int(max_length),step_size=step_size,max_angle=max_angle,
        pmf_threshold=0.,nbr_threads=2,random_seed=42,return_all=True)
    lines=[]; tested=0
    for line in generator:
        tested+=1
        if len(line)<2: continue
        length=float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum())
        if not min_length<=length<=max_length: continue
        world=nib.affines.apply_affine(model.to_reference,line).astype(np.float32)
        if not roi_contains_streamline(scene,roi1,world): continue
        if roi2 is not None and not roi_contains_streamline(scene,roi2,world): continue
        lines.append(world)
    offsets=np.r_[0,np.cumsum([len(line) for line in lines])].astype(np.int64)
    points=np.concatenate(lines) if lines else np.empty((0,3),np.float32)
    settings={'method':'DIPY tensor EuDX deterministic','fa_threshold':float(fa_threshold),'max_angle':float(max_angle),
        'step_size_mm':float(step_size),'min_length_mm':float(min_length),'max_length_mm':float(max_length),
        'max_seeds':int(max_seeds),'available_seeds':available,'used_seeds':len(indices),'tested_streamlines':tested,
        'roi1':roi_record(scene,roi1),'roi2':roi_record(scene,roi2) if roi2 else None,
        'seed_strategy':'one center per native voxel; deterministic subsampling',
        'coordinate_system':'scanner_RAS_mm','diffusion_fingerprint':model.quality.get('fingerprint'),
        'review_status':'candidate_not_anatomical_confirmation'}
    return TractBundle('tract_'+uuid4().hex,name,model.uid,points,offsets,settings)
