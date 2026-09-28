"""Local DICOM / NIfTI input with explicit physical geometry and no identifying metadata output."""
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from .imaging import InputError
from .dicom_input import dicom_series, read_dicom as _read_dicom


def validate_ct_intensities(data):
    """Reject masks/normalised images before HU registration or contact detection."""
    if not data.size or not np.isfinite(data).all():
        raise InputError('CTの画像値が不正、または全体が同じ値です。元DICOMを含む撮像フォルダから選び直してください。')
    low, high = float(data.min()), float(data.max())
    if high <= low:
        raise InputError('CTの画像値が不正、または全体が同じ値です。元DICOMを含む撮像フォルダから選び直してください。')
    if 0 <= low and high <= 1:
        raise InputError('この画像は0〜1のマスクまたは正規化画像で、CT濃度（HU）の画像ではありません。元DICOMを含むCT撮像フォルダを選択してください。')


def check_ct_candidate(image):
    if image.header.get_intent()[0] == 'label':
        raise InputError('このNIfTIは領域ラベル画像です。CT濃度（HU）の画像または元DICOMを選択してください。')
    # A small unscaled 8-bit image may be a binary mask even without label intent.
    # Bound this check; full float CT volumes are inspected only after selection.
    dtype = image.get_data_dtype()
    if (dtype.kind in 'bu' and dtype.itemsize == 1
            and np.prod(image.shape) <= 64*1024*1024
            and getattr(image.dataobj, 'slope', 1.) == 1.
            and getattr(image.dataobj, 'inter', 0.) == 0.):
        validate_ct_intensities(np.asanyarray(image.dataobj))


def read_volume(source, modality=None):
    if isinstance(source,nib.spatialimages.SpatialImage): return validate_volume(source)
    if isinstance(source,dict):
        if modality and source.get('modality') and source['modality'] != modality:
            raise InputError('選択した画像の種類とDICOMの種類が一致しません。撮像系列を選び直してください。')
        return validate_volume(_read_dicom(source))
    path=Path(source)
    if path.is_dir():
        scanned=dicom_series(path)
        series=[s for s in scanned if modality is None or s['modality']==modality]
        if not series and scanned.issues: raise InputError(scanned.issues[0])
        if len(series)!=1: raise InputError('DICOM系列を1つに特定できません。撮像フォルダを選択してください。')
        return validate_volume(_read_dicom(series[0]))
    image=nib.load(str(path))
    if len(image.shape)!=3 or image.header.get_xyzt_units()[0]!='mm':
        raise InputError('mm単位の3D NIfTIを選択してください。')
    image.extra['input_kind']='nifti'
    if modality == 'PT':
        sidecar=path.with_name(path.name.removesuffix('.gz').removesuffix('.nii')+'.json')
        try: metadata=json.loads(sidecar.read_text(encoding='utf-8-sig')) if sidecar.is_file() else {}
        except (OSError,ValueError): metadata={}
        if not isinstance(metadata,dict): metadata={}
        if metadata.get('Modality') and metadata['Modality'] != 'PT':
            raise InputError('選択した画像の種類とDICOMの種類が一致しません。撮像系列を選び直してください。')
        image.extra.update(value_units=str(metadata.get('Units',''))[:32],suv_type=str(metadata.get('SUVType',''))[:32],
                           value_scaling='NIfTI scaling',suv_conversion=False)
    return validate_volume(image)


def validate_volume(image):
    affine=image.affine
    if len(image.shape)!=3 or min(image.shape)<2 or image.header.get_xyzt_units()[0]!='mm':
        raise InputError('mm単位の3D画像を選択してください。')
    if affine.shape!=(4,4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<1e-8 or not np.allclose(affine[3],[0,0,0,1]):
        raise InputError('画像の座標情報が不正です。')
    qform,qcode=image.get_qform(coded=True)
    sform,scode=image.get_sform(coded=True)
    if not qcode and not scode:
        raise InputError('NIfTIに有効な座標情報がありません。元DICOMを指定してください。')
    if qcode and scode:
        from itertools import product
        corners=np.array(list(product(*[(0,n-1) for n in image.shape])))
        difference=nib.affines.apply_affine(qform,corners)-nib.affines.apply_affine(sform,corners)
        if np.linalg.norm(difference,axis=1).max()>.1:
            raise InputError('NIfTIのqformとsformが一致しません。座標を確認するか、元DICOMを指定してください。')
    return image
