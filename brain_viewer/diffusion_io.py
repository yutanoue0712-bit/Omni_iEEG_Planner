"""Local diffusion DICOM conversion and explicit FSL gradient conventions."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
from uuid import uuid4
import nibabel as nib
import numpy as np
from .imaging import InputError
from .dicom_input import scan_directories

B0_THRESHOLD = 50.


def gradient_paths(path):
    path = Path(path)
    stem = path.name.removesuffix('.gz').removesuffix('.nii')
    return path.with_name(stem+'.bval'), path.with_name(stem+'.bvec')


def read_gradients(path, shape, affine):
    bval, bvec = gradient_paths(path)
    if not bval.is_file() or not bvec.is_file():
        raise InputError('DTIには4D NIfTIと同名の.bval・.bvecが必要です。元DICOMフォルダからも読み込めます。')
    try:
        b = np.loadtxt(bval, dtype=float).reshape(-1)
        v = np.loadtxt(bvec, dtype=float)
    except (OSError, ValueError):
        raise InputError('DTIのb値・方向情報を読み込めません。')
    if v.shape == (3, len(b)): v = v.T.copy()
    if (len(b) != shape[-1] or v.shape != (len(b), 3) or not np.isfinite(b).all()
            or not np.isfinite(v).all() or np.any(b < 0)):
        raise InputError('DTIの画像数とb値・方向数が一致しないか、値が不正です。')
    b0 = b <= B0_THRESHOLD
    lengths = np.linalg.norm(v, axis=1)
    if not b0.any() or (~b0).sum() < 6:
        raise InputError('DTIの計算にはb0画像と6方向以上の拡散画像が必要です。')
    if np.any(np.abs(lengths[~b0]-1) > .05):
        raise InputError('DTIの拡散方向が単位ベクトルではありません。方向情報を確認してください。')
    v[~b0] /= lengths[~b0, None]
    v[b0] = 0
    # dcm2niix emits FSL bvecs: radiological voxel axes, not scanner RAS.
    # With right-handed storage undo FSL's x reflection before tensor fitting.
    if np.linalg.det(affine[:3, :3]) > 0: v[:, 0] *= -1
    x, y, z = v.T
    design = np.column_stack((b*x*x, 2*b*x*y, b*y*y, 2*b*x*z, 2*b*y*z, b*z*z, np.ones(len(b))))
    if np.linalg.matrix_rank(design) < 7:
        raise InputError('DTIの方向が不足・重複し、拡散テンソルを計算できません。')
    return b, v


def inspect_diffusion(path):
    try:
        image = nib.load(str(path))
    except (OSError, ValueError, nib.filebasedimages.ImageFileError):
        raise InputError('DTIのNIfTIヘッダーを読み込めません。')
    if (len(image.shape) != 4 or min(image.shape[:3]) < 2 or image.shape[3] < 7
            or image.header.get_xyzt_units()[0] != 'mm' or np.prod(image.shape) > 350_000_000):
        raise InputError('DTIはmm単位の4D画像を指定してください。FA・ADCなどの計算済み3D画像から線維は抽出できません。')
    a = image.affine
    if (not np.isfinite(a).all() or not np.allclose(a[3], [0,0,0,1])
            or abs(np.linalg.det(a[:3,:3])) < 1e-8):
        raise InputError('DTIの座標情報が不正です。')
    direction = a[:3,:3] / np.linalg.norm(a[:3,:3],axis=0)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise InputError('DTIの格子が直交していません。元DICOMの座標を確認してください。')
    q, qc = image.get_qform(coded=True); s, sc = image.get_sform(coded=True)
    if not qc and not sc: raise InputError('DTIに有効な座標情報がありません。')
    if qc and sc:
        from itertools import product
        corners = np.array(list(product(*[(0,n-1) for n in image.shape[:3]])))
        if np.linalg.norm(nib.affines.apply_affine(q,corners)-nib.affines.apply_affine(s,corners),axis=1).max() > .1:
            raise InputError('DTIのqformとsformが一致しません。元DICOMを指定してください。')
    b, v = read_gradients(path, image.shape, a)
    return image, b, v


def find_diffusion_sources(folder, progress=lambda _: None):
    import pydicom
    progress('DTIの撮像と拡散方向情報を確認しています…')
    groups = {}
    candidates = []
    tags = ['Modality','SeriesInstanceUID','SeriesNumber','SeriesDescription','ImageType',
            'MRDiffusionSequence','DiffusionBValue',(0x19,0x100c),(0x43,0x1039)]
    for directory, names in scan_directories(folder):
        for name in names:
            path = directory/name
            if name.lower().endswith(('.nii','.nii.gz')):
                if not all(p.is_file() for p in gradient_paths(path)): continue
                image, b, _ = inspect_diffusion(path)
                candidates.append((f'NIfTI / {name} / {len(b)} volumes', path))
                continue
            if path.suffix.lower() in ('.json','.bval','.bvec','.txt','.log','.csv','.png'): continue
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
            except (OSError, pydicom.errors.InvalidDicomError): continue
            if ds.get('Modality') != 'MR' or not ds.get('SeriesInstanceUID'): continue
            image_type = list(ds.get('ImageType', []))
            if 'DERIVED' in image_type: continue
            uid = str(ds.SeriesInstanceUID)
            record = groups.setdefault(uid, {'kind':'dwi_dicom', 'files':[], 'diffusion':False,
                'description':str(ds.get('SeriesDescription','DTI'))[:100]})
            record['files'].append(str(path))
            record['diffusion'] |= ('DIFFUSION' in image_type or 'MRDiffusionSequence' in ds or
                                   any(tag in ds for tag in [(0x18,0x9087),(0x19,0x100c),(0x43,0x1039)]))
    dicoms = [(f"DICOM / {r['description']} / {len(r['files'])} files", r)
              for r in groups.values() if r['diffusion']]
    if not dicoms and not candidates:
        raise InputError('拡散方向情報のあるDTIが見つかりません。DTIの元DICOM、または4D NIfTIと.bval・.bvecを指定してください。')
    return dicoms + candidates


def prepare_diffusion(source, cache, progress=lambda _: None):
    """Convert only the selected series; sources are never modified."""
    if not isinstance(source, dict):
        path = Path(source)
        inspect_diffusion(path)
        return path
    if source.get('kind') != 'dwi_dicom' or not source.get('files'):
        raise InputError('DTIの撮像を選び直してください。')
    import dcm2niix
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(p) for p in source['files'])
    stamps = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    digest = hashlib.sha256(json.dumps([dcm2niix.__version__, stamps]).encode()).hexdigest()
    folder = cache/('dwi_'+digest)
    marker = folder/'converted.json'
    if marker.is_file():
        stored = json.loads(marker.read_text(encoding='utf-8'))
        path = folder/stored['volume']
        if path.resolve().parent == folder.resolve():
            inspect_diffusion(path)
            progress('変換済みDTIを再利用しています…')
            return path
    folder.mkdir(exist_ok=True)
    # Numbered hardlinks/copies prevent conversion of adjacent patients/series.
    # Keep all generated files inside a uniquely-created application cache.
    staging = folder/('input_'+uuid4().hex)
    staging.mkdir()
    progress('DTIのDICOMを変換しています…')
    try:
        import shutil
        for index, path in enumerate(files):
            target = staging/f'{index:07d}.dcm'
            try: os.link(path, target)
            except OSError: shutil.copyfile(path, target)
        result = dcm2niix.main(['-b','y','-ba','y','-z','y','-i','y','-f','dwi','-w','0','-o',str(folder),str(staging)],
            capture_output=True, timeout=300,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        # Converter messages can contain scanner identifiers; stay local.
        (folder/'conversion.log').write_bytes(result.stdout+result.stderr)
        if result.returncode != 0:
            raise InputError('DTIのDICOM変換に失敗しました。撮像全体が揃っているか確認してください。')
        valid = []
        for path in folder.glob('*.nii*'):
            if not all(p.is_file() for p in gradient_paths(path)): continue
            try: inspect_diffusion(path)
            except InputError: continue
            valid.append(path)
        if len(valid) != 1:
            raise InputError('DTIを1つに特定できないか、b値・方向情報が不足しています。撮像系列を確認してください。')
        # Detect source changes while conversion was in progress.
        if any((p.stat().st_size,p.stat().st_mtime_ns) != stamp[1:] for p,stamp in zip(files,stamps)):
            raise InputError('DTIの変換中に元ファイルが変更されました。読み込み直してください。')
        from .patients import atomic_json
        atomic_json(marker, {'volume':valid[0].name,'converter':dcm2niix.__version__})
        return valid[0]
    finally:
        # Unlink only the numbered files created above; never recursive source deletion.
        for target in staging.iterdir(): target.unlink()
        staging.rmdir()
