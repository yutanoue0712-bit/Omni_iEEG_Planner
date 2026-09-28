"""Masked SyN candidates, isolated from Qt and the GUI's NumPy/SciPy environment."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from uuid import uuid4

import numpy as np
import SimpleITK as sitk

from .imaging import InputError
from .i18n import tr
from .postop_registration import (_normalized, _resolution, metric_masks, mask_record,
    deformation_qc, resample_postop, DEFORMATION_SPACE)
from .postop_deformation import field_record

VERSION = 'postop-masked-ants-syn-v1'
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CorrectionCancelled(Exception):
    pass


def ants_python():
    environment = PROJECT_ROOT / '.venv-ants'
    return environment / ('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def ants_available():
    environment = PROJECT_ROOT / '.venv-ants'
    package = environment / 'Lib/site-packages/ants/__init__.py'
    if os.name!='nt':
        return ants_python().is_file() and bool(list(environment.glob('lib/python*/site-packages/ants/__init__.py')))
    return ants_python().is_file() and package.is_file()


def _stop_worker(process):
    # Windows venv python.exe is a redirector that starts a second interpreter.
    # Terminating only the redirector leaves ANTs (and its file handles) alive.
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,timeout=15,check=False)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(timeout=10)


def _run_worker(folder, progress, cancel, timeout_seconds):
    interpreter = ants_python()
    if not ants_available():
        raise InputError('ANTsPyの準備が必要です。作業フォルダのSetup_Postop_ANTs.cmdを実行してください。')
    log = folder / 'worker.log'
    process = None
    previous = None
    started = time.monotonic()
    try:
        with log.open('wb') as output:
            process = subprocess.Popen([str(interpreter), str(Path(__file__).with_name('postop_ants_worker.py')), str(folder)],
                stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0), cwd=PROJECT_ROOT)
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    raise CorrectionCancelled()
                if time.monotonic()-started > timeout_seconds:
                    raise InputError('ANTsPyの計算が制限時間を超えました。範囲と設定を確認してください。')
                try:
                    stage = json.loads((folder/'status.json').read_text(encoding='utf-8'))['stage']
                    if stage != previous:
                        progress(stage); previous = stage
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(.2)
        if process.returncode != 0 or not (folder/'result.json').is_file():
            # Keep a text-only diagnostic under local_data; never send it anywhere.
            diagnostic = PROJECT_ROOT / 'local_data' / 'postop_ants_error.log'
            diagnostic.write_bytes(log.read_bytes()[-32000:])
            raise InputError('ANTsPyの補正を計算できませんでした。除外範囲と剛体合わせを確認してください。詳細はローカルのログに記録しました。')
        return json.loads((folder/'result.json').read_text(encoding='utf-8'))
    finally:
        if process is not None and process.poll() is None:
            _stop_worker(process)


def correct_postop_ants(scene, layer, exclusion=None, residual=None, *, margin_mm=3.,
        max_shift_mm=12., no_cavity=False, metric='CC', spacing_mm=2., n4=True,
        iterations=(80,50,20), progress=lambda _:None, cancel=None, timeout_seconds=1800):
    if layer.deformation is not None:
        raise InputError('剛体合わせの術後MRIを選択してください。')
    if exclusion is None and not no_cavity:
        raise InputError('除外範囲を指定するか、切除なしを選択してください。')
    if (not np.isfinite([margin_mm,max_shift_mm,spacing_mm]).all()
            or not 0<=margin_mm<=10 or not 1<=max_shift_mm<=15 or spacing_mm not in (1.5,2.)
            or metric not in ('CC','mattes') or len(iterations)!=3
            or any(not isinstance(v,int) or not 0<=v<=200 for v in iterations) or not any(iterations)):
        raise InputError('術後MRIの補正条件が不正です。')
    if not ants_available():
        raise InputError('ANTsPyの準備が必要です。作業フォルダのSetup_Postop_ANTs.cmdを実行してください。')
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise CorrectionCancelled()
    check_cancel()
    progress('残存脳と除外範囲を準備しています…')
    fixed = _resolution(_normalized(scene.data,scene.affine),spacing_mm,.6)
    moving = sitk.Resample(_normalized(layer.data,scene.affine),fixed,
        sitk.Transform(3,sitk.sitkIdentity),sitk.sitkLinear,0.,sitk.sitkFloat32)
    fixed_mask,moving_mask,count = metric_masks(scene,layer,fixed,exclusion,residual,margin_mm)
    root = PROJECT_ROOT / 'local_data' / 'postop_jobs'
    root.mkdir(parents=True,exist_ok=True)
    # This context owns only the newly created directory; no patient originals.
    with tempfile.TemporaryDirectory(prefix='ants_',dir=root) as temporary:
        folder = Path(temporary).resolve()
        if folder.parent != root.resolve():
            raise InputError('術後MRIの一時保存先が不正です。')
        for name,image in (('fixed',fixed),('moving',moving),('fixed_mask',fixed_mask),('moving_mask',moving_mask)):
            sitk.WriteImage(image,str(folder/(name+'.nii.gz')))
        options={'metric':metric,'iterations':list(iterations),'n4':bool(n4),'spacing_mm':spacing_mm}
        (folder/'options.json').write_text(json.dumps(options),encoding='utf-8')
        worker = _run_worker(folder,progress,cancel,timeout_seconds)
        check_cancel()
        field = sitk.ReadImage(str(folder/'displacement.nii.gz'),sitk.sitkVectorFloat64)
        if (field.GetSize()!=fixed.GetSize() or field.GetNumberOfComponentsPerPixel()!=3
                or not np.allclose(field.GetOrigin(),fixed.GetOrigin(),atol=1e-4)
                or not np.allclose(field.GetSpacing(),fixed.GetSpacing(),atol=1e-5)
                or not np.allclose(field.GetDirection(),fixed.GetDirection(),atol=1e-5)):
            raise InputError('術後MRIの変形情報が不正です。')
        record = field_record(field)
    from .postop_registration import read_deformation
    progress('変形量と折り返しを確認しています…')
    qc = deformation_qc(read_deformation(record),fixed)
    if (not qc['finite'] or qc['folding_voxels'] or qc['jacobian_min']<.2 or qc['jacobian_max']>5
            or qc['displacement_max_mm']>max_shift_mm+.01):
        raise InputError(tr('ANTsPyの補正量は最大 {maximum} mmでした。確認上限、または変形の検査条件を超えたため候補を追加しませんでした。').format(
            maximum=f"{qc['displacement_max_mm']:.2f}"))
    check_cancel()
    progress('元の術後MRIから補正画像を作成しています…')
    data,valid = resample_postop(scene,layer,record)
    check_cancel()
    quality={'version':VERSION,'engine':'ants','method':'rigid + masked SyNOnly / '+metric,
        'postop_role':'deformed','rigid_source_uid':layer.uid,'review_status':'candidate_requires_visual_review',
        'landmark_accuracy_mm':None,'created':datetime.now(timezone.utc).isoformat(),
        'exclusion':mask_record(exclusion),'residual':mask_record(residual),
        'residual_source':'manual_region' if residual else 'reference_FreeSurfer_brain_labels',
        'mask_policy':'fixed: reference brain & rigid validity minus cavity; moving: rigid validity minus cavity',
        'exclusion_margin_mm':float(margin_mm),'no_cavity_declared':bool(no_cavity and exclusion is None),
        'max_shift_mm':float(max_shift_mm),'max_shift_role':'post_registration_rejection_threshold_not_optimizer_bound',
        'working_spacing_mm':spacing_mm,'metric_voxels':count,'ants':worker,'numerical_qc':qc,
        'base_registration':dict(layer.quality),'transform_direction':DEFORMATION_SPACE,
        'missing_tissue_reconstructed':False,'postoperative_anatomy_labels_used':False}
    return replace(layer,uid='mr_'+uuid4().hex,name=layer.name+' · ANTsPy · '+tr('補正候補'),
                   data=data,valid=valid,deformation=record,quality=quality)
