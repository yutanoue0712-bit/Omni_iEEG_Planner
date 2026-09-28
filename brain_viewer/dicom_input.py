"""Header-only DICOM discovery and physical slice ordering, including mixed scouts."""
from pathlib import Path
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from .imaging import InputError


class DicomSources(list):
    def __init__(self):
        super().__init__()
        self.excluded_localizers = 0
        self.issues = []


def scan_directories(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise InputError('MRIまたはCTの撮像フォルダを指定してください。')
    entries = []
    for directory, children, files in os.walk(folder):
        children[:] = sorted(n for n in children if not n.lower().endswith(('_bs', '_fs')))
        entries.append((Path(directory), files))
        if len(entries) > 300:
            raise InputError('DICOM撮像のフォルダを直接指定してください。')
    return entries


def _stamp(file):
    info = Path(file).stat()
    return str(file), info.st_size, info.st_mtime_ns


def _header(file):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(file))
    reader.ReadImageInformation()
    def tag(key):
        return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ''
    return {'file': str(file), 'stamp': _stamp(file),
            'position': np.fromstring(tag('0020|0032'), sep='\\'),
            'orientation': np.fromstring(tag('0020|0037'), sep='\\'),
            'pixel_spacing': np.fromstring(tag('0028|0030'), sep='\\'),
            'size': (tag('0028|0010'), tag('0028|0011')),
            'modality': tag('0008|0060'), 'description': tag('0008|103e'),
            'image_type': tag('0008|0008').upper(),
            'pet_frame': tag('0054|1300') if tag('0008|0060') == 'PT' else '',
            'value_units': tag('0054|1001'), 'suv_type': tag('0054|1006'),
            'acquisition': (tag('0020|0052'), tag('0020|0012'), tag('0018|0086'), tag('0020|0100'))}


def _localizer(header):
    return bool({'LOCALIZER', 'SCOUT'} & set(header['image_type'].split('\\')))


def _same_stack(a, b):
    if any(a[key] != b[key] for key in ('size', 'modality', 'image_type', 'acquisition')):
        return False
    if a.get('modality') == 'PT' and any(a.get(key,'') != b.get(key,'') for key in ('pet_frame','value_units','suv_type')):
        return False
    return all(a[key].shape == b[key].shape and np.allclose(a[key], b[key], atol=1e-4, rtol=0)
               for key in ('orientation', 'pixel_spacing'))


def _ordered_headers(headers):
    if len(headers) < 2:
        raise InputError('3DのDICOM系列を選択してください。')
    first = headers[0]
    if any(h['position'].shape != (3,) or h['orientation'].shape != (6,)
           or h['pixel_spacing'].shape != (2,)
           or not np.isfinite(np.r_[h['position'], h['orientation'], h['pixel_spacing']]).all()
           or np.any(h['pixel_spacing'] <= 0) or not _same_stack(first, h) or _localizer(h)
           for h in headers):
        raise InputError('DICOMのスライス方向が不均一、または座標情報が不足しています。')
    row, column = first['orientation'].reshape(2, 3)
    if not np.allclose([row @ row, column @ column, row @ column], [1, 1, 0], atol=1e-4, rtol=0):
        raise InputError('DICOMのスライス方向が不均一、または座標情報が不足しています。')
    normal = np.cross(row, column)
    headers = sorted(headers, key=lambda h: float(h['position'] @ normal))
    positions = np.array([h['position'] for h in headers])
    differences = np.diff(positions, axis=0)
    step = (positions[-1] - positions[0]) / (len(headers) - 1)
    expected = positions[0] + np.arange(len(headers))[:, None] * step
    if (np.min(differences @ normal) < .01
            or np.max(np.linalg.norm(differences-step, axis=1)) > .05
            or np.max(np.linalg.norm(positions-expected, axis=1)) > .05):
        raise InputError('DICOMのスライス間隔が不均一、または重複しています。')
    return headers


def dicom_series(folder, *, entries=None, progress=lambda _: None):
    result = DicomSources()
    entries = scan_directories(folder) if entries is None else entries
    for directory, files in entries:
        # Avoid opening converted images, spreadsheets and meshes as DICOM.
        if not any(Path(f).suffix.lower() not in ('.nii', '.gz', '.json', '.xlsx', '.csv', '.mgz', '.vtk', '.vtp') for f in files):
            continue
        identifiers = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
        for identifier in identifiers:
            paths = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), identifier)
            groups = []
            excluded = 0
            try:
                for index, file in enumerate(paths):
                    if index % 50 == 0:
                        progress(f'DICOMの座標を確認：{index}/{len(paths)}枚')
                    header = _header(file)
                    if _localizer(header):
                        excluded += 1
                        continue
                    group = next((g for g in groups if _same_stack(g[0], header)), None)
                    if group is None:
                        groups.append([header])
                    else:
                        group.append(header)
            except (OSError, RuntimeError, ValueError):
                result.issues.append('DICOMのヘッダーを読み込めません。元データを確認してください。')
                continue
            result.excluded_localizers += excluded
            for group in groups:
                try:
                    ordered = _ordered_headers(group)
                except InputError as exc:
                    result.issues.append(str(exc))
                    continue
                first = ordered[0]
                result.append({'folder': directory, 'uid': identifier,
                               'files': tuple(h['file'] for h in ordered), 'count': len(ordered),
                               'modality': first['modality'], 'description': first['description'],
                               'pet_frame': first.get('pet_frame',''),
                               '_headers': ordered, 'excluded_localizers': excluded})
    return result


def read_dicom(series):
    paths = tuple(str(p) for p in series['files'])
    headers = series.get('_headers', [])
    if (len(headers) != len(paths)
            or any(h['file'] != p or h['stamp'] != _stamp(p) for h, p in zip(headers, paths))):
        headers = [_header(p) for p in paths]
    headers = _ordered_headers(headers)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([h['file'] for h in headers])
    if headers[0]['modality'] == 'PT':
        # PET rescale slopes can differ by slice; keep fractional calibrated values.
        reader.SetOutputPixelType(sitk.sitkFloat32)
    reader.ForceOrthogonalDirectionOff()
    image = reader.Execute()
    if image.GetDimension() != 3:
        raise InputError('3DのDICOM系列を選択してください。')
    affine = np.eye(4)
    affine[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    # Check ITK's geometry against the DICOM positions before using its volume.
    predicted = nib.affines.apply_affine(affine, [[0, 0, i] for i in range(len(headers))])
    if (image.GetSize()[2] != len(headers)
            or np.max(np.linalg.norm(predicted - np.array([h['position'] for h in headers]), axis=1)) > .05):
        raise InputError('DICOMの座標と読み込んだ画像の座標が一致しません。')
    affine = np.diag([-1., -1., 1., 1.]) @ affine
    data = sitk.GetArrayFromImage(image).transpose(2, 1, 0).astype(np.float32)
    result = nib.Nifti1Image(data, affine)
    result.header.set_xyzt_units('mm')
    result.extra.update(input_kind='dicom', dicom_slice_count=len(headers),
                        dicom_excluded_localizers=series.get('excluded_localizers', 0),
                        dicom_order='physical_position')
    result.extra['modality'] = headers[0]['modality']
    if headers[0]['modality'] == 'PT':
        result.extra.update(value_units=headers[0]['value_units'], suv_type=headers[0]['suv_type'],
                            value_scaling='DICOM RescaleSlope / RescaleIntercept', suv_conversion=False)
    return result
