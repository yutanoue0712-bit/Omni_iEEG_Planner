"""Portable, explicit LPS displacement fields; no temporary-file dependencies."""
import hashlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .imaging import InputError

SPACE = 'reference_LPS_mm_to_rigid_postop_LPS_mm'


def metadata(record):
    if record is None:
        return None
    return {key: value for key, value in record.items() if key != 'vectors'}


def field_record(image):
    vectors = np.ascontiguousarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    result = {'type': 'DisplacementField', 'space': SPACE,
              'size': list(image.GetSize()), 'spacing': list(image.GetSpacing()),
              'origin': list(image.GetOrigin()), 'direction': list(image.GetDirection()),
              'vector_basis': 'physical_LPS_mm', 'array_order': 'zyx_components',
              'sha256': hashlib.sha256(memoryview(vectors).cast('B')).hexdigest(),
              'vectors': vectors}
    validate_field(result)
    vectors.flags.writeable = False
    return result


def validate_field(record):
    try:
        size = np.asarray(record['size'])
        spacing = np.asarray(record['spacing'], float)
        origin = np.asarray(record['origin'], float)
        direction = np.asarray(record['direction'], float).reshape(3, 3)
        vectors = record['vectors']
        if (record['type'] != 'DisplacementField' or record['space'] != SPACE
                or record['vector_basis'] != 'physical_LPS_mm'
                or record['array_order'] != 'zyx_components'
                or size.shape != (3,) or size.dtype.kind not in 'iu'
                or np.any(size < 2) or np.any(size > 512) or np.prod(size) > 16_000_000
                or spacing.shape != (3,) or origin.shape != (3,)
                or not np.isfinite([spacing, origin]).all() or np.any(spacing <= 0)
                or not np.isfinite(direction).all()
                or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5)
                or not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32
                or vectors.shape != (*size[::-1], 3)
                or not np.isfinite(vectors).all() or np.max(np.abs(vectors)) > 100
                or hashlib.sha256(memoryview(np.ascontiguousarray(vectors)).cast('B')).hexdigest() != record['sha256']):
            raise ValueError()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InputError('術後MRIの変形情報が不正です。') from exc


def field_image(record):
    validate_field(record)
    image = sitk.GetImageFromArray(record['vectors'].astype(np.float64), isVector=True)
    image.SetSpacing(record['spacing']); image.SetOrigin(record['origin'])
    image.SetDirection(record['direction'])
    return image


def save_deformation(record, folder, stem, asset_root=None):
    if record is None or record.get('type') != 'DisplacementField':
        return record
    validate_field(record)
    from .local_assets import write_asset
    filename = stem + '_displacement.npz'
    vectors = record['vectors']
    write_asset(Path(folder) / filename, (vectors,),
                lambda path: np.savez_compressed(path, vectors=vectors), asset_root)
    return {**metadata(record), 'file': filename}


def load_deformation(record, folder):
    if record is None or record.get('type') != 'DisplacementField':
        return record
    from .imaging import _contained
    try:
        path = _contained(Path(folder), record['file'])
        with np.load(path, allow_pickle=False) as archive:
            vectors = archive['vectors']
        result = {key: value for key, value in record.items() if key != 'file'}
        result['vectors'] = vectors
        validate_field(result)
        vectors.flags.writeable = False
        return result
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise InputError('術後MRIの変形情報が不正です。') from exc
