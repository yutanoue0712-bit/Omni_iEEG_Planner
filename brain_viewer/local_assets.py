"""Immutable, content-addressed snapshot assets within one patient's local folder."""
import hashlib
import os
from pathlib import Path
import shutil
from uuid import uuid4
import numpy as np


def write_asset(destination, arrays, writer, asset_root=None):
    destination = Path(destination)
    if asset_root is None:
        writer(destination)
        return
    root = Path(asset_root)
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(destination.name.encode('utf-8'))
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode('ascii'))
        digest.update(memoryview(array).cast('B'))
    suffix = ''.join(destination.suffixes)
    asset = root / (digest.hexdigest() + suffix)
    if not asset.exists():
        temporary = root / (digest.hexdigest() + '.' + uuid4().hex + suffix)
        writer(temporary)
        temporary.replace(asset)
    try:
        os.link(asset, destination)
    except OSError:
        shutil.copyfile(asset, destination)
