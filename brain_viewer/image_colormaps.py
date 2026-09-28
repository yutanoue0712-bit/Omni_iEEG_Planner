"""Shared slice/surface scalar colors; native and registered values stay unchanged."""
from functools import lru_cache
import numpy as np

IMAGE_PALETTES = ('hot', 'inferno', 'turbo')


def default_image_mode(layer):
    return 'heatmap' if layer.sequence == 'PET' else 'full'


def value_units(layer):
    units = str(layer.quality.get('value_units', ''))
    return {'BQML': 'Bq/mL', 'CNTS': 'counts', 'CPS': 'counts/s'}.get(units, units)


@lru_cache(maxsize=3)
def image_lut(palette='hot'):
    from matplotlib import colormaps
    palette = palette if palette in IMAGE_PALETTES else 'hot'
    table = (colormaps[palette](np.linspace(0, 1, 256))[:, :3]*255).astype(np.uint8)
    table.setflags(write=False)
    return table


def heatmap_plane(values, width, level, palette='hot'):
    low = level-width/2
    unit = np.clip(np.nan_to_num((values-low)/max(.001, width), nan=0), 0, 1)
    color = image_lut(palette)[(unit*255).astype(np.uint8)]
    # Fade the lowest 10% of the display range so background counts do not hide MRI.
    alpha = np.minimum(unit*10, 1)*np.isfinite(values)
    return color, alpha
