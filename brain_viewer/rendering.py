"""Shared MRI/CT colour mapping for 2D images and 3D slice textures."""
import numpy as np
from .anatomy import nucleus_color
from .image_colormaps import default_image_mode, heatmap_plane


def composite_plane(scene, axis, index, low, high, ct_options=None, selected_label=0):
    raw = np.take(scene.data, int(index), axis=axis)
    grey = np.clip((raw - low) * (255 / max(.001, high-low)), 0, 255)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    options = ct_options or {}
    mri_weight = np.clip(options.get('mri_opacity',1.),0,1) if options.get('mri_visible',True) else 0.
    rgb *= mri_weight
    has_mri = mri_weight > 0
    for layer in scene.extra_mris:
        settings = options.get('extra_mris', {}).get(layer.uid, {})
        opacity = float(np.clip(settings.get('opacity', .5), 0, 1)) if settings.get('visible', True) else 0.
        if opacity <= 0: continue
        raw_layer = np.take(layer.data, int(index), axis=axis)
        valid = np.take(layer.valid, int(index), axis=axis)
        width, level = settings.get('window', layer.window), settings.get('level', layer.level)
        grey_layer = np.clip((raw_layer-level+width/2)*255/max(.001,width), 0, 255)
        alpha = valid.astype(float)*opacity
        mode = settings.get('mode', default_image_mode(layer))
        if mode == 'checker' and has_mri:
            axes = [i for i in range(3) if i != axis]
            grid = np.indices(raw_layer.shape)
            alpha *= (np.floor(grid[0]*scene.spacing[axes[0]]/12)+np.floor(grid[1]*scene.spacing[axes[1]]/12))%2
        layer_color = grey_layer[...,None]
        if mode == 'heatmap':
            layer_color, coverage = heatmap_plane(raw_layer, width, level, settings.get('palette', 'hot'))
            alpha *= coverage
        if mode == 'direction' and layer.sequence == 'DTI-FA':
            from .diffusion_rendering import direction_plane
            colored = direction_plane(scene,layer,axis,int(index))
            if colored is not None: layer_color = colored
        rgb = rgb*(1-alpha[...,None]) + layer_color*alpha[...,None]
        has_mri = True
    if scene.ct is not None and options.get("visible", False):
        ct = np.take(scene.ct, int(index), axis=axis)
        valid = np.take(scene.ct_valid, int(index), axis=axis)
        width, level = options.get("window", 2000), options.get("level", 1000)
        value = np.clip((ct - level + width/2) / max(.1, width), 0, 1)
        opacity = float(np.clip(options.get("opacity", .55),0,1))
        alpha = valid.astype(float) * opacity
        mode = options.get("mode", "bone")
        # A hidden/zero-weight MRI exposes the whole native CT, including soft tissue.
        if not has_mri: mode = 'ct'
        if mode == "bone":
            alpha *= np.clip((ct-150)/200, 0, 1)
        tint = np.array([1., .76, .38]) if options.get("color", "amber") == "amber" else np.ones(3)
        color = value[..., None] * 255 * tint
        if mode == "checker":
            # Physical 12 mm squares, independent of panel size and zoom.
            axes = [i for i in range(3) if i != axis]
            grid = np.indices(ct.shape)
            squares = (np.floor(grid[0]*scene.spacing[axes[0]]/12) + np.floor(grid[1]*scene.spacing[axes[1]]/12)) % 2
            alpha = valid * squares * opacity
            color = np.repeat(value[..., None]*255, 3, axis=2)
        elif mode == "ct":
            alpha = valid.astype(float) * opacity
            rgb = np.zeros_like(rgb)
            color = np.repeat(value[..., None]*255, 3, axis=2)
        elif mode == "edges":
            from scipy.ndimage import binary_erosion
            bone = (ct > 250) & valid
            alpha = (bone & ~binary_erosion(bone)) * options.get("opacity", .8)
            color = np.full_like(rgb, [255., 195., 85.])
        rgb = rgb * (1-alpha[..., None]) + color * alpha[..., None]
    if scene.nuclei_display is not None and options.get("nuclei_visible", False):
        plane = np.take(scene.nuclei_display, int(index), axis=axis)
        selected = options.get("nucleus", 0)
        labels = [selected] if selected else np.unique(plane[plane > 0])
        for label in labels:
            mask = plane == label
            rgb[mask] = rgb[mask]*.52 + np.array(nucleus_color(label))*255*.48
    labels = scene.nuclei_display if selected_label >= 8100 else scene.label_volume
    if selected_label and labels is not None and mri_weight>0 and options.get('mode')!='ct':
        mask = np.take(labels, int(index), axis=axis) == selected_label
        rgb[mask] = rgb[mask] * .77 + np.array([71, 224, 197]) * .23
    from .segmentation_sources import display_segments
    for segment in display_segments(scene, options.get('segmentation_context')):
        if segment.visible_2d:
            mask = np.take(segment.mask, int(index), axis=axis)
            rgb[mask] = rgb[mask]*.62 + np.asarray(segment.color)*255*.38
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
