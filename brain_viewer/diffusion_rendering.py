"""Direction colors use patient-reference RAS axes, independent of DWI storage."""
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


def tract_segments(bundle):
    """Adjacent vertices only; never connect the end of one fiber to the next."""
    return tract_segment_data(bundle)[0]


def tract_segment_data(bundle):
    key=(id(bundle.points),id(bundle.offsets),bundle.edit_revision)
    cached=getattr(bundle,'_segment_data',None)
    if cached is not None and cached[0]==key:return cached[1],cached[2]
    valid = np.ones(max(0, len(bundle.points)-1), bool)
    valid[bundle.offsets[1:-1]-1] = False
    ids=np.repeat(np.arange(bundle.total_count),np.diff(bundle.offsets)-1)
    segments=np.stack([bundle.points[:-1][valid],bundle.points[1:][valid]],axis=1)
    if bundle.excluded is not None:
        keep=~bundle.excluded[ids];segments=segments[keep];ids=ids[keep]
    bundle._segment_data=(key,segments,ids)
    return segments,ids


def clip_segments_to_slab(segments, axis, center, thickness, return_indices=False):
    """Clip segments in RAS millimetres, including crossings with both ends outside."""
    segments = np.asarray(segments)
    if not len(segments):
        result=(segments.reshape(0, 2, 3), np.empty((0, 3)))
        return (*result,np.array([],dtype=np.int64)) if return_indices else result
    a, b = segments[:, 0], segments[:, 1]
    dz = b[:, axis] - a[:, axis]
    low, high = center-thickness/2, center+thickness/2
    parallel = np.abs(dz) < 1e-10
    safe = np.where(parallel, 1, dz)
    first, last = (low-a[:, axis])/safe, (high-a[:, axis])/safe
    start = np.maximum(0, np.minimum(first, last))
    end = np.minimum(1, np.maximum(first, last))
    start[parallel], end[parallel] = 0, 1
    keep = (end >= start) & (~parallel | ((a[:, axis] >= low) & (a[:, axis] <= high)))
    delta = b[keep]-a[keep]
    colors = np.abs(delta)/np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
    clipped = np.stack([a[keep]+start[keep, None]*delta, a[keep]+end[keep, None]*delta], axis=1)
    return (clipped,colors,np.flatnonzero(keep)) if return_indices else (clipped,colors)


def paint_tracts(canvas, painter, rect):
    """Cache clipped geometry, not screen pixels, so zoom and crosshairs stay cheap."""
    from PySide6.QtCore import QLineF, QPointF, Qt
    from PySide6.QtGui import QColor, QPen
    from .imaging import plane_axes
    scene = canvas.scene
    if not scene or not scene.tracts:
        canvas._tract_cache = {}
        return
    cache = getattr(canvas, '_tract_cache', {})
    active = {b.uid for b in scene.tracts}
    cache = {k: v for k, v in cache.items() if k in active}
    canvas._tract_cache = cache
    h, v = plane_axes(canvas.axis)
    plane = scene.world(canvas.ijk)[canvas.axis]
    painter.save()
    painter.setClipRect(rect)
    painter.setRenderHint(painter.RenderHint.Antialiasing)
    for bundle in scene.tracts:
        if not bundle.visible or not bundle.visible_2d or bundle.opacity <= 0:
            continue
        key = (id(bundle.points), id(scene), canvas.axis, float(plane), bundle.slab_mm,
               bundle.edit_revision,getattr(bundle,'_highlight_revision',0))
        item = cache.get(bundle.uid)
        if item is None or item[0] != key:
            # Reuse immutable RAS segments between slices and panel instances.
            geometry,ids=tract_segment_data(bundle)
            lines,colors,kept=clip_segments_to_slab(geometry,canvas.axis,plane,bundle.slab_mm,return_indices=True)
            selected=getattr(bundle,'_highlight_ids',[])
            if len(selected):
                # Direction colors can already be red; mute the context during review.
                colors[:]=[.28,.34,.40]
                colors[np.isin(ids[kept],selected)]=[1.,.15,.15]
            ijk = scene.index(lines.reshape(-1, 3)).reshape(-1, 2, 3)
            # Quantized direction colors allow grouped native draw calls.
            rgb = np.clip(np.round(colors*7), 0, 7).astype(int)
            codes = rgb[:, 0]*64+rgb[:, 1]*8+rgb[:, 2]
            groups = []
            for code in np.unique(codes):
                groups.append(((code//64/7, (code//8)%8/7, code%8/7), ijk[codes == code]))
            item = (key, groups)
            cache[bundle.uid] = item
        for rgb, ijk in item[1]:
            color = QColor.fromRgbF(*rgb, bundle.opacity)
            pen = QPen(color, 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            x = rect.right()-(ijk[:, :, h]+.5)/scene.data.shape[h]*rect.width()
            y = rect.bottom()-(ijk[:, :, v]+.5)/scene.data.shape[v]*rect.height()
            painter.drawLines([QLineF(QPointF(a, b), QPointF(c, d))
                               for a, b, c, d in zip(x[:, 0], y[:, 0], x[:, 1], y[:, 1])])
    painter.restore()


def direction_plane(scene, layer, axis, index):
    model=next((m for m in scene.diffusions if m.uid==layer.quality.get('diffusion_uid')),None)
    if model is None:return None
    axes=[i for i in range(3) if i!=axis]
    shape=tuple(scene.data.shape[i] for i in axes)
    grid=np.indices(shape,dtype=float).reshape(2,-1)
    ijk=np.empty((len(grid[0]),3));ijk[:,axis]=index
    ijk[:,axes]=grid.T
    mapping=np.linalg.inv(model.to_reference @ model.affine) @ scene.affine
    native=nib.affines.apply_affine(mapping,ijk).T
    # Nearest tensor eigenvector avoids cancellation between opposite eigenvector signs.
    direction=np.column_stack([map_coordinates(model.directions[...,i],native,order=0,
                                               mode='constant',cval=0,prefilter=False) for i in range(3)])
    basis=model.affine[:3,:3]/np.linalg.norm(model.affine[:3,:3],axis=0)
    world=direction @ (model.to_reference[:3,:3] @ basis).T
    world=np.abs(world)/np.maximum(np.linalg.norm(world,axis=1,keepdims=True),1e-8)
    fa=np.take(layer.data,int(index),axis=axis)
    return (world.reshape(shape+(3,))*np.clip(fa,0,1)[...,None]*255).astype(np.float32)


def paint_roi_spheres(canvas,painter,rect):
    from PySide6.QtCore import QPointF,Qt
    from PySide6.QtGui import QColor,QPen
    from .imaging import plane_axes
    if not getattr(canvas,'diffusion_rois',None):return
    plane=canvas.scene.world(canvas.ijk)[canvas.axis]
    h,v=plane_axes(canvas.axis)
    painter.save();painter.setClipRect(rect)
    for i,roi in enumerate(canvas.diffusion_rois):
        if not roi or roi.get('kind')!='sphere':continue
        center=np.asarray(roi['center']);delta=abs(center[canvas.axis]-plane)
        if delta>roi['radius']:continue
        radius=np.sqrt(max(0,roi['radius']**2-delta**2))
        point=canvas.position_for_index(canvas.scene.index(center))
        rx=radius*rect.width()/(canvas.scene.data.shape[h]*canvas.scene.spacing[h])
        ry=radius*rect.height()/(canvas.scene.data.shape[v]*canvas.scene.spacing[v])
        color=QColor('#faac70' if i==0 else '#7edaf0')
        painter.setPen(QPen(color,1.8));painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(point,rx,ry);painter.drawText(point+QPointF(rx+3,-4),f'ROI {i+1}')
    painter.restore()
