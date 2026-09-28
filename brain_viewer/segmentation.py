"""Editable patient-space masks derived from existing anatomy or manual slice strokes."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import nibabel as nib
import numpy as np
from .imaging import InputError, Segment, plane_axes

# FreeSurfer's official LUT: distribution/FreeSurferColorLUT.txt.
# Selected label identifiers: see NOTICE and LICENSES/FreeSurfer-LICENSE.txt.
# Ventricles: lateral/inferior lateral, third and fourth; excludes CSF and choroid plexus.
PRESETS = {
    'ventricles': ('脳室', (4,5,14,15,43,44), (.26,.78,.96)),
    'callosum': ('脳梁', (192,251,252,253,254,255), (.97,.76,.30)),
    'thalamus': ('視床（左右）', (10,49), (.66,.49,.94)),
    'lesion': ('Lesion（手動）', (), (.95,.36,.47)),
    'artery': ('Artery（手動）', (), (.98,.33,.24)),
    'vessel': ('Vessel（手動）', (), (.28,.80,.68)),
    'manual': ('任意の領域（手動）', (), (.89,.58,.27)),
}


def create_segment(scene, preset, name):
    if preset not in PRESETS or not name.strip():
        raise InputError('領域の種類と名前を指定してください。')
    _, labels, color = PRESETS[preset]
    if labels:
        if scene.label_volume is None:
            raise InputError('対応するFreeSurferラベルがありません。手動領域を使用してください。')
        mask = np.isin(scene.label_volume, labels)
        found = [label for label in labels if np.any(scene.label_volume == label)]
        if not mask.any():
            raise InputError('この構造のFreeSurferラベルがありません。手動領域を使用してください。')
        provenance = {'method':'existing_FreeSurfer_labels', 'source':scene.label_source,
                      'requested_label_ids':list(labels), 'present_label_ids':found}
    else:
        mask = np.zeros(scene.data.shape, dtype=bool)
        provenance = {'method':'manual', 'source':'reference_MRI_display_grid'}
    provenance.update(created=datetime.now(timezone.utc).isoformat(), review_status='requires_visual_review',
                      grid='reference_MRI_display', spacing_mm=scene.spacing.tolist())
    return Segment('seg_'+uuid4().hex, name.strip()[:100], preset, mask, color, provenance=provenance)


def paint_stroke(mask, spacing, axis, start, end, radius_mm, value):
    """Paint a continuous physical-radius disk along one slice; never interpolate labels."""
    start, end = np.asarray(start,float), np.asarray(end,float)
    spacing = np.asarray(spacing,float)
    if axis not in (0,1,2) or start.shape!=(3,) or end.shape!=(3,) or not np.isfinite([*start,*end,radius_mm]).all():
        raise InputError('描画位置が不正です。')
    if not 0<radius_mm<=30 or np.any(spacing<=0): raise InputError('ブラシ半径が不正です。')
    index = int(round(end[axis]))
    if not 0<=index<mask.shape[axis]: return
    h,v = plane_axes(axis)
    if round(start[axis])!=index: start = end.copy()
    distance = np.linalg.norm((end-start)[[h,v]]*spacing[[h,v]])
    steps = max(1, int(np.ceil(distance/max(.2,min(radius_mm/2,spacing[[h,v]].min()/2)))))
    for point in np.linspace(start,end,steps+1):
        lower = np.maximum(0,np.floor(point[[h,v]]-radius_mm/spacing[[h,v]]).astype(int))
        upper = np.minimum(np.array(mask.shape)[[h,v]],np.ceil(point[[h,v]]+radius_mm/spacing[[h,v]]).astype(int)+1)
        if np.any(lower>=upper): continue
        x,y = np.ogrid[lower[0]:upper[0], lower[1]:upper[1]]
        disk = ((x-point[h])*spacing[h])**2+((y-point[v])*spacing[v])**2<=radius_mm**2
        slices = [slice(None)]*3
        slices[axis] = index; slices[h] = slice(lower[0],upper[0]); slices[v] = slice(lower[1],upper[1])
        view = mask[tuple(slices)]
        view[disk] = value


def fill_contour(mask, spacing, axis, vertices, value=True, thickness_mm=None):
    """Fill a closed contour at voxel centres, in a bounded physical slice slab."""
    from matplotlib.path import Path as PolygonPath
    vertices = np.asarray(vertices, float)
    spacing = np.asarray(spacing, float)
    if (axis not in (0, 1, 2) or vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all() or spacing.shape != (3,) or np.any(spacing <= 0)):
        raise InputError('描画位置が不正です。')
    if len(vertices) < 3:
        return
    index = int(round(vertices[0, axis]))
    if not 0 <= index < mask.shape[axis] or np.any(np.abs(vertices[:, axis] - index) > .01):
        raise InputError('同じ断面上で領域を囲んでください。')
    thickness = spacing[axis] if thickness_mm is None else float(thickness_mm)
    if not np.isfinite(thickness) or not 0 < thickness <= 30:
        raise InputError('ROIの厚みが不正です。')
    h, v = plane_axes(axis)
    polygon = vertices[:, [h, v]]
    # A click or a straight stroke is not an enclosed region.
    area = abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
               - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) / 2
    if area < .5:
        return
    lo = np.maximum(0, np.floor(polygon.min(0)).astype(int))
    hi = np.minimum(np.array(mask.shape)[[h, v]], np.ceil(polygon.max(0)).astype(int)+1)
    if np.any(hi <= lo):
        return
    grid = np.stack(np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing='ij'), -1)
    closed = np.vstack([polygon, polygon[0]])
    inside = PolygonPath(closed).contains_points(grid.reshape(-1, 2), radius=1e-7).reshape(grid.shape[:2])
    # Even widths use a half-open interval to avoid silently adding an extra slice.
    offsets = (np.arange(mask.shape[axis]) - index) * spacing[axis]
    planes = np.flatnonzero((offsets >= -thickness/2) & (offsets < thickness/2))
    if not len(planes):
        planes = [index]
    for plane in planes:
        slices = [slice(None)] * 3
        slices[axis] = int(plane)
        slices[h], slices[v] = slice(lo[0], hi[0]), slice(lo[1], hi[1])
        mask[tuple(slices)][inside] = value


def mask_surface(mask, affine):
    """Voxel boundary mesh in scanner RAS mm. No geometric smoothing changes the mask."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    points = np.argwhere(mask)
    output = vtk.vtkPolyData()
    if not len(points): return output
    lower, upper = points.min(0), points.max(0)+1
    cropped = np.pad(mask[tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))].astype(np.uint8),1)
    image = vtk.vtkImageData(); image.SetDimensions(*cropped.shape)
    image.GetPointData().SetScalars(numpy_to_vtk(cropped.ravel(order='F'),deep=True))
    contour = vtk.vtkFlyingEdges3D(); contour.SetInputData(image); contour.SetValue(0,.5)
    matrix = np.asarray(affine,float).copy()
    matrix[:3,3] = nib.affines.apply_affine(affine,lower-1)
    transform = vtk.vtkTransform(); transform.SetMatrix(matrix.ravel())
    physical = vtk.vtkTransformPolyDataFilter(); physical.SetTransform(transform); physical.SetInputConnection(contour.GetOutputPort())
    physical.Update(); output.DeepCopy(physical.GetOutput())
    field = vtk.vtkStringArray(); field.SetName('CoordinateSystem'); field.InsertNextValue('scanner_RAS_mm')
    output.GetFieldData().AddArray(field)
    return output


def export_mask(segment, scene, path):
    image = nib.Nifti1Image(segment.mask.astype(np.uint8),scene.affine)
    image.header.set_xyzt_units('mm'); image.header.set_intent('label')
    image.set_sform(scene.affine,code=1); image.set_qform(scene.affine,code=1)
    nib.save(image, str(Path(path)))


def export_surface(segment, scene, path):
    import vtk
    if not segment.mask.any(): raise InputError('空の領域は3Dモデルに出力できません。')
    writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(path))
    writer.SetInputData(mask_surface(segment.mask,scene.affine))
    if not writer.Write(): raise InputError('領域の3Dモデルを保存できませんでした。')
