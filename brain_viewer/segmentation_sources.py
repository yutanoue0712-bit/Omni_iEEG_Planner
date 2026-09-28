"""Label selection and sequence-specific extraction on the patient's reference grid."""
from dataclasses import dataclass
from datetime import datetime, timezone
import colorsys
import hashlib
from uuid import uuid4

import numpy as np
from scipy import ndimage
from .imaging import InputError, Segment
from .segmentation import PRESETS


def available_labels(scene):
    records = []
    for source, data, names in (('anatomy',scene.label_volume,scene.label_names),
                                ('nuclei',scene.nuclei_display,scene.nuclei_names)):
        if data is None: continue
        labels, counts = np.unique(data, return_counts=True)
        for label, count in zip(labels, counts):
            if label <= 0: continue
            label = int(label)
            records.append({'source':source,'label':label,'name':names.get(label,f'Label {label}'),
                            'voxels':int(count)})
    return records


def create_label_segment(scene, source, label, name=None):
    if source not in ('anatomy','nuclei'): raise InputError('解剖ラベルの種類が不正です。')
    data = scene.label_volume if source=='anatomy' else scene.nuclei_display
    names = scene.label_names if source=='anatomy' else scene.nuclei_names
    if label <= 0 or data is None or not np.any(data==label):
        raise InputError('選択した解剖ラベルは表示格子にありません。')
    color = colorsys.hsv_to_rgb(((int(label)%1000)*.61803398875)%1,.58,.96)
    provenance = {'method':'existing_FreeSurfer_labels','label_collection':source,
                  'source':scene.label_source if source=='anatomy' else scene.nuclei_source,
                  'requested_label_ids':[int(label)],'present_label_ids':[int(label)],
                  'created':datetime.now(timezone.utc).isoformat(), 'review_status':'requires_visual_review',
                  'grid':'reference_MRI_display','spacing_mm':scene.spacing.tolist()}
    return Segment('seg_'+uuid4().hex, name or names.get(label,f'Label {label}'), 'freesurfer_label',
                   (data==label), color, provenance=provenance)


def create_anatomy_roi(scene, records):
    """Copy selected existing labels into one editable ROI; retain label provenance."""
    if not records:
        raise InputError('解剖ラベルを選択してください。')
    first = records[0]
    segment = create_label_segment(scene, first['source'], int(first['label']))
    names = [segment.name]
    for record in records[1:]:
        other = create_label_segment(scene, record['source'], int(record['label']))
        segment.mask |= other.mask
        names.append(other.name)
    segment.name = (' + '.join(names))[:100]
    segment.provenance['roi_labels'] = [
        {'source':r['source'],'label':int(r['label']),'name':names[i]}
        for i,r in enumerate(records)]
    segment.provenance['requested_label_ids'] = [int(r['label']) for r in records]
    segment.provenance['present_label_ids'] = [int(r['label']) for r in records]
    segment.provenance['label_collection'] = 'combined' if len({r['source'] for r in records})>1 else first['source']
    return segment


@dataclass
class ImageSource:
    uid: str
    name: str
    sequence: str
    data: np.ndarray
    valid: np.ndarray | None
    units: str
    transform: np.ndarray
    native_affine: np.ndarray
    quality: dict
    deformation: dict | None = None


def image_sources(scene):
    if scene is None: return []
    sources = [ImageSource('mri','基準MRI','reference_MRI',scene.data,None,'intensity',np.eye(4),scene.source_affine,{})]
    for layer in scene.extra_mris:
        units = 'HU' if layer.sequence in ('CT-bone','CTA','CTV') else layer.quality.get('value_units','') or 'intensity'
        sources.append(ImageSource(layer.uid,layer.name,layer.sequence,layer.data,layer.valid,units,
                                   layer.to_reference,layer.raw_affine,layer.quality,layer.deformation))
    if scene.ct is not None:
        sources.append(ImageSource('ct','CT：電極留置後','post_implantation_CT',scene.ct,scene.ct_valid,'HU',
                                   scene.ct_to_mri,scene.raw_ct_affine,scene.ct_quality))
    return sources


def get_source(scene, uid):
    source = next((s for s in image_sources(scene) if s.uid==uid),None)
    if source is None: raise InputError('抽出元の画像がありません。画像を選び直してください。')
    return source


def extract_segment(scene, source_id, low, high, name, category='manual', seed=None, connected=True, radius_mm=30.):
    """Intensity interval plus optional six-connected component and physical sphere ROI.

    This creates a candidate, not a tissue classifier. Display window/level is never an input.
    """
    source = get_source(scene,source_id)
    if not name.strip() or category not in ('manual','lesion','artery','vessel'):
        raise InputError('領域の種類と名前を指定してください。')
    if not np.isfinite([low,high,radius_mm]).all() or low>high or not 0<=radius_mm<=250:
        raise InputError('画像値の下限・上限と抽出半径を確認してください。')
    needs_seed = connected or radius_mm>0
    point = None
    if needs_seed:
        point = np.asarray(seed,float)
        if point.shape!=(3,) or not np.isfinite(point).all():
            raise InputError('断面で位置を選び、抽出の起点を指定してください。')
        point = np.rint(point).astype(int)
        if np.any(point<0) or np.any(point>=scene.data.shape):
            raise InputError('抽出の起点が画像の範囲外です。')
    data = source.data
    valid = source.valid
    if valid is None:
        mapping = np.linalg.inv(scene.source_affine) @ scene.affine
        valid = ndimage.affine_transform(np.ones(scene.source_shape,np.uint8),mapping[:3,:3],mapping[:3,3],
                    output_shape=scene.data.shape,order=0,mode='constant',cval=0,prefilter=False).astype(bool)
    mask = np.isfinite(data) & valid & (data>=low) & (data<=high)
    if radius_mm>0:
        coordinates = np.ogrid[tuple(slice(0,n) for n in data.shape)]
        distance = sum(((axis-point[i])*scene.spacing[i])**2 for i,axis in enumerate(coordinates))
        mask &= distance <= radius_mm**2
    if connected:
        if not mask[tuple(point)]:
            raise InputError('起点の画像値が指定範囲外、または撮像範囲外です。起点と下限・上限を確認してください。')
        marker = np.zeros(mask.shape,bool); marker[tuple(point)]=True
        mask = ndimage.binary_propagation(marker,structure=ndimage.generate_binary_structure(3,1),mask=mask)
    if not mask.any(): raise InputError('この条件に合う領域がありません。画像値の範囲を調整してください。')
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast('B')).hexdigest()
    from .postop_deformation import metadata as deformation_metadata
    provenance = {'method':'image_intensity_candidate','source_id':source.uid,'source_name':source.name,
        'source_sequence':source.sequence,'source_units':source.units,'source_display_sha256':digest,
        'source_to_reference_ras_mm':source.transform.tolist(),
        'source_matrix_role':'rigid_component_only' if source.deformation else 'full_rigid_transform',
        'source_deformation':deformation_metadata(source.deformation),
        'source_native_affine':source.native_affine.tolist() if source.native_affine is not None else None,
        'source_registration_review':source.quality.get('review_status','reference_grid'),
        'threshold':[float(low),float(high)],'connected_6':bool(connected),
        'seed_reference_ijk':point.tolist() if needs_seed else None,'radius_mm':float(radius_mm),
        'created':datetime.now(timezone.utc).isoformat(),'review_status':'requires_visual_review',
        'grid':'reference_MRI_display','spacing_mm':scene.spacing.tolist()}
    return Segment('seg_'+uuid4().hex,name.strip()[:100],category,mask,PRESETS[category][2],provenance=provenance)


def display_segments(scene, context=None):
    """Editing masks have workflow visibility, independent of saved anatomy toggles."""
    context=context or {}
    workflow=context.get('workflow','images')
    result=[]
    for segment in scene.segmentations:
        exclusion=(segment.provenance.get('role')=='postop_registration_exclusion'
                   or bool(segment.provenance.get('postop_exclusion_for')))
        if exclusion and workflow!='segmentation':
            if workflow!='postop' or segment.uid!=context.get('exclusion'):continue
        result.append(segment)
    preview=scene.segmentation_preview
    if preview is not None:
        if preview.provenance.get('role')!='postop_exclusion_preview' or workflow=='postop':
            result.append(preview)
    return result
