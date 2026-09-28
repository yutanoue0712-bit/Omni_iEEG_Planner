"""Review operations with CT-native coordinates as the canonical contact positions."""
from copy import deepcopy
from dataclasses import replace
import colorsys
import numpy as np
import nibabel as nib
from .imaging import Contact, InputError
from .electrode_localization import refine_contact, fit_line
from scipy.spatial import cKDTree


def group_contacts(scene, name):
    return [c for c in scene.contacts if c.group==name]


def contact_axis(contacts):
    points=np.array([c.ct_position for c in contacts])
    if len(points)<2: return np.array([1.,0.,0.]),5.
    _,axis,_,_=fit_line(points)
    if np.dot(axis,points[-1]-points[0])<0: axis=-axis
    return axis,max(1.,float(np.median(np.linalg.norm(np.diff(points,axis=0),axis=1))))


def check_ct_bounds(scene, point):
    index=nib.affines.apply_affine(np.linalg.inv(scene.raw_ct_affine),point)
    if np.any(index<0) or np.any(index>np.array(scene.raw_ct.shape)-1):
        raise InputError('コンタクトがCTの撮像範囲外です。')


def refresh_quality(scene):
    groups=scene.electrode_quality.setdefault('groups',{})
    names=list(dict.fromkeys(c.group for c in scene.contacts))
    for old in list(groups):
        if old not in names: groups.pop(old)
    for name in names:
        contacts=group_contacts(scene,name)
        metadata=groups.setdefault(name,{})
        metadata['count']=len(contacts)
        metadata['status']='reviewed' if all(c.status=='reviewed' for c in contacts) else 'unreviewed'
        if len(contacts)>1:
            points=np.array([c.ct_position for c in contacts])
            metadata['spacing_mm']=float(np.median(np.linalg.norm(np.diff(points,axis=0),axis=1)))
            metadata['line_rms_mm']=float(np.sqrt(np.mean(fit_line(points)[3]**2)))
    for c in scene.contacts: c.evidence.pop('nearby_contact_uids',None)
    if scene.contacts:
        pairs=cKDTree(np.array([c.ct_position for c in scene.contacts])).query_pairs(.9)
        for a,b in pairs:
            scene.contacts[a].evidence.setdefault('nearby_contact_uids',[]).append(scene.contacts[b].uid)
            scene.contacts[b].evidence.setdefault('nearby_contact_uids',[]).append(scene.contacts[a].uid)
    scene.electrode_quality['contact_count']=len(scene.contacts)
    scene.electrode_quality['review_status']='reviewed' if scene.contacts and all(c.status=='reviewed' for c in scene.contacts) else 'unreviewed'


def renumber(contacts, name):
    for number,contact in enumerate(contacts,1):
        contact.group=name
        contact.name=f'{name}-{number:02d}'


def set_contact_position(scene, contact, ct_position):
    point=np.asarray(ct_position,dtype=float)
    if point.shape!=(3,) or not np.isfinite(point).all(): raise InputError('電極座標が不正です。')
    check_ct_bounds(scene,point)
    previous=contact.ct_position.copy()
    contact.ct_position=point.copy()
    contact.position=nib.affines.apply_affine(scene.ct_to_mri,point)
    contact.status='unreviewed'; contact.provenance='native_ct_only'
    axis,spacing=contact_axis(group_contacts(scene,contact.group))
    _,evidence=refine_contact(scene.raw_ct,scene.raw_ct_affine,point,axis,spacing)
    contact.status='unreviewed' if evidence['supported'] else 'uncertain'
    contact.evidence={**evidence,'origin':'manual_adjustment',
        'history':[*contact.evidence.get('history',[]),{'previous_ct_ras_mm':previous.tolist(),'ct_ras_mm':point.tolist()}]}
    scene.electrode_quality.setdefault('groups',{}).setdefault(contact.group,{})['status']='unreviewed'


def create_lead(scene, tip, last, count, name):
    tip,last=np.asarray(tip,dtype=float),np.asarray(last,dtype=float)
    if count<2 or count>64 or tip.shape!=(3,) or last.shape!=(3,) or not np.isfinite([tip,last]).all():
        raise InputError('電極の端点とコンタクト数を確認してください。')
    if any(c.group==name for c in scene.contacts) or not name.strip(): raise InputError('電極名が重複しているため、読み込みを中止しました。')
    length=float(np.linalg.norm(last-tip)); spacing=length/(count-1)
    if not 1<=spacing<=20: raise InputError('コンタクト数と中心間隔の組み合わせを確認してください。')
    axis=(last-tip)/length
    check_ct_bounds(scene,tip); check_ct_bounds(scene,last)
    color=colorsys.hsv_to_rgb((len({c.group for c in scene.contacts})*.61803398875)%1,.66,.98)
    contacts=[]
    for i,guess in enumerate(np.linspace(tip,last,count)):
        center,evidence=refine_contact(scene.raw_ct,scene.raw_ct_affine,guess,axis,spacing)
        point=center if evidence['supported'] else guess
        contacts.append(Contact(f'{name}-{i+1:02d}',name,nib.affines.apply_affine(scene.ct_to_mri,point),color,
            ct_position=point,provenance='native_ct_only',status='unreviewed' if evidence['supported'] else 'uncertain',
            evidence={**evidence,'origin':'user_seeded_trajectory'}))
    quality=deepcopy(scene.electrode_quality)
    quality['source']='native_ct_only'
    quality.setdefault('groups',{})[name]={'spacing_mm':spacing,'expected_count':count,
        'status':'unreviewed','numbering':'user selected tip first','manufacturer':'AD-Tech','model':None}
    quality['contact_count']=len(scene.contacts)+count
    result=replace(scene,contacts=[*deepcopy(scene.contacts),*contacts],electrode_quality=quality)
    refresh_quality(result)
    return result


def export_records(scene):
    records=[]
    for contact in scene.contacts:
        label,name=scene.annotation(np.rint(scene.index(contact.position)).astype(int))
        nucleus,nucleus_name=scene.nucleus_at(contact.position)
        if nucleus: label,name=nucleus,nucleus_name
        records.append({'uid':contact.uid,'name':contact.name,'group':contact.group,'type':contact.kind,
            'mri_ras_mm':contact.position.tolist(),'ct_ras_mm':contact.ct_position.tolist() if contact.ct_position is not None else None,
            'ct_voxel_zero_based':nib.affines.apply_affine(np.linalg.inv(scene.raw_ct_affine),contact.ct_position).tolist() if contact.ct_position is not None else None,
            'anatomy_id':label,'anatomy_name':name,'status':contact.status,'source':contact.provenance,'evidence':contact.evidence})
    return {'schema':'cortex-contacts/1','coordinate_system':'scanner_RAS_mm','contacts':records,
        'ct_to_mri_ras_mm':scene.ct_to_mri.tolist() if scene.ct_to_mri is not None else None,'provenance':scene.electrode_quality}
