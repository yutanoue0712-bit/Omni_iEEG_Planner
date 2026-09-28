"""SEEG contact candidates from native CT only, in scanner RAS millimetres.

No MRI, anatomy segmentation, saved contacts or external application are inputs.
Automatic results always require review; an image-derived chain is not a device specification.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import colorsys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from .imaging import InputError

LOCALIZATION_VERSION = "native-ct-seeg-3"


@dataclass(frozen=True)
class DetectionSettings:
    threshold_hu: float = 2500.
    minimum_contacts: int = 5
    minimum_spacing_mm: float = 2.
    maximum_spacing_mm: float = 10.
    line_tolerance_mm: float = 1.35


def physical_points(affine, indices):
    return nib.affines.apply_affine(affine, indices)


def sample_ct(data, affine, points, order=1):
    indices = physical_points(np.linalg.inv(affine), np.asarray(points))
    return ndi.map_coordinates(data, indices.T, order=order, mode="constant", cval=-1024.)


def orthogonal_basis(direction):
    axis = np.asarray(direction,dtype=float)
    axis /= np.linalg.norm(axis)
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis,reference); u /= np.linalg.norm(u)
    return axis,u,np.cross(axis,u)


def metal_candidates(data, affine, settings=DetectionSettings(), progress=lambda _:None):
    spacing = np.linalg.norm(affine[:3,:3],axis=0)
    voxel_volume = abs(np.linalg.det(affine[:3,:3]))
    candidates=[]
    # Higher thresholds can separate metal contacts joined at a lower threshold.
    maximum=float(np.max(data))
    thresholds=sorted(set([settings.threshold_hu,min(maximum*.93,settings.threshold_hu+400)]))
    for threshold in thresholds:
        components,count=ndi.label(data>=threshold,structure=np.ones((3,3,3)))
        sizes=np.bincount(components.ravel())
        for label,box in enumerate(ndi.find_objects(components),1):
            volume=sizes[label]*voxel_volume
            if box is None or not .25 <= volume <= 45: continue
            extents=np.array([s.stop-s.start for s in box])*spacing
            if extents.max()>9: continue
            local=np.argwhere(components[box]==label)
            indices=local+np.array([s.start for s in box])
            world=physical_points(affine,indices)
            candidates.append({"ct_ras_mm":world.mean(0).tolist(),"volume_mm3":float(volume),
                               "threshold_hu":float(threshold),"voxels":int(len(indices))})
    # Prefer lower-threshold centroids when two thresholds identify the same contact.
    unique=[]
    for item in candidates:
        p=np.asarray(item['ct_ras_mm'])
        if unique and np.min(np.linalg.norm(np.array([c['ct_ras_mm'] for c in unique])-p,axis=1)) < 1.25: continue
        unique.append(item)
    progress(f"CT金属候補：{len(unique)}点")
    return unique


def head_depth(data, affine, points):
    # CT-only external head outline. Filling is slice-wise to tolerate intracranial air.
    mask=np.zeros(data.shape,bool)
    for z in range(data.shape[2]):
        labels,n=ndi.label(data[:,:,z]>-300)
        if n:
            size=np.bincount(labels.ravel()); size[0]=0
            mask[:,:,z]=ndi.binary_fill_holes(labels==size.argmax())
    depth=ndi.distance_transform_edt(mask,sampling=np.linalg.norm(affine[:3,:3],axis=0))
    indices=physical_points(np.linalg.inv(affine),points)
    return ndi.map_coordinates(depth,indices.T,order=1,mode='constant',cval=0.)


def fit_line(points):
    center=points.mean(0)
    _,_,vectors=np.linalg.svd(points-center,full_matrices=False)
    axis=vectors[0]
    t=(points-center) @ axis
    residual=np.linalg.norm(points-center-t[:,None]*axis,axis=1)
    return center,axis,t,residual


def regular_chain(points, indices, settings):
    center,axis,t,perp=fit_line(points[indices])
    order=np.argsort(t); indices=indices[order]; t=t[order]; perp=perp[order]
    differences=np.diff(t)
    hypotheses=np.unique(np.round(np.concatenate([differences,differences/2]),1))
    hypotheses=hypotheses[(hypotheses>=settings.minimum_spacing_mm)&(hypotheses<=settings.maximum_spacing_mm)]
    best=None
    for spacing in hypotheses:
        for phase in t:
            numbers=np.rint((t-phase)/spacing).astype(int)
            error=np.abs(t-phase-numbers*spacing)
            hits=np.flatnonzero((error<min(.8,spacing*.18)) & (perp<settings.line_tolerance_mm))
            chosen=[]
            for number in np.unique(numbers[hits]):
                choices=hits[numbers[hits]==number]
                chosen.append(choices[np.argmin(error[choices]+perp[choices]*.4)])
            if len(chosen)<settings.minimum_contacts: continue
            chosen=np.array(chosen)
            # Do not join distant collinear hardware into one shaft.
            for segment in np.split(chosen,np.flatnonzero(np.diff(numbers[chosen])>2)+1):
                if len(segment)<settings.minimum_contacts: continue
                slots=numbers[segment]
                missing=int(slots[-1]-slots[0]+1-len(segment))
                score=len(segment)-.7*missing-.25*float(perp[segment].mean())-.2*float(error[segment].mean())
                if best is None or score>best['score']:
                    used=indices[segment]
                    line_center,line_axis,projection,residual=fit_line(points[used])
                    if np.dot(line_axis,axis)<0: line_axis=-line_axis; projection=-projection
                    # Regress the observed centres against the integer contact indices.
                    numbers_used=slots-slots.min()
                    step,origin=np.polyfit(numbers_used,projection,1)
                    best={'score':float(score),'indices':used,'numbers':numbers_used,'spacing_mm':float(abs(step)),
                          'origin':line_center+line_axis*origin,'axis':line_axis if step>0 else -line_axis,
                          'line_rms_mm':float(np.sqrt(np.mean(residual**2))),'missing':missing}
    return best


def find_chains(points, depths, settings, progress=lambda _:None):
    if len(points)<settings.minimum_contacts: return []
    distances=squareform(pdist(points))
    pairs=np.argwhere(np.triu((distances>8)&(distances<90),1))
    if not len(pairs): return []
    origins=points[pairs[:,0]]
    directions=points[pairs[:,1]]-origins
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    proposals=[]
    # Evaluate line hypotheses in bounded blocks, with no stochastic RANSAC seed.
    for start in range(0,len(pairs),256):
        delta=points[None,:,:]-origins[start:start+256,None,:]
        direction=directions[start:start+256]
        t=np.einsum('bnj,bj->bn',delta,direction)
        d2=np.maximum(np.sum(delta*delta,axis=2)-t*t,0)
        mask=d2<settings.line_tolerance_mm**2
        count=mask.sum(1)
        for j in np.flatnonzero(count>=settings.minimum_contacts):
            members=np.flatnonzero(mask[j])
            if np.count_nonzero(depths[members]>7)<settings.minimum_contacts-1: continue
            proposals.append((int(count[j]),members))
    proposals.sort(key=lambda x:-x[0])
    unique=set(); chains=[]
    for _,members in proposals:
        key=tuple(members)
        if key in unique: continue
        unique.add(key)
        candidate=regular_chain(points,members,settings)
        if candidate is not None: chains.append(candidate)
        if len(unique)>=2000: break
    chains.sort(key=lambda x:-x['score'])
    used=set(); accepted=[]
    for candidate in chains:
        indices=candidate['indices']
        remaining=np.array([i for i in indices if int(i) not in used])
        if len(remaining)<settings.minimum_contacts: continue
        if len(remaining)!=len(indices):
            candidate=regular_chain(points,remaining,settings)
            if candidate is None: continue
        accepted.append(candidate); used.update(map(int,candidate['indices']))
    progress(f"CTから電極の並びを検出：{len(accepted)}本")
    return accepted


def refine_contact(data, affine, guess, axis, spacing):
    """Local CT centroid and independent gap contrast; never return an unsupported point as detected."""
    axis,u,v=orthogonal_basis(axis)
    guess=np.asarray(guess)
    # Native voxels within a small physical tube; includes anti-aliased metal, excludes the next contact.
    native=physical_points(np.linalg.inv(affine),guess)
    voxsize=np.linalg.norm(affine[:3,:3],axis=0)
    radius=np.ceil(3.5/voxsize).astype(int)
    lower=np.maximum(np.floor(native).astype(int)-radius,0)
    upper=np.minimum(np.ceil(native).astype(int)+radius+1,data.shape)
    if np.any(upper<=lower): return guess,{'supported':False,'peak_hu':-1024.,'gap_contrast_hu':0.}
    coords=np.indices(tuple(upper-lower)).reshape(3,-1).T+lower
    world=physical_points(affine,coords)
    diff=world-guess; along=diff @ axis
    radial=np.linalg.norm(diff-along[:,None]*axis,axis=1)
    values=data[tuple(coords.T)]
    mask=(np.abs(along)<min(1.8,spacing*.38))&(radial<1.6)
    peak=float(values[mask].max()) if mask.any() else -1024.
    metal=mask&(values>max(1600.,min(2400.,peak*.78)))
    center=guess.copy()
    if metal.sum()>=2:
        weights=np.maximum(values[metal]-1500,1.) * np.exp(-radial[metal]**2/(2*1.1**2))
        center=np.average(world[metal],axis=0,weights=weights)
    offsets=np.array([[0,0],[.5,0],[-.5,0],[0,.5],[0,-.5]])
    disk=offsets[:,0,None]*u+offsets[:,1,None]*v
    middle=float(np.max(sample_ct(data,affine,center+disk)))
    flank=np.array([np.max(sample_ct(data,affine,center+sign*spacing*.5*axis+disk)) for sign in (-1,1)])
    contrast=middle-float(np.max(flank))
    return center,{'supported':bool(peak>=2200 and metal.sum()>=2 and contrast>=250),
        'peak_hu':peak,'gap_contrast_hu':float(contrast),'metal_voxels':int(metal.sum()),
        'centroid_shift_mm':float(np.linalg.norm(center-guess))}


def remove_crossing_extensions(groups):
    """A neighbouring lead's observed metal contact cannot extend another lead."""
    observed=[(i,c) for i,g in enumerate(groups) for c in g['contacts'] if c['evidence']['origin']!='end_extension']
    if not observed: return
    positions=np.array([c['ct_ras_mm'] for _,c in observed]); tree=cKDTree(positions)
    for i,group in enumerate(groups):
        kept=[]
        for contact in group['contacts']:
            overlap=False
            if contact['evidence']['origin']=='end_extension':
                overlap=any(observed[j][0]!=i for j in tree.query_ball_point(contact['ct_ras_mm'],.9))
            if not overlap: kept.append(contact)
        group['contacts']=kept


def detect_electrodes(data, affine, settings=DetectionSettings(), progress=lambda _:None):
    if data.ndim!=3 or not np.isfinite(data).all(): raise InputError('CTの表示格子が不正です。')
    if not 500<=settings.threshold_hu or not 2<=settings.minimum_contacts<=64 or not 1<=settings.minimum_spacing_mm<settings.maximum_spacing_mm<=20:
        raise InputError('CTの金属抽出しきい値が画像の範囲に合いません。')
    if settings.threshold_hu>=float(np.max(data)):
        return {'version':LOCALIZATION_VERSION,'source':'native_ct_only','settings':settings.__dict__,
                'groups':[],'candidates':[],'review_status':'unreviewed'}
    candidates=metal_candidates(data,affine,settings,progress)
    points=np.array([c['ct_ras_mm'] for c in candidates]).reshape(-1,3)
    if len(points)<settings.minimum_contacts:
        return {'version':LOCALIZATION_VERSION,'source':'native_ct_only','groups':[],'candidates':candidates}
    depths=head_depth(data,affine,points)
    chains=find_chains(points,depths,settings,progress)
    groups=[]
    for chain in chains:
        count=int(chain['numbers'].max())+1
        axis=chain['axis']; origin=chain['origin']; spacing=chain['spacing_mm']
        observed={int(n):int(i) for n,i in zip(chain['numbers'],chain['indices'])}
        contacts=[]
        for number in range(count):
            guess=points[observed[number]] if number in observed else origin+number*spacing*axis
            center,evidence=refine_contact(data,affine,guess,axis,spacing)
            evidence['origin']='component' if number in observed else 'internal_gap'
            contacts.append({'ct_ras_mm':(center if evidence['supported'] else guess).tolist(),'evidence':evidence,
                             'status':'unreviewed' if evidence['supported'] else 'uncertain'})
        if sum(c['evidence']['supported'] for c in contacts)<settings.minimum_contacts: continue
        # Add a neighbouring centre only with its own CT peak and a dark inter-contact gap.
        for end in (0,-1):
            for _ in range(2):
                guess=np.array(contacts[end]['ct_ras_mm'])+(-1 if end==0 else 1)*spacing*axis
                center,evidence=refine_contact(data,affine,guess,axis,spacing)
                if not evidence['supported']: break
                evidence['origin']='end_extension'
                item={'ct_ras_mm':center.tolist(),'evidence':evidence,'status':'uncertain'}
                contacts.insert(0,item) if end==0 else contacts.append(item)
        # Depths at the observed end candidates avoid recomputing a whole CT mask.
        end_depths=depths[chain['indices'][[0,-1]]]
        if end_depths[0]<end_depths[1]: contacts.reverse()
        groups.append({'contacts':contacts,'spacing_mm':spacing,'line_rms_mm':chain['line_rms_mm'],
            'status':'unreviewed','numbering':'CT depth heuristic; tip-first requires review',
            'direction_ambiguous':bool(abs(end_depths[0]-end_depths[1])<3),'expected_count':None})
    remove_crossing_extensions(groups)
    groups.sort(key=lambda g:tuple(np.round(g['contacts'][0]['ct_ras_mm'],1)))
    for i,group in enumerate(groups):
        group['name']=f'E{i+1:02d}'
        group['color']=list(colorsys.hsv_to_rgb((i*.61803398875)%1,.66,.98))
        for j,contact in enumerate(group['contacts']): contact['name']=f"{group['name']}-{j+1:02d}"
    return {'version':LOCALIZATION_VERSION,'source':'native_ct_only','settings':settings.__dict__,
            'groups':groups,'candidates':candidates,'review_status':'unreviewed'}


def cached_detection(data, affine, cache, settings=DetectionSettings(), progress=lambda _:None):
    from .volume_io import validate_ct_intensities
    validate_ct_intensities(data)
    progress('電極候補の保存済み結果を確認しています…')
    fingerprint=hashlib.sha256(LOCALIZATION_VERSION.encode())
    for array in (data,affine): fingerprint.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    fingerprint.update(json.dumps(settings.__dict__,sort_keys=True).encode())
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    path=cache/(fingerprint.hexdigest()+'.json')
    if path.is_file():
        obj=json.loads(path.read_text(encoding='utf-8'))
        if obj.get('source')=='native_ct_only' and obj.get('version')==LOCALIZATION_VERSION:
            progress('保存済みの電極候補を読み込みました。')
            return obj
    progress('CTの金属像から電極候補を抽出しています…')
    result=detect_electrodes(data,affine,settings,progress)
    result['ct_fingerprint']=fingerprint.hexdigest()
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)
    return result


def attach_detection(scene, result):
    from dataclasses import replace
    from .imaging import Contact
    contacts=[]
    for group in result['groups']:
        for entry in group['contacts']:
            raw=np.array(entry['ct_ras_mm'],dtype=float)
            contacts.append(Contact(entry['name'],group['name'],physical_points(scene.ct_to_mri,raw),
                tuple(group['color']),ct_position=raw,provenance='native_ct_only',status=entry['status'],
                evidence=dict(entry['evidence'])))
    quality={key:value for key,value in result.items() if key not in ('groups',)}
    quality['groups']={group['name']:{key:value for key,value in group.items() if key not in ('contacts','name','color')} for group in result['groups']}
    quality['manufacturer']='AD-Tech (user specified); model not specified'
    quality['contact_count']=len(contacts)
    attached=replace(scene,contacts=contacts,electrode_quality=quality)
    from .electrode_editing import refresh_quality
    refresh_quality(attached)
    return attached
