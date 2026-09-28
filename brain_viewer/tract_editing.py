"""Reversible whole-streamline exclusions using physical slabs or camera-space lassos."""
from datetime import datetime,timezone
import numpy as np
from matplotlib.path import Path
from .imaging import InputError,plane_axes
from .diffusion_rendering import tract_segment_data,clip_segments_to_slab


def polygon_vertices(vertices):
    polygon=np.asarray(vertices,float)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or not np.isfinite(polygon).all():
        raise InputError('囲んだ範囲が不正です。')
    if len(polygon)<3:return np.empty((0,2))
    if np.linalg.norm(polygon[0]-polygon[-1])<1e-10:polygon=polygon[:-1]
    if len(polygon)<3:return np.empty((0,2))
    area=abs(np.dot(polygon[:,0],np.roll(polygon[:,1],-1))-np.dot(polygon[:,1],np.roll(polygon[:,0],-1)))
    return polygon if area>1e-10 else np.empty((0,2))


def segments_hit_polygon(lines,polygon):
    """Include crossings even when neither endpoint falls inside a concave polygon."""
    polygon=polygon_vertices(polygon);lines=np.asarray(lines,float)
    hits=np.zeros(len(lines),bool)
    if not len(polygon) or not len(lines):return hits
    lo,hi=polygon.min(0),polygon.max(0)
    candidate=np.flatnonzero(np.all(lines.max(1)>=lo,axis=1)&np.all(lines.min(1)<=hi,axis=1))
    if not len(candidate):return hits
    a,b=lines[candidate,0],lines[candidate,1]
    path=Path(np.vstack([polygon,polygon[0]]))
    hits[candidate]=path.contains_points(a)|path.contains_points(b)
    def cross(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    delta=b-a
    scale=max(1.,float(np.ptp(polygon,axis=0).max()));epsilon=1e-10*scale**2
    for c,d in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=d-c;denom=cross(delta,edge);parallel=np.abs(denom)<epsilon
        safe=np.where(parallel,1.,denom)
        t=cross(c-a,edge)/safe;s=cross(c-a,delta)/safe
        crossing=(~parallel)&(t>=-1e-9)&(t<=1+1e-9)&(s>=-1e-9)&(s<=1+1e-9)
        # Include tangencies and point intersections on polygon boundaries.
        collinear=parallel&(np.abs(cross(c-a,edge))<epsilon)
        overlap=np.all(np.maximum(np.minimum(a,b),np.minimum(c,d))<=
                       np.minimum(np.maximum(a,b),np.maximum(c,d))+1e-9,axis=1)
        hits[candidate]|=crossing|(collinear&overlap)
    return hits


def select_slice(bundle,scene,axis,vertices,thickness_mm):
    vertices=np.asarray(vertices,float)
    if axis not in (0,1,2) or vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise InputError('囲んだ範囲が不正です。')
    if not len(vertices):return np.array([],dtype=np.int64)
    if (not np.isfinite(thickness_mm) or not .5<=thickness_mm<=30
            or np.ptp(vertices[:,axis])>1e-4):
        raise InputError('同じ断面上で領域を囲んでください。')
    world=scene.world(vertices)
    segments,ids=tract_segment_data(bundle)
    clipped,_,kept=clip_segments_to_slab(segments,axis,float(world[0,axis]),thickness_mm,return_indices=True)
    h,v=plane_axes(axis)
    hit=segments_hit_polygon(clipped[:,:,[h,v]],world[:,[h,v]])
    return np.unique(ids[kept[hit]])


def project_segments(segments,matrix):
    """Clip in homogeneous camera coordinates before dividing by w."""
    matrix=np.asarray(matrix,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():
        raise InputError('3Dの表示座標が不正です。')
    if not len(segments):return np.empty((0,2,2)),np.array([],dtype=np.int64)
    homogeneous=np.concatenate([segments,np.ones(segments.shape[:2]+(1,))],axis=2)@matrix.T
    a,b=homogeneous[:,0],homogeneous[:,1];delta=b-a
    start=np.zeros(len(a));end=np.ones(len(a));valid=np.ones(len(a),bool)
    # vtk composite projection uses normalized clip coordinates in [-1, 1].
    for axis in (0,1,2):
        for sign in (-1,1):
            f=a[:,3]+sign*a[:,axis];slope=delta[:,3]+sign*delta[:,axis]
            parallel=np.abs(slope)<1e-12
            valid&=~parallel|(f>=0)
            boundary=-f/np.where(parallel,1,slope)
            start=np.where(slope>1e-12,np.maximum(start,boundary),start)
            end=np.where(slope<-1e-12,np.minimum(end,boundary),end)
    indices=np.flatnonzero(valid&(start<=end))
    clipped=np.stack([a[indices]+start[indices,None]*delta[indices],a[indices]+end[indices,None]*delta[indices]],axis=1)
    good=np.all(clipped[:,:,3]>1e-12,axis=1)
    clipped=clipped[good];indices=indices[good]
    return clipped[:,:,:2]/clipped[:,:,3,None],indices


def select_projection(bundle,polygon,matrix):
    segments,ids=tract_segment_data(bundle)
    projected,kept=project_segments(segments,matrix)
    return np.unique(ids[kept[segments_hit_polygon(projected,polygon)]])


def exclude(bundle,ids,selection):
    ids=np.unique(np.asarray(ids,dtype=np.int64))
    if np.any(ids<0) or np.any(ids>=bundle.total_count):raise InputError('線維の除外情報が不正です。')
    if bundle.excluded is None:bundle.excluded=np.zeros(bundle.total_count,bool)
    ids=ids[~bundle.excluded[ids]]
    if not len(ids):return 0
    bundle.excluded[ids]=True
    bundle.edits.append({'action':'exclude','ids':ids.tolist(),'selection':selection,
                         'time':datetime.now(timezone.utc).isoformat()})
    bundle.edit_revision+=1
    return len(ids)


def restore_all(bundle):
    ids=np.flatnonzero(bundle.excluded) if bundle.excluded is not None else np.array([],dtype=int)
    if not len(ids):return 0
    bundle.excluded[ids]=False
    bundle.edits.append({'action':'restore','ids':ids.tolist(),'time':datetime.now(timezone.utc).isoformat()})
    bundle.edit_revision+=1
    return len(ids)


def undo(bundle):
    if not bundle.edits:return False
    edit=bundle.edits.pop()
    if bundle.excluded is None:bundle.excluded=np.zeros(bundle.total_count,bool)
    bundle.excluded[np.asarray(edit['ids'],dtype=np.int64)]=edit['action']=='restore'
    bundle.edit_revision+=1
    return True
