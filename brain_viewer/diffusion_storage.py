"""Self-contained tensor/tract snapshots and local streamline export."""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from .diffusion import DiffusionModel, TractBundle
from .local_assets import write_asset
from .imaging import _contained, InputError


def save_diffusion(scene, folder, manifest, asset_root):
    manifest['diffusions']=[]
    for model in scene.diffusions:
        filename=model.uid+'.npz'
        arrays=dict(affine=model.affine,to_reference=model.to_reference,fa=model.fa,directions=model.directions,
                    mask=model.mask,bvals=model.bvals,bvecs=model.bvecs)
        write_asset(folder/filename,tuple(arrays.values()),lambda p:np.savez_compressed(p,**arrays),asset_root)
        manifest['diffusions'].append({'uid':model.uid,'name':model.name,'data':filename,
                                     'layer_ids':model.layer_ids,'quality':model.quality})
    manifest['tracts']=[]
    for bundle in scene.tracts:
        filename=bundle.uid+'.npz'
        excluded=bundle.excluded if bundle.excluded is not None else np.zeros(bundle.total_count,bool)
        write_asset(folder/filename,(bundle.points,bundle.offsets,excluded),
                    lambda p:np.savez_compressed(p,points=bundle.points,offsets=bundle.offsets,excluded=excluded),asset_root)
        manifest['tracts'].append({'uid':bundle.uid,'name':bundle.name,'diffusion_uid':bundle.diffusion_uid,
            'data':filename,'settings':bundle.settings,'visible':bundle.visible,'opacity':bundle.opacity,
            'visible_2d':bundle.visible_2d,'slab_mm':bundle.slab_mm,'edits':bundle.edits})


def load_diffusion(scene, folder, manifest):
    for item in manifest.get('diffusions',[]):
        with np.load(_contained(folder,item['data']),allow_pickle=False) as data:
            scene.diffusions.append(DiffusionModel(item['uid'],item['name'],
                **{k:data[k].copy() for k in ('affine','to_reference','fa','directions','mask','bvals','bvecs')},
                layer_ids=item['layer_ids'],quality=item['quality']))
    for item in manifest.get('tracts',[]):
        with np.load(_contained(folder,item['data']),allow_pickle=False) as data:
            scene.tracts.append(TractBundle(item['uid'],item['name'],item['diffusion_uid'],
                data['points'].copy(),data['offsets'].copy(),item['settings'],
                bool(item.get('visible',True)),float(item.get('opacity',.9)),
                bool(item.get('visible_2d',True)),float(item.get('slab_mm',2.)),
                data['excluded'].copy() if 'excluded' in data else None,item.get('edits',[])))


def tract_polydata(bundle, highlight=True):
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    lines=list(bundle.streamlines())
    vertices=np.concatenate(lines) if lines else np.empty((0,3),dtype=np.float32)
    offsets=np.r_[0,np.cumsum([len(line) for line in lines])].astype(np.int64)
    points=vtk.vtkPoints(); points.SetData(numpy_to_vtk(vertices,deep=True))
    cells=vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(offsets,deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(vertices),dtype=np.int64),deep=True))
    poly=vtk.vtkPolyData(); poly.SetPoints(points); poly.SetLines(cells)
    colors=np.zeros_like(vertices)
    highlighted=set(getattr(bundle,'_highlight_ids',[])) if highlight else set()
    for index,a,b in zip(bundle.active_ids(),offsets[:-1],offsets[1:]):
        tangent=np.gradient(vertices[a:b],axis=0)
        colors[a:b]=np.abs(tangent)/np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-8)
        if highlighted:colors[a:b]=[1.,.15,.15] if index in highlighted else [.28,.34,.40]
    rgb=numpy_to_vtk((colors*255).astype(np.uint8),deep=True); rgb.SetName('DirectionRGB')
    poly.GetPointData().SetScalars(rgb)
    return poly


def export_tract(bundle, path, scene):
    if not bundle.count:raise InputError('出力する線維がありません。除外を戻してください。')
    path=Path(path)
    if path.suffix.lower()=='.vtp':
        import vtk
        writer=vtk.vtkXMLPolyDataWriter(); writer.SetFileName(str(path))
        writer.SetInputData(tract_polydata(bundle,highlight=False))
        if not writer.Write(): raise OSError('tract export failed')
    else:
        tractogram=nib.streamlines.Tractogram(list(bundle.streamlines()),affine_to_rasmm=np.eye(4))
        if path.suffix.lower()=='.trk':
            header=nib.streamlines.TrkFile.create_empty_header()
            header['voxel_to_rasmm']=scene.affine
            header['voxel_sizes']=scene.spacing
            header['dimensions']=scene.data.shape
            header['voxel_order']='RAS'
            nib.streamlines.save(nib.streamlines.TrkFile(tractogram,header=header),str(path))
        else:
            nib.streamlines.save(tractogram,str(path))
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(
        {'coordinate_system':'scanner_RAS_mm','name':bundle.name,'count':bundle.count,
         'reference_affine':scene.affine.tolist(),'settings':bundle.settings,
         'original_count':bundle.total_count,'excluded_count':bundle.total_count-bundle.count,'edits':bundle.edits},
        ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def export_scalar(scene,model,key,path):
    layer=next(layer for layer in scene.extra_mris if layer.uid==model.layer_ids[key])
    image=nib.Nifti1Image(layer.data,scene.affine)
    image.header.set_xyzt_units('mm')
    image.set_qform(scene.affine,1); image.set_sform(scene.affine,1)
    nib.save(image,str(path))
