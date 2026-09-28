"""Patient-space PET sampling on existing cortical meshes; no surface registration."""
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from .image_colormaps import heatmap_plane

BASE_COLOR = np.array([.80, .78, .75])*255


def sample_cortical_pet(scene, layer, hemisphere):
    pial = scene.surfaces['pial_'+hemisphere]
    white = scene.surfaces.get('white_'+hemisphere)
    paired = (white is not None and white.vertices.shape == pial.vertices.shape
              and np.array_equal(white.faces, pial.faces))
    points = ([white.vertices*(1-fraction)+pial.vertices*fraction for fraction in (.25,.5,.75)]
              if paired else [pial.vertices])
    inverse = np.linalg.inv(scene.affine)
    total = np.zeros(len(pial.vertices), np.float32)
    count = np.zeros(len(pial.vertices), np.uint8)
    for sample in points:
        indices = nib.affines.apply_affine(inverse, sample).T
        values = map_coordinates(layer.data, indices, order=1, mode='constant', cval=0, prefilter=False)
        valid = map_coordinates(layer.valid, indices, order=0, mode='constant', cval=0, prefilter=False)
        valid &= np.isfinite(values)
        total += np.where(valid, values, 0)
        count += valid
    return total/np.maximum(count,1), count>0, 'cortical_ribbon_mean' if paired else 'pial_sample'


class PETSurfaceColors:
    """Cache vertex values, recolor only when a layer's display settings change."""
    def __init__(self):
        self.samples = {}
        self.signature = None
        self.base_colors = {}

    def clear(self):
        self.samples.clear(); self.signature = None; self.base_colors.clear()

    def apply(self, panel):
        scene = panel.scene
        if scene is None: return
        layers = []
        for layer in scene.extra_mris:
            options = panel.ct_options.get('extra_mris',{}).get(layer.uid,{})
            if (layer.sequence=='PET' and options.get('surface_visible',False)
                    and options.get('visible',True) and options.get('opacity',.5)>0):
                layers.append((layer,options))
        signature = (id(scene),tuple((l.uid,id(l.data),id(l.valid),o.get('window',l.window),
            o.get('level',l.level),o.get('palette','hot'),o.get('opacity',.5)) for l,o in layers))
        if signature == self.signature: return
        self.signature = signature
        self.base_colors.clear()
        active = {l.uid for l in scene.extra_mris if l.sequence=='PET'}
        self.samples = {k:v for k,v in self.samples.items() if k[0] in active}
        if not layers:
            for key,actor in panel.actors.items():
                actor.GetMapper().ScalarVisibilityOff()
                panel.polys[key].GetPointData().SetScalars(None)
            return
        from vtk.util.numpy_support import numpy_to_vtk
        for hemisphere in ('lh','rh'):
            pial = scene.surfaces.get('pial_'+hemisphere)
            if pial is None: continue
            color = np.broadcast_to(BASE_COLOR,(len(pial.vertices),3)).copy()
            for layer,options in layers:
                key = (layer.uid,hemisphere,id(layer.data),id(layer.valid))
                if key not in self.samples:
                    self.samples[key] = sample_cortical_pet(scene,layer,hemisphere)
                values,valid,_ = self.samples[key]
                rgb,alpha = heatmap_plane(values,options.get('window',layer.window),
                    options.get('level',layer.level),options.get('palette','hot'))
                alpha *= valid*np.clip(options.get('opacity',.5),0,1)
                color = color*(1-alpha[:,None])+rgb*alpha[:,None]
            for kind in ('pial','white'):
                key = kind+'_'+hemisphere
                if key not in panel.actors: continue
                mesh = scene.surfaces[key]
                mapper = panel.actors[key].GetMapper()
                if (len(mesh.vertices)!=len(pial.vertices) or not np.array_equal(mesh.faces,pial.faces)):
                    mapper.ScalarVisibilityOff(); continue
                scalars = numpy_to_vtk(np.ascontiguousarray(color,dtype=np.uint8),deep=True)
                scalars.SetName('PET_display_RGB')
                self.base_colors[key]=np.ascontiguousarray(color,dtype=np.uint8)
                panel.polys[key].GetPointData().SetScalars(scalars)
                mapper.SetScalarModeToUsePointData(); mapper.SetColorModeToDirectScalars(); mapper.ScalarVisibilityOn()
