"""Bounded inverse-distance surface display in patient RAS, independent of Brainstorm.

This is a visualization, not an estimate of the tissue sampled by a contact.
Original channel values and positions are never replaced by projected values.
"""
import numpy as np
from scipy.spatial import cKDTree
from .analysis_results import resolved_positions,colors,color_limits
from .result_display import spatial_options,marker_options,metric_display,visible_values
from .pet_surface import BASE_COLOR


def surface_neighbors(vertices,positions):
    if not len(positions): return np.empty((len(vertices),0),np.float32),np.empty((len(vertices),0),np.int32)
    count=min(4,len(positions))
    distance,indices=cKDTree(positions).query(vertices,k=list(range(1,count+1)))
    return distance.astype(np.float32),indices.astype(np.int32)


def interpolate_surface(distance,indices,values,radius):
    """At most four neighbors; renormalize finite values within the physical radius."""
    output=np.full(len(distance),np.nan,np.float32)
    if not indices.size: return output
    samples=np.asarray(values)[indices]
    valid=(distance<=radius)&np.isfinite(samples)
    weights=np.where(valid,1/np.maximum(distance.astype(float),1e-6)**2,0)
    exact=valid&(distance<1e-6)
    exact_rows=exact.any(axis=1)
    weights[exact_rows]=exact[exact_rows]
    total=weights.sum(axis=1); covered=total>0
    output[covered]=(weights*np.where(valid,samples,0)).sum(axis=1)[covered]/total[covered]
    return output


class ResultSurfaceColors:
    def __init__(self): self.clear()

    def clear(self):
        self.result=None; self.active=False; self.metric=0; self.frame=0
        self.geometry=None; self.neighbors={}; self.signature=None; self.painted=False
        self.visible_vertices=0

    def set_result(self,panel,result,metric=0,frame=0,active=True):
        self.result=result; self.metric=metric; self.frame=frame; self.active=active
        self.apply(panel)

    def apply(self,panel):
        scene=panel.scene; result=self.result
        options=spatial_options(result); markers=marker_options(result)
        enabled=bool(scene and result and self.active and options['mode']!='points' and markers['opacity']>0)
        if not enabled:
            if self.painted and scene:
                panel.pet_colors.signature=None; panel.pet_colors.apply(panel)
            self.painted=False; self.signature=None; self.visible_vertices=0
            return
        channels,positions=resolved_positions(result,scene.contacts)
        geometry=(id(scene),tuple(channels),positions.tobytes(),
                  tuple((key,id(mesh.vertices),id(mesh.faces)) for key,mesh in scene.surfaces.items()))
        if geometry!=self.geometry:
            self.geometry=geometry; self.neighbors.clear(); self.signature=None
        display=metric_display(result,self.metric); limits=color_limits(result,self.metric)
        values=result.values[channels,self.frame,self.metric]
        signature=(geometry,values.tobytes(),tuple(display.items()),limits,options['distance_mm'],
                   markers['opacity'],panel.pet_colors.signature)
        if signature==self.signature: return
        self.signature=signature; self.visible_vertices=0
        from vtk.util.numpy_support import numpy_to_vtk
        for key,actor in panel.actors.items():
            mesh=scene.surfaces[key]
            if not key.startswith(('pial_','white_')): continue
            if key not in self.neighbors: self.neighbors[key]=surface_neighbors(mesh.vertices,positions)
            distance,indices=self.neighbors[key]
            projected=interpolate_surface(distance,indices,values,options['distance_mm'])
            visible=visible_values(projected,display)
            if key.startswith(panel.mode+'_'): self.visible_vertices+=int(visible.sum())
            base=panel.pet_colors.base_colors.get(key)
            color=(base.astype(float).copy() if base is not None else np.tile(BASE_COLOR,(len(mesh.vertices),1)))
            rgb=colors(projected,limits,display['colormap'],display['reverse'])*255
            alpha=markers['opacity']
            color[visible]=color[visible]*(1-alpha)+rgb[visible]*alpha
            scalars=numpy_to_vtk(np.ascontiguousarray(color,dtype=np.uint8),deep=True)
            scalars.SetName('analysis_surface_RGB')
            panel.polys[key].GetPointData().SetScalars(scalars)
            mapper=actor.GetMapper(); mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToDirectScalars(); mapper.ScalarVisibilityOn()
        self.painted=True
