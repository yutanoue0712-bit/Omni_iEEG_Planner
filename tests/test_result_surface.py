"""Physical support, scalar interpolation and non-destructive cortical display."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from brain_viewer.analysis_results import AnalysisResult,exact_bindings,export_record
from brain_viewer.imaging import Contact
from brain_viewer.pet_surface import PETSurfaceColors
from brain_viewer.result_surface import surface_neighbors,interpolate_surface,ResultSurfaceColors
from brain_viewer.surface_view import mesh_polydata
from test_pet_surface import pet_scene


class SurfaceInterpolationTests(unittest.TestCase):
    def test_physical_distance_exact_centers_zero_and_boundary(self):
        positions=np.array([[0.,0,0],[10.,0,0]])
        vertices=np.array([[0.,0,0],[2.5,0,0],[5.,0,0],[10.,0,0],[25.,0,0],[25.1,0,0]])
        distance,indices=surface_neighbors(vertices,positions)
        values=interpolate_surface(distance,indices,np.array([0.,100.]),15)
        np.testing.assert_allclose(values[:5],[0.,10.,50.,100.,100.])
        self.assertTrue(np.isnan(values[-1]))

    def test_missing_is_not_zero_and_signed_values_are_not_rectified(self):
        distance,indices=surface_neighbors(np.array([[0.,0,0],[5.,0,0],[10.,0,0]]),np.array([[0.,0,0],[10.,0,0]]))
        np.testing.assert_allclose(interpolate_surface(distance,indices,[-4.,4.],15),[-4.,0.,4.])
        np.testing.assert_allclose(interpolate_surface(distance,indices,[np.nan,4.],15),[4.,4.,4.])
        self.assertTrue(np.isnan(interpolate_surface(distance,indices,[np.nan,np.nan],15)).all())
        self.assertTrue(np.isnan(interpolate_surface(distance,indices,[np.nan,4.],3)[0]))

    def test_single_contact_and_empty_geometry(self):
        distance,indices=surface_neighbors(np.array([[0.,0,0],[1.,0,0],[2.,0,0]]),np.zeros((1,3)))
        values=interpolate_surface(distance,indices,[8.],1)
        np.testing.assert_allclose(values[:2],[8.,8.]);self.assertTrue(np.isnan(values[2]))
        distance,indices=surface_neighbors(np.zeros((3,3)),np.empty((0,3)))
        self.assertTrue(np.isnan(interpolate_surface(distance,indices,[],15)).all())

    def test_surface_recolors_without_requery_and_restores_pet(self):
        scene,layer=pet_scene();polys={};actors={}
        for key,mesh in scene.surfaces.items():
            poly=mesh_polydata(mesh);mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(poly)
            actor=vtk.vtkActor();actor.SetMapper(mapper);polys[key]=poly;actors[key]=actor
        panel=SimpleNamespace(scene=scene,polys=polys,actors=actors,mode='pial',pet_colors=PETSurfaceColors(),
            ct_options={'extra_mris':{layer.uid:{'surface_visible':True,'visible':True,'opacity':.7}}})
        panel.pet_colors.apply(panel)
        base=vtk_to_numpy(polys['pial_lh'].GetPointData().GetScalars()).copy()
        scene.contacts=[Contact('LPH1','LPH',scene.surfaces['pial_lh'].vertices[0].copy(),(1,1,1))]
        result=AnalysisResult('Synthetic','time_series',['LPH1'],['value'],np.array([[[0.],[8.]]]),np.array([0.,1.]),[''])
        result.bindings=exact_bindings(result,scene.contacts)
        result.settings={'spatial':{'mode':'surface','distance_mm':15},'display':{'value':{'colormap':'jet'}}}
        surface=ResultSurfaceColors();original=result.values.copy()
        with patch('brain_viewer.result_surface.surface_neighbors',wraps=surface_neighbors) as query:
            surface.set_result(panel,result,frame=0)
            zero_colors=vtk_to_numpy(polys['pial_lh'].GetPointData().GetScalars()).copy()
            self.assertEqual(query.call_count,len(actors));self.assertGreater(surface.visible_vertices,0)
            surface.set_result(panel,result,frame=1)
            self.assertFalse(np.array_equal(zero_colors,vtk_to_numpy(polys['pial_lh'].GetPointData().GetScalars())))
            result.settings['spatial']['distance_mm']=5;surface.apply(panel)
            self.assertEqual(query.call_count,len(actors))
            scene.contacts[0].position=scene.contacts[0].position+[.1,0,0];surface.apply(panel)
            self.assertEqual(query.call_count,2*len(actors))
            surface.set_result(panel,result,active=False)
            np.testing.assert_array_equal(base,vtk_to_numpy(polys['pial_lh'].GetPointData().GetScalars()))
            self.assertTrue(actors['pial_lh'].GetMapper().GetScalarVisibility())
        np.testing.assert_array_equal(result.values,original)
        record=export_record(result,scene.contacts,0,[0,1])
        self.assertEqual(record['spatial_display']['distance_mm'],5)
        self.assertFalse(record['spatial_display']['recording_range_estimate'])


if __name__=='__main__':unittest.main()
