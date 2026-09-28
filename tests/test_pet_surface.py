"""Independent physical sampling, invalid-FOV handling and surface color caching."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from brain_viewer.imaging import MRILayer,Mesh
from brain_viewer.pet_surface import sample_cortical_pet,PETSurfaceColors,BASE_COLOR
from brain_viewer.surface_view import mesh_polydata
from test_multimodal import small_scene


def pet_scene():
    scene=small_scene()
    xyz=nib.affines.apply_affine(scene.affine,np.indices(scene.data.shape).reshape(3,-1).T)
    data=(xyz@np.array([1.,2.,3.])).reshape(scene.data.shape).astype(np.float32)
    layer=MRILayer('mr_pet','PET','PET',data,np.ones(data.shape,bool),data.copy(),scene.affine.copy(),np.eye(4),{},250.,125.)
    scene.extra_mris=[layer]
    for hemisphere in ('lh','rh'):
        pial=scene.surfaces['pial_'+hemisphere]
        scene.surfaces['white_'+hemisphere]=Mesh(pial.vertices-[0,0,2],pial.faces.copy())
    return scene,layer


class PETSurfaceTests(unittest.TestCase):
    def test_ribbon_sampling_uses_patient_mri_grid_and_preserves_values(self):
        scene,layer=pet_scene(); before=layer.data.copy()
        values,valid,method=sample_cortical_pet(scene,layer,'lh')
        expected=(scene.surfaces['pial_lh'].vertices-[0,0,1])@np.array([1.,2.,3.])
        np.testing.assert_allclose(values,expected,atol=1e-5)
        self.assertTrue(valid.all()); self.assertEqual(method,'cortical_ribbon_mean')
        np.testing.assert_array_equal(layer.data,before)

    def test_missing_white_and_unmatched_topology_fall_back_to_pial(self):
        scene,layer=pet_scene(); scene.surfaces.pop('white_lh')
        values,valid,method=sample_cortical_pet(scene,layer,'lh')
        np.testing.assert_allclose(values,scene.surfaces['pial_lh'].vertices@np.array([1.,2.,3.]))
        self.assertEqual(method,'pial_sample')
        scene.surfaces['white_lh']=Mesh(scene.surfaces['pial_lh'].vertices.copy(),np.array([[0,2,1]],np.int32))
        self.assertEqual(sample_cortical_pet(scene,layer,'lh')[2],'pial_sample')
        layer.valid[:]=False
        self.assertFalse(sample_cortical_pet(scene,layer,'lh')[1].any())

    def test_color_changes_reuse_samples_and_hiding_restores_plain_surface(self):
        scene,layer=pet_scene(); polys={}; actors={}
        for key,mesh in scene.surfaces.items():
            poly=mesh_polydata(mesh); mapper=vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
            actor=vtk.vtkActor(); actor.SetMapper(mapper); polys[key]=poly; actors[key]=actor
        settings={'surface_visible':True,'visible':True,'opacity':.7,'palette':'hot','window':250.,'level':125.}
        panel=SimpleNamespace(scene=scene,polys=polys,actors=actors,ct_options={'extra_mris':{layer.uid:settings}})
        display=PETSurfaceColors()
        with patch('brain_viewer.pet_surface.sample_cortical_pet',wraps=sample_cortical_pet) as sample:
            display.apply(panel); colors=vtk_to_numpy(polys['pial_lh'].GetPointData().GetScalars()).copy()
            self.assertTrue(actors['pial_lh'].GetMapper().GetScalarVisibility())
            display.apply(panel); self.assertEqual(sample.call_count,2)
            settings['level']=250.; display.apply(panel)
            self.assertEqual(sample.call_count,2)
            self.assertFalse(np.array_equal(colors,vtk_to_numpy(polys['pial_lh'].GetPointData().GetScalars())))
            settings['visible']=False; display.apply(panel)
            self.assertFalse(actors['pial_lh'].GetMapper().GetScalarVisibility())
            self.assertIsNone(polys['pial_lh'].GetPointData().GetScalars())
            settings['visible']=True; display.apply(panel); self.assertEqual(sample.call_count,2)
        layer.valid[:]=False; display.clear(); display.apply(panel)
        expected=np.tile(BASE_COLOR.astype(np.uint8),(len(scene.surfaces['pial_lh'].vertices),1))
        np.testing.assert_array_equal(vtk_to_numpy(polys['pial_lh'].GetPointData().GetScalars()),expected)
