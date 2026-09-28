import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from brain_viewer.imaging import InputError, load_scene, save_scene, validate_scene
from brain_viewer.rendering import composite_plane
from brain_viewer.segmentation import create_segment, paint_stroke, mask_surface, export_mask, export_surface
from test_multimodal import small_scene


class SegmentationTests(unittest.TestCase):
    def labelled_scene(self):
        scene = small_scene()
        scene.label_volume = np.zeros(scene.data.shape, np.uint16)
        labels = [4,5,14,15,43,44,10,49,192,251,252,253,254,255,24,31,63,2,41,72]
        scene.label_volume.ravel()[:len(labels)] = labels
        scene.label_source = 'synthetic labels'
        return scene

    def test_anatomy_unions_and_original_labels_unchanged(self):
        scene = self.labelled_scene(); original = scene.label_volume.copy()
        expected = {'ventricles':range(6), 'thalamus':range(6,8), 'callosum':range(8,14)}
        for preset, indices in expected.items():
            segment = create_segment(scene, preset, preset)
            np.testing.assert_array_equal(np.flatnonzero(segment.mask), list(indices))
            self.assertFalse(np.shares_memory(segment.mask, scene.label_volume))
            self.assertEqual(segment.provenance['source'], 'synthetic labels')
            segment.mask[:] = False
        np.testing.assert_array_equal(scene.label_volume, original)

    def test_missing_labels_do_not_claim_extraction_manual_starts_empty(self):
        scene = small_scene()
        with self.assertRaises(InputError): create_segment(scene,'thalamus','Thalamus')
        scene.label_volume = np.zeros(scene.data.shape,np.uint16)
        with self.assertRaises(InputError): create_segment(scene,'ventricles','Ventricles')
        scene.label_volume[1,2,3] = 4
        segment = create_segment(scene,'ventricles','Ventricles')
        self.assertEqual(segment.provenance['present_label_ids'],[4])
        for preset in ('manual','lesion','artery','vessel'):
            segment = create_segment(scene,preset,'region')
            self.assertFalse(segment.mask.any())
            self.assertEqual(segment.provenance['method'],'manual')
        with self.assertRaises(InputError): create_segment(scene,'manual','  ')

    def test_brush_physical_radius_in_each_anisotropic_plane_and_erase(self):
        spacing = np.array([1.,2.,3.]); shape=(13,13,13); center=np.array([6,6,6])
        grid = np.indices(shape).transpose(1,2,3,0)
        for axis in range(3):
            mask = np.zeros(shape,bool)
            paint_stroke(mask,spacing,axis,center,center,3.,True)
            distance = np.sum(((grid-center)*spacing)**2,axis=-1)
            expected = (distance<=9)&(grid[...,axis]==center[axis])
            np.testing.assert_array_equal(mask,expected)
            paint_stroke(mask,spacing,axis,center,center,3.,False)
            self.assertFalse(mask.any())

    def test_continuous_stroke_clipped_and_no_unintended_slice_bridging(self):
        mask = np.zeros((20,20,20),bool)
        paint_stroke(mask,[1,1,1],2,[0,0,7],[19,19,7],1.,True)
        self.assertTrue(np.all(mask[np.arange(20),np.arange(20),7]))
        self.assertFalse(mask[:,:,:7].any()); self.assertFalse(mask[:,:,8:].any())
        paint_stroke(mask,[1,1,1],2,[0,0,7],[19,19,8],1.,True)
        self.assertTrue(mask[19,19,8]); self.assertFalse(mask[0,0,8])
        before=mask.copy()
        paint_stroke(mask,[1,1,1],2,[-20,-20,7],[-19,-19,7],1.,True)
        np.testing.assert_array_equal(mask,before)

    def test_mesh_uses_voxel_centres_affine_rotation_and_empty_mask(self):
        mask = np.zeros((8,8,8),bool); mask[2,3,4] = True
        affine = np.array([[0,-2,0,30],[1,0,0,-10],[0,0,3,7],[0,0,0,1.]])
        mesh=mask_surface(mask,affine)
        points=vtk_to_numpy(mesh.GetPoints().GetData())
        centre=nib.affines.apply_affine(affine,[2,3,4])
        expected=np.vstack([centre+affine[:3,axis]*sign*.5 for axis in range(3) for sign in (-1,1)])
        self.assertEqual(len(points),6)
        for point in expected: self.assertLess(np.min(np.linalg.norm(points-point,axis=1)),1e-5)
        self.assertTrue(mesh.GetNumberOfPolys()>0)
        self.assertEqual(mesh.GetFieldData().GetAbstractArray('CoordinateSystem').GetValue(0),'scanner_RAS_mm')
        self.assertEqual(mask_surface(np.zeros_like(mask),affine).GetNumberOfPoints(),0)

    def test_2d_overlay_is_independent_of_3d_visibility_and_opacity(self):
        scene=self.labelled_scene(); segment=create_segment(scene,'ventricles','region')
        baseline=composite_plane(scene,0,0,0,200)
        scene.segmentations=[segment]; segment.visible_3d=False; segment.opacity=0.
        self.assertFalse(np.array_equal(baseline,composite_plane(scene,0,0,0,200)))
        segment.visible_2d=False
        np.testing.assert_array_equal(baseline,composite_plane(scene,0,0,0,200))

    def test_persistent_masks_settings_edit_history_and_legacy_compatibility(self):
        scene=self.labelled_scene(); segment=create_segment(scene,'ventricles','Test region')
        segment.visible_3d=False; segment.visible_2d=False; segment.opacity=.37
        segment.edits=[{'action':'paint','changed_voxels':1}]; segment.revision=2
        segment.mask[7,8,9]=True; scene.segmentations=[segment]
        with tempfile.TemporaryDirectory() as directory:
            folder=save_scene(scene,Path(directory),{})
            loaded=load_scene(folder/'scene.json').segmentations[0]
            np.testing.assert_array_equal(loaded.mask,segment.mask)
            for name in ('uid','name','preset','color','visible_2d','visible_3d','opacity','provenance','edits','revision'):
                self.assertEqual(getattr(loaded,name),getattr(segment,name))
            manifest=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
            manifest['schema']='cortex-viewer/5'; manifest.pop('segmentations')
            (folder/'scene.json').write_text(json.dumps(manifest),encoding='utf-8')
            self.assertEqual(load_scene(folder/'scene.json').segmentations,[])

    def test_invalid_mask_geometry_values_paths_and_identifiers_rejected(self):
        scene=self.labelled_scene(); segment=create_segment(scene,'ventricles','region'); scene.segmentations=[segment]
        with tempfile.TemporaryDirectory() as directory:
            for failure in ('affine','values','path'):
                folder=save_scene(scene,Path(directory),{})
                manifest=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
                filename=manifest['segmentations'][0]['mask']
                if failure=='path':
                    manifest['segmentations'][0]['mask']='../outside.nii.gz'
                    (folder/'scene.json').write_text(json.dumps(manifest),encoding='utf-8')
                else:
                    mask=segment.mask.astype(np.uint8); affine=scene.affine.copy()
                    if failure=='affine': affine[0,3]+=5
                    else: mask[0,0,0]=2
                    nib.save(nib.Nifti1Image(mask,affine),folder/filename)
                with self.assertRaises(InputError): load_scene(folder/'scene.json')
        scene.segmentations.append(segment)
        with self.assertRaises(InputError): validate_scene(scene)

    def test_export_binary_mask_and_surface_in_reference_ras(self):
        scene=self.labelled_scene(); segment=create_segment(scene,'ventricles','region')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mask.nii.gz'; export_mask(segment,scene,path)
            image=nib.load(path)
            np.testing.assert_array_equal(np.asanyarray(image.dataobj),segment.mask)
            np.testing.assert_allclose(image.affine,scene.affine)
            self.assertEqual(image.header.get_xyzt_units()[0],'mm')
            self.assertEqual(image.header.get_intent()[0],'label')
            import vtk
            path=Path(directory)/'mask.vtp'; export_surface(segment,scene,path)
            reader=vtk.vtkXMLPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
            np.testing.assert_allclose(reader.GetOutput().GetBounds(),mask_surface(segment.mask,scene.affine).GetBounds())
            segment.mask[:]=False
            with self.assertRaises(InputError): export_surface(segment,scene,path)
