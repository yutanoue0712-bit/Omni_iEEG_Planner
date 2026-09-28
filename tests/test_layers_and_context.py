"""Opacity/visibility semantics and non-destructive reference labels in synthetic scenes."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import unittest
import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation
from brain_viewer.rendering import composite_plane
from brain_viewer.electrode_presets import REFERENCE_PRESETS,set_reference_preset
from brain_viewer.contact_overview import overview_plane
from brain_viewer.imaging import load_scene,save_scene
from test_multimodal import small_scene
from test_ct_localization import native_scene


class LayerCompositingTests(unittest.TestCase):
    def setUp(self):
        self.scene=small_scene()
        self.scene.ct=np.full(self.scene.data.shape,50.,np.float32)
        self.scene.ct_valid=np.ones(self.scene.data.shape,bool)
        self.scene.ct_valid[0,:,:]=False

    def test_mri_opacity_applies_independently(self):
        s=self.scene
        full=composite_plane(s,2,5,0,200,{'visible':False})
        half=composite_plane(s,2,5,0,200,{'visible':False,'mri_opacity':.5})
        np.testing.assert_array_equal(full,np.full_like(full,127))
        np.testing.assert_array_equal(half,np.full_like(half,63))

    def test_mri_off_exposes_full_grayscale_ct_even_with_bone_style(self):
        s=self.scene
        options={'visible':True,'mri_visible':False,'mode':'bone','window':100,'level':50,'opacity':1.}
        image=composite_plane(s,2,5,0,200,options)
        np.testing.assert_array_equal(image[1:],np.full_like(image[1:],127))
        np.testing.assert_array_equal(image[0],0)
        s.data[:]=9000
        np.testing.assert_array_equal(composite_plane(s,2,5,0,200,options),image)
        options.update(mri_visible=True,mri_opacity=0)
        np.testing.assert_array_equal(composite_plane(s,2,5,0,200,options),image)
        options['opacity']=.5
        np.testing.assert_array_equal(composite_plane(s,2,5,0,200,options)[1:],63)

    def test_hiding_both_images_does_not_leave_mri_label_fill(self):
        s=self.scene; s.label_volume=np.full(s.data.shape,1030,np.uint16)
        image=composite_plane(s,2,5,0,200,{'mri_visible':False,'visible':False},1030)
        np.testing.assert_array_equal(image,0)

    def test_ordinary_overlay_keeps_ct_opacity_independent(self):
        s=self.scene
        options={'visible':True,'mode':'full','color':'grey','window':100,'level':0,'opacity':.25,'mri_opacity':.5}
        image=composite_plane(s,2,5,0,200,options)
        np.testing.assert_array_equal(image[1:],np.uint8((127.5*.5)*.75+255*.25))
        np.testing.assert_array_equal(image[0],63)


class ReferenceAndOverviewTests(unittest.TestCase):
    def test_reference_selection_changes_metadata_only_and_roundtrips(self):
        scene=native_scene(); initial=deepcopy(scene.contacts)
        expected=scene.electrode_quality['groups']['A']['expected_count']
        for preset in REFERENCE_PRESETS:
            set_reference_preset(scene,'A',preset['id'])
            metadata=scene.electrode_quality['groups']['A']
            self.assertEqual(metadata['expected_count'],expected)
            self.assertEqual(metadata['reference_spec']['width_definition'],'unspecified')
            self.assertNotIn('pitch_mm',metadata['reference_spec'])
            self.assertEqual(len(scene.contacts),len(initial))
            for before,after in zip(initial,scene.contacts):
                self.assertEqual(before.uid,after.uid); self.assertEqual(before.status,after.status)
                np.testing.assert_array_equal(before.position,after.position)
                np.testing.assert_array_equal(before.ct_position,after.ct_position)
        parent=Path(__file__).resolve().parents[1]/'private_reports'
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            state={'image_layers':{'mri':{'visible':False,'opacity':.4},'ct':{'visible':True,'opacity':1.}},
                   'model_layers':{'brain':{'visible':False,'opacity':.23},'nuclei':{'visible':True,'opacity':.4}}}
            saved=save_scene(scene,Path(temp),state)
            restored=load_scene(saved/'scene.json')
            self.assertEqual(restored.view_state,state)
            self.assertEqual(restored.electrode_quality['groups']['A']['reference_spec'],REFERENCE_PRESETS[-1])
        set_reference_preset(scene,'A',None)
        self.assertNotIn('reference_spec',scene.electrode_quality['groups']['A'])
        self.assertEqual(scene.electrode_quality['groups']['A']['expected_count'],expected)

    def test_whole_ct_planes_contain_fov_and_pass_through_selected_point(self):
        shape=(55,61,43)
        affine=np.eye(4)
        affine[:3,:3]=Rotation.from_euler('xyz',[17,-23,13],degrees=True).as_matrix()@np.diag([-.7,.8,1.2])
        affine[:3,3]=[30,-21,-17]
        point=nib.affines.apply_affine(affine,[12,26,29])
        corners=nib.affines.apply_affine(affine,np.array(list(product(*[(-.5,n-.5) for n in shape]))))
        for axis in (0,1,2):
            center,u,v,extent=overview_plane(shape,affine,point,axis)
            self.assertAlmostEqual(center[axis],point[axis])
            self.assertTrue(np.all(np.abs((corners-center)@u)<extent[0]/2))
            self.assertTrue(np.all(np.abs((corners-center)@v)<extent[1]/2))
            np.testing.assert_allclose([np.linalg.norm(u),np.linalg.norm(v),np.dot(u,v)],[1,1,0])
