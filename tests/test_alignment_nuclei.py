import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import nibabel as nib
from brain_viewer.imaging import Contact, InputError, load_scene, save_scene
from brain_viewer.anatomy import prepare_nuclei_display
from brain_viewer.alignment_review import rigid_delta, apply_alignment, landmark_residuals
from brain_viewer.i18n import tr, set_language
from test_multimodal import small_scene


class AlignmentAndNucleiTests(unittest.TestCase):
    def test_sagittal_rotation_pivot_direction_and_translation(self):
        matrix = rigid_delta([90,0,0],[2,-1,3],[10,20,30])
        np.testing.assert_allclose(nib.affines.apply_affine(matrix,[10,20,30]),[12,19,33])
        np.testing.assert_allclose(nib.affines.apply_affine(matrix,[10,24,30]),[12,19,37])

    def test_adjustment_keeps_contacts_on_raw_ct_and_reset_avoids_resampling_drift(self):
        s=small_scene()
        s.raw_ct=np.zeros(s.data.shape,np.float32); s.raw_ct[7,7,7]=3000
        s.raw_ct_affine=s.affine.copy(); s.ct=s.raw_ct.copy(); s.ct_valid=np.ones(s.data.shape,bool)
        s.ct_to_mri=np.eye(4)
        point=s.world([7,7,7])
        s.contacts=[Contact('A1','A',point,(1.,0.,0.),point.copy())]
        matrix=rigid_delta([5,0,0],[.3,-.2,.5],point)
        changed=apply_alignment(s,matrix)
        np.testing.assert_allclose(nib.affines.apply_affine(np.linalg.inv(matrix),changed.contacts[0].position),point)
        np.testing.assert_array_equal(changed.contacts[0].source_position,point)
        restored=apply_alignment(changed,np.eye(4))
        np.testing.assert_array_equal(restored.ct,s.raw_ct)
        np.testing.assert_allclose(restored.contacts[0].position,point,atol=1e-10)
        self.assertEqual(changed.ct_quality['landmark_accuracy_mm'] if 'landmark_accuracy_mm' in changed.ct_quality else None,None)

    def test_review_landmark_distances_follow_current_transform(self):
        pairs=[{'ct_ras_mm':[1,2,3],'mri_ras_mm':[4,6,3]}]
        np.testing.assert_allclose(landmark_residuals(np.eye(4),pairs),[5])
        np.testing.assert_allclose(landmark_residuals(rigid_delta([0,0,0],[3,4,0],[0,0,0]),pairs),[0])

    def test_cropped_half_mm_nucleus_native_lookup_and_persistence(self):
        s=small_scene()
        s.nuclei=np.zeros((9,8,7),np.uint16); s.nuclei[3:6,2:5,2:5]=8106
        s.nuclei_affine=np.array([[0,-.5,0,14],[.5,0,0,28],[0,0,.5,28],[0,0,0,1.]])
        s.nuclei_names={8106:'Left-CM'}; s.nuclei_source='synthetic cropped half-mm labels'
        prepare_nuclei_display(s)
        point=nib.affines.apply_affine(s.nuclei_affine,[4,3,3])
        self.assertEqual(s.nucleus_at(point),(8106,'Left-CM'))
        self.assertEqual(s.nucleus_at(point+[30,0,0])[0],0)
        parent=Path(__file__).resolve().parents[1]/'private_reports'
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            assert Path(temp).resolve().is_relative_to(parent.resolve())
            folder=save_scene(s,Path(temp),{})
            restored=load_scene(folder/'scene.json')
            np.testing.assert_array_equal(restored.nuclei,s.nuclei)
            np.testing.assert_allclose(restored.nuclei_affine,s.nuclei_affine)
            self.assertEqual(restored.nucleus_at(point),(8106,'Left-CM'))
            self.assertEqual(restored.nuclei_display.shape,s.data.shape)

    def test_language_roundtrip_preserves_dynamic_numbers(self):
        jp='20本 / 200コンタクト\n2D：断面から±2 mmの中心'
        try:
            set_language('en'); english=tr(jp)
            self.assertEqual(english,'20 electrodes / 200 contacts\n2D: centres within ±2 mm of slice')
            set_language('ja'); self.assertEqual(tr(english),jp)
            self.assertEqual(tr('Left-CM'),'Left-CM')
        finally: set_language('ja')
