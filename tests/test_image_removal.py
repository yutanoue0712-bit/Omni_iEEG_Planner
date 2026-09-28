from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from brain_viewer.imaging import MRILayer, Contact, InputError, save_scene, load_scene, validate_scene
from brain_viewer.diffusion import DiffusionModel
from brain_viewer.image_removal import removal_ids, remove_images
from brain_viewer.segmentation import create_segment
from brain_viewer.segmentation_sources import display_segments
from brain_viewer.rendering import composite_plane
from test_multimodal import small_scene


def layer(scene,uid,sequence='postop-T1',source=None):
    return MRILayer(uid,uid,sequence,scene.data.copy(),np.ones(scene.data.shape,bool),
                    scene.data.copy(),scene.affine.copy(),np.eye(4),
                    {'rigid_source_uid':source} if source else {})


class ImageRemovalTests(unittest.TestCase):
    def test_removes_only_selected_candidate_or_source_family(self):
        scene=small_scene()
        scene.extra_mris=[layer(scene,'mr_base'),layer(scene,'mr_first',source='mr_base'),
                         layer(scene,'mr_second',source='mr_base'),layer(scene,'mr_pet','PET')]
        self.assertEqual(removal_ids(scene,'mr_first'),{'mr_first'})
        remove_images(scene,'mr_first')
        self.assertEqual(removal_ids(scene,'mr_base'),{'mr_base','mr_second'})
        remove_images(scene,'mr_base')
        self.assertEqual([s.uid for s in scene.extra_mris],['mr_pet'])
        with self.assertRaises(InputError):remove_images(scene,'mri')
        validate_scene(scene)

    def test_ct_deletion_keeps_contact_evidence_and_survives_save_reload(self):
        scene=small_scene(); scene.ct=scene.data.copy(); scene.ct_valid=np.ones(scene.data.shape,bool)
        scene.raw_ct=scene.ct.copy(); scene.raw_ct_affine=scene.affine.copy(); scene.ct_to_mri=np.eye(4)
        scene.contacts=[Contact('E1','E',np.array([10.,28.,28.]),(1.,0.,0.),
                                ct_position=np.array([10.,28.,28.]),provenance='native_ct_only')]
        expected=scene.contact_positions().copy()
        with TemporaryDirectory() as folder:
            root=Path(folder)
            original=root/'source.nii'; original.write_bytes(b'original source, never edited')
            before=save_scene(scene,root/'before',{})
            manifest=(before/'scene.json').read_bytes()
            remove_images(scene,'ct'); validate_scene(scene)
            after=save_scene(scene,root/'after',{}); loaded=load_scene(after/'scene.json')
            self.assertIsNone(loaded.ct); self.assertIsNotNone(loaded.raw_ct)
            np.testing.assert_array_equal(loaded.contact_positions(),expected)
            self.assertEqual(original.read_bytes(),b'original source, never edited')
            self.assertEqual((before/'scene.json').read_bytes(),manifest)
        scene.contacts=[]; scene.ct=scene.raw_ct
        remove_images(scene,'ct')
        self.assertIsNone(scene.raw_ct); self.assertIsNone(scene.ct_to_mri)

    def test_dti_scalar_removal_preserves_tracking_model(self):
        scene=small_scene(); fa=layer(scene,'mr_fa','DTI-FA'); b0=layer(scene,'mr_b0','DTI-b0')
        scene.extra_mris=[fa,b0]
        model=DiffusionModel('dti_test','DTI',scene.affine.copy(),np.eye(4),
            np.ones(scene.data.shape,np.float32)*.6,np.zeros(scene.data.shape+(3,),np.float32),
            np.ones(scene.data.shape,bool),np.array([0.,1000.]),np.zeros((2,3)),{'fa':fa.uid,'b0':b0.uid})
        scene.diffusions=[model]
        remove_images(scene,fa.uid); validate_scene(scene)
        self.assertIs(scene.diffusions[0],model); self.assertEqual(model.layer_ids,{'b0':b0.uid})
        with TemporaryDirectory() as folder:
            saved=save_scene(scene,Path(folder)/'saved',{}); loaded=load_scene(saved/'scene.json')
            self.assertEqual(loaded.diffusions[0].layer_ids,{'b0':b0.uid})
            np.testing.assert_array_equal(loaded.diffusions[0].fa,model.fa)

    def test_exclusions_are_contextual_without_changing_saved_visibility(self):
        scene=small_scene(); general=create_segment(scene,'manual','Anatomy')
        first=create_segment(scene,'manual','Exclusion 1'); second=create_segment(scene,'manual','Exclusion 2')
        general.mask[2:4,2:4,7]=True; first.mask[5:8,5:8,7]=True; second.mask[10:12,10:12,7]=True
        first.provenance['role']='postop_registration_exclusion'
        second.provenance['postop_exclusion_for']=['mr_postop']
        scene.segmentations=[general,first,second]
        ids=lambda context:[s.uid for s in display_segments(scene,context)]
        self.assertEqual(ids(None),[general.uid])
        self.assertEqual(ids({'workflow':'postop','exclusion':first.uid}),[general.uid,first.uid])
        self.assertEqual(ids({'workflow':'segmentation'}),[general.uid,first.uid,second.uid])
        normal=composite_plane(scene,2,7,0,200)
        editing=composite_plane(scene,2,7,0,200,{'segmentation_context':{'workflow':'postop','exclusion':first.uid}})
        self.assertFalse(np.array_equal(normal[6,6],editing[6,6]))
        np.testing.assert_array_equal(normal[10,10],editing[10,10])
        np.testing.assert_array_equal(normal[2,2],editing[2,2])
        self.assertTrue(all(s.visible_2d for s in scene.segmentations))
        scene.segmentation_preview=replace(first,uid='seg_preview',provenance={'role':'postop_exclusion_preview'})
        self.assertNotIn('seg_preview',ids(None))
        self.assertIn('seg_preview',ids({'workflow':'postop','exclusion':first.uid}))


if __name__=='__main__':unittest.main()
