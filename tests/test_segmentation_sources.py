from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib

from brain_viewer.imaging import MRILayer, Contact, InputError, save_scene, load_scene
from brain_viewer.segmentation_sources import available_labels, create_label_segment, image_sources, extract_segment
from brain_viewer.sequences import add_image, source_modality
from test_multimodal import small_scene


def extraction_scene():
    scene=small_scene()
    data=np.zeros(scene.data.shape,np.float32); valid=np.zeros(data.shape,bool)
    valid[4:13,4:13,4:13]=True
    data[6:8,6:8,6:8]=100; data[11,11,11]=100
    scene.extra_mris=[MRILayer('mr_contrast','Contrast scan','T1ce',data,valid,data.copy(),scene.affine.copy(),np.eye(4))]
    return scene


class LabelSelectionTests(unittest.TestCase):
    def test_all_present_ids_are_listed_even_without_a_name_and_nuclei_are_separate(self):
        scene=small_scene(); scene.label_volume=np.zeros(scene.data.shape,np.uint16)
        for i,label in enumerate((2,4,1001,2001,9999)): scene.label_volume[i,2,3]=label
        scene.label_names={2:'Left-WM',1001:'ctx-lh-demo',7777:'Absent label'}
        scene.nuclei_display=np.zeros(scene.data.shape,np.uint16); scene.nuclei_display[1,2,3]=8103
        scene.nuclei_names={8103:'Left-AV'}; scene.nuclei_source='nuclei'
        rows=available_labels(scene)
        self.assertEqual({(r['source'],r['label']) for r in rows},
                         {('anatomy',n) for n in (2,4,1001,2001,9999)}|{('nuclei',8103)})
        for row in rows:
            segment=create_label_segment(scene,row['source'],row['label'])
            source=scene.label_volume if row['source']=='anatomy' else scene.nuclei_display
            np.testing.assert_array_equal(segment.mask,source==row['label'])
            self.assertEqual(segment.provenance['present_label_ids'],[row['label']])
            self.assertFalse(np.shares_memory(segment.mask,source))
        with self.assertRaises(InputError): create_label_segment(scene,'anatomy',0)
        with self.assertRaises(InputError): create_label_segment(scene,'anatomy',7777)


class ImageExtractionTests(unittest.TestCase):
    def test_sequence_selection_uses_values_and_valid_fov_not_reference_mri(self):
        scene=extraction_scene()
        candidate=extract_segment(scene,'mr_contrast',90,110,'Lesion',seed=[7,7,7],connected=False,radius_mm=0)
        self.assertEqual(candidate.mask.sum(),9)
        reference=extract_segment(scene,'mri',90,110,'Reference',connected=False,radius_mm=0)
        self.assertEqual(reference.mask.sum(),scene.data.size)
        zeros=extract_segment(scene,'mr_contrast',0,0,'Zero',connected=False,radius_mm=0)
        self.assertFalse(zeros.mask[0].any())
        self.assertEqual(zeros.mask.sum(),scene.extra_mris[0].valid.sum()-9)
        self.assertEqual(candidate.provenance['source_id'],'mr_contrast')
        self.assertEqual(candidate.provenance['source_sequence'],'T1ce')

    def test_connected_component_and_physical_roi_do_not_include_other_islands(self):
        scene=extraction_scene()
        candidate=extract_segment(scene,'mr_contrast',90,110,'Connected',seed=[7,7,7],radius_mm=0)
        self.assertEqual(candidate.mask.sum(),8); self.assertFalse(candidate.mask[11,11,11])
        scene.affine[0,0]=2.
        scene.source_affine=scene.affine.copy()
        region=extract_segment(scene,'mri',90,110,'ROI',seed=[7,7,7],connected=False,radius_mm=2.)
        self.assertTrue(region.mask[8,7,7]); self.assertFalse(region.mask[9,7,7]); self.assertTrue(region.mask[7,9,7])

    def test_ct_uses_hu_and_native_transform_without_moving_electrodes(self):
        scene=extraction_scene(); scene.ct=scene.extra_mris[0].data.copy(); scene.ct[6:8,6:8,6:8]=350
        scene.ct_valid=scene.extra_mris[0].valid.copy(); scene.ct_to_mri=np.eye(4); scene.ct_to_mri[0,3]=5
        scene.raw_ct_affine=scene.affine.copy()
        scene.contacts=[Contact('E01-01','E01',np.array([8.,29,30]),(.2,.5,.9))]
        before=scene.contact_positions().copy()
        candidate=extract_segment(scene,'ct',300,400,'Candidate','artery',[7,7,7],True,10.)
        self.assertEqual(candidate.mask.sum(),8); self.assertEqual(candidate.provenance['source_units'],'HU')
        np.testing.assert_array_equal(candidate.provenance['source_to_reference_ras_mm'],scene.ct_to_mri)
        np.testing.assert_array_equal(scene.contact_positions(),before)

    def test_window_changes_leave_mask_and_source_fingerprint_unchanged(self):
        scene=extraction_scene()
        a=extract_segment(scene,'mr_contrast',90,110,'Before',seed=[7,7,7])
        scene.extra_mris[0].window=99; scene.extra_mris[0].level=-23
        b=extract_segment(scene,'mr_contrast',90,110,'After',seed=[7,7,7])
        np.testing.assert_array_equal(a.mask,b.mask)
        self.assertEqual(a.provenance['source_display_sha256'],b.provenance['source_display_sha256'])
        self.assertEqual(scene.extra_mris[0].data[7,7,7],100)

    def test_invalid_seed_interval_or_missing_source_do_not_create_regions(self):
        scene=extraction_scene()
        for uid,lo,hi,seed in [('mr_contrast',90,110,[0,0,0]),('mr_contrast',110,90,[7,7,7]),
                              ('mr_missing',90,110,[7,7,7]),('mr_contrast',90,110,[40,40,40]),
                              ('mr_contrast',90,float('nan'),[7,7,7]),('mr_contrast',90,110,None)]:
            with self.assertRaises(InputError): extract_segment(scene,uid,lo,hi,'Test',seed=seed)
        self.assertEqual(scene.segmentations,[])

    def test_reference_outside_native_field_is_excluded(self):
        scene=small_scene(); scene.source_shape=(5,5,5)
        mask=extract_segment(scene,'mri',90,110,'FOV',connected=False,radius_mm=0).mask
        self.assertEqual(mask.sum(),125); self.assertFalse(mask[5:].any())

    def test_source_provenance_roundtrip_and_preview_not_saved(self):
        scene=extraction_scene()
        committed=extract_segment(scene,'mr_contrast',90,110,'Committed',seed=[7,7,7])
        preview=extract_segment(scene,'mri',90,110,'Preview',seed=[7,7,7])
        scene.segmentations=[committed]; scene.segmentation_preview=preview
        with tempfile.TemporaryDirectory() as directory:
            folder=save_scene(scene,Path(directory),{})
            restored=load_scene(folder/'scene.json')
            self.assertEqual(len(restored.segmentations),1); self.assertIsNone(restored.segmentation_preview)
            self.assertEqual(restored.segmentations[0].provenance,committed.provenance)
            np.testing.assert_array_equal(restored.segmentations[0].mask,committed.mask)

    def test_cta_ctv_append_separately_from_post_ct_and_preserve_old_masks(self):
        scene=extraction_scene(); scene.ct=scene.data.copy(); scene.ct_valid=np.ones(scene.data.shape,bool)
        scene.ct_to_mri=np.eye(4); scene.raw_ct=scene.ct.copy(); scene.raw_ct_affine=scene.affine.copy()
        scene.contacts=[Contact('E01-01','E01',np.array([8.,29,30]),(.2,.5,.9))]
        candidate=extract_segment(scene,'mr_contrast',90,110,'Old mask',seed=[7,7,7]); scene.segmentations=[candidate]
        original_ct=scene.ct; original_contacts=scene.contact_positions().copy()
        raw=scene.data*3; matrix=np.eye(4); matrix[:3,3]=[2,-3,1]
        with tempfile.TemporaryDirectory() as directory:
            for sequence in ('CTA','CTV'):
                with patch('brain_viewer.registration.cached_registration',return_value=(raw,np.ones(raw.shape,bool),matrix,{},raw,scene.affine.copy())):
                    scene=add_image(scene,'synthetic',sequence,sequence,Path(directory))
                self.assertIs(scene.ct,original_ct)
                np.testing.assert_array_equal(scene.contact_positions(),original_contacts)
                self.assertIs(scene.segmentations[0],candidate)
                layer=scene.extra_mris[-1]
                self.assertEqual(get_source_units(scene,layer.uid),'HU')
            folder=save_scene(scene,Path(directory),{})
            restored=load_scene(folder/'scene.json')
            self.assertEqual([s.sequence for s in restored.extra_mris],['T1ce','CTA','CTV'])
            np.testing.assert_array_equal(restored.ct,original_ct)
        self.assertEqual(source_modality('T1ce'),'MR'); self.assertEqual(source_modality('CTA'),'CT'); self.assertEqual(source_modality('CTV'),'CT')


def get_source_units(scene,uid):
    return next(source.units for source in image_sources(scene) if source.uid==uid)
