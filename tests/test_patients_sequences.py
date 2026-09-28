"""Patient isolation, atomic saved versions, and independent MRI sequence geometry."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from brain_viewer.imaging import MRILayer, InputError, Scene, Mesh, save_scene, load_scene
from brain_viewer.patients import PatientStore
from brain_viewer.rendering import composite_plane
from brain_viewer.sequences import register_mri, resample_mri, find_image_sources
from brain_viewer.alignment_review import rigid_delta
from test_multimodal import small_scene


def sequence_scene():
    scene = small_scene()
    raw = np.full(scene.data.shape, 25., np.float32)
    valid = np.ones(raw.shape, bool); valid[0] = False
    scene.extra_mris = [MRILayer('mr_test1', 'T2 test', 'T2', raw.copy(), valid, raw,
        scene.affine.copy(), np.eye(4), {'review_status':'automatic_alignment_requires_visual_review'}, 100., 50.)]
    return scene


class PatientAndSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = PatientStore(self.root/'patients')

    def test_patient_isolation_and_self_contained_history(self):
        first = self.store.create('Case A'); second = self.store.create('Case B')
        a = sequence_scene(); b = small_scene(); b.data = b.data*2
        state = {'image_layers': {'mr_test1': {'visible':False, 'opacity':.23}}, 'window_target':'mr_test1'}
        original = self.store.save(first['id'], a, state)
        self.store.save(second['id'], b, {})
        a.extra_mris[0].level = 40
        self.store.save(first['id'], a, state)
        restored = self.store.load(first['id'])
        self.assertEqual(restored.extra_mris[0].level, 40)
        self.assertEqual(load_scene(original/'scene.json').extra_mris[0].level, 50)
        self.assertEqual(restored.view_state, state)
        np.testing.assert_array_equal(self.store.load(second['id']).data, b.data)
        self.assertEqual(self.store.load(second['id']).extra_mris, [])
        # A saved work folder loads directly; it has no external asset references.
        manifest = json.loads((original/'scene.json').read_text(encoding='utf-8'))
        self.assertTrue((original/manifest['extra_mris'][0]['raw']).is_file())
        self.store.remember(first['id'])
        self.assertEqual(PatientStore(self.store.root).last_selected(), first['id'])

    def test_failed_snapshot_keeps_previous_latest_and_other_patient(self):
        record = self.store.create('Case A'); scene = sequence_scene()
        self.store.save(record['id'], scene, {'level':1})
        before = self.store.read(record['id'])
        with patch('brain_viewer.local_assets.os.link', side_effect=OSError('disk')), patch('brain_viewer.local_assets.shutil.copyfile', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.store.save(record['id'], scene, {'level':2})
        self.assertEqual(self.store.read(record['id']), before)
        self.assertEqual(self.store.load(record['id']).view_state, {'level':1})

    def test_names_are_labels_and_paths_stay_inside_patient(self):
        record = self.store.create('Case / A : example')
        self.assertTrue(self.store.folder(record['id']).is_relative_to(self.store.root))
        with self.assertRaises(InputError): self.store.create('Case / A : example')
        with self.assertRaises(InputError): self.store.folder('../outside')
        with self.assertRaises(InputError): self.store.scene_path(record['id'], '../../scene.json')
        old_folder = self.store.folder(record['id'])
        self.store.rename(record['id'], 'Case renamed')
        self.assertEqual(self.store.folder(record['id']), old_folder)
        self.assertEqual(self.store.read(record['id'])['name'], 'Case renamed')

    def test_extra_mri_mixes_only_valid_field_and_does_not_move_contacts(self):
        scene = sequence_scene()
        options = {'mri_visible':False, 'extra_mris':{'mr_test1':{'visible':True,'opacity':1.}}}
        rendered = composite_plane(scene, 2, 5, 0, 200, options)
        np.testing.assert_array_equal(rendered[1:],63)
        np.testing.assert_array_equal(rendered[0],0)
        options['mri_visible'] = True
        options['extra_mris']['mr_test1']['opacity'] = .5
        rendered = composite_plane(scene,2,5,0,200,options)
        np.testing.assert_array_equal(rendered[1:],95)
        np.testing.assert_array_equal(rendered[0],127)
        options['extra_mris']['mr_test1']['visible'] = False
        np.testing.assert_array_equal(composite_plane(scene,2,5,0,200,options),127)

    def test_additional_mri_does_not_trigger_ct_only_mode(self):
        scene = sequence_scene()
        scene.ct = np.full(scene.data.shape,50.,np.float32)
        scene.ct_valid = np.ones(scene.data.shape,bool)
        options = {'mri_visible':False,'visible':True,'mode':'bone','opacity':1.,'window':100.,'level':50.,
                   'extra_mris':{'mr_test1':{'visible':True,'opacity':1.}}}
        # Soft-tissue CT is transparent in bone mode; the added MRI remains visible.
        np.testing.assert_array_equal(composite_plane(scene,2,5,0,200,options)[1:],63)

    def test_unknown_new_layer_has_default_and_legacy_scene_loads(self):
        scene = sequence_scene()
        saved = save_scene(scene,self.root,{})
        obj = json.loads((saved/'scene.json').read_text(encoding='utf-8'))
        obj['extra_mris'][0]['to_reference_ras_mm'][0][0] = 2
        (saved/'scene.json').write_text(json.dumps(obj),encoding='utf-8')
        with self.assertRaises(InputError): load_scene(saved/'scene.json')
        obj.pop('extra_mris'); obj['schema'] = 'cortex-viewer/4'
        (saved/'scene.json').write_text(json.dumps(obj),encoding='utf-8')
        self.assertEqual(load_scene(saved/'scene.json').extra_mris,[])

    def test_folder_picker_does_not_descend_brainstorm_or_freesurfer(self):
        image = nib.Nifti1Image(np.ones((4,4,4),np.float32),np.eye(4))
        image.header.set_xyzt_units('mm')
        for name in ('scan', 'case_bs', 'case_fs'):
            folder = self.root/name; folder.mkdir(); nib.save(image,folder/'image.nii.gz')
        sources = find_image_sources(self.root, 'MR')
        self.assertEqual(len(sources),1)
        self.assertEqual(sources[0][1].parent.name,'scan')


class MRIRegistrationTests(unittest.TestCase):
    def test_contrast_changed_oblique_anisotropic_mri_recovers_known_pose(self):
        self.check_known_pose(None)

    def test_postoperative_sampling_grid_recovers_known_pose(self):
        self.check_known_pose(1.5)

    def check_known_pose(self, final_spacing):
        shape=(76,90,72)
        affine=np.diag([1.3,1.3,1.3,1.]); affine[:3,3]=[-49,-58,-46]
        grid=np.indices(shape).reshape(3,-1).T
        x,y,z=nib.affines.apply_affine(affine,grid).T.reshape(3,*shape)
        radius=(x/42)**2+(y/50)**2+(z/39)**2
        t1=np.zeros(shape,np.float32); t2=np.zeros(shape,np.float32)
        t1[radius<1]=220; t2[radius<1]=700
        t1[radius<.8]=800; t2[radius<.8]=220
        for a,b,c,rx,ry,rz,u,v in ((-15,-10,8,8,18,8,75,1100),(12,4,12,7,15,6,100,950),
                (-6,-29,-22,15,11,12,450,400),(20,25,-9,7,10,11,600,300)):
            mask=((x-a)/rx)**2+((y-b)/ry)**2+((z-c)/rz)**2<1
            t1[mask]=u; t2[mask]=v
        vertices=np.array([[-36,-42,-32],[36,42,32],[-36,42,32],[36,-42,-32]],np.float32)
        mesh=Mesh(vertices,np.array([[0,1,2],[1,2,3]],np.int32))
        scene=Scene(t1,affine,affine.copy(),shape,{'pial_lh':mesh,'pial_rh':mesh})
        truth=rigid_delta([12,-5,7],[3,-4,5],[0,0,0])
        native=np.diag([1.1,1.3,1.7,1.]); native[:3,3]=[-66,-75,-62]
        raw=resample_from_to(nib.Nifti1Image(t2,affine),((122,116,76),truth@native),order=1).get_fdata(dtype=np.float32)
        estimated,quality=register_mri(scene,raw,native,final_spacing_mm=final_spacing)
        self.assertEqual(quality['final_sampling_mm'],final_spacing)
        points=np.array([[a,b,c] for a in (-22,0,22) for b in (-26,0,26) for c in (-22,0,22)])
        errors=np.linalg.norm(nib.affines.apply_affine(estimated@np.linalg.inv(truth),points)-points,axis=1)
        p95=float(np.percentile(errors,95)); print(f'MRI phantom landmark p95 error: {p95:.3f} mm',flush=True)
        self.assertLess(p95,1.5)
        self.assertIsNone(quality['landmark_accuracy_mm'])
        aligned,valid=resample_mri(raw,native,scene,estimated)
        self.assertEqual(aligned.shape,scene.data.shape)
        self.assertGreater(float(valid.mean()),.8)
