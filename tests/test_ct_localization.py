"""Known contact centres in oblique, anisotropic synthetic CT; no patient inputs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib
from scipy.spatial.transform import Rotation
from brain_viewer.electrode_localization import detect_electrodes,attach_detection,remove_crossing_extensions
from brain_viewer.electrode_editing import create_lead,set_contact_position,export_records,refresh_quality
from brain_viewer.imaging import Contact,InputError,validate_scene,save_scene,load_scene
from brain_viewer.alignment_review import apply_alignment,rigid_delta
from brain_viewer.volume_io import read_volume
from test_multimodal import small_scene


def phantom(missing=False):
    shape=(132,116,76)
    affine=np.eye(4); affine[:3,:3]=Rotation.from_euler('xyz',[11,-17,9],degrees=True).as_matrix()@np.diag([.6,.6,1.])
    affine[:3,3]=[-33,-37,-31]
    indices=np.indices(shape).reshape(3,-1).T
    local=indices*np.array([.6,.6,1.])
    head=np.sum(((local-[39,35,38])/[38,34,37])**2,axis=1)<1
    data=np.where(head,35.,-1024.).astype(np.float32)
    truth=[]
    for count,start,axis,spacing in ((6,np.array([20.,22.,27.]),np.array([1.,.10,.04]),5.),
                                     (9,np.array([22.,48.,38.]),np.array([.90,-.08,.22]),4.)):
        axis=axis/np.linalg.norm(axis)
        points=start+np.arange(count)[:,None]*spacing*axis
        truth.append(nib.affines.apply_affine(affine,points/[.6,.6,1.]))
        for i,point in enumerate(points):
            if missing and count==9 and i==4: continue
            delta=local-point; along=delta@axis
            radial=np.linalg.norm(delta-along[:,None]*axis,axis=1)
            data[(np.abs(along)<.95)&(radial<.95)]=3071
    return data.reshape(shape),affine,truth


def native_scene():
    s=small_scene(); s.raw_ct=np.zeros(s.data.shape,np.float32)
    s.raw_ct[7,7,7]=3000; s.raw_ct_affine=s.affine.copy()
    s.ct=s.raw_ct.copy(); s.ct_valid=np.ones(s.data.shape,bool); s.ct_to_mri=np.eye(4)
    point=s.world([7,7,7])
    s.contacts=[Contact('A-01','A',point,(1.,.5,.2),ct_position=point.copy(),provenance='native_ct_only')]
    s.electrode_quality={'source':'native_ct_only','groups':{'A':{'expected_count':1}}}
    return s


class DetectorTests(unittest.TestCase):
    def test_different_counts_and_spacings_from_oblique_native_ct(self):
        data,affine,truth=phantom()
        result=detect_electrodes(data,affine)
        self.assertEqual(sorted(len(g['contacts']) for g in result['groups']),[6,9])
        for points in truth:
            group=next(g for g in result['groups'] if len(g['contacts'])==len(points))
            found=np.array([c['ct_ras_mm'] for c in group['contacts']])
            errors=np.linalg.norm(points[:,None,:]-found[None,:,:],axis=2).min(1)
            self.assertLess(float(errors.max()),.5)
            self.assertTrue(all(c['status']=='unreviewed' for c in group['contacts']))
        self.assertEqual(result['source'],'native_ct_only')

    def test_missing_ct_peak_is_never_silently_confirmed(self):
        data,affine,truth=phantom(missing=True)
        result=detect_electrodes(data,affine)
        contacts=[c for g in result['groups'] for c in g['contacts']]
        gap=[c for c in contacts if c['evidence']['origin']=='internal_gap']
        self.assertTrue(gap)
        self.assertTrue(all(c['status']=='uncertain' and not c['evidence']['supported'] for c in gap))

    def test_metal_absent_returns_empty_not_error(self):
        result=detect_electrodes(np.zeros((10,10,10),np.float32),np.eye(4))
        self.assertEqual(result['groups'],[])

    def test_extension_cannot_reuse_crossing_leads_observed_contact(self):
        groups=[{'contacts':[{'ct_ras_mm':[0,0,0],'evidence':{'origin':'component'}},
                             {'ct_ras_mm':[0,0,5],'evidence':{'origin':'end_extension'}}]},
                {'contacts':[{'ct_ras_mm':[.2,0,5],'evidence':{'origin':'component'}}]}]
        remove_crossing_extensions(groups)
        self.assertEqual([len(g['contacts']) for g in groups],[1,1])


class NativeContactTests(unittest.TestCase):
    def test_ambiguous_ct_keeps_mri_available_for_explicit_selection(self):
        from brain_viewer.multimodal import load_workspace
        with patch('brain_viewer.multimodal.discover_inputs',return_value=('mr','fs')), \
             patch('brain_viewer.multimodal.load_mri',return_value=small_scene()), \
             patch('brain_viewer.multimodal.discover_ct',side_effect=InputError('ambiguous CT')):
            scene=load_workspace(Path('unused'),Path('unused-cache'))
        self.assertIsNone(scene.ct)
        self.assertTrue(scene.quality['ct_input_selection_required'])

    def test_manual_location_and_registration_preserve_ct_canonical_position(self):
        s=native_scene(); raw=s.contacts[0].ct_position.copy()+[.1,.2,.3]
        set_contact_position(s,s.contacts[0],raw)
        matrix=rigid_delta([5,-3,2],[.3,-.2,.5],raw)
        changed=apply_alignment(s,matrix)
        np.testing.assert_array_equal(changed.contacts[0].ct_position,raw)
        np.testing.assert_allclose(changed.contacts[0].position,nib.affines.apply_affine(matrix,raw),atol=1e-10)
        self.assertEqual(changed.contacts[0].uid,s.contacts[0].uid)
        self.assertIsNone(changed.contacts[0].source_position)
        self.assertEqual(len(changed.contacts[0].evidence['history']),1)
        validate_scene(changed)
        changed.contacts[0].position[0]+=1
        with self.assertRaises(InputError): validate_scene(changed)

    def test_seeded_counts_and_outside_ct(self):
        s=native_scene()
        created=create_lead(s,s.world([2,3,4]),s.world([12,3,4]),3,'B')
        self.assertEqual(len(created.contacts),4)
        self.assertEqual(len(s.contacts),1)
        self.assertTrue(all(c.status=='uncertain' for c in created.contacts[1:]))
        self.assertEqual(created.electrode_quality['groups']['B']['expected_count'],3)
        with self.assertRaises(InputError): create_lead(s,s.world([-4,3,4]),s.world([12,3,4]),3,'B')
        with self.assertRaises(InputError): set_contact_position(s,s.contacts[0],s.world([100,0,0]))

    def test_close_contact_warning_and_export_coordinate_basis(self):
        s=native_scene(); c=deepcopy(s.contacts[0]); c.uid='new'; c.name='B-01'; c.group='B'
        c.ct_position=c.ct_position+[.2,0,0]; c.position=c.ct_position.copy(); s.contacts.append(c)
        refresh_quality(s)
        self.assertEqual(s.contacts[0].evidence['nearby_contact_uids'],['new'])
        out=export_records(s)
        self.assertEqual(out['coordinate_system'],'scanner_RAS_mm')
        self.assertEqual(out['contacts'][0]['source'],'native_ct_only')
        np.testing.assert_allclose(out['contacts'][0]['ct_voxel_zero_based'],[7,7,7])
        self.assertNotIn('original_brainstorm_mri_ras_mm',out['contacts'][0])

    def test_saved_ct_contact_review_roundtrip_and_old_import_discard(self):
        s=native_scene(); s.contacts[0].status='reviewed'; s.contacts[0].evidence={'human_reviewed':True}
        parent=Path(__file__).resolve().parents[1]/'private_reports'
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            path=save_scene(s,Path(folder),{})/'scene.json'
            restored=load_scene(path)
            self.assertEqual(restored.contacts[0].uid,s.contacts[0].uid)
            self.assertEqual(restored.contacts[0].status,'reviewed')
            self.assertEqual(restored.contacts[0].evidence,{'human_reviewed':True})
            np.testing.assert_array_equal(restored.contacts[0].ct_position,s.contacts[0].ct_position)
            manifest=json.loads(path.read_text(encoding='utf-8'))
            manifest['electrode_quality']['source']='Brainstorm legacy test'
            path.write_text(json.dumps(manifest))
            self.assertEqual(load_scene(path).contacts,[])

    def test_nifti_conflicting_coordinate_matrices_are_rejected(self):
        image=nib.Nifti1Image(np.zeros((10,10,10),np.float32),np.eye(4)); image.header.set_xyzt_units('mm')
        image.set_qform(np.eye(4),code=1)
        wrong=np.eye(4); wrong[0,3]=5; image.set_sform(wrong,code=1)
        with self.assertRaises(InputError): read_volume(image)
