import csv
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from brain_viewer.contact_labels import contact_label_records,write_contact_labels
from brain_viewer.imaging import Contact
from test_ct_localization import native_scene


class ContactLabelsTests(unittest.TestCase):
    def test_base_anatomy_and_native_thalamic_nucleus_are_kept_separate(self):
        scene=native_scene(); scene.label_volume=np.full(scene.data.shape,10,np.uint16)
        scene.label_names={10:'Left-Thalamus'}; scene.label_source='aparc+aseg'
        scene.nuclei=np.full((40,40,40),8101,np.uint16); scene.nuclei_affine=scene.affine.copy()
        scene.nuclei_affine[:3,:3]*=.5; scene.nuclei_names={8101:'Left-Pulvinar'}; scene.nuclei_source='thalamic nuclei'
        scene.contacts[0].position+=.2; scene.contacts[0].status='reviewed'
        row=contact_label_records(scene)[0]
        self.assertEqual((row['freesurfer_id'],row['freesurfer_label']),(10,'Left-Thalamus'))
        self.assertEqual((row['thalamic_nucleus_id'],row['thalamic_nucleus']),(8101,'Left-Pulvinar'))
        self.assertEqual(row['mri_r_mm'],scene.contacts[0].position[0])
        self.assertEqual(row['review_status'],'reviewed')

    def test_labels_recomputed_after_edit_and_missing_or_outside_is_explicit(self):
        scene=native_scene(); scene.label_volume=np.zeros(scene.data.shape,np.uint16)
        scene.label_volume[7,7,7]=10; scene.label_volume[8,7,7]=49
        scene.label_names={10:'Left-Thalamus',49:'Right-Thalamus'}
        self.assertEqual(contact_label_records(scene)[0]['freesurfer_id'],10)
        scene.contacts[0].position=scene.world([8,7,7])
        self.assertEqual(contact_label_records(scene)[0]['freesurfer_id'],49)
        scene.contacts[0].position=scene.world([30,7,7])
        self.assertEqual(contact_label_records(scene)[0]['label_status'],'outside_volume')
        scene.label_volume=None
        self.assertEqual(contact_label_records(scene)[0]['label_status'],'not_loaded')

    def test_csv_utf8_names_status_coordinates_and_filtered_rows(self):
        scene=native_scene(); scene.contacts[0].name='=確認,"接点"'; scene.contacts[0].status='uncertain'
        second=deepcopy(scene.contacts[0]); second.group='B'; second.name='B-01'; second.uid='other'
        scene.contacts.append(second); rows=contact_label_records(scene)
        self.assertEqual([r['contact_number'] for r in rows],[1,1])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'labels.csv'; write_contact_labels(path,rows[:1])
            self.assertTrue(path.read_bytes().startswith(b'\xef\xbb\xbf'))
            with path.open(encoding='utf-8-sig',newline='') as stream: records=list(csv.DictReader(stream))
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]['contact'],"'=確認,\"接点\"")
            self.assertEqual(records[0]['review_status'],'uncertain')
            self.assertAlmostEqual(float(records[0]['mri_a_mm']),scene.contacts[0].position[1])
