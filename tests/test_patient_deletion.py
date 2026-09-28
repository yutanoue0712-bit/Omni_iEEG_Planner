"""Deleting a managed patient must never delete imported source files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from brain_viewer.imaging import InputError
from brain_viewer.patients import PatientStore
from brain_viewer.workspace_ui import WorkspaceUI
from test_multimodal import small_scene


class PatientDeletionTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1]/'private_reports'
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(parent.resolve())
        self.addCleanup(self.temp.cleanup)
        self.store = PatientStore(self.root/'patients')
        self.source = self.root/'original_CT.nii'
        self.source.write_bytes(b'Original source bytes; must remain unchanged')
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def fake_trash(self, folder):
        folder = Path(folder)
        self.assertEqual(folder.resolve().parent, self.store.root)
        trash = self.root/'simulated_trash'
        trash.mkdir(exist_ok=True)
        folder.rename(trash/folder.name)
        return True

    def assert_source_preserved(self):
        self.assertTrue(self.source.is_file())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_whole_folder_history_assets_and_exports_removed_sources_and_other_patient_preserved(self):
        first = self.store.create('Delete A'); second = self.store.create('Keep B')
        for record in (first, second): self.store.save(record['id'], small_scene(), {})
        self.store.save(first['id'], small_scene(), {'level': 10})
        folder = self.store.folder(first['id'])
        (folder/'exports'/'image.png').write_bytes(b'synthetic export')
        (folder/'cache').mkdir(); (folder/'cache'/'derived.json').write_text('{}')
        # Even a hard link in the managed folder must not remove the outside name.
        os.link(self.source, folder/'raw_copy.nii')
        self.store.remember(first['id'])
        second_record = self.store.read(second['id'])
        with patch('brain_viewer.patients.move_to_trash', side_effect=self.fake_trash) as trash:
            self.store.delete(first['id'])
        trash.assert_called_once_with(folder)
        self.assertFalse(folder.exists())
        self.assertEqual([p['id'] for p in self.store.list_patients()], [second['id']])
        self.assertEqual(self.store.read(second['id']), second_record)
        self.assertIsNotNone(self.store.load(second['id']))
        self.assertIsNone(self.store.last_selected())
        self.assert_source_preserved()

    def test_failed_move_keeps_scene_manifest_and_selection(self):
        patient = self.store.create('Keep A')
        self.store.save(patient['id'], small_scene(), {})
        self.store.remember(patient['id'])
        before = self.store.read(patient['id'])
        with patch('brain_viewer.patients.move_to_trash', return_value=False):
            with self.assertRaisesRegex(InputError, '削除できません'):
                self.store.delete(patient['id'])
        self.assertEqual(self.store.read(patient['id']), before)
        self.assertEqual(self.store.last_selected(), patient['id'])
        self.assertIsNotNone(self.store.load(patient['id']))
        self.assert_source_preserved()

    def test_selection_write_failure_prevents_deletion(self):
        patient = self.store.create('Keep A'); self.store.remember(patient['id'])
        with patch('brain_viewer.patients.atomic_json', side_effect=OSError('disk full')), \
             patch('brain_viewer.patients.move_to_trash') as trash:
            with self.assertRaises(OSError): self.store.delete(patient['id'])
        trash.assert_not_called()
        self.assertTrue(self.store.folder(patient['id']).is_dir())

    def test_invalid_id_or_unmanaged_folder_never_reaches_trash(self):
        with patch('brain_viewer.patients.move_to_trash') as trash:
            for value in ('', '..', str(self.root), '../original_CT.nii', 'patient_'+'a'*16):
                with self.subTest(value=value), self.assertRaises((InputError, OSError)):
                    self.store.delete(value)
            patient = self.store.create('Invalid manifest')
            manifest = self.store.folder(patient['id'])/'patient.json'
            obj = json.loads(manifest.read_text()); obj['id'] = 'patient_'+'b'*16
            manifest.write_text(json.dumps(obj))
            with self.assertRaises(InputError): self.store.delete(patient['id'])
        trash.assert_not_called()
        self.assert_source_preserved()

    def test_reparse_point_at_root_or_inside_is_blocked_before_move(self):
        patient = self.store.create('Links')
        folder = self.store.folder(patient['id'])
        child = folder/'linked_file'; child.write_text('placeholder')
        original_stat = Path.lstat
        for target in (folder, child):
            def patched_lstat(path):
                if path == target:
                    return SimpleNamespace(st_mode=stat.S_IFDIR,
                                           st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
                return original_stat(path)
            with patch.object(Path, 'lstat', patched_lstat), \
                 patch('brain_viewer.patients.move_to_trash') as trash:
                with self.assertRaises(InputError): self.store.delete(patient['id'])
            trash.assert_not_called()
        self.assert_source_preserved()

    def test_delete_other_patient_keeps_current_selection(self):
        first = self.store.create('Delete A'); second = self.store.create('Keep B')
        self.store.remember(second['id'])
        with patch('brain_viewer.patients.move_to_trash', side_effect=self.fake_trash):
            self.store.delete(first['id'])
        self.assertEqual(self.store.last_selected(), second['id'])

    def test_last_patient_deletion_does_not_reimport_samples_on_restart(self):
        patient = self.store.create('Last patient')
        with patch('brain_viewer.patients.move_to_trash', side_effect=self.fake_trash):
            self.store.delete(patient['id'])
        restarted = PatientStore(self.store.root)
        owner = SimpleNamespace(_smoke=False, patient_store=restarted, patient_id=None,
                                clear_scene=Mock(), refresh_patients=Mock(), message=Mock(), run_job=Mock())
        WorkspaceUI.start_workspace(owner, self.root/'original_inputs')
        owner.run_job.assert_not_called()
        owner.clear_scene.assert_called_once()
        self.assertEqual(restarted.list_patients(), [])

    def test_real_os_trash_from_background_thread_preserves_original_hardlink(self):
        patient = self.store.create('Synthetic recycle test')
        folder = self.store.folder(patient['id'])
        os.link(self.source, folder/'source_copy.nii')
        self.store.remember(patient['id'])
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(pool.submit(self.store.delete, patient['id']).result(timeout=20), patient['id'])
        self.assertFalse(folder.exists())
        self.assertIsNone(self.store.last_selected())
        self.assert_source_preserved()


if __name__ == '__main__':
    unittest.main()
