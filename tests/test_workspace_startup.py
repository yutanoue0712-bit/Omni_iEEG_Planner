"""Clean source installations must open without bundled sample images."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from brain_viewer.patients import PatientStore
from brain_viewer.workspace_ui import WorkspaceUI


class WorkspaceStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.owner = SimpleNamespace(
            _smoke=False, project_root=self.root,
            patient_store=PatientStore(self.root / 'patients'), patient_id=None,
            load_default=Mock(), switch_patient=Mock(), clear_scene=Mock(),
            refresh_patients=Mock(), message=Mock(),
            next_patient_name=Mock(return_value='Case 001'), run_job=Mock(),
        )

    def test_clean_install_opens_empty_without_trying_to_load_samples(self):
        WorkspaceUI.start_workspace(self.owner, self.root / 'sample_images')
        self.owner.run_job.assert_not_called()
        self.owner.clear_scene.assert_called_once_with()
        self.owner.refresh_patients.assert_called_once_with()
        self.assertIsNone(self.owner.patient_id)
        self.assertEqual(self.owner.patient_store.list_patients(), [])

    def test_demo_does_not_need_a_sample_folder(self):
        source = self.root / 'sample_images'
        WorkspaceUI.start_workspace(self.owner, source, demo=True)
        self.owner.load_default.assert_called_once_with(source, True)
        self.owner.clear_scene.assert_not_called()

    def test_existing_patient_is_reopened_without_samples(self):
        record = self.owner.patient_store.create('Synthetic saved case')
        WorkspaceUI.start_workspace(self.owner, self.root / 'sample_images')
        self.owner.switch_patient.assert_called_once_with(record['id'])
        self.owner.clear_scene.assert_not_called()

    def test_existing_sample_directory_keeps_the_import_path(self):
        source = self.root / 'sample_images'
        source.mkdir()
        WorkspaceUI.start_workspace(self.owner, source)
        self.owner.run_job.assert_called_once()
        self.owner.clear_scene.assert_not_called()

    def test_explicit_missing_input_still_uses_normal_error_reporting(self):
        WorkspaceUI.start_workspace(self.owner, self.root / 'explicit_input')
        self.owner.run_job.assert_called_once()
        self.owner.clear_scene.assert_not_called()


if __name__ == '__main__':
    unittest.main()
