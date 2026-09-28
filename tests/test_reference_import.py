"""Regression coverage for the two distinct reference-MRI input folders."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brain_viewer.imaging import InputError, load_mri, validate_freesurfer_folder
from brain_viewer.sequences import find_image_sources
from brain_viewer.workspace_ui import WorkspaceUI


class ImportHarness:
    scene = None

    def __init__(self, root):
        self.project_root = root
        self.errors = []
        self.imports = []

    def _job_failed(self, message):
        self.errors.append(message)

    def import_image(self, source, sequence, **kwargs):
        self.imports.append((source, sequence, kwargs))


class ReferenceImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scan = self.root / 'scan' / 't1'
        self.scan.mkdir(parents=True)
        self.fs = self.root / 'synthetic_fs'
        (self.fs / 'mri').mkdir(parents=True)
        (self.fs / 'surf').mkdir()
        # These placeholders exercise selection only, never image decoding.
        (self.fs / 'mri' / 'orig.mgz').touch()
        for hemi in ('lh', 'rh'):
            (self.fs / 'surf' / f'{hemi}.pial.T1').touch()

    def test_scan_folder_is_rejected_before_decoding_mri(self):
        with patch('brain_viewer.volume_io.read_volume') as read:
            with self.assertRaisesRegex(InputError, 'FreeSurfer.*mri/orig.mgz') as error:
                load_mri('unread-input.nii.gz', self.scan)
        read.assert_not_called()
        self.assertNotIn(str(self.scan), str(error.exception))

    def test_mri_subfolder_requires_subject_parent(self):
        with self.assertRaisesRegex(InputError, 'mriとsurfを含むフォルダ'):
            validate_freesurfer_folder(self.fs / 'mri')

    def test_pial_t1_fallback_passes_and_missing_hemisphere_fails(self):
        self.assertEqual(validate_freesurfer_folder(self.fs), self.fs)
        (self.fs / 'surf' / 'rh.pial.T1').unlink()
        with self.assertRaisesRegex(InputError, '左右の脳表'):
            validate_freesurfer_folder(self.fs)

    def test_wrong_second_folder_can_be_corrected_without_reselecting_mri(self):
        owner = ImportHarness(self.root)
        source = {'synthetic_selected_series': 1}
        with patch('brain_viewer.workspace_ui.QFileDialog.getExistingDirectory',
                   side_effect=[str(self.scan), str(self.fs)]) as dialog:
            WorkspaceUI.prepare_image_import(owner, source, 'reference')
        self.assertEqual(len(owner.errors), 1)
        self.assertEqual(owner.imports, [(source, 'reference', {'fs': self.fs})])
        self.assertEqual(dialog.call_count, 2)
        self.assertTrue(all(call.args[1].startswith('2/2') for call in dialog.call_args_list))

    def test_cancel_after_wrong_second_folder_does_not_import(self):
        owner = ImportHarness(self.root)
        with patch('brain_viewer.workspace_ui.QFileDialog.getExistingDirectory',
                   side_effect=[str(self.scan), '']):
            WorkspaceUI.prepare_image_import(owner, 'synthetic.nii.gz', 'reference')
        self.assertEqual(len(owner.errors), 1)
        self.assertEqual(owner.imports, [])

    def test_freesurfer_at_scan_step_explains_selection_order(self):
        with patch('brain_viewer.sequences.dicom_series') as scan:
            with self.assertRaisesRegex(InputError, '2番目の画面'):
                find_image_sources(self.fs, 'MR')
        scan.assert_not_called()


if __name__ == '__main__':
    unittest.main()
