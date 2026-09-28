"""Regression cases for mixed localizers, input preflight and CT import stages."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from brain_viewer.imaging import InputError
from brain_viewer.volume_io import dicom_series, read_volume, check_ct_candidate
from brain_viewer.sequences import find_image_sources
from brain_viewer.registration import cached_registration, validate_registration_metric
from brain_viewer.electrode_localization import cached_detection
from brain_viewer.workspace_ui import WorkspaceUI
from brain_viewer.window import ViewerWindow
from test_dicom_input import write_dicom
from test_multimodal import small_scene


def write_scout(folder, image, filename='scout.dcm'):
    part = image[:, :, 0]
    for key, value in {
        '0008|0060': 'CT', '0008|0016': '1.2.840.10008.5.1.4.1.1.2',
        '0008|0008': 'DERIVED\\PRIMARY\\LOCALIZER',
        '0020|000d': '2.25.778811010', '0020|000e': '2.25.778811011',
        '0020|0013': '1', '0020|0032': '17\\-21\\-8',
        '0020|0037': '1\\0\\0\\0\\0\\1', '0028|0030': '.8\\.7',
    }.items():
        part.SetMetaData(key, value)
    writer = sitk.ImageFileWriter(); writer.KeepOriginalImageUIDOn()
    writer.SetFileName(str(folder / filename)); writer.Execute(part)


class CTImportTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / 'private_reports'
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)
        assert self.root.resolve().is_relative_to(parent.resolve())
        self.addCleanup(self.temp.cleanup)

    def test_same_uid_scout_is_excluded_and_voxels_keep_physical_locations(self):
        original, data = write_dicom(self.root)
        write_scout(self.root, original)
        series = dicom_series(self.root)
        self.assertEqual(len(series), 1)
        self.assertEqual(series.excluded_localizers, 1)
        self.assertEqual(series[0]['count'], 6)
        image = read_volume(series[0])
        np.testing.assert_array_equal(image.get_fdata(), data.transpose(2, 1, 0))
        expected = np.array(original.TransformIndexToPhysicalPoint((3, 5, 4))) * [-1, -1, 1]
        np.testing.assert_allclose(nib.affines.apply_affine(image.affine, [3, 5, 4]), expected)

    def test_file_order_is_replaced_by_physical_slice_order(self):
        _, data = write_dicom(self.root)
        series = dicom_series(self.root)[0]
        series = dict(series, files=tuple(reversed(series['files'])))
        image = read_volume(series)
        np.testing.assert_array_equal(image.get_fdata(), data.transpose(2, 1, 0))

    def test_changed_file_invalidates_cached_header_before_read(self):
        original, _ = write_dicom(self.root)
        series = dicom_series(self.root)[0]
        write_scout(self.root, original, filename='005.dcm')
        with self.assertRaises(InputError):
            read_volume(series)

    def test_missing_middle_slice_is_rejected_without_filling_gap(self):
        write_dicom(self.root)
        (self.root/'003.dcm').unlink()
        with self.assertRaisesRegex(InputError, 'スライス間隔'):
            read_volume(self.root, modality='CT')

    def test_folder_picker_prefers_dicom_and_rejects_conflicting_nifti_headers(self):
        original, _ = write_dicom(self.root)
        write_scout(self.root, original)
        image = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4))
        image.header.set_xyzt_units('mm')
        image.set_qform(np.eye(4), code=1)
        shifted = np.eye(4); shifted[0, 3] = 50
        image.set_sform(shifted, code=1)
        nib.save(image, self.root/'conflict.nii.gz')
        image.set_sform(np.eye(4), code=1)
        nib.save(image, self.root/'valid.nii.gz')
        # Discovery must not inflate MRI/CT arrays merely to show choices.
        with patch.object(nib.Nifti1Image, 'get_fdata', side_effect=AssertionError('pixel read')):
            sources = find_image_sources(self.root, 'CT')
        self.assertEqual(len(sources), 2)
        self.assertIsInstance(sources[0][1], dict)
        self.assertEqual(len(sources.rejected), 1)
        self.assertIn('qform', sources.rejected[0][1])
        self.assertEqual(sources.excluded_localizers, 1)

    def test_blank_ct_fails_before_registration_or_cache_creation(self):
        image = nib.Nifti1Image(np.zeros((5, 5, 5), np.float32), np.eye(4))
        image.header.set_xyzt_units('mm')
        with patch('brain_viewer.registration.register_ct') as register:
            with self.assertRaisesRegex(InputError, '全体が同じ値'):
                cached_registration(small_scene(), image, None, self.root/'cache')
        register.assert_not_called()
        self.assertFalse((self.root/'cache').exists())

    def test_unmarked_binary_nifti_is_excluded_from_ct_folder_choices(self):
        write_dicom(self.root)
        values = np.zeros((8, 9, 10), np.uint8)
        values[2:6, 2:7, 2:8] = 1
        image = nib.Nifti1Image(values, np.eye(4))
        image.header.set_xyzt_units('mm')
        nib.save(image, self.root/'derived.nii.gz')
        sources = find_image_sources(self.root, 'CT')
        self.assertEqual(len(sources), 1)
        self.assertIsInstance(sources[0][1], dict)
        self.assertEqual(len(sources.rejected), 1)
        self.assertIn('0〜1', sources.rejected[0][1])

    def test_normalized_float_ct_fails_before_registration_or_cache_creation(self):
        data = np.linspace(0, 1, 125, dtype=np.float32).reshape(5, 5, 5)
        image = nib.Nifti1Image(data, np.eye(4))
        image.header.set_xyzt_units('mm')
        with patch('brain_viewer.registration.register_ct') as register:
            with self.assertRaisesRegex(InputError, '0〜1'):
                cached_registration(small_scene(), image, None, self.root/'cache')
        register.assert_not_called()
        self.assertFalse((self.root/'cache').exists())

    def test_scaled_eight_bit_ct_keeps_hu_values(self):
        data = np.arange(125, dtype=np.uint8).reshape(5, 5, 5)
        image = nib.Nifti1Image(data, np.eye(4))
        image.header.set_xyzt_units('mm')
        image.header.set_slope_inter(20., -1024.)
        nib.save(image, self.root/'scaled.nii.gz')
        sources = find_image_sources(self.root, 'CT')
        self.assertEqual(len(sources), 1)
        loaded = read_volume(sources[0][1])
        np.testing.assert_array_equal(loaded.get_fdata(), data.astype(float)*20-1024)
        check_ct_candidate(loaded)

    def test_nifti_label_intent_is_rejected(self):
        image = nib.Nifti1Image(np.arange(125, dtype=np.int16).reshape(5, 5, 5), np.eye(4))
        image.header.set_intent('label')
        with self.assertRaisesRegex(InputError, '領域ラベル'):
            check_ct_candidate(image)

    def test_saved_mask_cannot_be_used_for_contact_detection(self):
        data = np.zeros((5, 5, 5), np.float32); data[2, 2, 2] = 1
        with patch('brain_viewer.electrode_localization.detect_electrodes') as detect:
            with self.assertRaisesRegex(InputError, '0〜1'):
                cached_detection(data, np.eye(4), self.root/'cache')
        detect.assert_not_called()
        self.assertFalse((self.root/'cache').exists())

    def test_registration_without_image_information_is_rejected(self):
        for value in (0., 1e-12, -1e-12, float('nan')):
            with self.subTest(value=value), self.assertRaises(InputError):
                validate_registration_metric(value)
        validate_registration_metric(-.3)

    def owner(self):
        owner = SimpleNamespace(worker=None, scene=small_scene(), patient_id='synthetic',
                                view_state=lambda: {}, patient_store=Mock(),
                                clear_import_feedback=Mock(),
                                import_ct_contacts=SimpleNamespace(isChecked=lambda: False))
        owner.patient_store.folder.return_value = self.root
        owner.run_job = lambda operation, callback, description, **kwargs: operation(lambda _: None)
        return owner

    def test_invalid_ct_does_not_create_patient_snapshots(self):
        owner = self.owner()
        with patch('brain_viewer.multimodal.attach_ct', side_effect=InputError('invalid coordinates')):
            with self.assertRaises(InputError):
                WorkspaceUI.import_image(owner, 'synthetic', 'CT')
        owner.patient_store.save.assert_not_called()

    def test_ct_display_can_finish_without_contact_detection(self):
        owner = self.owner(); result = replace(owner.scene)
        with patch('brain_viewer.multimodal.attach_ct', return_value=(result, None, None)), \
             patch('brain_viewer.multimodal.create_ct_contacts') as detect:
            WorkspaceUI.import_image(owner, 'synthetic', 'CT')
        detect.assert_not_called()
        self.assertEqual(owner.patient_store.save.call_count, 2)
        self.assertIs(owner.patient_store.save.call_args_list[0].args[1], owner.scene)
        self.assertIs(owner.patient_store.save.call_args_list[1].args[1], result)

    def test_contact_detection_can_be_explicitly_included(self):
        owner = self.owner(); result = replace(owner.scene)
        owner.install_scene = Mock(); owner.side_tabs = Mock(); owner.message = Mock()
        owner.detect_contacts = Mock(); owner._after_job = None
        def run_job(operation, completed, description, **kwargs):
            imported = operation(lambda _: None)
            owner.detect_contacts.assert_not_called()
            completed(imported)
        owner.run_job = run_job
        with patch('brain_viewer.multimodal.attach_ct', return_value=(result, None, None)), \
             patch('brain_viewer.multimodal.create_ct_contacts', return_value=result) as detect:
            WorkspaceUI.import_image(owner, 'synthetic', 'CT', detect_contacts=True)
        detect.assert_not_called()
        owner.install_scene.assert_called_once_with(result)
        self.assertEqual(owner.patient_store.save.call_count, 2)
        owner.detect_contacts.assert_not_called()
        owner._after_job()
        owner.detect_contacts.assert_called_once()

    def test_zero_candidates_leave_ct_and_existing_contacts_available(self):
        scene = small_scene(); candidate = replace(scene, contacts=[])
        owner = SimpleNamespace(scene=scene, message=Mock(), electrode_info=Mock(),
                                write_status=Mock(), _after_job=None)
        with patch('brain_viewer.window.ContactEditor') as editor:
            ViewerWindow._contacts_detected(owner, candidate)
            ViewerWindow.edit_contacts(owner, candidate=candidate)
        editor.assert_not_called()
        self.assertIs(owner.scene, scene)
        self.assertIsNone(owner._after_job)
        self.assertIn('0個', owner.message.call_args.args[0])
        self.assertEqual(owner.write_status.call_args.args[1]['candidate_count'], 0)

    def test_cached_registration_reuses_transform_and_preserves_samples(self):
        scene = small_scene()
        data = np.arange(np.prod(scene.data.shape), dtype=np.float32).reshape(scene.data.shape)
        image = nib.Nifti1Image(data, scene.affine)
        image.header.set_xyzt_units('mm')
        quality = {'version': 'rigid-mi-multistart-v3', 'landmark_accuracy_mm': None,
                   'final_negative_mi': -.3}
        with patch('brain_viewer.registration.register_ct', return_value=(np.eye(4), quality)) as register:
            first = cached_registration(scene, image, None, self.root/'cache')
            second = cached_registration(scene, image, None, self.root/'cache')
        self.assertEqual(register.call_count, 1)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertTrue(second[3]['registration_cache_hit'])
        self.assertIn('resample_seconds', second[3]['import_timings'])
