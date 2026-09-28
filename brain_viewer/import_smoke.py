"""Synthetic GUI checks through file/folder pickers and recoverable input errors."""
from dataclasses import replace
from pathlib import Path
import time
from unittest.mock import patch
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QInputDialog
from .registration import to_sitk
from .workspace_smoke import wait_job


def settled(window, predicate=lambda: True):
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if window.worker is None and window._after_job is None and predicate():
            return
        time.sleep(.01)
    raise AssertionError('Image import did not finish')


def choose_source(window, accept=True):
    observed = []
    deadline = time.monotonic()+8
    def find_dialog():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QInputDialog):
            observed.append((window.worker is None, window.import_folder_button.isEnabled(),
                             window._job_timer.isActive()))
            # Keep a real nested event loop open long enough to deliver finished signals.
            QTimer.singleShot(80, dialog.accept if accept else dialog.reject)
        elif time.monotonic() < deadline:
            QTimer.singleShot(10, find_dialog)
    QTimer.singleShot(10, find_dialog)
    return observed


def dicom_phantom(folder, data, affine):
    volume = to_sitk(data.astype(np.int16), affine)
    direction = np.asarray(volume.GetDirection()).reshape(3, 3)
    orientation = '\\'.join(str(v) for v in np.r_[direction[:, 0], direction[:, 1]])
    for index in range(volume.GetDepth()):
        part = volume[:, :, index]
        for key, value in {'0008|0060': 'CT', '0008|0016': '1.2.840.10008.5.1.4.1.1.2',
                           '0008|0008': 'ORIGINAL\\PRIMARY\\AXIAL',
                           '0020|000d': '2.25.9212026330', '0020|000e': '2.25.9212026331',
                           '0020|0013': str(index+1), '0020|0037': orientation,
                           '0020|0032': '\\'.join(str(v) for v in volume.TransformIndexToPhysicalPoint((0, 0, index))),
                           '0028|0030': '\\'.join(str(v) for v in volume.GetSpacing()[1::-1])}.items():
            part.SetMetaData(key, value)
        writer = sitk.ImageFileWriter(); writer.KeepOriginalImageUIDOn()
        writer.SetFileName(str(folder/f'{index:03d}.dcm')); writer.Execute(part)


def run_import_checks(window, capture_path=None):
    assert window.scene.kind == 'synthetic'
    original_patient, original = window.patient_id, window.scene
    store = window.patient_store
    patient = store.create('Demo input recovery')
    window.switch_patient(patient['id']); wait_job(window)
    baseline = replace(original, ct=None, ct_valid=None, ct_to_mri=None, raw_ct=None,
                       raw_ct_affine=None, ct_quality={}, contacts=[], electrode_quality={},
                       extra_mris=[], segmentations=[], results=[], view_state={})
    with patch('brain_viewer.workspace_ui.load_mri', return_value=baseline):
        window.import_image('synthetic', 'reference', fs='synthetic_fs'); wait_job(window)
    folder = store.folder(patient['id'])/'input_test'; folder.mkdir()
    image = nib.Nifti1Image(original.raw_ct.copy(), original.raw_ct_affine)
    image.header.set_xyzt_units('mm')
    image.set_qform(image.affine, code=1); image.set_sform(image.affine, code=1)
    for name in ('ct.nii', 'ct.nii.gz'): nib.save(image, folder/name)
    shifted = image.affine.copy(); shifted[0, 3] += 50
    image.set_sform(shifted, code=1); nib.save(image, folder/'invalid.nii.gz')
    window.sequence_combo.setCurrentIndex(window.sequence_combo.findData('CT'))
    quality = {'version': 'rigid-mi-multistart-v3', 'final_negative_mi': -.3, 'review_status': 'synthetic'}
    with patch('brain_viewer.registration.register_ct', return_value=(np.eye(4), quality)):
        for name in ('ct.nii', 'ct.nii.gz'):
            previous = window.scene
            with patch('brain_viewer.workspace_ui.QFileDialog.getOpenFileName', return_value=(str(folder/name), '')):
                window.import_file_button.click()
                settled(window, lambda: window.scene is not previous)
            assert window.scene.ct is not None and window.ct_visible.isChecked()
        picker = choose_source(window)
        previous = window.scene
        with patch('brain_viewer.workspace_ui.QFileDialog.getExistingDirectory', return_value=str(folder)):
            window.import_folder_button.click()
            settled(window, lambda: window.scene is not previous)
        assert picker == [(True, True, False)]
        previous, saved = window.scene, store.read(patient['id'])['latest_scene']
        canceled = choose_source(window, accept=False)
        with patch('brain_viewer.workspace_ui.QFileDialog.getExistingDirectory', return_value=str(folder)):
            window.import_folder_button.click()
            settled(window, lambda: bool(canceled))
        assert window.scene is previous and store.read(patient['id'])['latest_scene'] == saved
        assert 'キャンセル' in window.statusBar().currentMessage()
        for recovery in ('missing', 'available'):
            with patch('brain_viewer.workspace_ui.QFileDialog.getOpenFileName', return_value=(str(folder/'invalid.nii.gz'), '')):
                window.import_file_button.click()
                settled(window, lambda: window.import_feedback.isVisible())
            assert window.scene is previous and store.read(patient['id'])['latest_scene'] == saved
            assert window.import_dicom_button.isVisible() and window.import_folder_button.isEnabled()
            assert not window._job_timer.isActive() and QApplication.activeModalWidget() is None
            if capture_path and recovery == 'missing':
                for language, suffix in ((1, 'en'), (0, 'ja')):
                    window.language_tabs.setCurrentIndex(language); QApplication.processEvents()
                    assert window.import_feedback.height() >= window.import_feedback.heightForWidth(window.import_feedback.width())
                    assert ('Check geometry' in window.import_feedback.text()) if language else ('座標を確認' in window.import_feedback.text())
                    assert window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+'_input_error_'+suffix)), 'PNG')
            if recovery == 'available': dicom_phantom(folder, original.raw_ct, original.raw_ct_affine)
            window.import_dicom_button.click()
            settled(window, lambda: window.scene is not previous if recovery == 'available' else window.import_feedback.isVisible())
            if recovery == 'missing':
                assert window.scene is previous and store.read(patient['id'])['latest_scene'] == saved
                assert '元DICOMがありません' in window.import_feedback.text()
        assert window.scene.ct_quality['input_kind'] == 'dicom'
        assert window.ct_visible.isChecked() and not window.import_feedback.isVisible()
    window.switch_patient(original_patient); wait_job(window)
    assert window._import_recovery is None and not window.import_dicom_button.isVisible()
    return {'nifti_nii_and_gz_file_buttons': True, 'folder_picker_to_import': True,
            'folder_cancel_preserves_scene': True, 'nifti_error_dicom_recovery': True,
            'input_errors_leave_viewer_responsive': True}
