"""Synthetic-only checks of the patient menu, confirmation and deletion lifecycle."""
from dataclasses import replace
import os
import time
from unittest.mock import patch
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QMessageBox
from .i18n import tr
from .workspace_smoke import wait_job


def answer_delete(window, accept, capture_path=None):
    result = []
    deadline = time.monotonic()+5
    def answer():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            assert dialog.windowTitle() == tr('患者を削除')
            assert 'Demo delete me' in dialog.text()
            assert dialog.textFormat() == Qt.TextFormat.PlainText
            assert dialog.defaultButton().text() == tr('キャンセル')
            assert dialog.escapeButton() is dialog.defaultButton()
            assert dialog.informativeText() and dialog.detailedText()
            if capture_path:
                assert window.capture_widget(dialog).save(str(capture_path), 'PNG')
            role = QMessageBox.ButtonRole.DestructiveRole if accept else QMessageBox.ButtonRole.RejectRole
            button = next(button for button in dialog.buttons() if dialog.buttonRole(button) == role)
            result.append(True)
            button.click()
        elif time.monotonic() < deadline:
            QTimer.singleShot(10, answer)
    QTimer.singleShot(10, answer)
    return result


def run_patient_delete_checks(window, capture_path=None):
    assert window.scene.kind == 'synthetic'
    store, original_patient, original = window.patient_store, window.patient_id, window.scene
    source = store.root.parent/'original_to_keep.nii'
    source.write_bytes(b'Synthetic original input outside managed patient folders')
    original_bytes = source.read_bytes()
    patient = store.create('Demo delete me')
    scene = replace(original, extra_mris=[], segmentations=[], results=[], view_state={})
    store.save(patient['id'], scene, {})
    folder = store.folder(patient['id'])
    os.link(source, folder/'input_copy.nii')
    (folder/'exports'/'synthetic.png').write_bytes(b'synthetic export')
    window.switch_patient(patient['id']); wait_job(window)
    active = window.scene
    before = store.read(patient['id'])
    # Cancel leaves both the current scene and its saved history untouched.
    for language, suffix in ((1, 'en'), (0, 'ja')):
        window.language_tabs.setCurrentIndex(language); QApplication.processEvents()
        capture = capture_path.with_stem(capture_path.stem+'_delete_'+suffix) if capture_path else None
        answered = answer_delete(window, False, capture)
        window.delete_patient_action.trigger()
        assert answered and window.worker is None
        assert window.scene is active and window.patient_id == patient['id']
        assert store.read(patient['id']) == before and source.read_bytes() == original_bytes
    # A failed OS operation must not clear the scene or lose the selected patient.
    with patch('brain_viewer.patients.move_to_trash', return_value=False):
        answered = answer_delete(window, True)
        window.delete_patient_action.trigger(); wait_job(window)
        assert answered and window.scene is active and window.patient_id == patient['id']
        assert store.read(patient['id']) == before and store.last_selected() == patient['id']
        assert window.delete_patient_action.isEnabled() and window.save_button.isEnabled()
    # Now use the real OS recycle operation on this synthetic patient only.
    answered = answer_delete(window, True)
    window.delete_patient_action.trigger(); wait_job(window)
    assert answered and not folder.exists()
    assert window.patient_id is None and window.scene is None
    assert not window.surface.actors and not window.surface.contact_actors
    assert all(panel.canvas.scene is None for panel in window.slices.values())
    assert window.patient_combo.findData(patient['id']) < 0
    assert not window.delete_patient_action.isEnabled() and not window.save_button.isEnabled()
    assert source.read_bytes() == original_bytes and store.last_selected() is None
    window.save_work()
    assert not folder.exists() and all(p['id'] != patient['id'] for p in store.list_patients())
    window.switch_patient(original_patient); wait_job(window)
    assert window.scene is not None and window.patient_id == original_patient
    assert source.read_bytes() == original_bytes and not folder.exists()
    return {'patient_delete_confirmation_and_cancel': True, 'patient_delete_failure_preserves_scene': True,
            'patient_folder_deleted_originals_preserved': True, 'deleted_patient_cannot_autosave_back': True}
