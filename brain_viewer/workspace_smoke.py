"""Synthetic-only integration checks for patient switching and MRI imports."""
from dataclasses import replace
import time
from unittest.mock import patch
import nibabel as nib
import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


def wait_job(window, timeout=90):
    deadline = time.monotonic()+timeout
    while window.worker is not None and time.monotonic()<deadline:
        QApplication.processEvents(); time.sleep(.01)
    assert window.worker is None, 'Workspace operation did not finish'
    QTest.qWait(180); QApplication.processEvents()


def run_workspace_checks(window, capture_path=None):
    assert window.scene.kind == 'synthetic'
    store = window.patient_store
    patient_a, patient_b = store.create('Demo case A'), store.create('Demo case B')
    window.patient_id = patient_a['id']; window.refresh_patients()
    original_contacts = window.scene.contact_positions().copy()
    initial = window.scene
    # Exercise the real extra-MRI importer, resampler, saving and list insertion.
    t2 = np.where(initial.data>0, 1300.-initial.data, 0).astype(np.float32)
    for sequence, data in (('T2',t2), ('FLAIR',np.where(t2>1050,0,t2).astype(np.float32)),('T1ce',initial.data*1.3)):
        image = nib.Nifti1Image(data, initial.affine)
        image.header.set_xyzt_units('mm'); image.set_sform(initial.affine,code=1)
        # Registration recovery is covered by the independent known-pose test.
        # These GUI phantoms already share physical coordinates; isolate import/saving here.
        with patch('brain_viewer.sequences.register_mri', return_value=(np.eye(4), {
                'method':'synthetic shared coordinates','review_status':'synthetic'})):
            window.import_image(image,sequence,name=sequence+' demo')
            wait_job(window)
        assert window.scene.extra_mris[-1].sequence == sequence
        assert window.image_layers.rows[window.scene.extra_mris[-1].uid].check.isVisible()
        np.testing.assert_array_equal(window.scene.contact_positions(),original_contacts)
    for sequence in ('CTA','CTV'):
        image=nib.Nifti1Image(initial.ct.copy(),initial.affine)
        image.header.set_xyzt_units('mm'); image.set_sform(initial.affine,code=1)
        with patch('brain_viewer.registration.register_ct',return_value=(np.eye(4),{'review_status':'synthetic'})):
            window.import_image(image,sequence,name=sequence+' demo'); wait_job(window)
        np.testing.assert_array_equal(window.scene.ct,initial.ct)
        np.testing.assert_array_equal(window.scene.contact_positions(),original_contacts)
    assert len(window.scene.extra_mris)==5
    from .segmentation_sources_smoke import check_added_source
    for layer in window.scene.extra_mris[2:]: check_added_source(window,layer)
    segment_ids = [s.uid for s in window.scene.segmentations]
    initial = window.scene
    first, second = window.scene.extra_mris[:2]
    window.extra_combo.setCurrentIndex(window.extra_combo.findData(first.uid))
    window.compare_extra()
    assert not window.ct_visible.isChecked()
    assert window.extra_styles[first.uid]=='checker'
    assert not window.image_layers.rows[second.uid].check.isChecked()
    width = first.window
    window.drag_window(20,10)
    assert first.window > width
    window.image_layers.rows[first.uid].slider.setValue(43)
    window.extra_level.setValue(375.)
    window.planes_check.setChecked(True)
    window.save_work(); wait_job(window)
    saved = window.view_state()
    assert len(store.read(patient_a['id'])['images'])==7
    window.switch_patient(patient_b['id']); wait_job(window)
    assert window.scene is None and not window.surface.actors and not window.surface.contact_actors
    assert not window.surface.segment_actors and not window.segmentation.history
    assert all(panel.canvas.scene is None for panel in window.slices.values())
    assert window.import_folder_button.isEnabled() and window.sequence_combo.currentData()=='reference'
    assert not window.image_layers.rows['mri'].isVisible()
    # A fresh patient's baseline import must not inherit another patient's CT/contacts/sequences.
    baseline = replace(initial,data=initial.data*.7,ct=None,ct_valid=None,ct_to_mri=None,raw_ct=None,
                       raw_ct_affine=None,ct_quality={},contacts=[],electrode_quality={},extra_mris=[],segmentations=[],view_state={})
    with patch('brain_viewer.workspace_ui.load_mri',return_value=baseline):
        window.import_image('synthetic-input','reference',fs='synthetic-freesurfer')
        wait_job(window)
    assert window.patient_id==patient_b['id'] and window.scene.ct is None
    assert not window.scene.contacts and not window.scene.extra_mris
    assert not window.scene.segmentations and not window.surface.segment_actors
    assert not window.image_layers.rows['ct'].isVisible()
    # The optional detector starts after CT is displayed and durably saved.
    # Empty results must complete without opening a blocking, empty editor.
    ct_image = nib.Nifti1Image(initial.raw_ct.copy(), initial.raw_ct_affine)
    ct_image.header.set_xyzt_units('mm')
    detected = []
    def empty_detection(imported, cache, progress):
        assert window.scene is imported and imported.ct is not None
        assert store.load(patient_b['id']).ct is not None
        detected.append(True)
        return replace(imported, contacts=[])
    with patch('brain_viewer.registration.register_ct', return_value=(np.eye(4), {
            'version': 'rigid-mi-multistart-v3', 'final_negative_mi': -.3, 'review_status': 'synthetic'})), \
         patch('brain_viewer.multimodal.create_ct_contacts', side_effect=empty_detection), \
         patch('brain_viewer.window.ContactEditor', side_effect=AssertionError('Empty editor opened')):
        window.import_image(ct_image, 'CT', detect_contacts=True)
        wait_job(window)
        wait_job(window)
        assert detected and not window.scene.contacts
        assert window.worker is None and window._after_job is None
        assert window.import_folder_button.isEnabled()
        assert '0個' in window.statusBar().currentMessage()
        assert window.image_layers.rows['ct'].isVisible()
        images = [panel.canvas._image.copy() for panel in window.slices.values()]
        window.ct_visible.setChecked(False)
        assert all(panel.canvas._image != image for panel, image in zip(window.slices.values(), images))
        window.ct_visible.setChecked(True)
    window.patient_combo.setCurrentIndex(window.patient_combo.findData(patient_a['id']))
    window.patient_selected(window.patient_combo.currentIndex()); wait_job(window)
    assert window.patient_id==patient_a['id'] and len(window.scene.extra_mris)==5
    assert [s.uid for s in window.scene.segmentations]==segment_ids and not window.segmentation.history
    for before,after in zip(initial.segmentations,window.scene.segmentations):
        np.testing.assert_array_equal(before.mask,after.mask)
    np.testing.assert_array_equal(window.scene.contact_positions(),original_contacts)
    assert window.image_layers.rows[first.uid].slider.value()==43
    assert window.extra_layer(first.uid).level==375.
    assert window.extra_styles[first.uid]=='checker'
    assert window.patient_combo.currentText()=='Demo case A'
    assert store.last_selected()==patient_a['id']
    assert window.view_state()['image_layers']==saved['image_layers']
    # The new entry points and dynamic image rows must fit both languages.
    window.side_tabs.setCurrentIndex(0)
    for language in (0,1):
        window.language_tabs.setCurrentIndex(language); QApplication.processEvents()
        assert window.patient_combo.currentText()=='Demo case A'
        assert window.extra_combo.itemText(window.extra_combo.findData(first.uid))=='T2 demo'
        for row in window.image_layers.rows.values():
            assert row.check.geometry().right()<row.slider.geometry().left()
        if capture_path:
            suffix = '_patients_ja' if language==0 else '_patients_en'
            assert window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+suffix)),'PNG')
    window.language_tabs.setCurrentIndex(0)
    return {'patient_switch_autosave_and_isolation':True,'multiple_mri_import_save_display':True,
            'ct_visible_before_detection_and_zero_candidates_finish':True,
            'contrast_t1_cta_ctv_import_and_extraction':True,'cta_ctv_preserve_post_ct_and_electrodes':True}
