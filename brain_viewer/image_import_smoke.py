"""Synthetic UI checks for PET/bone CT import, range controls and saved display."""
from unittest.mock import patch
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from PySide6.QtWidgets import QApplication
from .workspace_smoke import wait_job


def run_image_import_checks(window, capture_path=None):
    assert window.scene.kind == 'synthetic'
    initial=window.scene
    contacts=initial.contact_positions().copy()
    patient=window.patient_store.create('Synthetic imaging demo')
    window.patient_id=patient['id']; window.refresh_patients()
    for key in ('PET','CT-bone'):
        index=window.sequence_combo.findData(key)
        assert index>=0 and window.sequence_combo.model().item(index).isEnabled()
    window.sequence_combo.setCurrentIndex(window.sequence_combo.findData('PET'))
    raw=gaussian_filter(np.where(initial.data>0,initial.data*25,0),1.4).astype(np.float32)
    image=nib.Nifti1Image(raw,initial.affine); image.header.set_xyzt_units('mm')
    image.extra.update(value_units='BQML',input_kind='dicom',suv_conversion=False)
    with patch('brain_viewer.sequences.register_mri',return_value=(np.eye(4),{'review_status':'synthetic'})):
        window.import_image(image,'PET',name='PET demo'); wait_job(window)
    pet=window.scene.extra_mris[-1]
    assert window.extra_layer() is pet
    assert window.extra_mode.currentData()=='heatmap'
    assert window.extra_palette.isVisible() and window.extra_lower.isVisible()
    assert not window.extra_window.isVisible() and 'Bq/mL' in window.extra_info.text()
    assert window.extra_options()[pet.uid]['opacity']==.6
    before=[p.canvas._image.copy() for p in window.slices.values()]
    row=window.image_layers.rows[pet.uid]
    row.check.setChecked(False)
    assert all(p.canvas._image!=image for p,image in zip(window.slices.values(),before))
    row.check.setChecked(True)
    row.slider.setValue(73)
    window.extra_lower.setValue(1200); window.extra_upper.setValue(24000)
    assert abs(pet.level-pet.window/2-1200)<.01
    assert abs(pet.level+pet.window/2-24000)<.01
    window.extra_palette.setCurrentIndex(window.extra_palette.findData('inferno'))
    window.compare_extra()
    assert window.extra_mode.currentData()=='checker' and window.extra_window.isVisible()
    window.extra_mode.setCurrentIndex(window.extra_mode.findData('heatmap'))
    assert window.extra_lower.isVisible() and window.extra_palette.currentData()=='inferno'
    window.window_target.setCurrentIndex(window.window_target.findData(pet.uid))
    old_range=pet.window; window.drag_window(15,5)
    assert pet.window>old_range and abs(window.extra_lower.value()-(pet.level-pet.window/2))<.01
    window.extra_lower.setValue(1200); window.extra_upper.setValue(24000)
    row.slider.setValue(73)
    np.testing.assert_array_equal(pet.raw,raw)
    bone=nib.Nifti1Image(initial.raw_ct.copy(),initial.raw_ct_affine); bone.header.set_xyzt_units('mm')
    with patch('brain_viewer.registration.register_ct',return_value=(np.eye(4),{'review_status':'synthetic'})):
        window.import_image(bone,'CT-bone',name='CT bone demo'); wait_job(window)
    assert window.scene.extra_mris[-1].sequence=='CT-bone'
    assert window.extra_mode.currentData()=='full' and window.extra_window.value()==2000
    assert window.extra_level.value()==500
    np.testing.assert_array_equal(window.scene.ct,initial.ct)
    np.testing.assert_array_equal(window.scene.contact_positions(),contacts)
    for layer in window.scene.extra_mris:
        window.image_layers.rows[layer.uid].check.setChecked(layer.uid==pet.uid)
    window.ct_visible.setChecked(False)
    window.extra_combo.setCurrentIndex(window.extra_combo.findData(pet.uid))
    window.planes_check.setChecked(True)
    window.save_work(); wait_job(window)
    restored=window.patient_store.load(patient['id']); window.install_scene(restored)
    assert window.extra_layer().uid==pet.uid and window.extra_mode.currentData()=='heatmap'
    assert window.extra_palette.currentData()=='inferno'
    assert window.image_layers.rows[pet.uid].slider.value()==73
    assert window.extra_lower.value()==1200 and window.extra_upper.value()==24000
    np.testing.assert_array_equal(window.extra_layer().raw,raw)
    assert window.extra_options()[pet.uid]['palette']=='inferno'
    for language in (0,1):
        window.language_tabs.setCurrentIndex(language); QApplication.processEvents()
        assert window.extra_combo.currentText()=='PET demo'
        assert 'Bq/mL' in window.extra_info.text()
        assert ('Image units:' if language else '画像値の単位：') in window.extra_info.text()
        assert window.extra_palette.width()>100 and window.extra_lower.height()<40
        if capture_path:
            # Scroll the narrow controls into view for layout inspection.
            from PySide6.QtWidgets import QScrollArea
            parent=window.extra_group.parentWidget()
            while parent is not None and not isinstance(parent,QScrollArea): parent=parent.parentWidget()
            if parent is not None: parent.ensureWidgetVisible(window.extra_group)
            QApplication.processEvents()
            path=capture_path if language==0 else capture_path.with_stem(capture_path.stem+'_en')
            assert window.capture_widget(window).save(str(path),'PNG')
    window.language_tabs.setCurrentIndex(0)
    return {'pet_heatmap_import_and_saved_display':True,'palette_range_opacity_visibility_and_right_drag':True,
            'bone_ct_preserves_postoperative_ct_and_contacts':True,'slice_textures_and_bilingual_ui':True}
