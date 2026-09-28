"""Synthetic PET surface, contact editor, hover and CSV integration checks."""
from pathlib import Path
from unittest.mock import patch
import csv
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from PySide6.QtCore import Qt,QPoint,QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QLabel
from .image_import_smoke import run_image_import_checks
from .workspace_smoke import wait_job
from .contact_editor import ContactEditor
from .contact_labels_dialog import ContactLabelsDialog
from .i18n import set_language,translate_widgets


def run_review_refinement_checks(window,capture_path=None):
    result=run_image_import_checks(window)
    pet=next(layer for layer in window.scene.extra_mris if layer.sequence=='PET')
    window.extra_combo.setCurrentIndex(window.extra_combo.findData(pet.uid))
    assert window.extra_lower_slider.isVisible() and window.extra_upper_slider.isVisible()
    window.extra_surface.setChecked(True)
    assert window.surface.actors['pial_lh'].GetMapper().GetScalarVisibility()
    old=vtk_to_numpy(window.surface.polys['pial_lh'].GetPointData().GetScalars()).copy()
    cached=list(window.surface.pet_colors.samples.values())
    window.extra_upper_slider.setValue(350); QTest.qWait(100)
    assert window.extra_upper.value()>window.extra_lower.value()
    assert not np.array_equal(old,vtk_to_numpy(window.surface.polys['pial_lh'].GetPointData().GetScalars()))
    assert all(a is b for a,b in zip(cached,window.surface.pet_colors.samples.values()))
    window.extra_lower_slider.setValue(1000); QTest.qWait(70)
    assert window.extra_lower.value()<window.extra_upper.value()
    window.extra_lower_slider.setValue(60); window.extra_upper_slider.setValue(700); QTest.qWait(70)
    row=window.image_layers.rows[pet.uid]; row.check.setChecked(False)
    assert not window.surface.actors['pial_lh'].GetMapper().GetScalarVisibility()
    row.check.setChecked(True)
    assert window.surface.actors['pial_lh'].GetMapper().GetScalarVisibility()
    window.planes_check.setChecked(False)
    window.save_work(); wait_job(window)
    window.install_scene(window.patient_store.load(window.patient_id))
    assert window.extra_surface.isChecked() and window.surface.actors['pial_lh'].GetMapper().GetScalarVisibility()
    assert not window.findChildren(QLabel,'localBadge') and not hasattr(window,'subtitle')
    canvas=window.slices[2].canvas
    ijk=window.ijk.copy(); target=ijk.copy(); target[0]+=8
    QTest.mouseMove(canvas,canvas.position_for_index(target).toPoint()); QApplication.processEvents()
    assert canvas.hover_annotation() is not None
    np.testing.assert_array_equal(window.ijk,ijk)
    QApplication.sendEvent(canvas,QEvent(QEvent.Type.Leave))
    assert canvas.hover_annotation() is None
    if capture_path:
        assert window.capture_widget(window).save(str(capture_path),'PNG')
    original=window.scene.contact_positions().copy()
    editor=ContactEditor(window.scene,window); editor.show(); QApplication.processEvents()
    assert editor.move_check.isChecked() and all(s.edit for s in editor.sections)
    editor.overview.selector.setCurrentIndex(editor.overview.selector.findData('mri_2'))
    QApplication.processEvents(); assert editor.overview.head_ct.modality=='MRI'
    np.testing.assert_array_equal(editor.overview.head_ct.positions,[c.position for c in editor.current()])
    width=editor.width_spin.value(); mri_width=editor.overview.mri_width.value()
    editor.overview.image_windowing(10,5)
    assert editor.width_spin.value()==width and editor.overview.mri_width.value()>mri_width
    section=editor.sections[0]; section.zoom_by(1.5)
    selected=editor.current()[1]
    QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,section.project(selected.ct_position).toPoint())
    assert editor.index==1 and not editor.undo
    position=selected.ct_position.copy(); point=section.project(position).toPoint()+QPoint(3,-2)
    normal=np.cross(section.u,section.v)
    expected=section.plane_at(point)+normal*np.dot(position-section.view_center,normal)
    QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
    np.testing.assert_allclose(editor.current()[1].ct_position,expected,atol=1e-8)
    assert len(editor.undo)==1
    np.testing.assert_array_equal(editor.overview.head_ct.positions[1],editor.current()[1].position)
    editor.undo_edit(); np.testing.assert_array_equal(editor.current()[1].ct_position,position)
    if capture_path:
        assert editor.capture().save(str(capture_path.with_stem(capture_path.stem+'_t1_review')),'PNG')
    editor.reject(); np.testing.assert_array_equal(window.scene.contact_positions(),original)
    dialog=ContactLabelsDialog(window.scene,window.export_directory(),window); dialog.show(); QApplication.processEvents()
    assert dialog.table.rowCount()==len(window.scene.contacts)
    dialog.group.setCurrentIndex(1); group=dialog.group.currentData()
    assert all(row['electrode']==group for row in dialog.visible_records())
    directory=Path(window._patient_temp.name); output=directory/'contact_anatomy.csv'
    with patch('brain_viewer.contact_labels_dialog.QFileDialog.getSaveFileName',return_value=(str(output),'CSV (*.csv)')):
        dialog.export_button.click()
    with output.open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==dialog.table.rowCount() and all(row['electrode']==group for row in rows)
    for language in ('ja','en'):
        set_language(language); translate_widgets(dialog); dialog.refresh(); QApplication.processEvents()
        if capture_path:
            assert dialog.grab().save(str(capture_path.with_stem(capture_path.stem+'_labels_'+language)),'PNG')
    dialog.accept(); window.language_tabs.setCurrentIndex(1); QApplication.processEvents()
    assert window.extra_surface.text()=='Show PET on the 3D cortex'
    if capture_path:
        assert window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+'_en')),'PNG')
    window.language_tabs.setCurrentIndex(0)
    result.update(pet_surface_and_sliders_and_restore=True,hover_without_cursor_change=True,
                  contact_t1_context_and_click_edit=True,contact_anatomy_table_and_csv=True,header_simplified=True)
    return result
