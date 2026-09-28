"""Synthetic CT-editor navigation and editing checks in the complete Qt/VTK dialog."""
import time
import numpy as np
import nibabel as nib
from PySide6.QtCore import Qt,QPoint,QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from .contact_editor import ContactEditor
from .i18n import set_language,translate_widgets


def run_contact_navigation_checks(window,capture_path=None):
    scene=window.scene
    assert scene.kind=='synthetic' and scene.contacts and scene.raw_ct is not None
    original=np.array([c.ct_position for c in scene.contacts])
    editor=ContactEditor(scene,window);editor.show();QApplication.processEvents()
    lead_before=np.array([c.ct_position for c in editor.current()])
    durations=[]
    for section in editor.sections:
        start=time.perf_counter();section.zoom_out.click();section.zoom_out.click()
        QApplication.processEvents();durations.append(round((time.perf_counter()-start)*1000,2))
        assert np.isclose(section.zoom,.64)
        point=section.image_rect().center().toPoint()+QPoint(14,9)
        anchor=section.plane_at(point).copy()
        event=QWheelEvent(QPointF(point),QPointF(section.mapToGlobal(point)),QPoint(),QPoint(0,120),
                         Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        QApplication.sendEvent(section,event)
        np.testing.assert_allclose(section.plane_at(point),anchor,atol=1e-8)
        editor.move_check.setChecked(True)
        for button,modifier in ((Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ShiftModifier),
                                (Qt.MouseButton.MiddleButton,Qt.KeyboardModifier.NoModifier)):
            pan=section.pan.copy();QTest.mousePress(section,button,modifier,point)
            QTest.mouseMove(section,point+QPoint(25,11),20)
            QTest.mouseRelease(section,button,modifier,point+QPoint(25,11))
            assert not np.array_equal(section.pan,pan)
        editor.move_check.setChecked(False)
    assert not editor.undo
    np.testing.assert_array_equal([c.ct_position for c in editor.current()],lead_before)
    view_states=[(s.zoom,s.pan.copy(),s.raw) for s in editor.sections]
    editor.windowing(10,5)
    for section,(zoom,pan,raw) in zip(editor.sections,view_states):
        assert section.zoom==zoom and section.raw is raw
        np.testing.assert_array_equal(section.pan,pan)
    # Select and edit using the currently zoomed/panned image, then undo.
    section=editor.sections[0]
    point=section.project(editor.current()[1].ct_position).toPoint()
    QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
    assert editor.index==1 and section.zoom==view_states[0][0]
    editor.move_check.setChecked(True);point=point+QPoint(4,-2)
    normal=np.cross(section.u,section.v)
    expected=section.plane_at(point)+normal*np.dot(editor.current()[1].ct_position-section.view_center,normal)
    QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
    np.testing.assert_allclose(editor.current()[1].ct_position,expected,atol=1e-8)
    np.testing.assert_allclose(editor.current()[1].position,nib.affines.apply_affine(scene.ct_to_mri,expected),atol=1e-8)
    editor.undo_edit();editor.move_check.setChecked(False)
    np.testing.assert_array_equal([c.ct_position for c in editor.current()],lead_before)
    for section in editor.sections:
        section.setFocus();QTest.keyClick(section,Qt.Key.Key_Home)
        assert section.zoom==1. and not section.pan.any()
    # Deliberately different fields of view make independent controls visible.
    editor.sections[0].zoom_by(.5);editor.sections[1].zoom_by(.8);editor.sections[2].zoom_by(2.)
    for size in ((1080,720),(1480,930)):
        editor.resize(*size);QApplication.processEvents()
        for section in editor.sections:
            assert section.header.geometry().bottom()<section.image_rect().top()
            assert section.reset_button.geometry().right()<section.header.width()
            assert section.zoom_out.width()>=24 and section.zoom_in.width()>=24
            assert section.reset_button.width()>=60
            assert section.title.geometry().right()<section.zoom_out.geometry().left()
            assert section.image_rect().height()>40
        assert editor.rect().contains(editor.table.mapTo(editor,editor.table.rect().bottomRight()))
    if capture_path:assert editor.capture().save(str(capture_path),'PNG')
    set_language('en');translate_widgets(editor);QApplication.processEvents()
    assert editor.sections[0].zoom_out.toolTip()=='Zoom out'
    assert editor.sections[0].reset_button.text()=='Reset'
    if capture_path:assert editor.capture().save(str(capture_path.with_stem(capture_path.stem+'_en')),'PNG')
    editor.reject();set_language('ja')
    np.testing.assert_array_equal([c.ct_position for c in scene.contacts],original)
    return {'ct_sections_independent_zoom_and_pan':True,'ct_wheel_anchor_fixed':True,
            'ct_pan_does_not_edit_contacts':True,'ct_zoomed_click_and_mri_transform':True,
            'ct_window_reuses_native_plane':True,'ct_reset_and_small_window_layout':True,
            'ct_editor_zoom_two_click_render_ms':durations,'ct_editor_originals_unchanged':True}
