"""Automated interaction checks. Captures are allowed only for synthetic data."""
from pathlib import Path
import tempfile

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import vtk

from .imaging import load_scene, save_scene


def run_ui_checks(window, capture_path: Path | None = None) -> dict:
    scene = window.scene
    assert scene is not None
    assert window.surface.render_count > 0
    assert window.surface.actors["pial_lh"].GetProperty().GetColor() == window.surface.actors["pial_rh"].GetProperty().GetColor()
    original = window.ijk.copy()
    for axis in (0, 1, 2):
        window.slices[axis].slider.setValue(int(original[axis]) + 3)
        assert window.ijk[axis] == original[axis] + 3
        assert all(np.array_equal(panel.canvas.ijk, window.ijk) for panel in window.slices.values())
    # Click an independently chosen voxel in the axial view.
    target = window.ijk.copy()
    target[0] += 13
    target[1] -= 11
    canvas = window.slices[2].canvas
    point = canvas.position_for_index(target).toPoint()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    assert np.max(np.abs(window.ijk - target)) <= 1
    assert canvas.show_annotation
    assert scene.annotation(window.ijk)[1] in window.anatomy_label.text()
    # Right-button drags change window/level, independently of slice selection.
    before_width, before_level = window.window_spin.value(), window.level_spin.value()
    QTest.mousePress(canvas, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseMove(canvas, point + QPoint(24, 17), 20)
    QTest.mouseRelease(canvas, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, point + QPoint(24, 17))
    assert window.window_spin.value() > before_width
    assert window.level_spin.value() > before_level
    before = window.ijk.copy()
    wheel = QWheelEvent(QPointF(point), QPointF(canvas.mapToGlobal(point)), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(canvas, wheel)
    assert window.ijk[2] == before[2] + 1
    zoom_before = canvas.zoom
    zoom = QWheelEvent(QPointF(point), QPointF(canvas.mapToGlobal(point)), QPoint(), QPoint(0, 120),
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(canvas, zoom)
    assert canvas.zoom > zoom_before
    if scene.ct is not None:
        window.ct_visible.setChecked(False)
        mri_image = canvas._image.copy()
        window.ct_visible.setChecked(True)
        assert canvas._image != mri_image
        window.window_target.setCurrentIndex(1)
        before = window.ct_window.value()
        window.drag_window(20, 10)
        assert window.ct_window.value() > before
        window.window_target.setCurrentIndex(0)
    if scene.contacts:
        window.electrode_group.setCurrentIndex(1)
        group = window.electrode_group.currentData()
        assert all(not actor.GetVisibility() for name, actor in window.surface.contact_actors if name != group)
        window.contact_combo.setCurrentIndex(1)
        window.select_contact(1)
        contact_index = window.contact_combo.currentData()
        assert np.linalg.norm(scene.world(window.ijk)-scene.contacts[contact_index].position) <= np.linalg.norm(scene.spacing)/2+.01
        assert not window.surface.contact_source
        assert all(c.ct_position is not None and c.provenance == 'native_ct_only' for c in scene.contacts)
        window.electrode_group.setCurrentIndex(0)
    window.opacity_slider.setValue(46)
    assert abs(window.surface.actors["pial_lh"].GetProperty().GetOpacity() - .46) < 1e-8
    window.left_check.setChecked(False)
    assert not window.surface.actors["pial_lh"].GetVisibility()
    window.left_check.setChecked(True)
    window.surface_mode.setCurrentIndex(1)
    assert window.surface.actors["white_lh"].GetVisibility()
    assert not window.surface.actors["pial_lh"].GetVisibility()
    window.surface_mode.setCurrentIndex(0)
    window.planes_check.setChecked(True)
    assert all(actor.GetVisibility() for actor, _, _ in window.surface.planes.values())
    low_before = canvas.low
    window.level_spin.setValue(window.level_spin.value() + 23)
    assert canvas.low != low_before
    window.auto_contrast()
    window.opacity_slider.setValue(100)
    window.planes_check.setChecked(False)
    window.reset_views()
    # Verify an actual VTK camera rotation using native mouse events.
    widget = window.surface.widget
    before_camera = np.array(window.surface.camera_state()["position"])
    center = widget.rect().center()
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)
    QTest.mouseMove(widget, center + QPoint(45, 18), 20)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center + QPoint(45, 18))
    assert not np.allclose(before_camera, window.surface.camera_state()["position"])
    before_camera = window.surface.camera_state()
    before_width = window.window_spin.value()
    QTest.mousePress(widget, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, center)
    QTest.mouseMove(widget, center+QPoint(20, 9), 20)
    QTest.mouseRelease(widget, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, center+QPoint(20, 9))
    assert window.window_spin.value() != before_width
    assert window.surface.camera_state() == before_camera
    window.surface.set_camera_preset(1)
    # Find a visible cortex point with the picker, then exercise double-click linkage.
    render_window = widget.GetRenderWindow()
    width, height = render_window.GetSize()
    found = None
    for x in (.4, .6, .3, .7, .5):
        for y in (.5, .4, .6, .3, .7):
            if widget.picker.Pick(width*x, height*y, 0, window.surface.renderer):
                world = np.array(widget.picker.GetPickPosition())
                px = round(x * widget.width())
                py = round(widget.height() - 1 - y * widget.height())
                found = (QPoint(px, py), world)
                break
        if found:
            break
    assert found is not None
    QTest.mouseDClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, found[0])
    assert np.linalg.norm(scene.world(window.ijk) - found[1]) < 4.0
    image = window.surface.capture_image()
    pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
    # A rendered cortex must contribute many pixels above the dark background.
    assert np.count_nonzero(pixels > 110) > 1000
    # Switching languages changes text only, including dynamically built controls.
    state_before_language = window.view_state()
    window.language_tabs.setCurrentIndex(1)
    assert window.import_folder_button.text() == "Open folder"
    assert window.side_tabs.tabText(4) == "Review"
    assert window.view_state() == state_before_language
    window.set_cursor(window.ijk)
    assert window.voxel_label.text().startswith("Native MRI voxel")
    if scene.nuclei is not None:
        assert len(window.surface.nucleus_actors) == len(scene.nuclei_names)
        window.nuclei_check.setChecked(True)
        window.nucleus_combo.setCurrentIndex(1)
        window.select_nucleus()
        selected = window.nucleus_combo.currentData()
        assert window.surface.nucleus_actors[selected].GetVisibility()
        assert sum(bool(actor.GetVisibility()) for actor in window.surface.nucleus_actors.values()) == 1
        assert "Native grid" in window.nucleus_info.text()
        window.nuclei_check.setChecked(False)
    if scene.ct is not None:
        for mode in ("checker", "full", "edges", "bone"):
            window.ct_mode.setCurrentIndex(window.ct_mode.findData(mode))
            assert window.slices[2].canvas.ct_options["mode"] == mode
        window.review.capture_mri()
        assert window.review.ct_button.isEnabled()
        window.review.capture_ct()
        from .alignment_review import landmark_residuals
        assert landmark_residuals(scene.ct_to_mri,scene.ct_quality["landmarks"])[-1] < 1e-7
        window.review.clear()
        window.restore_state(state_before_language)
    layers_before=window.view_state()
    before_tab=window.side_tabs.currentIndex()
    for tab,listing in enumerate((window.image_layers,window.model_layers)):
        window.side_tabs.setCurrentIndex(tab); QApplication.processEvents()
        for row in listing.rows.values():
            assert row.check.geometry().right()<row.slider.geometry().left()
    window.side_tabs.setCurrentIndex(before_tab)
    window.mri_opacity.setValue(42)
    window.ct_opacity.setValue(83)
    window.mri_visible.setChecked(False)
    assert window.ct_options()['mri_visible'] is False
    assert not window.ct_mode.isEnabled()
    if scene.ct is not None:
        pixels=window.slices[2].canvas._image
        assert not pixels.isNull()
        window.ct_visible.setChecked(False)
    window.mri_visible.setChecked(True)
    assert window.mri_opacity.value()==42
    window.brain_visible.setChecked(False)
    assert not any(a.GetVisibility() for a in window.surface.actors.values())
    window.nuclei_check.setChecked(True); window.nuclei_opacity.setValue(27)
    if scene.nuclei is not None:
        assert all(abs(a.GetProperty().GetOpacity()-.27)<1e-6 for a in window.surface.nucleus_actors.values())
        assert any(a.GetVisibility() for a in window.surface.nucleus_actors.values())
    state_layers=window.view_state()
    window.restore_state(layers_before); window.restore_state(state_layers)
    assert window.view_state()==state_layers
    window.restore_state(layers_before)
    if capture_path is not None:
        assert scene.kind=='synthetic'
        window.side_tabs.setCurrentIndex(1)
        window.nuclei_check.setChecked(True); window.nuclei_opacity.setValue(45)
        QApplication.processEvents()
        assert window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+'_models_en')),'PNG')
        window.restore_state(layers_before)
    if capture_path is not None:
        assert scene.kind == "synthetic"
        window.side_tabs.setCurrentIndex(4)
        QApplication.processEvents()
        # Labels and sliders must fit in their actual 2D panels after translation.
        for panel in window.slices.values():
            assert panel.counter.geometry().right() < panel.width()
            assert panel.location.geometry().right() < panel.width()
        assert window.centralWidget().geometry().right() <= window.width()
        assert window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+"_review_en")),"PNG")
    window.language_tabs.setCurrentIndex(0)
    assert window.import_folder_button.text() == "フォルダを開く"
    window.side_tabs.setCurrentIndex(0)
    window.restore_state(state_before_language)
    if scene.contacts and scene.raw_ct is not None:
        from .contact_editor import ContactEditor
        from .electrode_editing import export_records
        from .i18n import set_language
        from copy import deepcopy
        from dataclasses import replace
        previous=deepcopy(scene.contacts)
        previous_quality=deepcopy(scene.electrode_quality)
        camera=window.surface.camera_state()
        editor=ContactEditor(scene,window)
        editor.show(); QApplication.processEvents()
        assert editor.overview.initialized
        assert editor.overview.surface.render_count>0
        assert editor.rect().contains(editor.overview.mapTo(editor,QPoint(0,0)))
        assert editor.rect().contains(editor.table.mapTo(editor,editor.table.rect().bottomRight()))
        initial=editor.current(); uid=initial[0].uid
        raw=initial[0].ct_position.copy()
        n=len(initial)
        editor.reverse(); assert editor.current()[-1].uid==uid
        assert editor.overview.contacts[-1].uid==uid
        editor.undo_edit(); assert editor.current()[0].uid==uid
        before_positions=np.array([c.ct_position for c in editor.current()])
        before_expected=editor.expected.value()
        editor.model_combo.setCurrentIndex(1)
        assert np.array_equal(before_positions,np.array([c.ct_position for c in editor.current()]))
        assert editor.expected.value()==before_expected
        assert editor.scene.electrode_quality['groups'][editor.groups.currentText()]['reference_spec']['contact_count']==4
        editor.undo_edit(); assert editor.model_combo.currentIndex()==0
        editor.overview.pick(editor.current()[-1].position)
        assert editor.index==n-1
        for axis in (2,1,0):
            editor.overview.selector.setCurrentIndex(editor.overview.selector.findData(axis))
            assert abs(editor.overview.head_ct.center[axis]-editor.current()[editor.index].ct_position[axis])<1e-8
            assert not editor.overview.head_ct.image.isNull()
            assert all(editor.overview.head_ct.image_rect().contains(editor.overview.head_ct.project(c.ct_position)) for c in editor.current())
        if capture_path is not None:
            assert scene.kind=='synthetic'
            assert editor.capture().save(str(capture_path.with_stem(capture_path.stem+'_context_ct')),'PNG')
        editor.overview.selector.setCurrentIndex(0)
        editor.select(0)
        section=editor.sections[0]
        p=section.project(editor.current()[1].ct_position).toPoint()
        QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,p)
        assert editor.index==1
        editor.select(0)
        editor.move_check.setChecked(True)
        p=section.project(raw).toPoint()+QPoint(2,0)
        QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,p)
        assert np.linalg.norm(editor.current()[0].ct_position-raw)>.01
        assert np.array_equal(scene.contacts[0].ct_position,raw)  # Working copy isolation.
        editor.undo_edit(); editor.move_check.setChecked(False)
        editor.delete_contact(); assert len(editor.current())==n-1
        editor.undo_edit(); assert len(editor.current())==n
        editor.extend(False); assert len(editor.current())==n+1
        editor.undo_edit()
        editor.expected.setValue(n); editor.confirm_group()
        assert all(c.status=='reviewed' for c in editor.current())
        editor.undo_edit(); assert not all(c.status=='reviewed' for c in editor.current())
        width=editor.width_spin.value()
        QTest.mousePress(section,Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier,p)
        QTest.mouseMove(section,p+QPoint(15,8),20)
        QTest.mouseRelease(section,Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier,p+QPoint(15,8))
        assert editor.width_spin.value()>width
        editor.group_name.setText('Review-Test'); editor.rename_group()
        assert editor.current()[0].group=='Review-Test'
        editor.undo_edit()
        if capture_path is not None:
            assert scene.kind=='synthetic'
            import json
            widgets={'editor':editor,'context':editor.overview.parentWidget(),'overview':editor.overview,
                     'selector':editor.overview.selector,'stack':editor.overview.stack,'surface':editor.overview.surface,
                     'vtk':editor.overview.surface.widget,'table':editor.table}
            report={key:{'rect':widget.geometry().getRect(),'visible':widget.isVisible(),
                         'hidden':widget.isHidden(),'in_editor':widget.mapTo(editor,QPoint()).toTuple()} for key,widget in widgets.items()}
            report['screen']=QApplication.primaryScreen().availableGeometry().getRect()
            (capture_path.parent/'review_layout.json').write_text(json.dumps(report,indent=2))
            assert editor.capture().save(str(capture_path.with_stem(capture_path.stem+'_contacts_ja')),'PNG')
        editor.reject()
        assert [c.uid for c in scene.contacts]==[c.uid for c in previous]
        set_language('en')
        editor=ContactEditor(scene,window); editor.show(); QApplication.processEvents()
        assert editor.windowTitle()=='Review and edit CT contacts'
        if capture_path is not None:
            assert scene.kind=='synthetic'
            assert editor.capture().save(str(capture_path.with_stem(capture_path.stem+'_contacts_en')),'PNG')
        editor.move(editor.current()[0].ct_position+[.1,0,0]); editor.accept()
        changed=editor.scene.contacts[0].ct_position.copy()
        window.replace_electrodes(editor.scene)
        assert np.array_equal(window.scene.contacts[0].ct_position,changed)
        assert window.surface.camera_state()==camera
        assert export_records(window.scene)['contacts'][0]['ct_ras_mm']==changed.tolist()
        # Restore the actual case after reversible test edits.
        restored=replace(scene,contacts=previous,electrode_quality=previous_quality)
        window.replace_electrodes(restored)
        set_language('ja')
        window.change_language(0)
    state = window.view_state()
    # Saving/reopening must retain geometry and display position without source folders.
    parent = Path(__file__).resolve().parents[1] / "private_reports"
    with tempfile.TemporaryDirectory(prefix="viewer_check_", dir=parent) as directory:
        assert Path(directory).resolve().is_relative_to(parent.resolve())
        folder = save_scene(scene, Path(directory), state)
        restored = load_scene(folder / "scene.json")
        assert np.array_equal(restored.data, scene.data)
        assert np.allclose(restored.affine, scene.affine)
        assert np.allclose(restored.source_affine, scene.source_affine)
        assert restored.view_state == state
        if scene.ct is not None:
            assert np.array_equal(restored.ct, scene.ct)
            assert np.array_equal(restored.ct_valid, scene.ct_valid)
            assert np.array_equal(restored.raw_ct, scene.raw_ct)
            assert np.allclose(restored.raw_ct_affine,scene.raw_ct_affine)
        if scene.nuclei is not None:
            assert np.array_equal(restored.nuclei,scene.nuclei)
            assert np.allclose(restored.nuclei_affine,scene.nuclei_affine)
        if scene.label_volume is not None:
            assert np.array_equal(restored.label_volume, scene.label_volume)
            assert restored.label_names == scene.label_names
        assert len(restored.contacts) == len(scene.contacts)
        for original_contact, restored_contact in zip(scene.contacts, restored.contacts):
            assert np.array_equal(original_contact.position, restored_contact.position)
            assert np.array_equal(original_contact.source_position, restored_contact.source_position)
            assert np.array_equal(original_contact.ct_position, restored_contact.ct_position)
            assert original_contact.status == restored_contact.status
            assert original_contact.uid == restored_contact.uid
        for key in scene.surfaces:
            assert np.array_equal(scene.surfaces[key].vertices, restored.surfaces[key].vertices)
        window.surface.export_surface(Path(directory) / "cortex.vtp")
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(Path(directory) / "cortex.vtp"))
        reader.Update()
        expected = sum(len(m.faces) for key, m in scene.surfaces.items() if key.startswith("pial"))
        assert reader.GetOutput().GetNumberOfPolys() == expected
        assert reader.GetOutput().GetFieldData().GetAbstractArray("CoordinateSystem").GetValue(0) == "scanner_RAS_mm"
        # The production screenshot path composes the native 3D framebuffer.
        assert window.capture_widget(window.views).save(str(Path(directory) / "views.png"), "PNG")
    window.reset_views()
    window.auto_contrast()
    window.opacity_slider.setValue(32 if scene.contacts else 100)
    window.ct_window.setValue(2000)
    window.ct_level.setValue(1000)
    if capture_path is not None:
        assert scene.kind == "synthetic"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        geometry={key:{'text':row.check.text(),'check':row.check.geometry().getRect(),'slider':row.slider.geometry().getRect(),
                       'minimum':row.check.minimumSize().toTuple(),'check_parent':row.check.parentWidget().objectName(),
                       'visible':row.check.isVisible(),'row':row.geometry().getRect()} for key,row in window.image_layers.rows.items()}
        (capture_path.parent/'layers_layout.json').write_text(json.dumps(geometry,indent=2))
        assert window.capture_widget(window).save(str(capture_path), "PNG")
    from .segmentation_smoke import run_segmentation_checks
    segmentation_checks = run_segmentation_checks(window, capture_path)
    from .segmentation_sources_smoke import run_source_checks
    source_checks = run_source_checks(window,capture_path)
    workspace_checks = {}
    if scene.kind == 'synthetic':
        from .workspace_smoke import run_workspace_checks
        workspace_checks = run_workspace_checks(window, capture_path)
    from .results_smoke import run_result_checks
    from .import_smoke import run_import_checks
    import_checks = run_import_checks(window, capture_path)
    result_checks=run_result_checks(window,capture_path)
    from .patient_delete_smoke import run_patient_delete_checks
    delete_checks = run_patient_delete_checks(window, capture_path)
    return {**delete_checks, **import_checks, **result_checks, **workspace_checks, **segmentation_checks, **source_checks, "linked_slices": True, "wheel_and_zoom": True, "surface_rotation": True,
            "surface_picking": True, "scene_roundtrip": True, "png_export": True,
            "mesh_export": True, "nonempty_3d_render": True, "matching_hemisphere_colors": True,
            "right_drag_window": True, "anatomy_callout": True, "ct_overlay": scene.ct is not None,
            "electrode_selection": bool(scene.contacts), "native_ct_contact_editor":bool(scene.contacts), "language_switch_preserves_state":True,
            "independent_image_and_model_layers":True,"contact_whole_head_context":bool(scene.contacts),"reference_presets_preserve_detection":bool(scene.contacts),
            "nuclei_3d_and_native_roundtrip":scene.nuclei is not None, "registration_comparison_modes":scene.ct is not None}
