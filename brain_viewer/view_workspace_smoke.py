"""Synthetic end-to-end checks for ROI editing, tract slices and configurable panels."""
import json
import time
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from .workspace_smoke import wait_job
from .imaging import load_scene


def run_view_checks(w,capture_path=None):
    from tests.test_diffusion import synthetic_model
    assert w.scene.kind=='synthetic'
    scene=w.scene;original_data=scene.data;original_labels=scene.label_volume.copy()
    scene,model=synthetic_model(scene)
    w.diffusion.bind_scene();w.workflow_tabs.setCurrentIndex(5)
    d=w.diffusion;views=w.views
    # Anatomy selection uses existing patient labels, copies masks, and binds ROI 1.
    d.add_anatomy_roi(d.roi1,[{'source':'anatomy','label':10},{'source':'anatomy','label':49}])
    wait_job(w)
    anatomy=scene.segmentations[-1]
    np.testing.assert_array_equal(anatomy.mask,np.isin(original_labels,[10,49]))
    assert d.roi1.value()=={'kind':'segment','uid':anatomy.uid}
    d.edit_roi(d.roi1,False)
    assert d.manual_uid==anatomy.uid
    # Draw a polygon through actual mouse events, then undo, cancel and redraw.
    d.roi1.manual.click();manual=scene.segmentations[-1]
    assert not manual.mask.any()
    w.set_cursor(scene.index([0,0,0]))
    canvas=w.slices[2].canvas
    center=w.ijk.copy()
    points=[center+delta for delta in np.array([[-4,-4,0],[4,-4,0],[4,4,0],[-4,4,0],[-4,-4,0]])]
    screen=[canvas.position_for_index(p).toPoint() for p in points]
    def enclose():
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,screen[0])
        for point in screen[1:]:QTest.mouseMove(canvas,point,20)
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,screen[-1])
        QApplication.processEvents()
    enclose()
    assert manual.mask[tuple(center)] and manual.edits[-1]['action']=='contour'
    assert np.ptp(np.argwhere(manual.mask)[:,2])>0
    painted=manual.mask.copy()
    w.segmentation.undo();assert not manual.mask.any()
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,screen[0])
    QTest.mouseMove(canvas,screen[1]);QTest.keyClick(canvas,Qt.Key.Key_Escape)
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,screen[2])
    assert not manual.mask.any()
    enclose();np.testing.assert_array_equal(manual.mask,painted)
    d.draw_mode.setCurrentIndex(0)
    d.max_seeds.setValue(100);d.track_button.click();wait_job(w)
    assert scene.tracts and scene.tracts[-1].count>0
    bundle=scene.tracts[-1]
    # Frame changes really include tract pixels. Deselect 3D independently.
    d.show_2d.setChecked(False);QApplication.processEvents();off=canvas.grab().toImage()
    d.show_2d.setChecked(True);QApplication.processEvents();on=canvas.grab().toImage()
    assert on!=off and canvas._tract_cache
    d.bundles.currentItem().setCheckState(Qt.CheckState.Unchecked)
    assert bundle.visible_2d and not bundle.visible
    d.bundles.currentItem().setCheckState(Qt.CheckState.Checked)
    d.slab.setValue(3.5)
    # Double-click focus round trip preserves image object, cursor and camera.
    before_cursor=w.ijk.copy();camera=w.surface.camera_state()
    QTest.mouseDClick(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,canvas.rect().center())
    QTest.qWait(80);assert views.count==1 and views.hosts[0].isVisible()
    assert not views.hosts[3].isVisible()
    QTest.mouseDClick(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,canvas.rect().center())
    QTest.qWait(80);assert views.count==4 and views.hosts[3].isVisible()
    np.testing.assert_array_equal(w.ijk,before_cursor);assert w.surface.camera_state()==camera
    views.set_count(6);QTest.qWait(100)
    assert all(host.isVisible() for host in views.hosts)
    trajectory=views.hosts[5].content
    assert not trajectory.canvas.image.isNull() and trajectory.frame is not None
    trajectory.slider.setValue(12);QApplication.processEvents()
    np.testing.assert_allclose(scene.world(w.ijk),trajectory.frame[0]+6*trajectory.frame[1],atol=scene.spacing.max())
    assert all(np.array_equal(panel.canvas.ijk,w.ijk) for panel in w.slice_panels())
    # A second 3D shares the same actors/mappers and has an independent camera.
    views.hosts[4].choice.setCurrentIndex(views.hosts[4].choice.findData('surface'))
    QTest.qWait(100);mirror=views.hosts[4].content
    assert mirror.renderer is not w.surface.renderer
    assert mirror.renderer.HasViewProp(w.surface.tract_actors[bundle.uid])
    w.opacity_slider.setValue(35)
    assert w.surface.actors['pial_lh'].GetProperty().GetOpacity()==.35
    mirror.set_camera_preset(2);assert mirror.camera_state()!=w.surface.camera_state()
    views.toggle_focus(4);QTest.qWait(60);assert views.count==1
    views.toggle_focus(4);QTest.qWait(60);assert views.count==6
    trajectory=views.hosts[5].content
    trajectory.plane.setCurrentIndex(1)
    before_layout=w.view_state()['panels']
    w.save_work();wait_job(w)
    saved=load_scene(w.patient_store.scene_path(w.patient_id))
    assert saved.view_state['panels']==before_layout
    assert saved.tracts[-1].visible_2d and saved.tracts[-1].slab_mm==3.5
    np.testing.assert_array_equal(saved.segmentations[-1].mask,manual.mask)
    w.install_scene(saved);QTest.qWait(100)
    assert views.count==6 and views.hosts[4].choice.currentData()=='surface'
    assert views.hosts[5].content.plane.currentData()=='along'
    actual_layout=w.view_state()['panels']
    if actual_layout!=before_layout:
        Path('private_reports/panel_roundtrip_difference.json').write_text(
            json.dumps({'expected':before_layout,'actual':actual_layout},indent=2))
    assert actual_layout==before_layout
    # Save while one panel is enlarged; hidden panels retain their camera/zoom.
    views.toggle_focus(4);focused_state=views.state()
    views.set_count(4);views.restore(focused_state)
    assert views.count==1 and views.state()==focused_state
    views.toggle_focus(4)
    assert views.count==6
    assert views.state()['panels']==before_layout['panels']
    if capture_path:
        w.workflow_tabs.setCurrentIndex(5)
        d.mode.setCurrentIndex(1);d.roi2.source.setCurrentIndex(d.roi2.source.findData(anatomy.uid))
        views.hosts[4].choice.setCurrentIndex(views.hosts[4].choice.findData('axial'))
        QTest.qWait(100);w.capture_widget(w).save(str(capture_path),'PNG')
        views.toggle_focus(0);QTest.qWait(100)
        w.capture_widget(w).save(str(capture_path.with_stem(capture_path.stem+'_single')),'PNG')
        views.toggle_focus(0)
        w.language_tabs.setCurrentIndex(1)
        assert views.tabs.tabText(0)=='4 panels'
        w.capture_widget(w).save(str(capture_path.with_stem(capture_path.stem+'_en')),'PNG')
        w.language_tabs.setCurrentIndex(0)
    np.testing.assert_array_equal(saved.label_volume,original_labels)
    # Empty patient must release references from extra panels too.
    w.patient_id=None;w.clear_scene()
    assert all(p.canvas.scene is None for p in w.slice_panels())
    assert all(p.scene is None for p in views.surface_panels())
    assert all(p.canvas.image.isNull() for p in views.trajectories)
    return {'manual_roi_draw_undo_cancel':True,'freesurfer_roi_union':True,'tract_slice_overlay':True,
            'six_panel_layout_and_double_click':True,'shared_3d_geometry':True,
            'trajectory_scroll':True,'layout_and_roi_roundtrip':True,'extra_panel_patient_isolation':True}
