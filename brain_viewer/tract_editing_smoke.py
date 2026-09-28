"""Synthetic GUI checks for reversible 2D/3D tract exclusions and region deletion."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QScrollArea
from .diffusion import TractBundle
from .imaging import load_scene
from .segmentation import create_segment
from .workspace_smoke import wait_job


def run_editing_checks(w,capture_path=None):
    from tests.test_diffusion import synthetic_model
    assert w.scene.kind=='synthetic'
    scene,model=synthetic_model(w.scene)
    points=np.array([[-25,-10,0],[25,-10,0],[-25,10,0],[25,10,0],
                     [-25,0,10],[25,0,10],[0,0,-20],[0,0,20]],np.float32)
    bundle=TractBundle('tract_manualtest','Synthetic fibers','dti_synthetic',points,
                       np.arange(0,10,2,dtype=np.int64),{'fa_threshold':.2,'max_angle':35})
    scene.tracts=[bundle];w.diffusion.bind_scene();w.workflow_tabs.setCurrentIndex(5)
    d=w.diffusion;editor=d.exclusion;w.set_cursor(scene.index([0,0,0]));d.refresh_tract_views()
    canvas=w.slices[2].canvas
    d.show_2d.setChecked(False);QApplication.processEvents();baseline=canvas.grab().toImage()
    d.show_2d.setChecked(True);QApplication.processEvents();assert canvas.grab().toImage()!=baseline
    d.bundles.currentItem().setCheckState(Qt.CheckState.Unchecked);QApplication.processEvents()
    assert canvas.grab().toImage()==baseline
    assert not w.surface.tract_actors[bundle.uid].GetVisibility()
    assert not editor.mode.isEnabled()
    d.bundles.currentItem().setCheckState(Qt.CheckState.Checked)
    editor.mode.setCurrentIndex(editor.mode.findData('2d'))
    assert canvas.brush_mode=='tract_exclude'
    d.show_2d.setChecked(False)
    assert editor.mode.currentData()=='navigate' and canvas.brush_mode=='navigate'
    assert not editor.mode.model().item(editor.mode.findData('2d')).isEnabled()
    d.show_2d.setChecked(True)
    editor.mode.setCurrentIndex(editor.mode.findData('2d'))
    previous_opacity=d.opacity.value();d.opacity.setValue(0)
    assert not editor.mode.isEnabled() and canvas.brush_mode=='navigate'
    d.opacity.setValue(previous_opacity);editor.mode.setCurrentIndex(editor.mode.findData('2d'))
    polygon=scene.index(np.array([[-3,-12,0],[3,-12,0],[3,-8,0],[-3,-8,0],[-3,-12,0]]))
    def stroke(widget,positions):
        QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,positions[0])
        for point in positions[1:]:QTest.mouseMove(widget,point,20)
        QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,positions[-1])
        wait_job(w)
    stroke(canvas,[canvas.position_for_index(p).toPoint() for p in polygon])
    np.testing.assert_array_equal(editor.pending_ids,[0])
    assert bundle.count==4 and editor.apply_button.isEnabled()
    editor.cancel_button.click();assert bundle.count==4 and not len(editor.pending_ids)
    stroke(canvas,[canvas.position_for_index(p).toPoint() for p in polygon])
    if capture_path:
        for scroll in w.findChildren(QScrollArea):
            if scroll.isAncestorOf(editor):scroll.ensureWidgetVisible(editor,0,5)
        QApplication.processEvents();w.capture_widget(w).save(str(capture_path),'PNG')
    editor.apply_button.click();assert bundle.count==3
    np.testing.assert_array_equal(bundle.points,points)
    assert w.surface.tract_actors[bundle.uid].GetMapper().GetInput().GetNumberOfLines()==3
    editor.undo_button.click();assert bundle.count==4
    # Real mouse lasso in a second 3D pane uses that pane's camera and DPI conversion.
    w.views.set_count(6)
    w.views.hosts[4].choice.setCurrentIndex(w.views.hosts[4].choice.findData('surface'))
    surface=w.views.hosts[4].content;surface.set_camera_preset(4)
    editor.mode.setCurrentIndex(editor.mode.findData('3d'))
    QTest.qWait(100)
    widget=surface.widget;assert widget.tract_lasso.enabled
    def project(world):
        surface.renderer.SetWorldPoint(*world,1);surface.renderer.WorldToDisplay()
        x,y,_=surface.renderer.GetDisplayPoint();width,height=widget.GetRenderWindow().GetSize()
        return QPoint(round(x/width*widget.width()),round(widget.height()-y/height*widget.height()))
    vertices=[project(p) for p in [(-3,8,0),(3,8,0),(3,12,0),(-3,12,0),(-3,8,0)]]
    before_camera=surface.camera_state()
    stroke(widget,vertices)
    np.testing.assert_array_equal(editor.pending_ids,[1])
    assert surface.camera_state()==before_camera
    assert widget.tract_lasso.actor is None
    if capture_path:
        w.views.toggle_focus(4);QApplication.processEvents()
        w.capture_widget(w).save(str(capture_path.with_stem(capture_path.stem+'_3d')),'PNG')
        w.views.toggle_focus(4)
    editor.apply_button.click();assert bundle.count==3
    # A partial lasso cancelled with Esc never changes the tract.
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,vertices[0])
    QTest.mouseMove(widget,vertices[1]);QTest.keyClick(widget,Qt.Key.Key_Escape)
    QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,vertices[-1])
    assert bundle.count==3 and not len(editor.pending_ids)
    w.save_work();wait_job(w)
    saved=load_scene(w.patient_store.scene_path(w.patient_id))
    assert saved.tracts[0].count==3 and saved.tracts[0].excluded[1]
    w.install_scene(saved);w.workflow_tabs.setCurrentIndex(5);QApplication.processEvents()
    d=w.diffusion;editor=d.exclusion;bundle=w.scene.tracts[0]
    assert editor.undo_button.isEnabled()
    editor.undo_button.click();assert bundle.count==4
    editor.restore_button.click();assert bundle.count==4
    # Compact preset and expanded extraction form can be switched repeatedly.
    w.views.set_count(4);w.workflow_tabs.setCurrentIndex(1);seg=w.segmentation
    for size in ((1460,960),(1100,740),(1460,960)):
        w.resize(*size);QApplication.processEvents()
        seg.creation_method.setCurrentIndex(0);QApplication.processEvents()
        assert seg.preset.height()<=seg.creation_method.height()+5
        assert seg.creation_stack.height()<=seg.creation_method.height()+5
        seg.creation_method.setCurrentIndex(2);QApplication.processEvents()
        assert seg.extraction.height()>100 and seg.extraction.preview_button.isVisible()
    seg.creation_method.setCurrentIndex(0)
    segment=create_segment(w.scene,'manual','ROI for deletion');segment.mask[45:50,50:55,50]=True
    labels=w.scene.label_volume.copy();seg.add_segments([segment]);d.refresh_regions()
    d.roi1.center=[10.,10.,10.];d.roi1.source.setCurrentIndex(d.roi1.source.findData(segment.uid))
    assert d.roi1.value()['uid']==segment.uid
    seg.delete_button.click()
    assert not w.scene.segmentations and d.roi1.value() is None
    assert segment.uid not in w.surface.segment_actors
    assert seg.undo_button.isEnabled()
    seg.undo_button.click()
    assert w.scene.segmentations[-1].uid==segment.uid and d.roi1.value()['uid']==segment.uid
    np.testing.assert_array_equal(w.scene.label_volume,labels)
    if capture_path:
        for scroll in w.findChildren(QScrollArea):
            if scroll.isAncestorOf(seg):scroll.verticalScrollBar().setValue(0)
        QApplication.processEvents();w.capture_widget(w).save(str(capture_path.with_stem(capture_path.stem+'_segmentation')),'PNG')
    seg.delete_button.click();w.save_work();wait_job(w)
    saved=load_scene(w.patient_store.scene_path(w.patient_id))
    assert not saved.segmentations and saved.tracts[0].count==4
    w.language_tabs.setCurrentIndex(1)
    assert seg.delete_button.text()=='Delete selected region'
    w.language_tabs.setCurrentIndex(0)
    return {'tract_checkbox_hides_2d_and_3d':True,'slice_lasso_preview_apply_undo':True,
            'second_3d_lasso_camera_and_escape':True,'exclusion_save_reload_undo':True,
            'compact_segmentation_preset':True,'region_delete_undo_and_roi_cleanup':True}
