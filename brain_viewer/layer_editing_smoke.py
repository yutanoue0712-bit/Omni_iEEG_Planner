"""Synthetic integration checks for image removal and selected postoperative masks."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import SimpleITK as sitk
from PySide6.QtCore import Qt,QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QMessageBox,QScrollArea
from .imaging import MRILayer,load_scene
from .postop_registration import deformation_record
from .registration import to_sitk
from .segmentation import create_segment
from .segmentation_sources import display_segments
from .workspace_smoke import wait_job


def run_layer_editing_checks(w,capture_path=None):
    assert w.scene.kind=='synthetic'
    scene=w.scene; contacts=scene.contact_positions().copy()
    def new_layer(uid,name,sequence):
        return MRILayer(uid,name,sequence,scene.data.copy(),np.ones(scene.data.shape,bool),
                        scene.data.copy(),scene.affine.copy(),np.eye(4),{'review_status':'visually_reviewed'})
    base=new_layer('mr_base','Postoperative T1','postop-T1')
    pet=new_layer('mr_pet','PET demo','PET')
    warp=deformation_record(sitk.BSplineTransformInitializer(to_sitk(scene.data,scene.affine),[1,1,1]))
    candidate=replace(base,uid='mr_candidate',name='Correction candidate',deformation=warp,
                      quality={'rigid_source_uid':base.uid,'engine':'bspline'})
    other=replace(candidate,uid='mr_other',name='Second correction candidate')
    w.install_scene(replace(scene,extra_mris=[pet,base,candidate,other],view_state={}))
    p=w.postop; s=w.segmentation
    w.workflow_tabs.setCurrentIndex(3);p.refresh(base.uid);p.apply_view()
    p.create_mask.click();first=w.scene.segmentations[-1]
    assert p.manual_uid==first.uid and s.active_mode()=='contour'
    center=np.array(w.scene.data.shape)//2
    def polygon(dx=0,size=5):
        return np.array([center+[x+dx,y,0] for x,y in [(-size,-size),(size,-size),(size,size),(-size,size),(-size,-size)]])
    w.set_cursor(center)
    s.enclose(2,polygon());count=int(first.mask.sum())
    p.add_mask.click();s.enclose(2,polygon(dx=8))
    assert first.mask.sum()>count and len(w.scene.segmentations)==1
    expanded=first.mask.copy()
    p.subtract_mask.click()
    canvas=w.slices[2].canvas
    points=[canvas.position_for_index(point).toPoint() for point in polygon(size=2)]
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,points[0])
    for point in points[1:]:QTest.mouseMove(canvas,point,20)
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,points[-1])
    QApplication.processEvents()
    assert 0<first.mask.sum()<expanded.sum() and not first.mask[tuple(center)]
    # Unrelated segmentation changes must not be undone by the postoperative button.
    general=create_segment(w.scene,'manual','Anatomical region');general.mask[5:8,5:8,5:8]=True
    s.add_segments([general]);p.undo_mask.click()
    assert general in w.scene.segmentations
    np.testing.assert_array_equal(first.mask,expanded)
    s.refresh_list(general.uid);p.delete_mask.click()
    assert first not in w.scene.segmentations and general in w.scene.segmentations
    assert p.manual_uid is None and not p.draw_mode.isEnabled()
    p.undo_mask.click()
    assert p.manual_uid==first.uid and first in w.scene.segmentations
    np.testing.assert_array_equal(first.mask,expanded)
    p.create_mask.click();second=w.scene.segmentations[-1];s.enclose(2,polygon(dx=-14))
    for mask in (first,second):mask.visible_3d=True
    w.set_cursor(center);w.update_layers()
    assert first.uid not in w.surface.segment_actors and second.uid in w.surface.segment_actors
    assert [r.uid for r in display_segments(w.scene,w.ct_options()['segmentation_context'])]==[general.uid,second.uid]
    w.workflow_tabs.setCurrentIndex(0)
    assert s.active_mode()=='navigate' and w.scene.segmentation_preview is None
    assert first.uid not in w.surface.segment_actors and second.uid not in w.surface.segment_actors
    assert general.uid in w.surface.segment_actors
    for panel in w.slice_panels():
        assert [r.uid for r in display_segments(w.scene,panel.canvas.ct_options['segmentation_context'])]==[general.uid]
    assert first.visible_2d and second.visible_2d
    w.workflow_tabs.setCurrentIndex(3);p.exclusion.setCurrentIndex(p.exclusion.findData(first.uid))
    assert first.uid in w.surface.segment_actors and second.uid not in w.surface.segment_actors
    p.no_cavity.setChecked(True);assert first.uid not in w.surface.segment_actors
    p.no_cavity.setChecked(False);assert first.uid in w.surface.segment_actors
    # Selection-scoped undo/delete and mask visibility persist through a patient save.
    record=w.patient_store.create('Synthetic layer editing');w.patient_id=record['id'];w.refresh_patients()
    w.save_work();wait_job(w)
    loaded=w.patient_store.load(record['id']);w.install_scene(loaded)
    assert p.manual_uid==first.uid
    w.workflow_tabs.setCurrentIndex(0)
    assert first.uid not in w.surface.segment_actors
    w.workflow_tabs.setCurrentIndex(3)
    for language in (0,1):
        w.language_tabs.setCurrentIndex(language);QApplication.processEvents()
        scroll=p.parentWidget()
        while not isinstance(scroll,QScrollArea):scroll=scroll.parentWidget()
        scroll.ensureWidgetVisible(p.draw_mode);QApplication.processEvents()
        for button in (p.add_mask,p.subtract_mask,p.delete_mask):
            assert button.mapTo(p,QPoint()).x()+button.width()<=p.width(),(button.text(),button.geometry(),p.width())
            assert button.fontMetrics().horizontalAdvance(button.text())+20<=button.width(),button.text()
        if capture_path:
            target=capture_path if language==0 else capture_path.with_stem(capture_path.stem+'_en')
            assert w.capture_widget(w).save(str(target),'PNG')
    w.language_tabs.setCurrentIndex(0)
    # Cancel removal first, then remove a candidate and its parent family.
    w.workflow_tabs.setCurrentIndex(0);w.image_layers.rows[pet.uid].slider.setValue(43)
    w.extra_combo.setCurrentIndex(w.extra_combo.findData(candidate.uid))
    with patch.object(QMessageBox,'exec',return_value=QMessageBox.StandardButton.Cancel):
        w.image_layers.rows[candidate.uid].remove_button.click()
    assert w.extra_layer(candidate.uid) is not None
    with patch.object(QMessageBox,'exec',return_value=QMessageBox.StandardButton.Yes):
        w.image_layers.rows[candidate.uid].remove_button.click()
    assert w.extra_layer(candidate.uid) is None and w.extra_layer(other.uid) is not None
    assert w.image_layers.rows[pet.uid].slider.value()==43
    assert p.stage.findData(candidate.uid)<0 and w.extra_combo.findData(candidate.uid)<0
    assert w.segmentation.extraction.source.findData(candidate.uid)<0
    with patch.object(QMessageBox,'exec',return_value=QMessageBox.StandardButton.Yes):
        w.image_layers.rows[base.uid].remove_button.click()
    assert [l.uid for l in w.scene.extra_mris]==[pet.uid]
    assert p.base() is None and p.layer() is None and p.stage.count()==0
    assert w.window_target.findData(base.uid)<0 and w.window_target.findData(other.uid)<0
    with patch.object(QMessageBox,'exec',return_value=QMessageBox.StandardButton.Yes):
        w.image_layers.rows['ct'].remove_button.click()
    assert w.scene.ct is None and w.scene.raw_ct is not None and not w.image_layers.rows['ct'].isVisible()
    assert w.edit_contacts_button.isEnabled()
    np.testing.assert_array_equal(w.scene.contact_positions(),contacts)
    w.save_work();wait_job(w);loaded=load_scene(w.patient_store.scene_path(w.patient_id));w.install_scene(loaded)
    assert [l.uid for l in w.scene.extra_mris]==[pet.uid] and w.scene.ct is None
    assert w.image_layers.rows[pet.uid].slider.value()==43
    assert {r.uid for r in w.scene.segmentations}=={first.uid,second.uid,general.uid}
    np.testing.assert_array_equal(w.scene.contact_positions(),contacts)
    if capture_path:
        w.side_tabs.setCurrentIndex(0);QApplication.processEvents()
        assert w.capture_widget(w).save(str(capture_path.with_stem(capture_path.stem+'_layers')),'PNG')
    # Removing one DTI display map keeps the tensor model and other maps usable.
    from .diffusion import DiffusionModel
    b0=new_layer('mr_b0','DTI b0','DTI-b0');fa=new_layer('mr_fa','DTI FA','DTI-FA')
    model=DiffusionModel('dti_test','DTI demo',np.eye(4),np.eye(4),np.full((4,4,4),.6,np.float32),
        np.zeros((4,4,4,3),np.float32),np.ones((4,4,4),bool),np.array([0.,1000.]),np.zeros((2,3)),
        {'b0':b0.uid,'fa':fa.uid})
    state=w.view_state()
    w.scene.extra_mris.extend([b0,fa]);w.scene.diffusions.append(model)
    w.scene.view_state=state;w.install_scene(w.scene)
    d=w.diffusion
    assert d.overlay_button.isEnabled() and d.compare_button.isEnabled()
    with patch.object(QMessageBox,'exec',return_value=QMessageBox.StandardButton.Yes):
        w.image_layers.rows[fa.uid].remove_button.click()
    assert not d.overlay_button.isEnabled() and d.track_button.isEnabled()
    d.export_map('fa');assert '削除' in w.statusBar().currentMessage()
    d.map.setCurrentIndex(d.map.findData('b0'));assert d.overlay_button.isEnabled()
    d.show_map();assert w.extra_layer().uid==b0.uid
    with patch.object(QMessageBox,'exec',return_value=QMessageBox.StandardButton.Yes):
        w.image_layers.rows[b0.uid].remove_button.click()
    assert not d.compare_button.isEnabled() and not d.overlay_button.isEnabled()
    w.save_work();wait_job(w);w.install_scene(w.patient_store.load(record['id']))
    assert w.scene.diffusions[0].layer_ids=={} and w.diffusion.track_button.isEnabled()
    w.clear_scene()
    return {'selected_mask_add_erase_delete_undo':True,'unrelated_segmentation_preserved':True,
            'workflow_scoped_2d_3d_exclusion_visibility':True,'image_removal_cancel_candidate_family':True,
            'remaining_display_settings_and_contacts_preserved':True,'save_reload_after_deletion':True,
            'bilingual_controls_fit':True,'removed_dti_map_controls_and_model_preserved':True}
