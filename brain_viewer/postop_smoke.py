"""Synthetic UI integration for import, manual exclusion, review, save and export."""
import json
from unittest.mock import patch
import nibabel as nib
import numpy as np
from PySide6.QtCore import Qt,QPoint,QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea, QPushButton
from .workspace_smoke import wait_job
from .postop_registration import correct_postop, export_postop
from .postop_ants import correct_postop_ants


def run_postop_checks(w,capture_path=None):
    assert w.scene.kind=='synthetic'
    original=w.scene; contacts=original.contact_positions().copy(); labels=original.label_volume.copy()
    record=w.patient_store.create('Synthetic postoperative MRI'); w.patient_id=record['id']; w.refresh_patients()
    image=nib.Nifti1Image(original.data.copy(),original.affine);image.header.set_xyzt_units('mm')
    with patch('brain_viewer.sequences.register_mri',return_value=(np.eye(4),{'review_status':'synthetic'})):
        w.import_image(image,'postop-T1',name='Postoperative T1 demo'); wait_job(w)
    p=w.postop; base=w.scene.extra_mris[-1]
    assert w.workflow_tabs.currentIndex()==3 and p.isVisible()
    assert p.base() is base and p.layer() is base and p.display_mode.currentData()=='checker'
    assert not p.compute.isEnabled()
    for key in ('preop','postop','overlay','checker'):
        p.display_mode.setCurrentIndex(p.display_mode.findData(key)); QApplication.processEvents()
        assert w.mri_visible.isChecked()==(key!='postop')
        assert w.image_layers.rows[base.uid].check.isChecked()==(key!='preop')
        assert not w.ct_visible.isChecked()
    p.mark_reviewed(); assert base.quality['review_status']=='visually_reviewed' and not p.compute.isEnabled()
    p.create_exclusion(); region=w.scene.segmentations[-1]
    assert p.manual_uid==region.uid and p.layer() is base and p.draw_mode.currentData()=='contour'
    assert w.segmentation.active_mode()=='contour'
    center=np.array(w.scene.data.shape)//2+np.array([-20,-15,0]); w.set_cursor(center)
    canvas=w.slices[2].canvas
    polygon=[center+offset for offset in np.array([[-5,-5,0],[5,-5,0],[5,5,0],[-5,5,0],[-5,-5,0]])]
    points=[canvas.position_for_index(point).toPoint() for point in polygon]
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,points[0])
    for point in points[1:]:QTest.mouseMove(canvas,point,20)
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,points[-1])
    QApplication.processEvents()
    assert region.mask[tuple(center)] and np.ptp(np.argwhere(region.mask)[:,2])>0
    mask=region.mask.copy(); p.undo(); assert not region.mask.any()
    w.segmentation.enclose(2,np.asarray(polygon)); assert region.mask.any()
    w.workflow_tabs.setCurrentIndex(0); assert w.segmentation.active_mode()=='navigate'
    w.workflow_tabs.setCurrentIndex(3); assert p.draw_mode.currentData()=='navigate'
    assert p.compute.isEnabled()
    current=p.state()
    legacy={key:value for key,value in current.items() if key not in ('engine','limits')}
    legacy['maximum']=1.
    p.restore(legacy)
    assert p.engine.currentData()=='ants' and p.max_shift.value()==12.
    p.engine.setCurrentIndex(p.engine.findData('bspline')); assert p.max_shift.value()==1.
    p.engine.setCurrentIndex(p.engine.findData('ants')); assert p.max_shift.value()==12.
    p.restore(current)
    p.preview_button.click(); wait_job(w)
    assert w.scene.segmentation_preview is not None and p.preview_button.isChecked()
    assert w.scene.segmentation_preview.mask.sum()>=region.mask.sum()
    p.margin.setValue(3.5)
    assert w.scene.segmentation_preview is None and not p.preview_button.isChecked()
    p.margin.setValue(3.)
    p.engine.setCurrentIndex(p.engine.findData('bspline'))
    assert not p.ants_options.isVisible() and p.grid.isVisible()
    # Exercise the complete asynchronous correction/save path; the optimizer's
    # nonzero-warp accuracy is tested independently against known landmarks.
    with patch('brain_viewer.postop_panel.correct_postop',side_effect=lambda *a,**kw:correct_postop(*a,iterations=6,**kw)):
        p.compute.click(); wait_job(w,120)
    assert len(w.scene.extra_mris)==2
    candidate=w.scene.extra_mris[-1]
    assert p.layer() is candidate and candidate.deformation and w.workflow_tabs.currentIndex()==3
    assert candidate.quality['review_status']=='candidate_requires_visual_review'
    autosaved=w.patient_store.load(record['id'])
    assert autosaved.view_state['image_layers'][candidate.uid]['visible']
    assert not autosaved.view_state['image_layers'][base.uid]['visible']
    del autosaved
    assert p.display_mode.currentData()=='checker'
    np.testing.assert_array_equal(w.scene.contact_positions(),contacts)
    np.testing.assert_array_equal(w.scene.label_volume,labels)
    np.testing.assert_array_equal(w.scene.data,original.data)
    p.mark_reviewed(); assert candidate.quality['review_status']=='visually_reviewed'
    p.show_rigid(); assert p.layer().uid==base.uid
    p.stage.setCurrentIndex(p.stage.findData(candidate.uid))
    p.draw_mode.setCurrentIndex(p.draw_mode.findData('paint'))
    assert p.layer().uid==base.uid, 'Manual exclusion must use rigid image'
    p.draw_mode.setCurrentIndex(0); p.stage.setCurrentIndex(p.stage.findData(candidate.uid))
    p.display_mode.setCurrentIndex(p.display_mode.findData('overlay')); p.opacity.setValue(47)
    old_width=p.window_spin.value(); w.drag_window(12,1)
    assert p.window_spin.value()>old_width and abs(p.window_spin.value()-p.layer().window)<.02
    w.save_work(); wait_job(w)
    loaded=w.patient_store.load(record['id']); w.install_scene(loaded); w.workflow_tabs.setCurrentIndex(3)
    assert p.layer().uid==candidate.uid and p.opacity.value()==47
    assert p.exclusion.currentData()==region.uid and p.layer().deformation==candidate.deformation
    exported=export_postop(w.scene,p.layer(),w.export_directory())
    assert (exported/'postop_in_preop.nii.gz').is_file()
    assert json.loads((exported/'registration.json').read_text(encoding='utf-8'))['deformation']
    # Native ANTs worker through the real asynchronous GUI/save path, including N4.
    p.engine.setCurrentIndex(p.engine.findData('ants'))
    assert p.ants_options.isVisible() and not p.grid.isVisible()
    p.n4.setChecked(True)
    with patch('brain_viewer.postop_panel.correct_postop_ants',side_effect=lambda *a,**kw:correct_postop_ants(*a,iterations=(6,4,2),**kw)):
        p.compute.click(); wait_job(w,120)
    assert len(w.scene.extra_mris)==3
    candidate=w.scene.extra_mris[-1]
    assert candidate.quality['engine']=='ants' and candidate.quality['ants']['n4']
    assert candidate.deformation['type']=='DisplacementField' and p.layer().uid==candidate.uid
    assert p.stage.itemText(p.stage.currentIndex()).startswith('ANTsPy')
    np.testing.assert_array_equal(w.scene.contact_positions(),contacts)
    np.testing.assert_array_equal(w.scene.label_volume,labels)
    loaded=w.patient_store.load(record['id']); w.install_scene(loaded); w.workflow_tabs.setCurrentIndex(3)
    assert p.engine.currentData()=='ants' and p.n4.isChecked() and p.layer().uid==candidate.uid
    np.testing.assert_array_equal(p.layer().deformation['vectors'],candidate.deformation['vectors'])
    assert w.scene.segmentation_preview is None
    exported=export_postop(w.scene,p.layer(),w.export_directory())
    assert (exported/'reference_to_native_postop_LPS.h5').is_file()
    p.compute.click()
    cancel_button=w.findChild(QPushButton,'postopCancel')
    assert cancel_button is not None and cancel_button.isEnabled()
    QTimer.singleShot(700,cancel_button.click); wait_job(w,45)
    assert len(w.scene.extra_mris)==3 and p.layer().uid==candidate.uid
    assert '中止' in w.statusBar().currentMessage()
    for language in (0,1):
        w.language_tabs.setCurrentIndex(language); QApplication.processEvents()
        assert p.base().name=='Postoperative T1 demo'
        assert p.layer().uid==candidate.uid
        assert p.sequence.height()<40 and p.images.height()<40 and p.stage.height()<40
        for spin in (p.window_spin,p.level_spin,p.radius,p.max_shift):
            assert spin.width()>=100 and spin.height()<40
            assert spin.mapTo(p,QPoint()).x()+spin.width()<=p.width(), (spin.geometry(),p.width())
        assert ('Status:' if language else '状態：') in p.info.text()
        if capture_path:
            parent=p.parentWidget()
            while parent is not None and not isinstance(parent,QScrollArea):parent=parent.parentWidget()
            parent.verticalScrollBar().setValue(0); QApplication.processEvents()
            target=capture_path if language==0 else capture_path.with_stem(capture_path.stem+'_en')
            assert w.capture_widget(w).save(str(target),'PNG')
            parent.ensureWidgetVisible(p.compute); QApplication.processEvents()
            assert w.capture_widget(w).save(str(target.with_stem(target.stem+'_correction')),'PNG')
    w.language_tabs.setCurrentIndex(0)
    # Old patient controls must not leak into the next patient's workflow.
    w.clear_scene(); assert p.base() is None and not w.workflow_tabs.isTabEnabled(3)
    return {'postop_import_and_independent_layers':True,'cavity_contour_and_undo':True,
            'bounded_correction_candidate_and_review':True,'preserved_reference_labels_contacts':True,
            'portable_warp_save_and_export':True,'bilingual_comparison_controls':True,
            'ants_native_worker_n4_gui_save_reload':True,'exclusion_margin_preview':True,
            'ants_cancel_keeps_candidates':True,'legacy_method_limit_migration':True}
