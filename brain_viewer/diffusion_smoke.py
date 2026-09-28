"""Artificial diffusion data only: real widgets, jobs, overlays and saving."""
from pathlib import Path
import time
from unittest.mock import patch
import nibabel as nib
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from .imaging import load_scene, InputError
from .rendering import composite_plane


def run_diffusion_checks(window,capture_path=None):
    from tests.test_diffusion import phantom
    assert window.scene.kind=='synthetic'
    def wait():
        from .workspace_smoke import wait_job
        wait_job(window)
    folder=Path(window._patient_temp.name)/'dti_test';folder.mkdir(exist_ok=True)
    data,a,b,v,mask=phantom()
    image=nib.Nifti1Image(data,a);image.header.set_xyzt_units('mm');image.set_sform(a,1);image.set_qform(a,1)
    path=folder/'dwi.nii.gz';nib.save(image,path)
    np.savetxt(folder/'dwi.bval',b[None])
    v[:,0]*=-1;np.savetxt(folder/'dwi.bvec',v.T)
    previous_ct=window.scene.ct
    previous_contacts=window.scene.contact_positions().copy()
    window.diffusion.correct_motion.setChecked(False)
    window.sequence_combo.setCurrentIndex(window.sequence_combo.findData('DTI'))
    with patch('PySide6.QtWidgets.QFileDialog.getOpenFileName',return_value=(str(path),'NIfTI')), \
         patch('PySide6.QtWidgets.QInputDialog.getText',return_value=('Synthetic DTI',True)), \
         patch('brain_viewer.sequences.register_mri',return_value=(np.eye(4),{'review_status':'synthetic','method':'fixture'})):
        window.import_file_button.click()
        wait()
    assert len(window.scene.diffusions)==1
    assert window.workflow_tabs.currentIndex()==5
    assert window.scene.ct is previous_ct
    np.testing.assert_array_equal(window.scene.contact_positions(),previous_contacts)
    panel=window.diffusion;model=panel.selected()
    assert model is not None and len(model.layer_ids)==3
    current=window.scene
    with patch('PySide6.QtWidgets.QFileDialog.getOpenFileName',return_value=(str(path),'NIfTI')), \
         patch('PySide6.QtWidgets.QInputDialog.getText',return_value=('Invalid DTI',True)), \
         patch('brain_viewer.diffusion_io.read_gradients',side_effect=InputError('DTIのb値・方向情報を読み込めません。')):
        window.import_file_button.click();wait()
    assert window.scene is current and panel.feedback.isVisible()
    assert panel.file_button.isEnabled() and not window._job_timer.isActive()
    window.clear_import_feedback()
    uid=model.layer_ids['fa'];layer=window.extra_layer(uid)
    assert layer is not None and window.extra_styles[uid]=='direction'
    center=np.argwhere(model.mask & (model.fa>.5))[len(np.argwhere(model.mask & (model.fa>.5)))//2]
    world=nib.affines.apply_affine(model.affine,center)
    window.set_cursor(window.scene.index(world));panel.roi1.capture.click()
    assert len(window.surface.roi_actors)==1
    assert panel.roi1.value()['center'] is not None
    panel.max_seeds.setValue(100)
    panel.parameters['min_length'].setValue(5)
    panel.track_button.click();wait()
    assert window.scene.tracts and window.scene.tracts[-1].count>0
    bundle=window.scene.tracts[-1]
    assert window.surface.tract_actors[bundle.uid].GetVisibility()
    panel.bundles.currentItem().setCheckState(Qt.CheckState.Unchecked)
    assert not window.surface.tract_actors[bundle.uid].GetVisibility()
    panel.bundles.currentItem().setCheckState(Qt.CheckState.Checked)
    panel.opacity.setValue(45)
    assert abs(window.surface.tract_actors[bundle.uid].GetProperty().GetOpacity()-.45)<1e-6
    panel.mode.setCurrentIndex(1)
    world2=world+np.array([8,0,0])
    window.set_cursor(window.scene.index(world2));panel.roi2.capture.click()
    panel.track_button.click();wait()
    assert len(window.scene.tracts)==2
    # FA and direction color are distinct and obey normal image visibility controls.
    options=window.ct_options();idx=int(window.ijk[2])
    colored=composite_plane(window.scene,2,idx,0,1000,options)
    panel.map.setCurrentIndex(panel.map.findData('fa'));panel.show_map()
    grey=composite_plane(window.scene,2,idx,0,1000,window.ct_options())
    assert np.any(grey!=colored)
    panel.map.setCurrentIndex(0);panel.show_map()
    panel.compare_button.click()
    assert window.extra_styles[model.layer_ids['b0']]=='checker'
    panel.show_map()
    # All saved data must allow tracking again without the original DWI.
    window.save_work();wait()
    saved=load_scene(window.patient_store.scene_path(window.patient_id))
    assert len(saved.diffusions)==1 and len(saved.tracts)==2
    np.testing.assert_array_equal(saved.tracts[-1].points,window.scene.tracts[-1].points)
    for language in (0,1,0):
        window.language_tabs.setCurrentIndex(language)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents();QTest.qWait(100);QApplication.processEvents()
        assert panel.selected().uid==model.uid
        if capture_path:
            target=Path(capture_path).with_name(Path(capture_path).stem+('_dti_en' if language else '_dti_ja')+'.png')
            window.capture_widget(window).save(str(target))
    if capture_path:
        from PySide6.QtWidgets import QScrollArea
        scroll=window.control_stack.parentWidget().parentWidget()
        if isinstance(scroll,QScrollArea):
            scroll.ensureWidgetVisible(panel.track_button)
            QApplication.processEvents()
            target=Path(capture_path).with_name(Path(capture_path).stem+'_dti_settings.png')
            window.capture_widget(window).save(str(target))
    # Switching away removes ROI helpers while preserving the saved bundles.
    window.workflow_tabs.setCurrentIndex(0);assert not window.surface.roi_actors
    window.workflow_tabs.setCurrentIndex(5);assert len(window.surface.roi_actors)==2
    panel.mode.setCurrentIndex(panel.mode.findData(0))
    assert not window.surface.roi_actors and not panel.roi1.isVisible()
    panel.track_button.click();wait()
    assert window.scene.tracts[-1].settings['used_seeds']<=100
    assert window.scene.tracts[-1].settings['roi1']['kind']=='brain'
    return {'dti_file_button_import':True,'dti_preserves_ct_contacts':True,'dti_scalar_color_overlay':True,
            'dti_one_two_roi_widgets':True,'dti_tract_visibility_opacity':True,'dti_snapshot_roundtrip':True,
            'dti_invalid_input_preserves_scene':True,'dti_bounded_preview':True}
