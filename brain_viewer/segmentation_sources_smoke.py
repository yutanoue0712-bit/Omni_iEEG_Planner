"""GUI checks for label selection and extraction preview lifecycle."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QDialog
from .segmentation_create import LabelPicker
from .workspace_smoke import wait_job


def run_source_checks(window,capture_path=None):
    panel=window.segmentation; scene=window.scene
    before=len(scene.segmentations); contacts=scene.contact_positions().copy()
    window.workflow_tabs.setCurrentIndex(1); panel.creation_method.setCurrentIndex(1)
    picker=LabelPicker(scene,set(),panel)
    label_count=picker.labels.count()
    selected=[]
    for collection in ('anatomy','nuclei'):
        for i in range(picker.labels.count()):
            item=picker.labels.item(i); row=item.data(Qt.ItemDataRole.UserRole)
            if row['source']==collection and (collection=='nuclei' or row['label']>=1000):
                item.setCheckState(Qt.CheckState.Checked); selected.append(row); break
    assert len(selected)==2
    picker.search.setText(str(selected[0]['label'])); QApplication.processEvents()
    assert len(picker.checked())==2 and any(picker.labels.item(i).isHidden() for i in range(picker.labels.count()))
    if capture_path:
        assert scene.kind=='synthetic'; picker.show(); QApplication.processEvents()
        picker.grab().save(str(capture_path.with_stem(capture_path.stem+'_labels')),'PNG')
        picker.hide()
    with patch('brain_viewer.segmentation_panel.LabelPicker',return_value=picker), patch.object(picker,'exec',return_value=QDialog.DialogCode.Accepted):
        panel.choose_labels()
    panel.create_button.click(); wait_job(window)
    assert len(scene.segmentations)==before+2
    for segment,row in zip(scene.segmentations[-2:],selected):
        assert segment.provenance['present_label_ids']==[row['label']]
        assert window.surface.segment_actors[segment.uid].GetVisibility()
    panel.undo_button.click(); assert len(scene.segmentations)==before
    panel.create_button.click(); wait_job(window)
    panel.creation_method.setCurrentIndex(2); pane=panel.extraction
    point=np.argwhere(scene.label_volume==4)[len(np.argwhere(scene.label_volume==4))//2]
    window.set_cursor(point); pane.capture_seed()
    value=float(scene.data[tuple(point)]); pane.lower.setValue(value-5); pane.upper.setValue(value+5); pane.radius.setValue(6)
    panel.name.setText('Source extraction test')
    pane.preview_button.click(); wait_job(window)
    candidate=scene.segmentation_preview
    assert candidate is not None and candidate.mask[tuple(point)] and len(scene.segmentations)==before+2
    assert panel.create_button.isEnabled() and window.surface.segment_actors[candidate.uid].GetVisibility()
    window.drag_window(8,5)
    assert scene.segmentation_preview is candidate
    pane.upper.setValue(pane.upper.value()+1)
    assert scene.segmentation_preview is None and candidate.uid not in window.surface.segment_actors
    assert not panel.create_button.isEnabled()
    pane.preview_button.click(); wait_job(window)
    candidate=scene.segmentation_preview; panel.create_button.click()
    assert scene.segmentation_preview is None and scene.segmentations[-1] is candidate
    assert candidate.provenance['source_id']=='mri'
    assert len(scene.segmentations)==before+3
    pane.preview_button.click(); wait_job(window)
    preview_id=scene.segmentation_preview.uid; pane.clear_button.click()
    assert scene.segmentation_preview is None and preview_id not in window.surface.segment_actors
    # A subsequent preview is discarded on tab exit, preserving the committed region.
    pane.preview_button.click(); wait_job(window)
    preview_id=scene.segmentation_preview.uid
    if capture_path:
        assert scene.kind=='synthetic'
        for language in (0,1):
            window.language_tabs.setCurrentIndex(language); QApplication.processEvents()
            suffix='_extraction_ja' if language==0 else '_extraction_en'
            window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+suffix)),'PNG')
        window.language_tabs.setCurrentIndex(0)
    window.workflow_tabs.setCurrentIndex(0)
    assert scene.segmentation_preview is None and preview_id not in window.surface.segment_actors
    assert candidate.uid in window.surface.segment_actors
    np.testing.assert_array_equal(scene.contact_positions(),contacts)
    return {'all_loaded_label_selection':True,'available_freesurfer_labels':label_count,
            'image_threshold_preview_commit_and_invalidation':True}


def check_added_source(window,layer):
    panel=window.segmentation; pane=panel.extraction
    window.workflow_tabs.setCurrentIndex(1); panel.creation_method.setCurrentIndex(2)
    pane.source.setCurrentIndex(pane.source.findData(layer.uid))
    point=np.array(window.scene.data.shape)//2
    window.set_cursor(point); pane.capture_seed(); value=float(layer.data[tuple(point)])
    pane.lower.setValue(value-1); pane.upper.setValue(value+1); pane.radius.setValue(4)
    pane.preview_button.click(); wait_job(window)
    candidate=window.scene.segmentation_preview
    assert candidate is not None and candidate.provenance['source_id']==layer.uid
    assert candidate.provenance['source_sequence']==layer.sequence
    assert window.window_target.currentData()==layer.uid
    panel.create_button.click()
    assert window.scene.segmentations[-1] is candidate
    window.workflow_tabs.setCurrentIndex(0)
