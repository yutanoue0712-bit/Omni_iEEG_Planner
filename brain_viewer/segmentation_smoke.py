"""Local GUI integration checks; only synthetic scenes may be captured."""
from pathlib import Path
import tempfile
import nibabel as nib
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from .imaging import save_scene, load_scene
from .segmentation import export_mask, export_surface
from .workspace_smoke import wait_job


def run_segmentation_checks(window, capture_path=None):
    scene=window.scene; panel=window.segmentation
    source_labels=scene.label_volume.copy(); source_mri=scene.data.copy()
    source_contacts=scene.contact_positions().copy()
    window.workflow_tabs.setCurrentIndex(1); panel.reference_view()
    window.reset_views(); QApplication.processEvents()
    assert panel.isVisible()
    for preset in ('ventricles','callosum','thalamus'):
        panel.preset.setCurrentIndex(panel.preset.findData(preset))
        panel.create_button.click(); wait_job(window)
        segment=panel.selected()
        assert segment.preset==preset and segment.mask.any()
        actor=window.surface.segment_actors[segment.uid]
        assert actor.GetVisibility() and actor.GetMapper().GetInput().GetNumberOfPolys()>0
        points=np.argwhere(segment.mask)
        lo=scene.world(points.min(0)-.5); hi=scene.world(points.max(0)+.5)
        np.testing.assert_allclose(actor.GetBounds(),np.column_stack([lo,hi]).ravel(),atol=1e-4)
    selected=panel.selected(); actor=window.surface.segment_actors[selected.uid]
    panel.regions.currentItem().setCheckState(Qt.CheckState.Unchecked)
    assert not actor.GetVisibility() and selected.visible_2d
    assert sum(a.GetVisibility() for a in window.surface.segment_actors.values())==2
    panel.regions.currentItem().setCheckState(Qt.CheckState.Checked)
    panel.opacity.setValue(48); assert abs(actor.GetProperty().GetOpacity()-.48)<1e-6
    image=window.slices[2].canvas._image.copy()
    panel.show_2d.setChecked(False)
    assert image!=window.slices[2].canvas._image and actor.GetVisibility()
    panel.show_2d.setChecked(True)
    # Empty manual masks must be editable in all three radiological slice views.
    panel.preset.setCurrentIndex(panel.preset.findData('manual')); panel.name.setText('Manual test')
    panel.create_button.click(); wait_job(window)
    manual=panel.selected(); assert not manual.mask.any()
    assert panel.mode.currentData()=='paint'
    for axis in (2,1,0):
        center=np.array(scene.data.shape)//2; center[(axis+1)%3]+=12
        window.set_cursor(center); canvas=window.slices[axis].canvas
        target=canvas.position_for_index(center).toPoint()
        actual=canvas._index_at(target)
        QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,target)
        assert manual.mask[tuple(actual)]
        filled=np.argwhere(manual.mask)
        assert np.all(filled[:,axis]==center[axis]) and len(manual.edits)==1
        assert window.surface.segment_actors[manual.uid].GetVisibility()
        panel.undo_button.click(); assert not manual.mask.any() and not manual.edits
    # A real drag paints a continuous path, and erasing/undo restore the mask.
    canvas=window.slices[2].canvas; center=np.array(scene.data.shape)//2
    window.set_cursor(center); end=center+np.array([10,0,0])
    a=canvas.position_for_index(center).toPoint(); b=canvas.position_for_index(end).toPoint()
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,a)
    QTest.mouseMove(canvas,b,20)
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,b)
    assert manual.mask[tuple(canvas._index_at(a))] and manual.mask[tuple(canvas._index_at(b))]
    painted=manual.mask.copy()
    panel.mode.setCurrentIndex(panel.mode.findData('erase'))
    QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,b)
    assert not manual.mask[tuple(canvas._index_at(b))]
    panel.undo_button.click(); np.testing.assert_array_equal(manual.mask,painted)
    panel.rename.setText('Edited region'); panel.rename.editingFinished.emit()
    assert manual.name=='Edited region'
    window.workflow_tabs.setCurrentIndex(0)
    assert all(p.canvas.brush_mode=='navigate' for p in window.slices.values())
    np.testing.assert_array_equal(scene.label_volume,source_labels)
    np.testing.assert_array_equal(scene.data,source_mri)
    np.testing.assert_array_equal(scene.contact_positions(),source_contacts)
    with tempfile.TemporaryDirectory() as directory:
        folder=save_scene(scene,Path(directory),window.view_state())
        restored=load_scene(folder/'scene.json')
        assert len(restored.segmentations)==4
        for before,after in zip(scene.segmentations,restored.segmentations):
            np.testing.assert_array_equal(before.mask,after.mask)
            assert before.edits==after.edits and before.opacity==after.opacity
        export_mask(manual,scene,Path(directory)/'manual.nii.gz')
        np.testing.assert_array_equal(np.asanyarray(nib.load(Path(directory)/'manual.nii.gz').dataobj),manual.mask)
        export_surface(manual,scene,Path(directory)/'manual.vtp')
    # Both interfaces keep the user's region names unchanged.
    window.workflow_tabs.setCurrentIndex(1); panel.regions.setCurrentRow(0); panel.jump()
    for language in (0,1):
        window.language_tabs.setCurrentIndex(language); QApplication.processEvents()
        assert panel.regions.item(3).text()=='Edited region'
        if capture_path is not None:
            assert scene.kind=='synthetic'
            suffix='_segmentation_ja' if language==0 else '_segmentation_en'
            window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+suffix)),'PNG')
    window.language_tabs.setCurrentIndex(0); window.workflow_tabs.setCurrentIndex(0)
    return {'segmentation_presets_and_3d_visibility':True,'segmentation_brush_erase_undo_three_planes':True,
            'segmentation_coordinates_export_roundtrip':True,'segmentation_original_labels_unchanged':True}
