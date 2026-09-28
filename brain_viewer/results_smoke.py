"""Synthetic UI checks for import, exact mapping, overlays and animated exports."""
from pathlib import Path
import tempfile
import time
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from .analysis_results import resolved_positions,exact_bindings
from .workspace_smoke import wait_job


def run_result_checks(window,capture_path=None):
    from tests.xlsx_fixture import write_xlsx
    if window.scene.kind!='synthetic': return {}
    scene=window.scene; panel=window.results; original=list(scene.results); state=window.view_state()
    parent=Path(__file__).resolve().parents[1]/'private_reports'
    try:
        with tempfile.TemporaryDirectory(prefix='result_smoke_',dir=parent) as d:
            root=Path(d); c=scene.contacts[:3]; channel=c[0].name+'-'+c[1].name
            run_mapping_surface_checks(window,capture_path,root)
            # Empty categories cannot keep a previous result or start playback.
            window.workflow_tabs.setCurrentIndex(4)
            if not original:
                panel.kind_tabs.setCurrentIndex(1)
                assert panel.selected() is None and not panel.png.isEnabled() and not panel.play.isEnabled()
                assert not panel.transport.isVisible() and not window.surface.result_glyphs.actor.GetVisibility()
                panel.kind_tabs.setCurrentIndex(0)
            h=root/'hfo.xlsx'; g=root/'gamma.xlsx'
            write_xlsx(h,{'Channel Ranking':[['channel_name','total_events','spike_associated'],[channel,8,3],['Unmatched',0,0]]})
            short=channel[:-3]
            write_xlsx(g,{'Data':[[None,-20,-10,0,10],[short,0,1,-2,3],['Unmatched',0,None,1,2]]})
            panel.import_files([h,g]); wait_job(window)
            assert window.workflow_tabs.currentIndex()==4 and len(scene.results)==len(original)+2
            assert panel.result_kind()=='static' and panel.library.count()==sum(r.kind=='static' for r in scene.results)
            assert panel.selected().kind=='static' and len(resolved_positions(panel.selected(),scene.contacts)[0])==1
            assert not panel.time_box.isVisible() and not panel.video.isVisible() and not panel.transport.isVisible()
            static=panel.selected();chart=panel.chart;glyphs=window.surface.result_glyphs
            assert window.surface.result_glyphs.actor.GetVisibility()
            # All visible representations use the same palette, range and mask.
            from vtkmodules.util.numpy_support import vtk_to_numpy
            from .analysis_results import overlay_points
            panel.display.palette.setCurrentIndex(panel.display.palette.findData('turbo'))
            panel.display.reverse.setChecked(True);panel.display.radius.setValue(3.)
            panel.display.slab.setValue(8.);panel.display.labels.setChecked(True)
            glyph=window.surface.result_glyphs
            expected=overlay_points(panel.selected(),scene.contacts,0,0)
            np.testing.assert_array_equal(vtk_to_numpy(glyph.data.GetPointData().GetScalars()),(expected[1]*255).astype(np.uint8))
            assert glyph.sphere.GetRadius()==3. and glyph.label_actor.GetVisibility()
            for pane in window.slice_panels():
                np.testing.assert_array_equal(pane.canvas.result_points,expected[0])
                assert pane.canvas.result_options['slab_mm']==8.
            panel.display.threshold_mode.setCurrentIndex(panel.display.threshold_mode.findData('above'))
            panel.display.threshold.setValue(9.)
            assert not glyph.actor.GetVisibility() and not glyph.label_actor.GetVisibility()
            assert all(not len(p.canvas.result_points) for p in window.slice_panels())
            panel.display.threshold_mode.setCurrentIndex(panel.display.threshold_mode.findData('none'))
            assert glyph.actor.GetVisibility()
            assert window.result_banner.result is panel.selected()
            panel.view.setCurrentIndex(panel.view.findData('channels')); QApplication.processEvents()
            panel.export_png(root/'hfo.png'); assert (root/'hfo.png').is_file()
            panel.open_mapping(); QApplication.processEvents()
            dialog=panel.mapping_dialog; dialog.jump(0,2); dialog.commit(); dialog.close()
            panel.kind_tabs.setCurrentIndex(1)
            assert panel.selected().kind=='time_series' and panel.video.isVisible() and panel.transport.isVisible()
            assert len(resolved_positions(panel.selected(),scene.contacts)[0])==0
            panel.open_mapping(); panel.mapping_dialog.suggest_aliases()
            assert not panel.selected().bindings
            panel.mapping_dialog.commit(); panel.mapping_dialog.close()
            assert len(resolved_positions(panel.selected(),scene.contacts)[0])==1
            panel.time_unit.setCurrentIndex(panel.time_unit.findData('ms'))
            assert panel.video.isEnabled(); panel.slider.setValue(1)
            dynamic=panel.selected();values=dynamic.values;times=dynamic.times
            panel.toggle_play();assert panel.timer.isActive()
            panel.kind_tabs.setCurrentIndex(0)
            assert not panel.timer.isActive() and panel.selected() is static and not panel.video.isVisible()
            assert panel.display.palette.currentData()=='turbo' and panel.display.radius.value()==3.
            panel.kind_tabs.setCurrentIndex(1)
            assert panel.selected() is dynamic and panel.slider.value()==1 and panel.time_unit.currentData()=='ms'
            assert panel.chart is chart and window.surface.result_glyphs is glyphs
            assert dynamic.values is values and dynamic.times is times
            # Re-import selects the correct category without duplicating values or losing units.
            panel.kind_tabs.setCurrentIndex(0);panel.import_files([g]);wait_job(window)
            assert len(scene.results)==len(original)+2 and panel.selected() is dynamic
            assert panel.result_kind()=='time_series' and panel.slider.value()==1 and dynamic.time_unit=='ms'
            panel.view.setCurrentIndex(panel.view.findData('channels')); panel.begin_video(root/'gamma.mp4',size=(960,640))
            deadline=time.monotonic()+45
            while panel.exporting and time.monotonic()<deadline:
                QApplication.processEvents(); time.sleep(.01)
            assert not panel.exporting and panel._export_error is None and panel.slider.value()==1
            import imageio_ffmpeg
            reader=imageio_ffmpeg.read_frames(str(root/'gamma.mp4')); meta=next(reader); frames=list(reader)
            assert len(frames)==4 and frames[0]!=frames[-1] and meta['fps']==20
            panel.selected().times[-1]=-20; panel.refresh(); assert not panel.video.isEnabled()
            panel.end.setValue(3); assert panel.video.isEnabled()
            panel.selected().times[-1]=10; panel.end.setValue(4)
            # Losing a contact leaves its channel unresolved, never at the origin.
            removed=scene.contacts.pop(0); panel.refresh()
            assert not window.surface.result_glyphs.actor.GetVisibility()
            scene.contacts.insert(0,removed); panel.refresh()
            panel.view.setCurrentIndex(panel.view.findData('brain')); panel.export_png(root/'brain.png')
            assert (root/'brain.png').is_file()
            # Include the color map in spatial movies, not just the channel chart.
            panel.display.palette.setCurrentIndex(panel.display.palette.findData('viridis'))
            panel.display.radius.setValue(3.);panel.display.labels.setChecked(True)
            panel.begin_video(root/'gamma_brain.mp4',size=(960,640))
            while panel.exporting and time.monotonic()<deadline+45:
                QApplication.processEvents();time.sleep(.01)
            assert not panel.exporting and panel._export_error is None
            reader=imageio_ffmpeg.read_frames(str(root/'gamma_brain.mp4'));meta=next(reader);frames=list(reader)
            assert len(frames)==4 and frames[0]!=frames[-1]
            panel.begin_video(root/'cancel.mp4',size=(960,640)); panel.cancel_export()
            while panel.exporting: QApplication.processEvents(); time.sleep(.01)
            assert not (root/'cancel.mp4').exists()
            window.workflow_tabs.setCurrentIndex(0); assert not window.surface.result_glyphs.actor.GetVisibility()
            assert not window.surface.result_glyphs.label_actor.GetVisibility()
            for _,actor in window.surface.contact_actors:
                assert actor.GetProperty().GetColor()==actor._result_base_color
            window.workflow_tabs.setCurrentIndex(4)
            if capture_path:
                from .analysis_results import AnalysisResult
                count=len(scene.contacts);times=np.arange(-100.,101.,10.)
                demo=AnalysisResult('Time series · synthetic electrodes','time_series',[c.name for c in scene.contacts],['value'],
                    np.sin(np.arange(count)[:,None,None]/3+times[None,:,None]/35),times,[''],time_unit='ms')
                demo.bindings=exact_bindings(demo,scene.contacts)
                demo.binding_modes={c:'monopolar' for c in demo.channels}
                demo.settings={'markers':{'radius_mm':3.,'slab_mm':8.},'display':{'value':{'colormap':'turbo'}},'frame':10}
                scene.results.append(demo);panel.bind_scene(preferred_uid=demo.uid)
                panel.show_on_electrodes();window.select_world(scene.contacts[len(scene.contacts)//2].position)
                window.opacity_slider.setValue(20);QApplication.processEvents();QTest.qWait(100)
                from PySide6.QtWidgets import QScrollArea
                window.findChild(QScrollArea).ensureWidgetVisible(panel.kind_tabs)
                assert panel.transport.isVisible()
                panel.transport.slider.setValue(11);assert panel.slider.value()==11
                panel.transport.previous.click();assert panel.slider.value()==10
                window.capture_widget(window).save(str(capture_path),'PNG')
                panel.kind_tabs.setCurrentIndex(0);QApplication.processEvents();QTest.qWait(50)
                window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+'_static')),'PNG')
                panel.kind_tabs.setCurrentIndex(1)
                window.change_language(1);QApplication.processEvents();QTest.qWait(50)
                window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+'_en')),'PNG')
                window.change_language(0)
                panel.view.setCurrentIndex(panel.view.findData('surface'));panel.export_png(capture_path.with_stem(capture_path.stem+'_3d'))
                # A dense, fully synthetic matrix exercises the 180-channel layout.
                demo=AnalysisResult('Gamma demo','time_series',[f'Demo{i+1:03}' for i in range(180)],['gamma'],
                    np.sin(np.arange(720).reshape(180,4,1)/17),np.array([-20.,-10.,0.,10.]),['demo units'],time_unit='ms')
                scene.results.append(demo); panel.bind_scene(preferred_uid=demo.uid)
                panel.view.setCurrentIndex(panel.view.findData('channels')); QApplication.processEvents(); QTest.qWait(100)
                window.grab().save(str(capture_path.with_name('analysis_demo.png')),'PNG')
                panel.export_png(capture_path.with_name('analysis_grid_demo.png'))
    finally:
        panel.stop(); scene.results=original; panel.bind_scene(); window.workflow_tabs.setCurrentIndex(0); window.restore_state(state)
    return {'analysis_excel_import':True,'analysis_mapping_and_missing_values':True,'analysis_png_mp4':True,
            'analysis_video_frame_order':True,'analysis_cancel':True,'analysis_time_axis_validation':True,
            'analysis_palette_threshold_labels_2d_3d':True,'analysis_spatial_video':True,'analysis_context_colors_restored':True,
            'analysis_static_time_series_tabs':True,'analysis_tab_state_and_shared_renderers':True,
            'analysis_duplicate_import_selects_category':True,'analysis_bipolar_name_candidates':True,
            'analysis_cortical_projection_radius_and_restore':True}


def run_mapping_surface_checks(window,capture_path=None,output_dir=None):
    """Exercise the new workflow with existing synthetic patient-space geometry."""
    from .analysis_results import AnalysisResult
    from .contact_matching import edf_contact_name
    from vtk.util.numpy_support import vtk_to_numpy
    scene=window.scene;panel=window.results;original=list(scene.results)
    channels=[]
    for group in dict.fromkeys(c.group for c in scene.contacts):
        lead=[c for c in scene.contacts if c.group==group]
        channels += [edf_contact_name(a.name,a.group)+'-'+edf_contact_name(b.name,b.group) for a,b in zip(lead,lead[1:])]
    channels.append('Unknown1-Unknown2')
    values=(10+10*np.sin(np.arange(len(channels))/2))[:,None,None]
    result=AnalysisResult('Synthetic bipolar heatmap','static',channels,['value'],values,np.array([0.]),['a.u.'])
    try:
        scene.results.append(result);window.workflow_tabs.setCurrentIndex(4);panel.bind_scene(preferred_uid=result.uid)
        panel.open_mapping();dialog=panel.mapping_dialog;QApplication.processEvents()
        assert sum(dialog.row_valid(i) for i in range(len(channels)))==len(channels)-1
        assert not result.bindings
        dialog.unresolved_only.setChecked(True)
        assert dialog.table.isRowHidden(0) and not dialog.table.isRowHidden(len(channels)-1)
        dialog.unresolved_only.setChecked(False)
        if capture_path: dialog.grab().save(str(capture_path.with_stem(capture_path.stem+'_mapping')),'PNG')
        dialog.reject();assert not result.bindings
        panel.open_mapping();panel.mapping_dialog.commit();panel.close_mapping()
        assert len(resolved_positions(result,scene.contacts)[0])==len(channels)-1
        display=panel.display
        display.spatial_mode.setCurrentIndex(display.spatial_mode.findData('surface'))
        display.palette.setCurrentIndex(display.palette.findData('jet'))
        surface=window.surface.result_surface
        assert surface.painted and not window.surface.result_glyphs.actor.GetVisibility()
        assert all(len(p.canvas.result_points)==len(channels)-1 for p in window.slice_panels())
        neighbors=dict(surface.neighbors)
        display.distance_slider.setValue(250);QTest.qWait(90);QApplication.processEvents()
        assert display.distance.value()==25 and result.settings['spatial']['distance_mm']==25
        assert all(surface.neighbors[key] is neighbors[key] for key in neighbors)
        assert surface.visible_vertices>0
        colors=vtk_to_numpy(window.surface.polys['pial_lh'].GetPointData().GetScalars()).copy()
        display.threshold_mode.setCurrentIndex(display.threshold_mode.findData('above'));display.threshold.setValue(1000)
        assert surface.visible_vertices==0
        display.threshold_mode.setCurrentIndex(display.threshold_mode.findData('none'))
        np.testing.assert_array_equal(colors,vtk_to_numpy(window.surface.polys['pial_lh'].GetPointData().GetScalars()))
        if capture_path:
            panel.view.setCurrentIndex(panel.view.findData('surface'));window.surface.set_camera_preset(4)
            window.opacity_slider.setValue(75);QApplication.processEvents()
            window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+'_surface_ui')),'PNG')
            panel.export_png(capture_path.with_stem(capture_path.stem+'_surface'))
            window.change_language(1);QApplication.processEvents()
            window.capture_widget(window).save(str(capture_path.with_stem(capture_path.stem+'_surface_en')),'PNG')
            window.change_language(0)
        display.spatial_mode.setCurrentIndex(display.spatial_mode.findData('both'))
        assert window.surface.result_glyphs.actor.GetVisibility() and surface.painted
        if output_dir:
            import json,imageio_ffmpeg
            result.kind='time_series';result.time_unit='ms';result.times=np.array([0.,10.,20.])
            result.values=np.concatenate([values,20-values,values*.5],axis=1)
            panel.bind_scene(preferred_uid=result.uid);panel.view.setCurrentIndex(panel.view.findData('surface'))
            panel.begin_video(output_dir/'surface_movie.mp4',size=(960,640))
            deadline=time.monotonic()+35
            while panel.exporting and time.monotonic()<deadline:
                QApplication.processEvents();time.sleep(.01)
            assert not panel.exporting and panel._export_error is None
            reader=imageio_ffmpeg.read_frames(str(output_dir/'surface_movie.mp4'));next(reader);frames=list(reader)
            assert len(frames)==3 and frames[0]!=frames[1]
            metadata=json.loads((output_dir/'surface_movie.json').read_text(encoding='utf-8'))
            assert metadata['spatial_display']['mode']=='both' and metadata['spatial_display']['distance_mm']==25
        panel.show_overlay.setChecked(False)
        assert not surface.painted and not window.surface.result_glyphs.actor.GetVisibility()
        assert all(not actor.GetMapper().GetScalarVisibility() for actor in window.surface.actors.values())
    finally:
        panel.close_mapping();scene.results=original;panel.bind_scene()
