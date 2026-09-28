"""Result library, channel review, fixed-scale playback and local PNG/MP4 export."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,
    QGroupBox,QDoubleSpinBox,QSpinBox,QSlider,QCheckBox,QFileDialog,QLineEdit,QMessageBox,QProgressDialog,QTabBar)
from .i18n import tr
from .analysis_results import (read_workbook,exact_bindings,resolved_positions,color_limits,overlay_points,
    frame_range,time_issue,export_record)
from .result_rendering import ResultCanvas,export_frame,time_text
from .result_export import VideoWriter,save_png
from .result_display import marker_options,spatial_options
from .result_display_controls import ResultDisplayControls


class ResultsPanel(QWidget):
    def __init__(self,window):
        super().__init__(); self.window=window; self.busy=False; self.exporting=False; self.mapping_dialog=None
        self._library_scene=None; self._selected_by_kind={}
        self.timer=QTimer(self); self.timer.timeout.connect(self.advance)
        self.chart=ResultCanvas(); self.chart.channel_clicked.connect(self.jump); self.chart.setMouseTracking(True)
        layout=QVBoxLayout(self); layout.setContentsMargins(0,3,10,0); layout.setSpacing(9)
        intro=QLabel(tr('静的な集計値と、時間で変わる解析結果を表示・出力します。'))
        intro.setWordWrap(True); intro.setObjectName('muted'); layout.addWidget(intro)
        self.import_button=QPushButton(tr('解析Excelを追加')); self.import_button.setObjectName('primary')
        self.import_button.clicked.connect(self.choose_files); layout.addWidget(self.import_button)
        self.import_button.setToolTip(tr('Excelの時間列の有無に応じて、対応するタブに追加します。'))
        self.kind_tabs=QTabBar();self.kind_tabs.setExpanding(True)
        for label,kind in (('静的結果','static'),('時系列結果','time_series')):
            index=self.kind_tabs.addTab(tr(label));self.kind_tabs.setTabData(index,kind)
        self.kind_tabs.currentChanged.connect(self.change_kind);layout.addWidget(self.kind_tabs)
        self.kind_hint=QLabel();self.kind_hint.setWordWrap(True);self.kind_hint.setObjectName('muted')
        layout.addWidget(self.kind_hint)
        self.library=QComboBox(); self.library.setProperty('userText',True); self.library.currentIndexChanged.connect(self.select)
        layout.addWidget(self.library)
        self.name=QLineEdit(); self.name.setProperty('userText',True); self.name.setMaxLength(120)
        self.name.editingFinished.connect(self.rename); layout.addWidget(self.name)
        self.details=QLabel(); self.details.setWordWrap(True); self.details.setObjectName('muted'); layout.addWidget(self.details)
        properties=QGroupBox(tr('表示する値')); prop=QVBoxLayout(properties); self.properties=properties
        self.metric_combo=QComboBox(); self.metric_combo.setProperty('userText',True); self.metric_combo.currentIndexChanged.connect(self.metric_changed)
        prop.addWidget(self.metric_combo)
        row=QHBoxLayout(); row.addWidget(QLabel(tr('値の単位')))
        self.unit=QLineEdit(); self.unit.setProperty('userText',True); self.unit.setPlaceholderText(tr('未確認'))
        self.unit.setMaxLength(40); self.unit.editingFinished.connect(self.update_unit); row.addWidget(self.unit); prop.addLayout(row)
        row=QHBoxLayout(); self.minimum=QDoubleSpinBox(); self.maximum=QDoubleSpinBox()
        for box in (self.minimum,self.maximum): box.setDecimals(5); box.setRange(-1e12,1e12); box.editingFinished.connect(self.change_limits)
        row.addWidget(QLabel(tr('色の下限'))); row.addWidget(self.minimum); prop.addLayout(row)
        row=QHBoxLayout(); row.addWidget(QLabel(tr('色の上限'))); row.addWidget(self.maximum); prop.addLayout(row)
        automatic=QPushButton(tr('全時間の範囲に戻す')); automatic.clicked.connect(self.reset_limits); prop.addWidget(automatic)
        layout.addWidget(properties)
        self.display=ResultDisplayControls(self);layout.addWidget(self.display)
        self.view=QComboBox()
        for text,key in (('チャンネル別','channels'),('脳画像（4画面）','brain'),('脳画像（3D出力）','surface')): self.view.addItem(tr(text),key)
        self.view.currentIndexChanged.connect(self.change_view); layout.addWidget(self.view)
        self.show_overlay=QCheckBox(tr('解析値を2D・3Dに重ねる')); self.show_overlay.setChecked(True)
        self.show_overlay.toggled.connect(self.refresh); layout.addWidget(self.show_overlay)
        self.mapping_button=QPushButton(tr('電極との対応を確認・指定')); self.mapping_button.clicked.connect(self.open_mapping)
        layout.addWidget(self.mapping_button)
        self.electrode_view=QPushButton(tr('電極位置にカラーマップを表示'))
        self.electrode_view.setObjectName('primary');self.electrode_view.clicked.connect(self.show_on_electrodes)
        layout.addWidget(self.electrode_view)
        self.mapping_info=QLabel(); self.mapping_info.setWordWrap(True); layout.addWidget(self.mapping_info)
        self.time_box=QGroupBox(tr('時間・再生')); time_layout=QVBoxLayout(self.time_box)
        self.time_unit=QComboBox()
        for name,key in (('時間単位：未確認',''),('時間単位：ms','ms'),('時間単位：s','s')): self.time_unit.addItem(tr(name),key)
        self.time_unit.currentIndexChanged.connect(self.time_unit_changed); time_layout.addWidget(self.time_unit)
        self.time_label=QLabel(); time_layout.addWidget(self.time_label)
        self.slider=QSlider(Qt.Orientation.Horizontal); self.slider.valueChanged.connect(self.refresh); time_layout.addWidget(self.slider)
        row=QHBoxLayout(); self.play=QPushButton(tr('再生')); self.play.clicked.connect(self.toggle_play); row.addWidget(self.play)
        row.addWidget(QLabel(tr('列/秒'))); self.fps=QSpinBox(); self.fps.setRange(1,60); self.fps.setValue(20)
        self.fps.valueChanged.connect(self.rate_changed); row.addWidget(self.fps); time_layout.addLayout(row)
        row=QHBoxLayout(); row.addWidget(QLabel(tr('範囲（列番号）')))
        self.start=QSpinBox(); self.end=QSpinBox()
        for box in (self.start,self.end): box.setRange(1,1); box.valueChanged.connect(self.range_changed); row.addWidget(box)
        time_layout.addLayout(row)
        note=QLabel(tr('1フレーム＝Excelの1時間列。補間せず順番に再生します。'))
        note.setWordWrap(True); note.setObjectName('muted'); time_layout.addWidget(note)
        layout.addWidget(self.time_box)
        self.issue=QLabel(); self.issue.setWordWrap(True); self.issue.setStyleSheet('color:#efc087'); layout.addWidget(self.issue)
        row=QHBoxLayout(); self.png=QPushButton(tr('静止画 PNG')); self.png.clicked.connect(self.export_png)
        self.video=QPushButton(tr('動画 MP4')); self.video.clicked.connect(self.export_video)
        row.addWidget(self.png); row.addWidget(self.video); layout.addLayout(row)
        for widget in (self.mapping_button,self.electrode_view):
            layout.removeWidget(widget);layout.insertWidget(layout.indexOf(properties),widget)
        layout.addStretch(); self.bind_scene()

    def selected(self):
        if self.window.scene is None: return None
        return next((r for r in self.window.scene.results if r.uid==self.library.currentData()),None)

    def close_mapping(self):
        if self.mapping_dialog:
            dialog=self.mapping_dialog;self.mapping_dialog=None;dialog.close();dialog.deleteLater()

    def result_kind(self):
        return self.kind_tabs.tabData(self.kind_tabs.currentIndex())

    def bind_scene(self,preferred_uid=None):
        self.stop()
        self.close_mapping()
        scene=self.window.scene; changed=scene is not self._library_scene
        if changed:
            self._selected_by_kind.clear();self._library_scene=scene
        results=scene.results if scene else []
        preferred=next((r for r in results if r.uid==preferred_uid),None)
        kind=self.result_kind()
        if preferred:
            kind=preferred.kind;self._selected_by_kind[kind]=preferred.uid
        elif changed and results and not any(r.kind==kind for r in results):
            kind=results[0].kind
        self.kind_tabs.blockSignals(True)
        self.kind_tabs.setCurrentIndex(0 if kind=='static' else 1)
        self.kind_tabs.blockSignals(False)
        self.rebuild_library()

    def change_kind(self,*_):
        if self.busy:return
        self.stop();self.close_mapping();self.rebuild_library()

    def rebuild_library(self):
        # Reuse results and renderers; switching tabs never reloads arrays or images.
        self.busy=True; self.library.clear()
        kind=self.result_kind()
        if self.window.scene:
            for result in self.window.scene.results:
                if result.kind==kind:self.library.addItem(result.name,result.uid)
        self.library.setCurrentIndex(max(0,self.library.findData(self._selected_by_kind.get(kind))))
        self.library.setEnabled(self.library.count()>0)
        hint=('時間を含まない結果を静止画で表示・出力します。' if kind=='static' else
              '時間に沿って再生・動画出力します。任意の時点を静止画にもできます。')
        self.kind_hint.setText(tr(hint))
        self.video.setVisible(kind=='time_series')
        self.busy=False; self.select()

    def choose_files(self):
        files,_=QFileDialog.getOpenFileNames(self,tr('解析Excelを追加'),str(self.window.project_root/'sample_images'),'Excel (*.xlsx)')
        if files: self.import_files([Path(p) for p in files])

    def import_files(self,paths,time_unit=''):
        self.stop()
        scene=self.window.scene
        def read(progress):
            result=[]
            for i,path in enumerate(paths):
                progress(tr('解析Excelを読み込み中 {current}/{total}').format(current=i+1,total=len(paths)))
                result.append(read_workbook(path,time_unit=time_unit))
            return result
        def done(results):
            if scene is not self.window.scene: return
            new=[]; imported=[]
            for result in results:
                existing=next((r for r in scene.results if r.provenance.get('sha256')==result.provenance.get('sha256')),None)
                if existing:
                    imported.append(existing);continue
                result.bindings=exact_bindings(result,scene.contacts); scene.results.append(result); new.append(result)
                result.binding_modes={key:'bipolar' if len(ids)==2 else 'monopolar' for key,ids in result.bindings.items()}
                imported.append(result)
            for result in imported:self._selected_by_kind[result.kind]=result.uid
            self.bind_scene(preferred_uid=imported[0].uid if imported else None)
            self.window.workflow_tabs.setCurrentIndex(4); self.change_view()
            self.window.message(tr('解析結果を追加しました。患者の保存で値と対応情報を保存できます。'))
            if new and self.window.patient_id and not self.window._smoke:
                self.window._after_job=lambda:self.window.save_work()
        self.window.run_job(read,done,tr('解析Excelを読み込んでいます…'))

    def select(self,*_):
        if self.busy: return
        self.stop(); self.close_mapping(); r=self.selected(); self.busy=True; self.metric_combo.clear()
        for widget in (self.properties,self.name,self.mapping_button,self.electrode_view,self.view,self.show_overlay): widget.setEnabled(r is not None)
        if r:
            self._selected_by_kind[r.kind]=r.uid
            self.name.setText(r.name)
            for metric in r.metrics: self.metric_combo.addItem(metric)
            self.metric_combo.setCurrentIndex(int(np.clip(r.settings.get('metric',0),0,len(r.metrics)-1)))
            self.slider.setRange(0,len(r.times)-1); self.slider.setValue(int(np.clip(r.settings.get('frame',0),0,len(r.times)-1)))
            self.time_unit.setCurrentIndex(max(0,self.time_unit.findData(r.time_unit)))
            self.start.setRange(1,len(r.times)); self.end.setRange(1,len(r.times))
            a,b=frame_range(r); self.start.setValue(a+1); self.end.setValue(b+1)
            self.fps.setValue(int(np.clip(r.settings.get('fps',20),1,60)))
            default='brain' if len(resolved_positions(r,self.window.scene.contacts)[0]) else 'channels'
            self.view.setCurrentIndex(max(0,self.view.findData(r.settings.get('view',default))))
            self.show_overlay.setChecked(bool(r.settings.get('overlay',True)))
            self.details.setText(tr('{channels}チャンネル · {frames}列 · 欠損 {missing}').format(
                channels=len(r.channels),frames=len(r.times),missing=int(np.isnan(r.values).sum())))
        else:
            self.name.clear(); self.unit.clear()
            self.details.setText(tr('このタブの結果はまだありません。「解析Excelを追加」から読み込めます。'))
            self.slider.setRange(0,0);self.view.setCurrentIndex(0)
        self.time_box.setVisible(bool(r and r.kind=='time_series'))
        self.busy=False; self.metric_changed(); self.change_view()

    def rename(self):
        r=self.selected()
        if r and self.name.text().strip():
            r.name=self.name.text().strip(); self.library.setItemText(self.library.currentIndex(),r.name); self.refresh()

    def metric_changed(self,*_):
        if self.busy: return
        r=self.selected()
        if r:
            i=self.metric_combo.currentIndex(); self.unit.setText(r.units[i]); lo,hi=color_limits(r,i)
            self.minimum.setValue(lo); self.maximum.setValue(hi); r.settings['metric']=i
        self.display.bind()
        self.refresh()

    def update_unit(self):
        r=self.selected()
        if r: r.units[self.metric_combo.currentIndex()]=self.unit.text().strip(); self.refresh()

    def time_unit_changed(self,*_):
        if self.busy: return
        r=self.selected()
        if r: r.time_unit=self.time_unit.currentData(); self.refresh()

    def change_limits(self):
        r=self.selected()
        if not r: return
        lo,hi=self.minimum.value(),self.maximum.value()
        if hi<=lo: self.window.message(tr('色の上限は下限より大きくしてください。')); self.metric_changed(); return
        r.settings.setdefault('limits',{})[r.metrics[self.metric_combo.currentIndex()]]=[lo,hi]; self.refresh()

    def reset_limits(self):
        r=self.selected()
        if r: r.settings.setdefault('limits',{}).pop(r.metrics[self.metric_combo.currentIndex()],None); self.metric_changed()

    def range_changed(self,*_):
        if self.busy: return
        self.stop(); r=self.selected()
        if r:
            r.settings['start']=self.start.value()-1; r.settings['end']=self.end.value()-1
            self.refresh()

    def rate_changed(self,*_):
        if self.busy: return
        r=self.selected()
        if r: r.settings['fps']=self.fps.value()
        if self.timer.isActive(): self.timer.setInterval(round(1000/self.fps.value()))

    def change_view(self,*_):
        if self.busy: return
        r=self.selected()
        if r: r.settings['view']=self.view.currentData()
        if hasattr(self.window,'right_stack'):
            self.window.right_stack.setCurrentIndex(1 if self.window.workflow_tabs.currentIndex()==4 and self.view.currentData()=='channels' else 0)
        self.refresh()

    def refresh(self,*_):
        if self.busy: return
        r=self.selected(); scene=self.window.scene; active=self.window.workflow_tabs.currentIndex()==4
        positions=np.empty((0,3)); rgb=np.empty((0,3)); count=0;labels=[]
        markers=marker_options(r)
        if r:
            frame=self.slider.value(); metric=max(0,self.metric_combo.currentIndex())
            r.settings['frame']=frame; r.settings['overlay']=self.show_overlay.isChecked()
            count=len(resolved_positions(r,scene.contacts)[0]); self.time_label.setText(time_text(r,frame))
            if active and self.show_overlay.isChecked():
                positions,rgb,indices=overlay_points(r,scene.contacts,metric,frame)
                labels=[r.aliases.get(r.channels[i],r.channels[i]) for i in indices]
            self.chart.set_result(r,scene.contacts,metric,frame)
            self.mapping_info.setText(tr('位置対応 {mapped}/{total} · 現在の着色 {shown}\n未対応のチャンネルは「電極との対応を確認・指定」で設定します。').format(mapped=count,total=len(r.channels),shown=len(positions)))
            issue=time_issue(r)
            if self.start.value()>self.end.value(): issue=tr('開始列は終了列以前にしてください。')
            if r.kind=='time_series' and not r.time_unit: issue=(tr(issue)+'\n' if issue else '')+tr('時間単位は未確認です。Excelの時間値をそのまま表示します。')
            self.issue.setText(tr(issue))
            available=self.view.currentData()=='channels' or (count>0 and self.show_overlay.isChecked())
            self.png.setEnabled(available and not self.exporting)
            self.video.setEnabled(available and r.kind=='time_series' and not time_issue(r) and self.start.value()<=self.end.value() and not self.exporting)
            self.play.setEnabled(r.kind=='time_series' and not time_issue(r) and self.start.value()<=self.end.value())
        else:
            self.chart.set_result(None); self.mapping_info.clear(); self.issue.clear(); self.png.setEnabled(False); self.video.setEnabled(False); self.play.setEnabled(False)
        if hasattr(self.window.surface,'result_glyphs'):
            show_points=spatial_options(r)['mode']!='surface'
            self.window.surface.result_glyphs.set_points(positions if show_points else [],rgb if show_points else [],radius=markers['radius_mm'],
                opacity=markers['opacity'],labels=labels if markers['labels'] and show_points else ())
            self.window.surface.result_surface.set_result(self.window.surface,r,
                max(0,self.metric_combo.currentIndex()),self.slider.value(),bool(active and r and self.show_overlay.isChecked()))
            self.window.surface.set_result_context(bool(active and r and self.show_overlay.isChecked()))
            self.window.surface.render()
        for panel in self.window.slice_panels():
            panel.canvas.result_points=positions; panel.canvas.result_colors=rgb
            panel.canvas.result_labels=labels;panel.canvas.result_options=markers
            panel.canvas.result_active=bool(active and r and self.show_overlay.isChecked());panel.canvas.update()
        if hasattr(self.window,'result_banner'):
            banner=self.window.result_banner; banner.setVisible(bool(active and r and self.view.currentData()!='channels'))
            banner.set_result(r,scene.contacts if scene else [],max(0,self.metric_combo.currentIndex()),self.slider.value())
        if hasattr(self,'transport'):self.transport.sync()

    def show_on_electrodes(self):
        r=self.selected()
        if not r:return
        if not len(resolved_positions(r,self.window.scene.contacts)[0]):
            self.open_mapping();return
        self.show_overlay.setChecked(True)
        self.view.setCurrentIndex(self.view.findData('brain'))
        self.refresh()

    def open_mapping(self):
        self.stop()
        from .result_mapping import MappingDialog
        self.close_mapping()
        self.mapping_dialog=MappingDialog(self); self.mapping_dialog.show()

    def jump(self,index):
        r=self.selected()
        if r:
            indices,positions=resolved_positions(r,self.window.scene.contacts)
            where=np.flatnonzero(indices==index)
            if len(where): self.window.select_world(positions[where[0]])

    def stop(self):
        self.timer.stop()
        if hasattr(self,'play'): self.play.setText(tr('再生'))
        if hasattr(self,'transport'):self.transport.sync()

    def toggle_play(self):
        if self.timer.isActive(): self.stop(); return
        r=self.selected()
        if not r or r.kind!='time_series' or time_issue(r): return
        a,b=frame_range(r)
        if not a<=self.slider.value()<b: self.slider.setValue(a)
        self.timer.start(round(1000/self.fps.value())); self.play.setText(tr('停止'))
        if hasattr(self,'transport'):self.transport.sync()

    def advance(self):
        r=self.selected()
        if not r: self.stop(); return
        _,b=frame_range(r)
        if self.slider.value()>=b: self.stop()
        else: self.slider.setValue(self.slider.value()+1)

    def export_png(self,path=None):
        self.stop(); r=self.selected()
        if not r: return
        if not path:
            directory=self.window.export_directory(); directory.mkdir(parents=True,exist_ok=True)
            path,_=QFileDialog.getSaveFileName(self,tr('解析の静止画を保存'),str(directory/'analysis_static.png'),'PNG (*.png)')
        if not path: return
        path=Path(path).with_suffix('.png')
        try:
            record=export_record(r,self.window.scene.contacts,self.metric_combo.currentIndex(),[self.slider.value()],view=self.view.currentData())
            save_png(path,export_frame(self),record); self.window.message(tr('解析の静止画と表示条件を保存しました。'))
        except Exception:
            from .window import log_error
            log_error(); self.window._job_failed(tr('静止画を保存できませんでした。保存先を確認してください。'))

    def export_video(self,path=None):
        self.stop(); r=self.selected()
        if not r or time_issue(r) or self.start.value()>self.end.value(): return
        if not path:
            directory=self.window.export_directory(); directory.mkdir(parents=True,exist_ok=True)
            path,_=QFileDialog.getSaveFileName(self,tr('解析の動画を保存'),str(directory/'analysis_time_series.mp4'),'MP4 (*.mp4)')
        if not path: return
        self.begin_video(Path(path).with_suffix('.mp4'))

    def begin_video(self,path,size=(1800,1120)):
        r=self.selected(); a,b=frame_range(r)
        if self.exporting: return
        if r.kind!='time_series' or self.start.value()>self.end.value(): raise ValueError('Invalid video range')
        if self.view.currentData()!='channels' and (not len(resolved_positions(r,self.window.scene.contacts)[0]) or not self.show_overlay.isChecked()):
            raise ValueError('No visible mapped channels')
        if time_issue(r): raise ValueError(time_issue(r))
        self._saved_frame=self.slider.value(); self._export_frames=list(range(a,b+1)); self._export_index=0
        self._export_size=size; self._export_error=None; self._cancelled=False; self._close_after_export=False
        self._export_record=export_record(r,self.window.scene.contacts,self.metric_combo.currentIndex(),self._export_frames,
                                         self.fps.value(),self.view.currentData())
        try: self._writer=VideoWriter(path,size,self.fps.value())
        except Exception:
            from .window import log_error
            log_error(); self.window._job_failed(tr('動画の保存を開始できませんでした。')); return
        self.exporting=True; self.window._set_busy(True)
        self.progress=QProgressDialog(tr('動画を書き出しています…'),tr('キャンセル'),0,len(self._export_frames),self.window)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal); self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False); self.progress.setAutoReset(False)
        self.progress.canceled.connect(self.cancel_export); self.progress.show(); QTimer.singleShot(0,self._export_step)

    def cancel_export(self): self._cancelled=True

    def _export_step(self):
        try:
            if self._cancelled:
                self._writer.abort(); self._finish_video(False); return
            if self._export_index==len(self._export_frames):
                self._writer.finish(self._export_record); self._finish_video(True); return
            self.slider.setValue(self._export_frames[self._export_index]); self.refresh()
            self._writer.write(export_frame(self,*self._export_size)); self._export_index+=1
            self.progress.setValue(self._export_index); QTimer.singleShot(0,self._export_step)
        except Exception as exc:
            self._export_error=str(exc)
            from .window import log_error
            log_error(); self._writer.abort(); self._finish_video(False)
            self.window._job_failed(tr('動画を保存できませんでした。保存先を確認してください。'))

    def _finish_video(self,success):
        self.progress.close(); self.progress.deleteLater(); self.exporting=False
        self.slider.setValue(self._saved_frame); self.window._set_busy(False); self.refresh()
        self.window.message(tr('動画と表示条件を保存しました。') if success else tr('動画出力を中止しました。'))
        if self._close_after_export: QTimer.singleShot(0,self.window.close)
