"""Postoperative MRI review, cavity exclusion and reversible deformation candidates."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QGroupBox,QLabel,QComboBox,
    QPushButton,QCheckBox,QSlider,QDoubleSpinBox,QFileDialog)

from .i18n import tr
from .segmentation import create_segment
from .postop_registration import correct_postop, export_postop, exclusion_preview
from .postop_ants import correct_postop_ants, ants_available, CorrectionCancelled


class PostopPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window=window; self.busy=False; self._deleted_exclusion_uid=None
        self._engine_limits={'ants':12.,'bspline':8.}; self._last_engine='ants'
        layout=QVBoxLayout(self); layout.setContentsMargins(0,3,10,0); layout.setSpacing(10)
        intro=QLabel(tr('術後MRIを基準の術前T1へ重ねます。剛体合わせを確認し、摘出腔などを除外して変形補正を作成します。'))
        intro.setWordWrap(True); layout.addWidget(intro)
        imports=QGroupBox(tr('術後MRIを追加')); rows=QVBoxLayout(imports)
        self.sequence=QComboBox()
        for name in ('T1','T1ce','T2','FLAIR','MRI'):
            self.sequence.addItem(tr({'T1ce':'造影T1','MRI':'その他のMRI'}.get(name,name)), 'postop-'+name)
        rows.addWidget(self.sequence)
        buttons=QHBoxLayout()
        folder=QPushButton(tr('フォルダを開く')); folder.clicked.connect(self.open_folder); buttons.addWidget(folder)
        file=QPushButton(tr('NIfTIファイル')); file.clicked.connect(self.open_file); buttons.addWidget(file)
        rows.addLayout(buttons); layout.addWidget(imports)
        from .workspace_ui import ImportFeedbackLabel
        self.feedback=ImportFeedbackLabel(); self.feedback.setWordWrap(True)
        self.feedback.setTextFormat(Qt.TextFormat.PlainText); self.feedback.setObjectName('importFeedback')
        self.feedback.hide(); layout.addWidget(self.feedback)
        self.images=QComboBox(); self.images.setObjectName('imageNames'); layout.addWidget(self.images)
        self.images.currentIndexChanged.connect(self.select_image)
        self.stage=QComboBox(); self.stage.setObjectName('imageNames'); layout.addWidget(self.stage)
        self.stage.currentIndexChanged.connect(self.select_stage)
        view=QGroupBox(tr('術前・術後を比較')); view_rows=QVBoxLayout(view)
        self.display_mode=QComboBox()
        for label,key in (('重ね合わせ','overlay'),('市松模様で照合','checker'),('術前T1のみ','preop'),('術後MRIのみ','postop')):
            self.display_mode.addItem(tr(label),key)
        self.display_mode.currentIndexChanged.connect(self.apply_view); view_rows.addWidget(self.display_mode)
        row=QHBoxLayout(); row.addWidget(QLabel(tr('術後MRIの不透明度')))
        self.opacity=QSlider(Qt.Orientation.Horizontal); self.opacity.setRange(0,100); self.opacity.setValue(50)
        self.opacity.valueChanged.connect(self.apply_view); row.addWidget(self.opacity,1); view_rows.addLayout(row)
        self.window_spin=self.spin(view_rows,'表示幅（コントラスト）',.001,1e9,1000,25)
        self.level_spin=self.spin(view_rows,'中心値（明るさ）',-1e9,1e9,500,25)
        self.window_spin.valueChanged.connect(self.change_contrast); self.level_spin.valueChanged.connect(self.change_contrast)
        note=QLabel(tr('右ドラッグ：術後MRIの濃淡。3Dの脳表と解剖ラベルは術前の形です。'))
        note.setWordWrap(True); note.setObjectName('muted'); view_rows.addWidget(note)
        self.review=QPushButton(tr('この重なりを確認済みにする')); self.review.clicked.connect(self.mark_reviewed)
        view_rows.addWidget(self.review)
        layout.addWidget(view)
        masks=QGroupBox(tr('摘出腔・変化した領域を除外')); mask_rows=QVBoxLayout(masks)
        self.exclusion=QComboBox(); self.exclusion.setObjectName('imageNames'); mask_rows.addWidget(self.exclusion)
        self.exclusion.currentIndexChanged.connect(self.select_mask)
        self.create_mask=QPushButton(tr('新しい除外範囲を作成'))
        self.create_mask.clicked.connect(self.create_exclusion); mask_rows.addWidget(self.create_mask)
        edit_row=QHBoxLayout()
        self.add_mask=QPushButton(tr('選択範囲に追加'))
        self.add_mask.clicked.connect(lambda:self.edit_exclusion('contour')); edit_row.addWidget(self.add_mask)
        self.subtract_mask=QPushButton(tr('選択範囲から消去'))
        self.subtract_mask.clicked.connect(lambda:self.edit_exclusion('contour_erase')); edit_row.addWidget(self.subtract_mask)
        mask_rows.addLayout(edit_row)
        self.delete_mask=QPushButton(tr('選択した除外範囲を削除'))
        self.delete_mask.clicked.connect(self.delete_exclusion); mask_rows.addWidget(self.delete_mask)
        self.no_cavity=QCheckBox(tr('切除なし・除外範囲なし')); mask_rows.addWidget(self.no_cavity)
        self.no_cavity.toggled.connect(self.exclusion_enabled_changed)
        self.draw_mode=QComboBox()
        for label,key in (('位置を確認','navigate'),('手描きで囲んで追加','contour'),('手描きで囲んで削除','contour_erase'),
                          ('ブラシで追加','paint'),('消しゴム','erase')):
            self.draw_mode.addItem(tr(label),key)
        self.draw_mode.currentIndexChanged.connect(self.set_draw_mode); mask_rows.addWidget(self.draw_mode)
        self.radius=self.spin(mask_rows,'半径 mm',.5,20,2.5,.5)
        self.radius.valueChanged.connect(self.update_tools)
        self.thickness=self.spin(mask_rows,'囲んだROIの厚み mm',.5,10,3,.5)
        self.undo_mask=QPushButton(tr('編集を1つ戻す')); self.undo_mask.clicked.connect(self.undo); mask_rows.addWidget(self.undo_mask)
        mask_note=QLabel(tr('追加・消去は上で選んだ1つの範囲を編集します。複数の断面で囲み、摘出腔など全体を覆ってください。補正には選択中の範囲を使います。'))
        mask_note.setWordWrap(True); mask_note.setObjectName('muted'); mask_rows.addWidget(mask_note)
        self.margin=self.spin(mask_rows,'除外範囲の余白 mm',0,10,3,.5)
        self.margin.valueChanged.connect(self.clear_mask_preview)
        self.preview_button=QPushButton(tr('余白を含む除外範囲を確認')); self.preview_button.setCheckable(True)
        self.preview_button.clicked.connect(self.preview_mask); mask_rows.addWidget(self.preview_button)
        layout.addWidget(masks)
        correction=QGroupBox(tr('残存脳を使って変形補正')); corr=QVBoxLayout(correction)
        self.engine=QComboBox()
        self.engine.addItem(tr('ANTsPy：マスク付き補正'),'ants')
        self.engine.addItem(tr('従来の補正：B-spline'),'bspline')
        corr.addWidget(self.engine)
        self.residual=QComboBox(); self.residual.setObjectName('imageNames'); corr.addWidget(self.residual)
        self.max_shift=self.spin(corr,'補正量の確認上限 mm',1,15,12,1)
        self.grid=self.spin(corr,'変形の滑らかさ mm',25,80,40,5)
        self.grid.setToolTip(tr('大きい値ほど広い範囲で滑らかに補正します。'))
        self.ants_options=QWidget(); ants_rows=QVBoxLayout(self.ants_options); ants_rows.setContentsMargins(0,0,0,0)
        self.metric=QComboBox()
        self.metric.addItem(tr('T1同士の比較'),'CC'); self.metric.addItem(tr('異なる撮像条件の比較'),'mattes')
        ants_rows.addWidget(self.metric)
        self.resolution=QComboBox()
        self.resolution.addItem(tr('標準：2 mmで計算'),2.); self.resolution.addItem(tr('詳細：1.5 mmで計算'),1.5)
        ants_rows.addWidget(self.resolution); self.resolution.currentIndexChanged.connect(self.clear_mask_preview)
        self.n4=QCheckBox(tr('濃淡むらを補正（N4）')); self.n4.setChecked(True); ants_rows.addWidget(self.n4)
        ants_note=QLabel(tr('術後FreeSurferは不要です。除外範囲と残存組織の画像を使います。確認上限を超えた結果は追加しません。'))
        ants_note.setWordWrap(True); ants_note.setObjectName('muted'); ants_rows.addWidget(ants_note)
        self.ants_status=QLabel(); self.ants_status.setWordWrap(True); ants_rows.addWidget(self.ants_status)
        corr.addWidget(self.ants_options)
        self.engine.currentIndexChanged.connect(self.engine_changed)
        self.compute=QPushButton(tr('変形補正の候補を作成')); self.compute.setObjectName('primary')
        self.compute.clicked.connect(self.correct); corr.addWidget(self.compute)
        note=QLabel(tr('先に剛体合わせを確認済みにしてください。欠損した組織は復元しません。補正後は残存脳表・脳室・深部構造を確認します。'))
        note.setObjectName('muted'); note.setWordWrap(True); corr.addWidget(note); layout.addWidget(correction)
        self.info=QLabel(); self.info.setWordWrap(True); self.info.setObjectName('muted'); layout.addWidget(self.info)
        self.revert=QPushButton(tr('剛体合わせに戻して表示')); self.revert.clicked.connect(self.show_rigid); layout.addWidget(self.revert)
        self.export=QPushButton(tr('術後MRI・変換を出力')); self.export.clicked.connect(self.export_current); layout.addWidget(self.export)
        layout.addStretch(); self.bind_scene()

    @staticmethod
    def spin(layout,label,minimum,maximum,value,step):
        caption=QLabel(tr(label)); caption.setWordWrap(True); layout.addWidget(caption)
        spin=QDoubleSpinBox(); spin.setRange(minimum,maximum); spin.setDecimals(2); spin.setValue(value)
        spin.setSingleStep(step); spin.setKeyboardTracking(False); spin.setMinimumWidth(112)
        spin.caption=caption
        layout.addWidget(spin); return spin

    def layer(self, uid=None):
        scene=self.window.scene
        if scene is None: return None
        uid=uid or self.stage.currentData()
        return next((layer for layer in scene.extra_mris if layer.uid==uid),None)

    def base(self): return self.layer(self.images.currentData()) if self.images.currentData() else None

    def segment(self, combo):
        if self.window.scene is None:return None
        return next((s for s in self.window.scene.segmentations if s.uid==combo.currentData()),None)

    @property
    def manual_uid(self): return self.exclusion.currentData() if not self.no_cavity.isChecked() else None

    def open_folder(self):
        folder=QFileDialog.getExistingDirectory(self,tr('術後MRIの撮像フォルダ'),str(self.window.project_root/'sample_images'))
        if folder:self.window.open_image_folder(Path(folder),self.sequence.currentData())

    def open_file(self):
        path,_=QFileDialog.getOpenFileName(self,tr('術後MRIを選択'),str(self.window.project_root/'sample_images'),'NIfTI (*.nii *.nii.gz)')
        if path:self.window.prepare_image_import(Path(path),self.sequence.currentData())

    def bind_scene(self):
        self._deleted_exclusion_uid=None
        self.draw_mode.setCurrentIndex(0)
        self.restore(self.window.scene.view_state.get('postop',{}) if self.window.scene else {})

    def refresh(self, uid=None):
        chosen=uid or self.images.currentData(); stage=self.stage.currentData()
        self.busy=True; self.images.clear(); self.images.addItem(tr('術後MRIを選択'),None)
        if self.window.scene:
            for layer in self.window.scene.extra_mris:
                if layer.deformation is None and (layer.sequence.startswith('postop-') or layer.sequence in ('T1','T1ce','T2','FLAIR','MRI','SWI','DWI')):
                    self.images.addItem(layer.name,layer.uid)
        self.images.setCurrentIndex(max(0,self.images.findData(chosen)))
        self.busy=False; self.refresh_stages(stage); self.refresh_masks(); self.update_controls()

    def refresh_stages(self, chosen=None):
        self.busy=True; self.stage.clear(); base=self.base()
        if base:
            self.stage.addItem(tr('剛体合わせ'),base.uid)
            number=0
            for layer in self.window.scene.extra_mris:
                if layer.deformation is not None and layer.quality.get('rigid_source_uid')==base.uid:
                    number+=1
                    label=tr('変形補正（確認済み）') if layer.quality.get('review_status')=='visually_reviewed' else tr('変形補正（確認前）')
                    method='ANTsPy' if layer.quality.get('engine')=='ants' else 'B-spline'
                    self.stage.addItem(method+' · '+label+f' {number}',layer.uid)
            self.stage.setCurrentIndex(max(0,self.stage.findData(chosen)))
        self.busy=False

    def refresh_masks(self):
        self.busy=True
        for combo,title in ((self.exclusion,'除外範囲を選択'),(self.residual,'FreeSurferの脳領域（自動選択）')):
            uid=combo.currentData(); combo.clear(); combo.addItem(tr(title),None)
            if self.window.scene:
                for region in self.window.scene.segmentations: combo.addItem(region.name,region.uid)
            combo.setCurrentIndex(max(0,combo.findData(uid)))
        self.busy=False

    def select_image(self,*_):
        if self.busy:return
        self.clear_mask_preview()
        self.draw_mode.setCurrentIndex(0)
        self.exclusion.setCurrentIndex(0); self.residual.setCurrentIndex(0); self.no_cavity.setChecked(False)
        self.refresh_stages(); self.update_controls(); self.apply_view()
        if self.base():self.metric.setCurrentIndex(0 if self.base().sequence in ('postop-T1','T1','postop-T1ce','T1ce') else 1)

    def select_stage(self,*_):
        if self.busy:return
        self.clear_mask_preview()
        self.draw_mode.setCurrentIndex(0); self.sync_contrast(); self.update_controls(); self.apply_view()

    def select_mask(self,*_):
        if self.busy:return
        self.window.segmentation.finish_stroke()
        self._deleted_exclusion_uid=None
        self.clear_mask_preview()
        self.draw_mode.setCurrentIndex(0)
        self.mark_exclusion_role()
        if self.manual_uid:self.window.segmentation.refresh_list(self.manual_uid)
        self.update_controls()
        self.window.update_layers()

    def mark_exclusion_role(self):
        region=self.segment(self.exclusion); base=self.base()
        if region and base:
            region.provenance['postop_exclusion_for']=sorted(set(region.provenance.get('postop_exclusion_for',[]))|{base.uid})

    def exclusion_enabled_changed(self,*_):
        if self.busy:return
        self.update_controls(); self.update_tools(); self.window.update_layers()

    def sync_contrast(self):
        layer=self.layer()
        if layer is None:return
        for spin,value in ((self.window_spin,layer.window),(self.level_spin,layer.level)):
            spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)

    def change_contrast(self,*_):
        layer=self.layer()
        if self.busy or layer is None:return
        layer.window=self.window_spin.value(); layer.level=self.level_spin.value()
        self.window.select_extra(); self.window.update_layers()

    def apply_view(self,*_):
        window=self.window; layer=self.layer()
        if self.busy or layer is None or window._restoring or window.workflow_tabs.currentIndex()!=3:return
        mode=self.display_mode.currentData()
        window._restoring=True
        try:
            for uid,row in window.image_layers.rows.items():
                row.check.setChecked((uid=='mri' and mode!='postop') or (uid==layer.uid and mode!='preop'))
                if uid=='mri':row.slider.setValue(100)
                if uid==layer.uid:row.slider.setValue(self.opacity.value() if mode=='overlay' else 100)
            window.extra_styles[layer.uid]='checker' if mode=='checker' else 'full'
            window.extra_combo.setCurrentIndex(window.extra_combo.findData(layer.uid))
            window.window_target.setCurrentIndex(window.window_target.findData('mri' if mode=='preop' else layer.uid))
            window.select_extra(); self.sync_contrast()
        finally:window._restoring=False
        window.update_layers()

    def show_rigid(self):
        base=self.base()
        if base:self.stage.setCurrentIndex(self.stage.findData(base.uid))

    def create_exclusion(self):
        base=self.base()
        if base is None or self.window.worker is not None:return
        self.draw_mode.setCurrentIndex(0); self.show_rigid()
        segment=create_segment(self.window.scene,'manual',tr('術後の除外範囲'))
        segment.visible_3d=False
        segment.provenance.update(postop_rigid_uid=base.uid,role='postop_registration_exclusion')
        self.window.segmentation.add_segments([segment]); self.refresh_masks()
        self.no_cavity.setChecked(False)
        self.exclusion.setCurrentIndex(self.exclusion.findData(segment.uid))
        self.display_mode.setCurrentIndex(self.display_mode.findData('postop'))
        self.draw_mode.setCurrentIndex(self.draw_mode.findData('contour'))
        self.window.update_layers(); self.update_controls()

    def edit_exclusion(self,mode):
        region=self.segment(self.exclusion)
        if region is None or self.base() is None or self.window.worker is not None:return
        self.no_cavity.setChecked(False); region.visible_2d=True
        self.window.segmentation.refresh_list(region.uid)
        self.draw_mode.setCurrentIndex(self.draw_mode.findData(mode))
        self.set_draw_mode(); self.window.update_layers()

    def delete_exclusion(self):
        region=self.segment(self.exclusion)
        if region is None or self.window.worker is not None:return
        self.clear_mask_preview(); self.draw_mode.setCurrentIndex(0)
        self.window.segmentation.delete_selected(uid=region.uid)
        self.refresh_masks(); self.exclusion.setCurrentIndex(0)
        self._deleted_exclusion_uid=region.uid
        self.update_controls(); self.window.update_layers()

    def set_draw_mode(self,*_):
        if self.busy:return
        requested=self.draw_mode.currentData()
        if requested!='navigate':
            self.clear_mask_preview()
            # Cavity masks are always drawn on the RIGID image, never a warped preview.
            base=self.base()
            if base:
                self.busy=True; self.stage.setCurrentIndex(self.stage.findData(base.uid)); self.busy=False
            self.apply_view()
        self.update_tools()

    def update_tools(self,*_):
        if not hasattr(self.window,'segmentation'):return
        if self.manual_uid and self.window.workflow_tabs.currentIndex()==3:
            selected=self.window.segmentation.selected()
            if selected is None or selected.uid!=self.manual_uid:self.window.segmentation.refresh_list(self.manual_uid)
            self.window.segmentation.radius.setValue(self.radius.value())
        self.window.segmentation.update_tools()

    def undo(self):
        self.clear_mask_preview()
        uid=self._deleted_exclusion_uid or self.manual_uid
        if not uid:return
        self.window.segmentation.undo(uid=uid)
        self.refresh_masks()
        self.exclusion.setCurrentIndex(max(0,self.exclusion.findData(uid)))
        self._deleted_exclusion_uid=None
        self.update_tools(); self.update_controls(); self.window.update_layers()

    def engine_changed(self,*_):
        if self.busy:return
        self._engine_limits[self._last_engine]=self.max_shift.value()
        self._last_engine=self.engine.currentData()
        self.max_shift.setValue(self._engine_limits[self._last_engine])
        self.clear_mask_preview(); self.update_controls()

    def clear_mask_preview(self,*_):
        if hasattr(self,'preview_button'):
            self.preview_button.setChecked(False)
        scene=self.window.scene
        if scene and scene.segmentation_preview and scene.segmentation_preview.provenance.get('role')=='postop_exclusion_preview':
            scene.segmentation_preview=None; self.window.update_layers()

    def preview_mask(self,checked):
        if not checked:
            self.clear_mask_preview(); return
        window=self.window; scene=window.scene; region=self.segment(self.exclusion)
        if scene is None or region is None or window.worker is not None:
            self.preview_button.setChecked(False); return
        self.draw_mode.setCurrentIndex(0); self.show_rigid()
        margin=self.margin.value(); spacing=self.resolution.currentData() if self.engine.currentData()=='ants' else 2.
        def complete(preview):
            scene.segmentation_preview=preview; self.preview_button.setChecked(True); window.update_layers()
            window.message(tr('ピンク色は余白を含む除外範囲です。設定変更・描画開始で確認表示を解除します。'))
        window.run_job(lambda _:exclusion_preview(scene,region,margin,spacing),complete,
                       tr('余白を含む除外範囲を準備しています…'),
                       on_error=lambda message:(self.clear_mask_preview(),window._job_failed(message)))

    def update_controls(self,*_):
        if self.busy:return
        base=self.base(); layer=self.layer()
        use_ants=self.engine.currentData()=='ants'
        self.ants_options.setVisible(use_ants); self.grid.setVisible(not use_ants); self.grid.caption.setVisible(not use_ants)
        ready=ants_available()
        self.ants_status.setText(tr('ANTsPy：準備済み') if ready else tr('ANTsPyの準備が必要です。作業フォルダのSetup_Postop_ANTs.cmdを実行してください。'))
        self.preview_button.setEnabled(base is not None and bool(self.manual_uid))
        self.create_mask.setEnabled(base is not None)
        editable=base is not None and self.segment(self.exclusion) is not None
        self.add_mask.setEnabled(editable); self.subtract_mask.setEnabled(editable)
        self.delete_mask.setEnabled(self.segment(self.exclusion) is not None)
        self.undo_mask.setEnabled(self.window.segmentation.can_undo(self._deleted_exclusion_uid or self.manual_uid))
        if self.no_cavity.isChecked():
            self.draw_mode.setCurrentIndex(0); self.clear_mask_preview()
        self.draw_mode.setEnabled(base is not None and bool(self.manual_uid))
        reviewed=base is not None and base.quality.get('review_status')=='visually_reviewed'
        self.compute.setEnabled(reviewed and (bool(self.manual_uid) or self.no_cavity.isChecked()) and (ready or not use_ants))
        self.review.setEnabled(layer is not None); self.revert.setEnabled(base is not None); self.export.setEnabled(layer is not None)
        self.sync_contrast()
        if layer is None:
            self.info.setText(tr('術後MRIを追加するか、既存のMRIを選択してください。'));return
        status=tr('確認済み') if layer.quality.get('review_status')=='visually_reviewed' else tr('目視確認前')
        text=tr('状態：{status}').format(status=status)
        qc=layer.quality.get('numerical_qc')
        if qc:
            text+='\n'+('ANTsPy' if layer.quality.get('engine')=='ants' else 'B-spline')
            text+='\n'+tr('補正量 最大 {maximum} mm / 95% {p95} mm\n折り返し {folds} voxel（{spacing} mm格子）').format(
                maximum=f"{qc['displacement_max_mm']:.2f}",p95=f"{qc['displacement_p95_mm']:.2f}",folds=qc['folding_voxels'],
                spacing=f"{layer.quality.get('working_spacing_mm',2.):g}")
            text+='\n'+tr('数値上の確認であり、解剖学的な位置精度は別途確認が必要です。')
        self.info.setText(text)

    def mark_reviewed(self):
        layer=self.layer()
        if layer is None:return
        layer.quality=dict(layer.quality,review_status='visually_reviewed',reviewed_at=datetime.now(timezone.utc).isoformat())
        self.refresh_stages(layer.uid); self.update_controls()
        self.window.message(tr('確認状態を更新しました。「保存」で患者フォルダに記録できます。'))

    def correct(self):
        window=self.window; base=self.base()
        if base is None or window.worker is not None or not self.compute.isEnabled():return
        self.clear_mask_preview()
        self.draw_mode.setCurrentIndex(0); window.segmentation.finish_stroke()
        scene=window.scene; state=window.view_state(); patient_id=window.patient_id
        exclusion=None if self.no_cavity.isChecked() else self.segment(self.exclusion)
        residual=self.segment(self.residual)
        use_ants=self.engine.currentData()=='ants'; cancel=Event(); cancel_button=None
        options=dict(margin_mm=self.margin.value(),max_shift_mm=self.max_shift.value(),no_cavity=self.no_cavity.isChecked())
        if use_ants:
            options.update(metric=self.metric.currentData(),spacing_mm=self.resolution.currentData(),n4=self.n4.isChecked(),cancel=cancel)
            cancel_button=QPushButton(tr('補正の計算を中止'),window)
            cancel_button.setObjectName('postopCancel')
            def request_cancel():
                cancel.set(); cancel_button.setEnabled(False); window._update_job_progress(tr('補正の計算を中止しています…'))
            cancel_button.clicked.connect(request_cancel); window.statusBar().addPermanentWidget(cancel_button)
        else: options['grid_mm']=self.grid.value()
        def cleanup():
            if cancel_button is not None:
                window.statusBar().removeWidget(cancel_button); cancel_button.deleteLater()
        def operation(progress):
            try:
                function=correct_postop_ants if use_ants else correct_postop
                candidate=function(scene,base,exclusion,residual,progress=progress,**options)
                if cancel.is_set():return None
            except CorrectionCancelled:return None
            if patient_id:
                progress('変更前の作業を保存しています…'); window.patient_store.save(patient_id,scene,state)
            result=replace(scene,extra_mris=[*scene.extra_mris,candidate],view_state=dict(state))
            result.view_state['postop']={**state.get('postop',{}),'image':base.uid,'stage':candidate.uid,'display':'checker'}
            result.view_state.update(extra_selected=candidate.uid,window_target=candidate.uid,
                extra_styles={**state.get('extra_styles',{}),candidate.uid:'checker'},
                image_layers={key:{'visible':key in ('mri',candidate.uid),'opacity':1.}
                              for key in ('mri','ct',*[item.uid for item in result.extra_mris])})
            if patient_id:
                progress('術後MRIの補正候補を保存しています…'); window.patient_store.save(patient_id,result,result.view_state)
            return result
        def complete(result):
            cleanup()
            if result is None:
                window.message(tr('補正の計算を中止しました。画像と既存の候補は保持されています。')); return
            window.install_scene(result); window.workflow_tabs.setCurrentIndex(3)
            self.restore(result.view_state['postop']); self.apply_view()
            window.message(tr('補正候補を作成しました。剛体合わせと切り替えて、残存脳の重なりを確認してください。'))
        def failed(message):
            cleanup(); window._job_failed(message)
        window.run_job(operation,complete,tr('術後MRIの変形補正を計算しています…'),on_error=failed)
        if cancel_button is not None:
            window.worker.progress.connect(lambda stage:cancel_button.setEnabled(not cancel.is_set() and '保存' not in stage))

    def export_current(self):
        scene=self.window.scene; layer=self.layer()
        if layer is None:return
        directory=self.window.export_directory(); directory.mkdir(exist_ok=True)
        parent=QFileDialog.getExistingDirectory(self,tr('術後MRIの出力先'),str(directory))
        if not parent:return
        self.window.run_job(lambda _:export_postop(scene,layer,parent),
            lambda _:self.window.message(tr('術後MRI・変換・補正条件を新しいフォルダに出力しました。')),
            tr('術後MRIを出力しています…'))

    def state(self):
        limits={**self._engine_limits,self.engine.currentData():self.max_shift.value()}
        return {'image':self.images.currentData(),'stage':self.stage.currentData(),
                'exclusion':self.exclusion.currentData(),'residual':self.residual.currentData(),
                'no_cavity':self.no_cavity.isChecked(),'display':self.display_mode.currentData(),
                'opacity':self.opacity.value(),'margin':self.margin.value(),'maximum':self.max_shift.value(),
                'grid':self.grid.value(),'radius':self.radius.value(),'thickness':self.thickness.value(),
                'engine':self.engine.currentData(),'metric':self.metric.currentData(),
                'resolution':self.resolution.currentData(),'n4':self.n4.isChecked(),'limits':limits}

    def restore(self,state):
        self.refresh(state.get('image')); self.refresh_stages(state.get('stage'))
        self.busy=True
        self._engine_limits={'ants':12.,'bspline':8.,**state.get('limits',{})}
        if 'maximum' in state:
            self._engine_limits[state.get('engine','bspline')]=float(state['maximum'])
        for combo,key in ((self.exclusion,'exclusion'),(self.residual,'residual'),(self.display_mode,'display')):
            combo.setCurrentIndex(max(0,combo.findData(state.get(key))))
        for combo,key,default in ((self.engine,'engine','ants'),(self.metric,'metric','CC'),(self.resolution,'resolution',2.)):
            combo.setCurrentIndex(max(0,combo.findData(state.get(key,default))))
        self.n4.setChecked(bool(state.get('n4',True)))
        self._last_engine=self.engine.currentData()
        self.max_shift.setValue(self._engine_limits[self._last_engine])
        self.no_cavity.setChecked(bool(state.get('no_cavity',False))); self.opacity.setValue(int(state.get('opacity',50)))
        for spin,key,default in ((self.margin,'margin',3),(self.grid,'grid',40),
                                 (self.radius,'radius',2.5),(self.thickness,'thickness',3)):
            spin.setValue(float(state.get(key,default)))
        self.busy=False; self.mark_exclusion_role(); self.update_controls()
