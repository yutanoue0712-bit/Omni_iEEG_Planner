"""DTI workflow controls shared with the existing three slices and 3D view."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QGroupBox,QLabel,QComboBox,
    QPushButton,QCheckBox,QDoubleSpinBox,QSpinBox,QListWidget,QListWidgetItem,QLineEdit,
    QSlider,QFileDialog,QMenu)
from .i18n import tr
from .diffusion import track_roi
from .diffusion_storage import export_tract,export_scalar


class ROIControls(QGroupBox):
    def __init__(self,panel,index):
        super().__init__(f'ROI {index}')
        self.panel=panel; self.index=index; self.center=None
        layout=QVBoxLayout(self)
        self.source=QComboBox(); self.source.setObjectName('dataNames')
        layout.addWidget(self.source)
        self.capture=QPushButton(tr('断面の選択位置をROI中心にする'))
        self.capture.clicked.connect(self.set_center);layout.addWidget(self.capture)
        row=QHBoxLayout();self.radius_label=QLabel(tr('球の半径 mm'));row.addWidget(self.radius_label)
        self.radius=QDoubleSpinBox();self.radius.setRange(1,50);self.radius.setValue(6);self.radius.setDecimals(1)
        row.addWidget(self.radius);layout.addLayout(row)
        self.info=QLabel(tr('断面上をクリックして中心を指定'));self.info.setWordWrap(True)
        self.info.setObjectName('muted');layout.addWidget(self.info)
        self.source.currentIndexChanged.connect(self.update)
        self.radius.valueChanged.connect(self.update)
        self.manual=QPushButton(tr('手描きROIを新規作成'))
        self.manual.clicked.connect(lambda:self.panel.edit_roi(self,True));layout.addWidget(self.manual)
        self.edit=QPushButton(tr('このROIを手描きで編集'))
        self.edit.clicked.connect(lambda:self.panel.edit_roi(self,False));layout.addWidget(self.edit)
        self.anatomy=QPushButton(tr('FreeSurferの解剖ラベルをROIにする'))
        self.anatomy.clicked.connect(lambda:self.panel.choose_anatomy(self));layout.addWidget(self.anatomy)

    def bind(self):
        value=self.source.currentData(); self.source.blockSignals(True);self.source.clear()
        self.source.addItem(tr('球形ROI（位置を指定）'),'sphere')
        if self.panel.window.scene:
            for segment in self.panel.window.scene.segmentations:self.source.addItem(segment.name,segment.uid)
        found=self.source.findData(value)
        if value and value!='sphere' and found<0:self.center=None
        self.source.setCurrentIndex(max(0,found))
        self.source.blockSignals(False);self.update()

    def set_center(self):
        window=self.panel.window
        if window.scene is None:return
        self.center=window.scene.world(window.ijk).tolist()
        self.update()

    def value(self):
        uid=self.source.currentData()
        if uid=='sphere':
            return {'kind':'sphere','center':self.center,'radius':self.radius.value()} if self.center is not None else None
        return {'kind':'segment','uid':uid} if uid else None

    def restore(self,item):
        self.center=item.get('center') if item else None
        uid=item.get('uid','sphere') if item else 'sphere'
        self.source.setCurrentIndex(max(0,self.source.findData(uid)))
        self.radius.setValue(float(item.get('radius',6)) if item else 6)
        self.update()

    def update(self,*_):
        sphere=self.source.currentData()=='sphere'
        self.capture.setEnabled(sphere);self.radius.setEnabled(sphere)
        self.capture.setVisible(sphere);self.radius.setVisible(sphere);self.radius_label.setVisible(sphere)
        if hasattr(self,'edit'):self.edit.setEnabled(not sphere and bool(self.source.currentData()))
        if sphere:
            text=tr('断面上をクリックして中心を指定') if self.center is None else 'RAS / mm: '+', '.join(f'{v:.1f}' for v in self.center)
        else:text=tr('セグメンテーションで編集した領域を使用')
        self.info.setText(text)
        if hasattr(self.panel,'roi2'):self.panel.update_roi_preview()


class DiffusionPanel(QWidget):
    def __init__(self,window):
        super().__init__()
        self.window=window;self._binding=False;self.manual_uid=None
        layout=QVBoxLayout(self);layout.setContentsMargins(0,3,10,0);layout.setSpacing(8)
        info=QLabel(tr('拡散MRIからFA画像と線維候補を作成します。ROIは球形、または作成済みの領域から選べます。'))
        info.setWordWrap(True);info.setObjectName('muted');layout.addWidget(info)
        buttons=QHBoxLayout()
        self.folder_button=QPushButton(tr('DTIフォルダ'));self.folder_button.clicked.connect(self.choose_folder)
        self.file_button=QPushButton(tr('4D NIfTI'));self.file_button.clicked.connect(self.choose_file)
        buttons.addWidget(self.folder_button);buttons.addWidget(self.file_button);layout.addLayout(buttons)
        from .workspace_ui import ImportFeedbackLabel
        self.feedback=ImportFeedbackLabel();self.feedback.setWordWrap(True)
        self.feedback.setObjectName('importFeedback');self.feedback.hide();layout.addWidget(self.feedback)
        self.correct_motion=QCheckBox(tr('体動を補正して読み込む'));self.correct_motion.setChecked(True);layout.addWidget(self.correct_motion)
        self.models=QComboBox();self.models.setObjectName('dataNames');self.models.currentIndexChanged.connect(self.select_model);layout.addWidget(self.models)
        self.info=QLabel();self.info.setWordWrap(True);self.info.setObjectName('muted');layout.addWidget(self.info)
        maps=QHBoxLayout()
        self.map=QComboBox()
        for text,key in [('方向カラーFA','color'),('FA','fa'),('b0','b0'),('MD','md')]:self.map.addItem(tr(text),key)
        self.map.currentIndexChanged.connect(self.select_model)
        maps.addWidget(self.map,1)
        self.overlay_button=QPushButton(tr('2Dに重ねる'));self.overlay_button.clicked.connect(self.show_map);maps.addWidget(self.overlay_button)
        layout.addLayout(maps)
        self.compare_button=QPushButton(tr('b0とT1の位置合わせを確認'))
        self.compare_button.clicked.connect(self.compare);layout.addWidget(self.compare_button)
        self.mode=QComboBox()
        self.mode.addItem(tr('1 ROI：起点から追跡'),1);self.mode.addItem(tr('2 ROI：両方を通過'),2)
        self.mode.addItem(tr('全体プレビュー（開始点を間引く）'),0)
        self.mode.currentIndexChanged.connect(self.mode_changed);layout.addWidget(self.mode)
        self.roi1=ROIControls(self,1);self.roi2=ROIControls(self,2)
        layout.addWidget(self.roi1);layout.addWidget(self.roi2)
        manual=QGroupBox(tr('手描きROIの編集')); manual_layout=QVBoxLayout(manual)
        self.manual_info=QLabel(tr('ROI 1 / ROI 2で作成または編集を選択'));self.manual_info.setWordWrap(True)
        manual_layout.addWidget(self.manual_info)
        self.draw_mode=QComboBox()
        for label,key in [('位置を確認','navigate'),('手描きで囲んで追加','contour'),
                          ('手描きで囲んで削除','contour_erase'),('ブラシで追加','paint'),('消しゴム','erase')]:
            self.draw_mode.addItem(tr(label),key)
        self.draw_mode.currentIndexChanged.connect(self.manual_mode_changed)
        manual_layout.addWidget(self.draw_mode)
        row=QHBoxLayout();row.addWidget(QLabel(tr('囲んだROIの厚み mm')))
        self.roi_thickness=QDoubleSpinBox();self.roi_thickness.setRange(.5,30);self.roi_thickness.setValue(4)
        row.addWidget(self.roi_thickness);manual_layout.addLayout(row)
        undo=QPushButton(tr('編集を1つ戻す'));undo.clicked.connect(window.segmentation.undo);manual_layout.addWidget(undo)
        guide=QLabel(tr('左ドラッグで囲み、離すと確定。別の断面にも追加できます。Escで描画を取消。ブラシ半径はセグメンテーションと共通です。'))
        guide.setWordWrap(True);guide.setObjectName('muted');manual_layout.addWidget(guide);layout.addWidget(manual)
        settings=QGroupBox(tr('線維追跡の条件'));form=QVBoxLayout(settings)
        self.parameters={}
        for key,label,lo,hi,value,step in [
            ('fa_threshold','FAしきい値',.01,.99,.2,.01),('max_angle','最大角度 °',5,80,35,5),
            ('min_length','最短の長さ mm',0,200,10,5),('max_length','最長の長さ mm',10,500,200,10),
            ('step_size','追跡の刻み mm',.2,2,.5,.1)]:
            row=QHBoxLayout();row.addWidget(QLabel(tr(label)));spin=QDoubleSpinBox()
            spin.setRange(lo,hi);spin.setValue(value);spin.setSingleStep(step);spin.setDecimals(2 if key=='fa_threshold' else 1)
            row.addWidget(spin);form.addLayout(row);self.parameters[key]=spin
        row=QHBoxLayout();row.addWidget(QLabel(tr('開始点の上限')))
        self.max_seeds=QSpinBox();self.max_seeds.setRange(50,20000);self.max_seeds.setValue(3000);self.max_seeds.setSingleStep(500)
        row.addWidget(self.max_seeds);form.addLayout(row)
        hint=QLabel(tr('FAは0〜1。低くすると追跡範囲が広がる一方、不要な線維も増えます。値は調整用の初期値です。'))
        hint.setWordWrap(True);hint.setObjectName('muted');form.addWidget(hint);layout.addWidget(settings)
        self.name=QLineEdit();self.name.setPlaceholderText(tr('線維束の名前'));self.name.setProperty('userText',True);self.name.setMaxLength(100);layout.addWidget(self.name)
        self.track_button=QPushButton(tr('線維を抽出'));self.track_button.setObjectName('primary')
        self.track_button.clicked.connect(self.track);layout.addWidget(self.track_button)
        layout.addWidget(QLabel(tr('作成した線維束（チェック＝2D・3D表示）')))
        self.bundles=QListWidget();self.bundles.setMaximumHeight(150);self.bundles.setMinimumHeight(90)
        self.bundles.itemChanged.connect(self.toggle);self.bundles.currentItemChanged.connect(self.select_bundle);layout.addWidget(self.bundles)
        self.show_2d=QCheckBox(tr('線維を2D断面に表示'));self.show_2d.toggled.connect(self.display_2d_changed);layout.addWidget(self.show_2d)
        row=QHBoxLayout();row.addWidget(QLabel(tr('2D表示の厚み mm')))
        self.slab=QDoubleSpinBox();self.slab.setRange(.5,20);self.slab.setValue(2);self.slab.setSingleStep(.5)
        self.slab.valueChanged.connect(self.display_2d_changed);row.addWidget(self.slab);layout.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel(tr('線維の不透明度')))
        self.opacity=QSlider(Qt.Orientation.Horizontal);self.opacity.setRange(0,100);self.opacity.setValue(90)
        self.opacity.valueChanged.connect(self.display_changed);row.addWidget(self.opacity);layout.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel(tr('脳表の不透明度')))
        self.brain_opacity=QSlider(Qt.Orientation.Horizontal);self.brain_opacity.setRange(0,100);self.brain_opacity.setValue(25)
        self.brain_opacity.valueChanged.connect(lambda v:window.opacity_slider.setValue(v))
        window.opacity_slider.valueChanged.connect(self.brain_opacity.setValue)
        row.addWidget(self.brain_opacity);layout.addLayout(row)
        self.bundle_info=QLabel();self.bundle_info.setWordWrap(True);self.bundle_info.setObjectName('muted');layout.addWidget(self.bundle_info)
        from .tract_editing_panel import TractEditingPanel
        self.exclusion=TractEditingPanel(self);layout.addWidget(self.exclusion)
        export=QPushButton(tr('DTI・線維を出力'));menu=QMenu(export)
        for suffix in ('tck','trk','vtp'):menu.addAction(suffix.upper(),lambda checked=False,s=suffix:self.export_bundle(s))
        for key in ('fa','md','b0'):menu.addAction(key.upper()+' NIfTI',lambda checked=False,k=key:self.export_map(k))
        export.setMenu(menu);layout.addWidget(export)
        remove=QPushButton(tr('選択した線維束を削除'));remove.clicked.connect(self.remove_bundle);layout.addWidget(remove)
        caution=QLabel(tr('線維は推定候補です。交差線維の分離・EPI歪み補正は未対応です。T1との重なりを確認してください。'))
        caution.setWordWrap(True);caution.setObjectName('muted');layout.addWidget(caution)
        legend=QLabel(tr('方向色：赤＝左右 / 緑＝前後 / 青＝上下'));legend.setWordWrap(True);layout.addWidget(legend)
        layout.addStretch();self.mode_changed();self.bind_scene()

    def selected(self):
        scene=self.window.scene
        return next((m for m in scene.diffusions if m.uid==self.models.currentData()),None) if scene else None

    def bundle(self):
        item=self.bundles.currentItem();scene=self.window.scene
        return next((b for b in scene.tracts if item and b.uid==item.data(Qt.ItemDataRole.UserRole)),None) if scene else None

    def bind_scene(self):
        self.exclusion.reset()
        self.manual_uid=None;self.draw_mode.setCurrentIndex(0)
        self.manual_info.setText(tr('ROI 1 / ROI 2で作成または編集を選択'))
        self._binding=True
        current=self.models.currentData();self.models.blockSignals(True);self.models.clear()
        self.models.addItem(tr('DTIを選択'),None)
        if self.window.scene:
            for m in self.window.scene.diffusions:self.models.addItem(m.name,m.uid)
        index=self.models.findData(current)
        self.models.setCurrentIndex(index if index>0 else max(0,self.models.count()-1));self.models.blockSignals(False)
        self.roi1.center=None;self.roi2.center=None;self.roi1.bind();self.roi2.bind()
        self.refresh_bundles()
        self._binding=False;self.select_model()

    def select_model(self,*_):
        model=self.selected()
        for button in (self.overlay_button,self.compare_button,self.track_button):button.setEnabled(model is not None)
        available=model.layer_ids if model else {}
        for index in range(self.map.count()):
            key=self.map.itemData(index); key='fa' if key=='color' else key
            self.map.model().item(index).setEnabled(key in available)
        key=self.map.currentData(); key='fa' if key=='color' else key
        self.overlay_button.setEnabled(key in available)
        self.compare_button.setEnabled('b0' in available)
        if model is None:self.info.setText(tr('DTI未読込'));return
        correction=tr('体動補正あり') if model.quality.get('motion',{}).get('corrected') else tr('体動補正なし')
        spacing=' × '.join(f'{s:.2f}' for s in np.linalg.norm(model.affine[:3,:3],axis=0))
        self.info.setText(tr('{volumes} volumes / {b0} b0 / {spacing} mm\nb = {bvalues}\n{correction} / EPI歪み補正なし').format(
            volumes=len(model.bvals),b0=int(np.sum(model.bvals<=50)),spacing=spacing,
            bvalues=', '.join(f'{b:g}' for b in np.unique(model.bvals)),correction=correction))

    def choose_folder(self):
        w=self.window;w.sequence_combo.setCurrentIndex(w.sequence_combo.findData('DTI'));w.choose_image_folder()

    def choose_file(self):
        w=self.window;w.sequence_combo.setCurrentIndex(w.sequence_combo.findData('DTI'));w.choose_image_file()

    def mode_changed(self,*_):
        if hasattr(self,'roi2'):
            self.roi1.setVisible(self.mode.currentData()!=0)
            self.roi2.setVisible(self.mode.currentData()==2);self.update_roi_preview()

    def update_roi_preview(self):
        if self._binding or not hasattr(self.window,'diffusion'):return
        rois=[self.roi1.value() if self.mode.currentData()!=0 else None,self.roi2.value() if self.mode.currentData()==2 else None]
        self.window.surface.set_diffusion_rois(rois if self.window.workflow_tabs.currentIndex()==5 else [])
        for panel in self.window.slice_panels():
            panel.canvas.diffusion_rois=rois if self.window.workflow_tabs.currentIndex()==5 else []
            panel.canvas.update()

    def refresh_regions(self):
        if self.manual_uid and not any(s.uid==self.manual_uid for s in (self.window.scene.segmentations if self.window.scene else [])):
            self.manual_uid=None;self.draw_mode.setCurrentIndex(0)
            self.manual_info.setText(tr('ROI 1 / ROI 2で作成または編集を選択'))
        self.roi1.bind();self.roi2.bind()

    def manual_mode_changed(self,*_):
        if self.draw_mode.currentData()!='navigate' and hasattr(self,'exclusion'):
            self.exclusion.mode.setCurrentIndex(0)
        self.window.segmentation.update_tools()

    def edit_roi(self,controls,create):
        w=self.window
        if w.scene is None or w.worker is not None:return
        if create:
            from .segmentation import create_segment
            segment=create_segment(w.scene,'manual',f'DTI ROI {controls.index}')
            segment.color=(.98,.67,.44) if controls.index==1 else (.49,.85,.94)
            w.segmentation.add_segments([segment])
            self.refresh_regions()
            controls.source.setCurrentIndex(controls.source.findData(segment.uid))
        else:
            segment=next((s for s in w.scene.segmentations if s.uid==controls.source.currentData()),None)
            if segment is None:return
            w.segmentation.refresh_list(segment.uid)
        self.manual_uid=segment.uid
        self.manual_info.setText(tr('編集中：{name}').format(name=segment.name))
        self.draw_mode.setCurrentIndex(self.draw_mode.findData('contour'))
        w.segmentation.update_tools()

    def choose_anatomy(self,controls):
        from PySide6.QtWidgets import QDialog
        from .segmentation_create import LabelPicker
        w=self.window
        if w.scene is None or w.worker is not None:return
        self.exclusion.mode.setCurrentIndex(0)
        picker=LabelPicker(w.scene,set(),self)
        # The same search supports cortical labels and detailed thalamic nuclei.
        for label in picker.findChildren(QLabel):
            if label.text()==tr('読み込み済みの解剖・視床核を選択します。各ラベルを独立した領域として追加します。'):
                label.setText(tr('視床などの解剖ラベルをROIにします。複数選ぶと、選択領域をまとめて1つのROIにします。'))
        accepted=picker.exec()==QDialog.DialogCode.Accepted
        records=picker.checked() if accepted else []
        picker.deleteLater()
        if records:self.add_anatomy_roi(controls,records)

    def add_anatomy_roi(self,controls,records):
        from .segmentation_sources import create_anatomy_roi
        w=self.window;scene=w.scene
        def completed(segment):
            w.segmentation.add_segments([segment])
            self.refresh_regions()
            controls.source.setCurrentIndex(controls.source.findData(segment.uid))
            self.draw_mode.setCurrentIndex(0);self.manual_uid=None
            self.manual_info.setText(tr('ROI 1 / ROI 2で作成または編集を選択'))
            self.update_roi_preview()
        w.run_job(lambda _:create_anatomy_roi(scene,records),completed,tr('解剖ラベルからROIを作成しています…'))

    def show_map(self,*_):
        w=self.window;m=self.selected()
        if not m:return
        key=self.map.currentData();kind='fa' if key=='color' else key
        if kind not in m.layer_ids:return
        for uid in m.layer_ids.values():w.image_layers.rows[uid].check.setChecked(uid==m.layer_ids[kind])
        uid=m.layer_ids[kind]
        w.extra_styles[uid]='direction' if key=='color' else 'full'
        w.extra_combo.setCurrentIndex(w.extra_combo.findData(uid));w.select_extra();w.update_layers()

    def compare(self):
        w=self.window;m=self.selected()
        if not m or 'b0' not in m.layer_ids:return
        w.extra_combo.setCurrentIndex(w.extra_combo.findData(m.layer_ids['b0']));w.compare_extra()

    def state(self):
        return {'model':self.models.currentData(),'mode':self.mode.currentData(),'map':self.map.currentData(),
                'roi1':self.roi1.value(),'roi2':self.roi2.value(),'motion':self.correct_motion.isChecked(),
                'parameters':{key:spin.value() for key,spin in self.parameters.items()},'max_seeds':self.max_seeds.value(),
                'roi_thickness':self.roi_thickness.value()}

    def restore(self,state):
        self._binding=True
        if state.get('model'):self.models.setCurrentIndex(max(0,self.models.findData(state['model'])))
        self.mode.setCurrentIndex(max(0,self.mode.findData(state.get('mode',1))))
        self.map.setCurrentIndex(max(0,self.map.findData(state.get('map','color'))))
        self.roi1.restore(state.get('roi1'));self.roi2.restore(state.get('roi2'))
        self.correct_motion.setChecked(state.get('motion',True))
        for key,value in state.get('parameters',{}).items():
            if key in self.parameters:self.parameters[key].setValue(float(value))
        self.max_seeds.setValue(int(state.get('max_seeds',3000)))
        self.roi_thickness.setValue(float(state.get('roi_thickness',4)))
        self._binding=False;self.mode_changed();self.select_model()

    def track(self):
        w=self.window;m=self.selected()
        if m is None or w.worker is not None:return
        roi1={'kind':'brain'} if self.mode.currentData()==0 else self.roi1.value()
        roi2=self.roi2.value() if self.mode.currentData()==2 else None
        if roi1 is None or (self.mode.currentData()==2 and roi2 is None):
            w._job_failed(tr('ROIの中心を指定するか、作成済みの領域を選択してください。'));return
        params={key:spin.value() for key,spin in self.parameters.items()}
        name=self.name.text().strip() or f'{m.name} · {len(w.scene.tracts)+1}'
        scene=w.scene;state=w.view_state();patient_id=w.patient_id;max_seeds=self.max_seeds.value()
        def operation(progress):
            bundle=track_roi(scene,m,roi1,roi2,name=name,max_seeds=max_seeds,progress=progress,**params)
            if bundle.count and patient_id:
                w.patient_store.save(patient_id,scene,state)
                result=replace(scene,tracts=[*scene.tracts,bundle])
                w.patient_store.save(patient_id,result,state)
            return bundle
        def completed(bundle):
            if bundle.count:
                scene.tracts.append(bundle);self.refresh_bundles(bundle.uid)
                self.refresh_tract_views();self.brain_opacity.setValue(min(25,w.opacity_slider.value()))
                w.message(tr('線維候補 {count}本を作成しました。').format(count=bundle.count))
            else:
                w.message(tr('条件を満たす線維は0本です。ROI・FA値・長さを確認してください。'))
                self.bundle_info.setText(tr('条件を満たす線維は0本です。ROI・FA値・長さを確認してください。'))
            w.write_status('ready')
        w.run_job(operation,completed,tr('DTIから線維候補を作成しています…'))

    def refresh_bundles(self,uid=None):
        current=self.bundle();uid=uid or (current.uid if current else None)
        self.bundles.blockSignals(True);self.bundles.clear()
        if self.window.scene:
            for b in self.window.scene.tracts:
                item=QListWidgetItem(f'{b.name} · {b.count}')
                item.setData(Qt.ItemDataRole.UserRole,b.uid)
                item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if b.visible else Qt.CheckState.Unchecked)
                self.bundles.addItem(item)
                if b.uid==uid:self.bundles.setCurrentItem(item)
        if self.bundles.currentRow()<0 and self.bundles.count():self.bundles.setCurrentRow(self.bundles.count()-1)
        self.bundles.blockSignals(False);self.select_bundle()

    def select_bundle(self,*_):
        b=self.bundle()
        self.opacity.setEnabled(b is not None)
        self.show_2d.setEnabled(b is not None);self.slab.setEnabled(b is not None)
        self.show_2d.blockSignals(True);self.slab.blockSignals(True)
        self.show_2d.setChecked(b.visible_2d if b else False);self.slab.setValue(b.slab_mm if b else 2)
        self.show_2d.blockSignals(False);self.slab.blockSignals(False)
        if b:
            self.opacity.blockSignals(True);self.opacity.setValue(round(b.opacity*100));self.opacity.blockSignals(False)
            self.bundle_info.setText(tr('{count}本 / FA ≥ {fa} / 最大角度 {angle}°\n条件とROIは患者保存に含まれます。').format(
                count=b.count,fa=b.settings['fa_threshold'],angle=b.settings['max_angle']))
        else:self.bundle_info.clear()
        if hasattr(self,'exclusion'):self.exclusion.bundle_changed()

    def toggle(self,item):
        if self.window.scene:
            b=next(b for b in self.window.scene.tracts if b.uid==item.data(Qt.ItemDataRole.UserRole))
            b.visible=item.checkState()==Qt.CheckState.Checked;self.refresh_tract_views()
            self.exclusion.bundle_changed()

    def display_changed(self,value):
        b=self.bundle()
        if b:
            previous=b.opacity;b.opacity=value/100;self.refresh_tract_views()
            if (previous==0)!=(b.opacity==0):self.exclusion.bundle_changed()

    def display_2d_changed(self,*_):
        b=self.bundle()
        if b:
            changed=b.visible_2d!=self.show_2d.isChecked()
            b.visible_2d=self.show_2d.isChecked();b.slab_mm=self.slab.value();self.refresh_tract_views()
            if changed:self.exclusion.bundle_changed()

    def refresh_tract_views(self):
        self.window.surface.refresh_tracts()
        for panel in self.window.slice_panels():panel.canvas.update()

    def remove_bundle(self):
        b=self.bundle()
        if not b:return
        self.exclusion.clear_preview(refresh=False)
        self.window.scene.tracts.remove(b);self.refresh_bundles();self.refresh_tract_views()

    def export_bundle(self,suffix):
        b=self.bundle()
        if b is None:return
        path,_=QFileDialog.getSaveFileName(self,tr('線維束を保存'),str(self.window.export_directory()/('tracts.'+suffix)),
                                          suffix.upper()+f' (*.{suffix})')
        if path:
            self.window.run_job(lambda _:export_tract(b,Path(path).with_suffix('.'+suffix),self.window.scene),
                lambda _:self.window.message(tr('線維束と抽出条件を保存しました。')),tr('線維束を保存しています…'))

    def export_map(self,key):
        m=self.selected()
        if m is None:return
        if key not in m.layer_ids:
            self.window.message(tr('このDTI画像レイヤーは削除されています。'));return
        path,_=QFileDialog.getSaveFileName(self,tr('DTI画像を保存'),str(self.window.export_directory()/(key+'.nii.gz')),'NIfTI (*.nii.gz)')
        if path:
            self.window.run_job(lambda _:export_scalar(self.window.scene,m,key,path),
                lambda _:self.window.message(tr('DTI画像を保存しました。')),tr('DTI画像を保存しています…'))
