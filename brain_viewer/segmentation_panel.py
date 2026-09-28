"""Segmentation workflow: anatomy presets, separate masks, slice brushes and local exports."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QGroupBox,QLabel,QComboBox,QPushButton,
    QLineEdit,QListWidget,QListWidgetItem,QCheckBox,QSlider,QDoubleSpinBox,QFileDialog,QMenu,QStackedWidget,QDialog,QSizePolicy)
from .i18n import tr
from .segmentation import PRESETS,create_segment,paint_stroke,fill_contour,export_mask,export_surface
from .segmentation_sources import create_label_segment
from .segmentation_create import LabelPicker, ImageExtractionPane


class CreationStack(QStackedWidget):
    """Fit the active creation form, without stretching a combo to the extraction form."""
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.currentChanged.connect(self.fit)

    def fit(self,*_):
        page=self.currentWidget()
        if page is None:return
        height=page.heightForWidth(self.width()) if page.hasHeightForWidth() else page.sizeHint().height()
        height=max(page.minimumSizeHint().height(),height,1)
        if self.minimumHeight()!=height or self.maximumHeight()!=height:self.setFixedHeight(height)

    def resizeEvent(self,event):
        super().resizeEvent(event);self.fit()


class SegmentationPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window=window; self.busy=False; self.history=[]; self.stroke=None; self.label_selection=[]
        layout=QVBoxLayout(self); layout.setContentsMargins(0,3,10,0); layout.setSpacing(10)
        intro=QLabel(tr('FreeSurferの解剖ラベル、または選んだ画像から領域を作成し、断面上で修正します。'))
        intro.setWordWrap(True); intro.setObjectName('muted'); layout.addWidget(intro)
        create_box=QGroupBox(tr('領域を作成')); create_layout=QVBoxLayout(create_box)
        self.creation_method=QComboBox()
        for text in ('解剖・手動プリセット','FreeSurferラベルを選択','画像から抽出'): self.creation_method.addItem(tr(text))
        create_layout.addWidget(self.creation_method)
        self.creation_stack=CreationStack(); create_layout.addWidget(self.creation_stack)
        self.preset=QComboBox()
        for key,(name,_,_) in PRESETS.items(): self.preset.addItem(tr(name),key)
        self.preset.currentIndexChanged.connect(self.preset_changed); self.creation_stack.addWidget(self.preset)
        label_page=QWidget(); label_layout=QVBoxLayout(label_page); label_layout.setContentsMargins(0,0,0,0)
        self.label_button=QPushButton(tr('ラベルを検索・選択')); self.label_button.clicked.connect(self.choose_labels); label_layout.addWidget(self.label_button)
        self.label_info=QLabel(); self.label_info.setWordWrap(True); self.label_info.setObjectName('muted'); label_layout.addWidget(self.label_info)
        self.creation_stack.addWidget(label_page)
        self.extraction=ImageExtractionPane(self); self.creation_stack.addWidget(self.extraction)
        self.name=QLineEdit(); self.name.setPlaceholderText(tr('領域の名前')); self.name.setProperty('userText',True); self.name.setMaxLength(100); create_layout.addWidget(self.name)
        self.create_button=QPushButton(tr('領域を作成')); self.create_button.setObjectName('primary')
        self.create_button.clicked.connect(self.create); create_layout.addWidget(self.create_button)
        self.preset_hint=QLabel(); self.preset_hint.setWordWrap(True); self.preset_hint.setObjectName('muted'); create_layout.addWidget(self.preset_hint)
        layout.addWidget(create_box)
        layout.addWidget(QLabel(tr('作成した領域（チェック＝3D表示）')))
        self.regions=QListWidget(); self.regions.setMinimumHeight(115); self.regions.setMaximumHeight(180)
        self.regions.currentItemChanged.connect(self.select); self.regions.itemChanged.connect(self.toggle_3d)
        layout.addWidget(self.regions)
        properties=QGroupBox(tr('選択した領域')); prop=QVBoxLayout(properties)
        self.rename=QLineEdit(); self.rename.setProperty('userText',True); self.rename.setMaxLength(100); self.rename.editingFinished.connect(self.rename_selected)
        prop.addWidget(self.rename)
        self.show_2d=QCheckBox(tr('2D断面に重ねる')); self.show_2d.toggled.connect(self.display_changed); prop.addWidget(self.show_2d)
        row=QHBoxLayout(); row.addWidget(QLabel(tr('3D不透明度')))
        self.opacity=QSlider(Qt.Orientation.Horizontal); self.opacity.setRange(0,100); self.opacity.valueChanged.connect(self.display_changed)
        row.addWidget(self.opacity,1); self.percent=QLabel('75%'); self.percent.setFixedWidth(34); row.addWidget(self.percent); prop.addLayout(row)
        self.info=QLabel(); self.info.setWordWrap(True); self.info.setObjectName('muted'); prop.addWidget(self.info)
        jump=QPushButton(tr('領域の位置へ移動')); jump.clicked.connect(self.jump); prop.addWidget(jump)
        self.delete_button=QPushButton(tr('選択した領域を削除'))
        self.delete_button.setToolTip(tr('削除は「編集を1つ戻す」で取り消せます。'))
        self.delete_button.clicked.connect(self.delete_selected);prop.addWidget(self.delete_button)
        layout.addWidget(properties); self.properties=properties
        edit=QGroupBox(tr('断面上で手動修正')); edit_layout=QVBoxLayout(edit)
        self.mode=QComboBox()
        for label,key in (('位置を確認','navigate'),('ブラシで追加','paint'),('消しゴム','erase'),
                          ('手描きで囲んで追加','contour'),('手描きで囲んで削除','contour_erase')):
            self.mode.addItem(tr(label),key)
        self.mode.currentIndexChanged.connect(self.update_tools); edit_layout.addWidget(self.mode)
        radius=QHBoxLayout(); radius.addWidget(QLabel(tr('半径 mm')))
        self.radius=QDoubleSpinBox(); self.radius.setRange(.5,20); self.radius.setDecimals(1); self.radius.setValue(2.5)
        self.radius.valueChanged.connect(self.update_tools); radius.addWidget(self.radius); edit_layout.addLayout(radius)
        self.undo_button=QPushButton(tr('編集を1つ戻す')); self.undo_button.clicked.connect(self.undo); edit_layout.addWidget(self.undo_button)
        guide=QLabel(tr('左ドラッグ：選択断面だけを描画\nShift＋左ドラッグ：移動\nホイール：断面送り / 右ドラッグ：濃淡'))
        guide.setWordWrap(True); guide.setObjectName('muted'); edit_layout.addWidget(guide); layout.addWidget(edit); self.edit_box=edit
        # Undo must remain reachable after deleting the last region.
        edit_layout.removeWidget(self.undo_button);layout.addWidget(self.undo_button)
        reference=QPushButton(tr('基準T1を表示')); reference.clicked.connect(self.reference_view); layout.addWidget(reference)
        self.export_button=QPushButton(tr('選択した領域を出力'))
        menu=QMenu(self.export_button); menu.addAction(tr('マスク NIfTI'),lambda:self.export('nifti'))
        menu.addAction(tr('3Dモデル VTP'),lambda:self.export('vtp')); self.export_button.setMenu(menu); layout.addWidget(self.export_button)
        self.creation_method.currentIndexChanged.connect(self.creation_changed)
        layout.addStretch(); self.preset_changed(); self.bind_scene()

    def creation_changed(self,index):
        self.finish_stroke(); self.mode.setCurrentIndex(0); self.extraction.clear_preview()
        self.creation_stack.setCurrentIndex(index)
        # Only the active form should determine the sidebar's vertical size.
        for i in range(self.creation_stack.count()):
            widget=self.creation_stack.widget(i)
            from PySide6.QtWidgets import QSizePolicy
            widget.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred if i==index else QSizePolicy.Policy.Ignored)
        self.creation_stack.adjustSize()
        self.name.setVisible(index!=1); self.preset_hint.setVisible(index==0)
        if index==0: self.preset_changed()
        if index==2:
            self.name.setText(tr('画像抽出領域')); self.extraction.show_source()
        self.update_create_button()
        self.creation_stack.fit()

    def update_create_button(self):
        method=self.creation_method.currentIndex()
        text=('領域を作成','選択したラベルを追加','領域として追加')[method]
        self.create_button.setText(tr(text))
        ready=(method==0 or (bool(self.label_selection) if method==1 else
                            bool(self.window.scene and self.window.scene.segmentation_preview is not None)))
        self.create_button.setEnabled(ready)
        self.creation_stack.fit()

    def choose_labels(self):
        if self.window.scene is None: return
        picker=LabelPicker(self.window.scene,{(r['source'],r['label']) for r in self.label_selection},self)
        if picker.exec()==QDialog.DialogCode.Accepted:
            self.label_selection=picker.checked()
            self.label_info.setText(tr('{count}ラベルを選択').format(count=len(self.label_selection))+'\n'+
                                    '\n'.join(r['name'] for r in self.label_selection[:4]))
            self.update_create_button()
        picker.deleteLater()

    def selected(self):
        item=self.regions.currentItem()
        if self.window.scene is None or item is None: return None
        return next((segment for segment in self.window.scene.segmentations if segment.uid==item.data(Qt.ItemDataRole.UserRole)),None)

    def preset_changed(self,*_):
        if not hasattr(self,'name'): return
        key=self.preset.currentData(); self.name.setText(tr(PRESETS[key][0]))
        self.preset_hint.setText(tr('既存FreeSurferのラベルから作成。抽出後に形を確認してください。') if PRESETS[key][1] else
                                tr('空の領域を作り、手動で指定します。自動抽出は今後追加します。'))

    def bind_scene(self):
        self.stroke=None; self.history=[]
        self.label_selection=[]; self.label_info.setText(tr('ラベルは未選択'))
        if self.window.scene: self.window.scene.segmentation_preview=None
        self.mode.setCurrentIndex(0)
        self.extraction.reset_scene()
        self.creation_method.setCurrentIndex(0); self.creation_changed(0)
        self.refresh_list()

    def refresh_list(self, selected_id=None):
        current=self.selected(); selected_id=selected_id or (current.uid if current else None)
        self.busy=True; self.regions.clear()
        if self.window.scene:
            for segment in self.window.scene.segmentations:
                item=QListWidgetItem(segment.name); item.setData(Qt.ItemDataRole.UserRole,segment.uid)
                item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if segment.visible_3d else Qt.CheckState.Unchecked)
                self.regions.addItem(item)
                if segment.uid==selected_id: self.regions.setCurrentItem(item)
        if self.regions.count() and self.regions.currentRow()<0: self.regions.setCurrentRow(0)
        self.busy=False; self.select()

    def select(self,*_):
        if self.busy: return
        self.finish_stroke()
        segment=self.selected(); self.busy=True
        for widget in (self.properties,self.edit_box,self.export_button): widget.setEnabled(segment is not None)
        if segment:
            self.rename.setText(segment.name); self.show_2d.setChecked(segment.visible_2d); self.opacity.setValue(round(segment.opacity*100))
        else:
            self.rename.clear(); self.info.clear()
        self.busy=False; self.refresh_info(); self.update_tools()

    def refresh_info(self):
        segment=self.selected()
        self.percent.setText(f'{self.opacity.value()}%')
        self.undo_button.setEnabled(bool(self.history))
        postop=getattr(self.window,'postop',None)
        if postop and self.window.workflow_tabs.currentIndex()==3:postop.update_controls()
        if not segment: return
        count=int(np.count_nonzero(segment.mask)); volume=count*float(np.prod(self.window.scene.spacing))/1000
        method=segment.provenance.get('method')
        source=tr('FreeSurfer由来') if method=='existing_FreeSurfer_labels' else (
            tr('画像抽出：{name}').format(name=segment.provenance.get('source_name','')) if method=='image_intensity_candidate' else tr('手動領域'))
        self.info.setText(tr('{source} / 修正履歴 {edits}件\n{count} voxel / {volume} mL\n形状は確認前・表示格子で計算').format(
            source=source,edits=len(segment.edits),count=count,volume=f'{volume:.2f}'))

    def create(self):
        if self.window.scene is None or self.window.worker is not None: return
        self.finish_stroke()
        method=self.creation_method.currentIndex()
        if method==1:
            scene=self.window.scene; records=list(self.label_selection)
            self.window.run_job(lambda _:[create_label_segment(scene,r['source'],r['label']) for r in records],
                                self.add_segments,tr('領域を作成しています…'))
            return
        if method==2:
            candidate=self.window.scene.segmentation_preview
            if candidate is None: return
            if not self.name.text().strip(): return
            candidate.name=self.name.text().strip()
            self.window.scene.segmentation_preview=None
            self.add_segments([candidate]); self.extraction.clear_preview(); self.update_create_button()
            return
        scene=self.window.scene; key=self.preset.currentData(); name=self.name.text().strip()
        names={s.name for s in scene.segmentations}; stem=name; number=2
        while name in names: name=f'{stem} {number}'; number+=1
        def completed(segment):
            self.add_segments([segment])
            if not PRESETS[key][1]: self.mode.setCurrentIndex(1)
        self.window.run_job(lambda _:create_segment(scene,key,name),completed,tr('領域を作成しています…'))

    def add_segments(self,segments):
        if not segments: return
        names={s.name for s in self.window.scene.segmentations}
        for segment in segments:
            stem=segment.name; number=2
            while segment.name in names:
                segment.name=f'{stem[:90]} {number}'; number+=1
            names.add(segment.name)
        self.window.scene.segmentations.extend(segments)
        self.history.append(('create_many',[s.uid for s in segments])); self.history=self.history[-30:]
        self.refresh_list(segments[-1].uid)
        self.window.opacity_slider.setValue(min(25,self.window.opacity_slider.value()))
        self.window.update_layers(); self.jump()

    def rename_selected(self):
        segment=self.selected()
        if not segment or self.busy: return
        name=self.rename.text().strip()
        if name: segment.name=name; self.regions.currentItem().setText(name)
        else: self.rename.setText(segment.name)

    def toggle_3d(self,item):
        if self.busy or self.window.scene is None: return
        self.finish_stroke()
        segment=next(s for s in self.window.scene.segmentations if s.uid==item.data(Qt.ItemDataRole.UserRole))
        segment.visible_3d=item.checkState()==Qt.CheckState.Checked
        self.window.surface.refresh_segmentations(self.window.scene)

    def display_changed(self,*_):
        segment=self.selected()
        if not segment or self.busy: return
        segment.visible_2d=self.show_2d.isChecked(); segment.opacity=self.opacity.value()/100
        self.percent.setText(f'{self.opacity.value()}%')
        self.window.surface.refresh_segmentations(self.window.scene); self.window.update_layers()

    def update_tools(self,*_):
        self.finish_stroke()
        if self.mode.currentData()!='navigate': self.extraction.clear_preview()
        if not hasattr(self.window,'slices'): return
        segment=self.selected()
        mode=self.active_mode() if segment else 'navigate'
        diffusion=getattr(self.window,'diffusion',None)
        if diffusion and hasattr(diffusion,'exclusion') and diffusion.exclusion.active() and diffusion.exclusion.mode.currentData()=='2d':
            mode='tract_exclude'
        for panel in self.window.slice_panels(): panel.canvas.set_brush(mode,self.radius.value())

    def active_mode(self):
        if self.window.workflow_tabs.currentIndex()==1:
            return self.mode.currentData()
        postop=getattr(self.window,'postop',None)
        if self.window.workflow_tabs.currentIndex()==3 and postop and postop.manual_uid:
            segment=self.selected()
            if segment and segment.uid==postop.manual_uid:
                return postop.draw_mode.currentData()
        diffusion=getattr(self.window,'diffusion',None)
        if diffusion and hasattr(diffusion,'exclusion') and diffusion.exclusion.mode.currentData()!='navigate':
            return 'navigate'
        if self.window.workflow_tabs.currentIndex()==5 and diffusion and diffusion.manual_uid:
            segment=self.selected()
            if segment and segment.uid==diffusion.manual_uid:
                return diffusion.draw_mode.currentData()
        return 'navigate'

    def enclose(self,axis,vertices):
        segment=self.selected(); mode=self.active_mode()
        if segment is None or mode not in ('contour','contour_erase') or self.window.worker is not None: return
        self.finish_stroke()
        thickness=(self.window.diffusion.roi_thickness.value() if self.window.workflow_tabs.currentIndex()==5 else
                   self.window.postop.thickness.value() if self.window.workflow_tabs.currentIndex()==3 else self.window.scene.spacing[axis])
        self.stroke={'segment':segment,'before':segment.mask.copy(),'axis':axis,
                     'radius':0.,'mode':mode,'thickness_mm':thickness}
        fill_contour(segment.mask,self.window.scene.spacing,axis,vertices,mode=='contour',thickness)
        segment.visible_2d=True
        self.finish_stroke()

    def draw(self,axis,point,phase):
        segment=self.selected()
        mode=self.active_mode()
        if segment is None or mode not in ('paint','erase') or self.window.worker is not None: return
        if phase=='end': self.finish_stroke(); return
        if phase=='start' or self.stroke is None:
            self.finish_stroke()
            self.stroke={'segment':segment,'before':segment.mask.copy(),'point':np.asarray(point,float),'axis':axis,
                         'radius':self.radius.value(),'mode':mode}
            segment.visible_2d=True; self.show_2d.setChecked(True)
        stroke=self.stroke
        paint_stroke(segment.mask,self.window.scene.spacing,axis,stroke['point'],point,stroke['radius'],stroke['mode']=='paint')
        stroke['point']=np.asarray(point,float)
        for panel in self.window.slice_panels(): panel.canvas._update_image()

    def finish_stroke(self):
        if self.stroke is None: return
        stroke,self.stroke=self.stroke,None; segment=stroke['segment']
        changed=np.flatnonzero(segment.mask.ravel()!=stroke['before'].ravel())
        if len(changed):
            before=stroke['before'].ravel()[changed]
            self.history.append(('paint',segment.uid,changed,before,len(segment.edits)))
            self.history=self.history[-30:]
            segment.edits.append({'action':stroke['mode'],'axis':stroke['axis'],'radius_mm':stroke['radius'],
                                  'thickness_mm':stroke.get('thickness_mm',self.window.scene.spacing[stroke['axis']]),
                                  'changed_voxels':int(len(changed)),'time':datetime.now(timezone.utc).isoformat()})
            segment.revision+=1
            self.window.surface.refresh_segmentations(self.window.scene)
            self.window.update_layers()
        self.refresh_info()

    def _undo_index(self,uid=None):
        for index in range(len(self.history)-1,-1,-1):
            action=self.history[index]
            ids=([action[1].uid] if action[0]=='delete' else
                 action[1] if action[0]=='create_many' else [action[1]])
            if uid is None or uid in ids:return index
        return None

    def can_undo(self,uid):
        return bool(uid) and self._undo_index(uid) is not None

    def undo(self,checked=False,*,uid=None):
        self.finish_stroke()
        if not self.history or self.window.scene is None: return
        index=self._undo_index(uid)
        if index is None:return
        action=self.history[index]
        if uid and action[0]=='create_many' and len(action[1])>1:
            self.history[index]=('create_many',[key for key in action[1] if key!=uid])
            action=('create_many',[uid])
        else:self.history.pop(index)
        if action[0]=='delete':
            _,segment,index,roi_state=action
            self.window.scene.segmentations.insert(min(index,len(self.window.scene.segmentations)),segment)
            self.refresh_list(segment.uid)
            self.window.diffusion.refresh_regions()
            for key,value in roi_state.items():
                if value and value.get('kind')=='segment' and value.get('uid')==segment.uid:
                    control=getattr(self.window.diffusion,key)
                    if control.value() is None:control.restore(value)
            self.window.update_layers();self.refresh_info();return
        if action[0]=='create_many':
            self.window.scene.segmentations=[s for s in self.window.scene.segmentations if s.uid not in action[1]]
            self.window.surface.refresh_segmentations(self.window.scene)
            self.refresh_list(); self.window.diffusion.refresh_regions();self.window.update_layers(); return
        segment=next((s for s in self.window.scene.segmentations if s.uid==action[1]),None)
        if segment is None: return
        if action[0]=='create': self.window.scene.segmentations.remove(segment)
        else:
            segment.mask.ravel()[action[2]]=action[3]; segment.edits=segment.edits[:action[4]]; segment.revision+=1
        self.window.surface.refresh_segmentations(self.window.scene)
        self.refresh_list(); self.window.diffusion.refresh_regions();self.window.update_layers()

    def delete_selected(self,checked=False,*,uid=None):
        w=self.window
        segment=(next((s for s in w.scene.segmentations if s.uid==uid),None)
                 if uid and w.scene else self.selected())
        if segment is None or w.worker is not None:return
        self.finish_stroke();self.mode.setCurrentIndex(0)
        for panel in w.slice_panels():panel.canvas._contour=[]
        index=w.scene.segmentations.index(segment)
        roi_state={key:getattr(w.diffusion,key).value() for key in ('roi1','roi2')}
        self.history.append(('delete',segment,index,roi_state));self.history=self.history[-30:]
        w.scene.segmentations.remove(segment)
        self.refresh_list();w.diffusion.refresh_regions();w.update_layers()
        self.refresh_info()
        w.message(tr('領域を削除しました。「編集を1つ戻す」で取り消せます。'))

    def jump(self):
        segment=self.selected()
        if segment is None: return
        points=np.argwhere(segment.mask)
        if len(points):
            center=points.mean(0); nearest=points[np.argmin(np.sum(((points-center)*self.window.scene.spacing)**2,axis=1))]
            self.window.set_cursor(nearest)

    def reference_view(self):
        self.window.mri_visible.setChecked(True); self.window.mri_opacity.setValue(100)
        self.window.ct_visible.setChecked(False)
        for layer in self.window.scene.extra_mris: self.window.image_layers.rows[layer.uid].check.setChecked(False)
        self.window.window_target.setCurrentIndex(0)

    def export(self,kind):
        self.finish_stroke(); segment=self.selected()
        if segment is None: return
        suffix='.nii.gz' if kind=='nifti' else '.vtp'
        path,_=QFileDialog.getSaveFileName(self,tr('領域を出力'),str(self.window.export_directory()/('segmentation'+suffix)),
                                        'NIfTI (*.nii.gz)' if kind=='nifti' else 'VTK (*.vtp)')
        if not path: return
        if not path.lower().endswith(suffix): path+=suffix
        operation=export_mask if kind=='nifti' else export_surface
        self.window.run_job(lambda _:operation(segment,self.window.scene,Path(path)),
                            lambda _:self.window.message(tr('領域を患者座標で出力しました。')),tr('領域を出力しています…'))
