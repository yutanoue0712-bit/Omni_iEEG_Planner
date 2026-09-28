"""Native CT sections and reversible contact editing; no external application dependency."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import nibabel as nib
from PySide6.QtCore import QPoint,QRectF,Qt
from PySide6.QtGui import QColor,QPainter
from PySide6.QtWidgets import (QWidget,QDialog,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,
    QPushButton,QComboBox,QCheckBox,QSpinBox,QDoubleSpinBox,QLineEdit,QTableWidget,QTableWidgetItem,
    QHeaderView,QAbstractItemView,QDialogButtonBox,QMessageBox,QScrollArea,QSplitter,QApplication)
from .imaging import Contact,InputError,validate_scene
from .i18n import tr,translate_widgets
from .electrode_localization import refine_contact
from .electrode_editing import group_contacts,contact_axis,renumber,set_contact_position,check_ct_bounds,refresh_quality
from .contact_overview import ContactOverview
from .ct_section import CTSection,STATUS_COLOR
from .electrode_presets import REFERENCE_PRESETS,set_reference_preset


STATUS_TEXT={'unreviewed':'未確認','uncertain':'要修正','reviewed':'確認済み'}


class ContactEditor(QDialog):
    def __init__(self,scene,parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('CTコンタクトの確認・修正'))
        screen=QApplication.primaryScreen().availableGeometry()
        self.resize(min(1480,screen.width()-45),min(930,screen.height()-60)); self.setMinimumSize(1080,720)
        self.scene=replace(scene,contacts=deepcopy(scene.contacts),electrode_quality=deepcopy(scene.electrode_quality))
        self.undo=[]; self.busy=False; self.index=0
        layout=QVBoxLayout(self)
        hint=QLabel(tr('元CTだけで作成した候補です。電極の名前・先端からの順序・コンタクト数を確認してください。'))
        hint.setWordWrap(True); layout.addWidget(hint)
        body=QHBoxLayout(); layout.addLayout(body,1)
        sidebar=QWidget(); sidebar.setFixedWidth(255); controls=QVBoxLayout(sidebar)
        controls.setContentsMargins(0,0,6,0)
        self.groups=QComboBox(); self.groups.setObjectName('dataNames')
        self.groups.currentTextChanged.connect(self.change_group); controls.addWidget(self.groups)
        self.summary=QLabel(); self.summary.setWordWrap(True); controls.addWidget(self.summary)
        self.group_name=QLineEdit(); controls.addWidget(self.group_name)
        rename=QPushButton(tr('電極名を変更')); rename.clicked.connect(self.rename_group); controls.addWidget(rename)
        controls.addWidget(QLabel(tr('型番・仕様（参考ラベル）')))
        self.model_combo=QComboBox()
        self.model_combo.addItem(tr('指定なし：CTで判断'),None)
        for preset in REFERENCE_PRESETS: self.model_combo.addItem(tr(preset['label']),preset['id'])
        self.model_combo.currentIndexChanged.connect(self.change_reference)
        controls.addWidget(self.model_combo)
        self.reference_info=QLabel(); self.reference_info.setWordWrap(True); self.reference_info.setObjectName('muted')
        controls.addWidget(self.reference_info)
        row=QHBoxLayout(); row.addWidget(QLabel(tr('予定コンタクト数')))
        self.expected=QSpinBox(); self.expected.setRange(0,64); self.expected.setSpecialValueText(tr('未指定'))
        self.expected.valueChanged.connect(self.set_expected); row.addWidget(self.expected); controls.addLayout(row)
        reverse=QPushButton(tr('先端・最後端を反転')); reverse.clicked.connect(self.reverse); controls.addWidget(reverse)
        self.move_check=QCheckBox(tr('クリックで選択点を移動'))
        self.move_check.setToolTip(tr('別の丸をクリックすると選択、それ以外のクリックで二重丸の点を移動します。'))
        self.move_check.toggled.connect(self.set_edit); controls.addWidget(self.move_check)
        coordinates=QGridLayout(); self.position=[]
        for i,axis in enumerate('RAS'):
            coordinates.addWidget(QLabel('CT '+axis),i,0)
            spin=QDoubleSpinBox(); spin.setRange(-2000,2000); spin.setDecimals(2); spin.setSingleStep(.1)
            coordinates.addWidget(spin,i,1); self.position.append(spin)
        controls.addLayout(coordinates)
        move=QPushButton(tr('入力座標を適用')); move.clicked.connect(lambda:self.move(np.array([s.value() for s in self.position]))); controls.addWidget(move)
        snap=QPushButton(tr('CT金属像の中心へ補正')); snap.clicked.connect(self.refine); controls.addWidget(snap)
        ends=QHBoxLayout()
        for text,at_start in (('先端側に追加',True),('最後端側に追加',False)):
            button=QPushButton(tr(text)); button.clicked.connect(lambda _,first=at_start:self.extend(first)); ends.addWidget(button)
        controls.addLayout(ends)
        delete=QPushButton(tr('選択コンタクトを削除')); delete.clicked.connect(self.delete_contact); controls.addWidget(delete)
        delete_group=QPushButton(tr('この電極を削除')); delete_group.clicked.connect(self.delete_group); controls.addWidget(delete_group)
        undo=QPushButton(tr('編集を1つ戻す')); undo.clicked.connect(self.undo_edit); controls.addWidget(undo)
        accept=QPushButton(tr('先端・個数を確認済みにする')); accept.setObjectName('primary')
        accept.clicked.connect(self.confirm_group); controls.addWidget(accept)
        controls.addStretch()
        controls.addWidget(QLabel(tr('右ドラッグ：CTのwindow / level')))
        self.width_spin=QDoubleSpinBox(); self.width_spin.setRange(100,12000); self.width_spin.setValue(4000)
        self.level_spin=QDoubleSpinBox(); self.level_spin.setRange(-1500,6000); self.level_spin.setValue(1000)
        self.width_spin.valueChanged.connect(self.refresh_sections); self.level_spin.valueChanged.connect(self.refresh_sections)
        levels=QHBoxLayout(); levels.addWidget(self.width_spin); levels.addWidget(self.level_spin); controls.addLayout(levels)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(sidebar); scroll.setFixedWidth(272)
        body.addWidget(scroll)
        split=QSplitter(Qt.Orientation.Horizontal); split.setChildrenCollapsible(False); body.addWidget(split,1)
        details=QWidget(); grid=QGridLayout(details); grid.setContentsMargins(0,0,0,0)
        split.addWidget(details)
        self.sections=[CTSection(i) for i in range(3)]
        for i,section in enumerate(self.sections):
            section.selected.connect(self.select); section.moved.connect(self.move); section.window_delta.connect(self.windowing)
            grid.addWidget(section,i,0)
        navigation=QLabel(tr('ホイール：拡大・縮小 / Shift＋左ドラッグ・中ドラッグ：表示移動 / Home：リセット'))
        navigation.setWordWrap(True);navigation.setObjectName('muted');grid.addWidget(navigation,3,0)
        edit_hint=QLabel(tr('別の丸：選択 / クリック移動がON：二重丸の点を修正 / 編集を1つ戻す：取消'))
        edit_hint.setWordWrap(True); edit_hint.setObjectName('muted'); grid.addWidget(edit_hint,4,0)
        # Parent the pane before VTK creates native Windows handles.
        context=QWidget(split); context_layout=QVBoxLayout(context); context_layout.setContentsMargins(0,0,0,0)
        self.overview=ContactOverview(self.scene,context); self.overview.selected.connect(self.select)
        self.overview.window_delta.connect(self.windowing)
        context_layout.addWidget(self.overview,3); split.addWidget(context); split.setSizes([700,420])
        self.table=QTableWidget(0,4)
        self.table.verticalHeader().hide()
        self.table.setHorizontalHeaderLabels([tr('番号'),tr('状態'),'CT peak HU',tr('隣接間隔 mm')])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(lambda row,_:self.select(row)); context_layout.addWidget(self.table,2)
        self.table.setMinimumHeight(160)
        grid.setColumnStretch(0,1)
        grid.setRowStretch(0,1); grid.setRowStretch(1,1); grid.setRowStretch(2,1)
        footer=QDialogButtonBox()
        apply=footer.addButton(tr('編集を適用'),QDialogButtonBox.ButtonRole.AcceptRole)
        cancel=footer.addButton(tr('変更を破棄して閉じる'),QDialogButtonBox.ButtonRole.RejectRole)
        apply.clicked.connect(self.accept); cancel.clicked.connect(self.reject); layout.addWidget(footer)
        self.move_check.setChecked(True)
        self.rebuild_groups()

    def current(self): return group_contacts(self.scene,self.groups.currentText())

    def snapshot(self):
        self.undo.append((deepcopy(self.scene.contacts),deepcopy(self.scene.electrode_quality),self.groups.currentText(),self.index))
        self.undo=self.undo[-50:]

    def rebuild_groups(self,selected=None):
        selected=selected or self.groups.currentText()
        self.groups.blockSignals(True); self.groups.clear()
        self.groups.addItems(list(dict.fromkeys(c.group for c in self.scene.contacts)))
        index=self.groups.findText(selected); self.groups.setCurrentIndex(max(index,0)); self.groups.blockSignals(False)
        self.change_group(self.groups.currentText())

    def change_group(self,name):
        self.busy=True; self.index=0; self.group_name.setText(name)
        metadata=self.scene.electrode_quality.get('groups',{}).get(name,{})
        self.expected.setValue(metadata.get('expected_count') or 0)
        self.model_combo.setCurrentIndex(max(0,self.model_combo.findData(metadata.get('reference_preset'))))
        self.busy=False
        self.refresh()

    def refresh(self):
        refresh_quality(self.scene)
        contacts=self.current(); self.index=min(self.index,max(0,len(contacts)-1))
        count=len(contacts); reviewed=sum(c.status=='reviewed' for c in contacts)
        self.summary.setText(tr('CT候補：{count}個 / 確認済み：{reviewed}個\nAD-Tech・型番未指定').format(count=count,reviewed=reviewed))
        metadata=self.scene.electrode_quality.get('groups',{}).get(self.groups.currentText(),{})
        reference=metadata.get('reference_spec')
        self.reference_info.setText(tr('参考：{count}極 / CT：{detected}個\n幅の定義は未確認。位置・個数はCTで判断。').format(count=reference['contact_count'],detected=count)
                                    if reference else tr('CTで検出した位置・個数を使用します。'))
        self.table.setRowCount(count)
        for i,c in enumerate(contacts):
            gap=np.linalg.norm(c.ct_position-contacts[i-1].ct_position) if i else None
            values=[str(i+1),tr(STATUS_TEXT.get(c.status,'未確認')),str(round(c.evidence.get('peak_hu',0))),f'{gap:.2f}' if gap is not None else '—']
            for j,value in enumerate(values):
                item=QTableWidgetItem(value); item.setForeground(QColor(STATUS_COLOR.get(c.status,'#ffca72'))); self.table.setItem(i,j,item)
                if c.evidence.get('nearby_contact_uids'):
                    item.setToolTip(tr('別のコンタクト中心が0.9 mm以内にあります。重複を確認してください。'))
                    if j==1: item.setText(tr('重複候補'))
        if contacts:
            self.table.selectRow(self.index)
            for spin,value in zip(self.position,contacts[self.index].ct_position): spin.setValue(float(value))
        self.refresh_sections()

    def refresh_sections(self,*_):
        if not hasattr(self,'sections'): return
        self.set_edit(self.move_check.isChecked())
        for section in self.sections:
            section.configure(self.scene,self.current(),self.index,self.width_spin.value(),self.level_spin.value())
        self.overview.configure(self.scene,self.current(),self.index,self.width_spin.value(),self.level_spin.value())

    def select(self,index):
        if not self.current(): return
        self.index=min(max(index,0),len(self.current())-1); self.refresh()

    def set_edit(self,enabled):
        for section in self.sections: section.edit=enabled

    def windowing(self,dx,dy):
        width=self.width_spin.value()
        self.width_spin.blockSignals(True);self.level_spin.blockSignals(True)
        self.width_spin.setValue(width*np.exp(np.clip(dx/160,-2,2)))
        self.level_spin.setValue(self.level_spin.value()+dy*width/180)
        self.width_spin.blockSignals(False);self.level_spin.blockSignals(False)
        self.refresh_sections()

    def move(self,point):
        if not self.current(): return
        if np.allclose(self.current()[self.index].ct_position,point,atol=1e-7,rtol=0): return
        self.snapshot()
        try: set_contact_position(self.scene,self.current()[self.index],point)
        except InputError as exc:
            self.undo.pop(); QMessageBox.warning(self,tr('位置の確認'),tr(str(exc)))
        self.refresh()

    def refine(self):
        contacts=self.current()
        if not contacts: return
        axis,spacing=contact_axis(contacts); c=contacts[self.index]
        center,evidence=refine_contact(self.scene.raw_ct,self.scene.raw_ct_affine,c.ct_position,axis,spacing)
        if not evidence['supported']:
            QMessageBox.information(self,tr('位置の確認'),tr('この近傍では独立したCT金属像を確認できません。画像を見て位置を修正してください。')); return
        self.move(center)

    def rename_group(self):
        name=self.group_name.text().strip(); old=self.groups.currentText()
        if not name or (name!=old and name in {c.group for c in self.scene.contacts}): return
        self.snapshot(); renumber(self.current(),name)
        metadata=self.scene.electrode_quality.setdefault('groups',{})
        metadata[name]=metadata.pop(old,{}) if name!=old else metadata.get(old,{})
        self.rebuild_groups(name)

    def set_expected(self,value):
        if self.busy or not self.current(): return
        self.snapshot(); name=self.groups.currentText()
        self.scene.electrode_quality.setdefault('groups',{}).setdefault(name,{})['expected_count']=value or None
        self.invalidate(); self.refresh()

    def change_reference(self,*_):
        if self.busy or not self.current(): return
        self.snapshot()
        set_reference_preset(self.scene,self.groups.currentText(),self.model_combo.currentData())
        self.refresh()

    def invalidate(self):
        for c in self.current():
            if c.status=='reviewed':
                c.status='unreviewed'; c.evidence.pop('human_reviewed',None)
        self.scene.electrode_quality.setdefault('groups',{}).setdefault(self.groups.currentText(),{})['status']='unreviewed'

    def replace_current(self,contacts):
        name=self.groups.currentText(); all_contacts=[]; inserted=False
        for c in self.scene.contacts:
            if c.group==name:
                if not inserted: all_contacts.extend(contacts); inserted=True
            else: all_contacts.append(c)
        self.scene.contacts=all_contacts; renumber(contacts,name)
        self.invalidate(); self.refresh()

    def reverse(self):
        if not self.current(): return
        self.snapshot(); self.replace_current(list(reversed(self.current())))

    def delete_contact(self):
        contacts=self.current()
        if not contacts: return
        self.snapshot(); contacts.pop(self.index); self.replace_current(contacts)
        if not contacts: self.rebuild_groups()

    def delete_group(self):
        if not self.current(): return
        self.snapshot(); name=self.groups.currentText()
        self.scene.contacts=[c for c in self.scene.contacts if c.group!=name]
        self.scene.electrode_quality.setdefault('groups',{}).pop(name,None); self.rebuild_groups()

    def extend(self,at_start):
        contacts=self.current()
        if not contacts: return
        axis,spacing=contact_axis(contacts)
        endpoint=contacts[0] if at_start else contacts[-1]
        guess=endpoint.ct_position+(-1 if at_start else 1)*spacing*axis
        try: check_ct_bounds(self.scene,guess)
        except InputError as exc:
            QMessageBox.warning(self,tr('位置の確認'),tr(str(exc))); return
        self.snapshot()
        center,evidence=refine_contact(self.scene.raw_ct,self.scene.raw_ct_affine,guess,axis,spacing)
        point=center if evidence['supported'] else guess
        contact=Contact('',endpoint.group,nib.affines.apply_affine(self.scene.ct_to_mri,point),endpoint.color,
            ct_position=point,provenance='native_ct_only',status='unreviewed' if evidence['supported'] else 'uncertain',
            evidence={**evidence,'origin':'user_extended'})
        contacts.insert(0,contact) if at_start else contacts.append(contact)
        self.replace_current(contacts); self.select(0 if at_start else len(contacts)-1)

    def confirm_group(self):
        contacts=self.current()
        if not contacts: return
        if self.expected.value() and self.expected.value()!=len(contacts):
            QMessageBox.warning(self,tr('個数の確認'),tr('予定コンタクト数と現在の個数が一致しません。')); return
        self.snapshot()
        for c in contacts:
            c.status='reviewed'; c.evidence['human_reviewed']=True
        metadata=self.scene.electrode_quality.setdefault('groups',{}).setdefault(self.groups.currentText(),{})
        metadata.update(status='reviewed',numbering='tip-first confirmed by user',direction_ambiguous=False)
        self.refresh()

    def undo_edit(self):
        if not self.undo: return
        contacts,quality,group,index=self.undo.pop()
        self.scene.contacts=contacts; self.scene.electrode_quality=quality
        self.rebuild_groups(group); self.select(index)

    def accept(self):
        refresh_quality(self.scene)
        try: validate_scene(self.scene)
        except InputError as exc:
            QMessageBox.warning(self,tr('電極の確認'),tr(str(exc))); return
        super().accept()

    def done(self,result):
        super().done(result)
        self.overview.shutdown()

    def capture(self):
        pixmap=self.grab()
        if self.overview.stack.currentIndex()==0:
            widget=self.overview.surface.widget
            painter=QPainter(pixmap); origin=widget.mapTo(self,QPoint(0,0))
            painter.drawImage(QRectF(origin.x(),origin.y(),widget.width(),widget.height()),self.overview.surface.capture_image())
            painter.end()
        return pixmap
