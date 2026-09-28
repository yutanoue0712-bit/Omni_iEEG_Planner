"""Automatic name candidates, with explicit channel-to-contact review before saving."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QTableWidget,
    QTableWidgetItem,QComboBox,QHeaderView,QDialogButtonBox,QTabWidget,QWidget,QCheckBox)
from .analysis_results import alias_candidates, channel_tokens, lead_binding, binding_arity, shared_binding_candidates,resolved_positions
from .contact_matching import ContactNameIndex,normalized_contact_name
from .i18n import tr


class MappingDialog(QDialog):
    def __init__(self,panel):
        super().__init__(panel.window); self.panel=panel; self.scene=panel.window.scene; self.result=panel.selected()
        self.name_index=ContactNameIndex(self.scene.contacts); self._updating=True
        self.sources=[]; self.issues={}
        self.setWindowTitle(tr('解析チャンネルと電極の対応')); self.resize(1100,680)
        layout=QVBoxLayout(self)
        hint=QLabel(tr('名前が一致する接点を自動照合します。候補を確認して保存してください。双極は2接点の中点に表示します。'))
        hint.setWordWrap(True); layout.addWidget(hint)
        self.tabs=tabs=QTabWidget(); layout.addWidget(tabs)
        page=QWidget(); body=QVBoxLayout(page)
        actions=QHBoxLayout(); body.addLayout(actions)
        self.auto_button=QPushButton(tr('名前から自動照合')); self.auto_button.clicked.connect(self.auto_match)
        actions.addWidget(self.auto_button)
        self.suggest=QPushButton(tr('他の結果から名前・接点の候補を表示'))
        self.suggest.clicked.connect(self.suggest_aliases); actions.addWidget(self.suggest)
        self.unresolved_only=QCheckBox(tr('未対応のみ表示'))
        self.unresolved_only.toggled.connect(self.update_count); body.addWidget(self.unresolved_only)
        self.table=QTableWidget(len(self.result.channels),5)
        self.table.setHorizontalHeaderLabels([tr('Excelのチャンネル'),tr('対応名（確認・編集）'),tr('接点1'),tr('接点2（双極）'),tr('照合状況')])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4,QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide(); body.addWidget(self.table)
        self.boxes=[]
        for row,channel in enumerate(self.result.channels):
            item=QTableWidgetItem(channel); item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable); self.table.setItem(row,0,item)
            self.table.setItem(row,1,QTableWidgetItem(self.result.aliases.get(channel,channel)))
            pair=[]; bindings=self.result.bindings.get(channel,[])
            self.sources.append('saved' if bindings else '')
            status=QTableWidgetItem(); status.setFlags(status.flags() & ~Qt.ItemFlag.ItemIsEditable); self.table.setItem(row,4,status)
            for col in (2,3):
                box=QComboBox(); box.setProperty('userText',True); box.addItem(tr('未対応'),None)
                for c in self.scene.contacts: box.addItem(c.name,c.uid)
                if len(bindings)>col-2: box.setCurrentIndex(max(0,box.findData(bindings[col-2])))
                box.currentIndexChanged.connect(lambda _,r=row:self.row_changed(r)); self.table.setCellWidget(row,col,box); pair.append(box)
            self.boxes.append(pair)
        self.table.cellDoubleClicked.connect(self.jump)
        self.table.itemChanged.connect(self.alias_changed)
        tabs.addTab(page,tr('チャンネルごと'))
        leads=QWidget(); lead_layout=QVBoxLayout(leads)
        guide=QLabel(tr('電極名と番号の方向を指定して候補を作成します。「チャンネルごと」で確認してから保存してください。'))
        guide.setWordWrap(True); lead_layout.addWidget(guide)
        self.leads=QTableWidget(0,3); self.leads.setHorizontalHeaderLabels([tr('解析側の電極名'),tr('画像側の電極名'),tr('番号を逆順にする')])
        self.leads.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); lead_layout.addWidget(self.leads)
        self.apply_button=QPushButton(tr('電極名の対応から候補を作成')); self.apply_button.clicked.connect(self.apply_leads); lead_layout.addWidget(self.apply_button)
        tabs.addTab(leads,tr('電極名でまとめて指定'))
        self.info=QLabel(); layout.addWidget(self.info)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(tr('確認して保存'))
        buttons.accepted.connect(self.commit); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.refresh_leads(); self._updating=False; self.auto_match()

    def fill_row(self,row,ids,source):
        for i,box in enumerate(self.boxes[row]):
            blocked=box.blockSignals(True)
            box.setCurrentIndex(max(0,box.findData(ids[i])) if i<len(ids) else 0)
            box.blockSignals(blocked)
        self.sources[row]=source

    def auto_match(self,*_,rows=None):
        for row in range(len(self.boxes)) if rows is None else rows:
            match=self.name_index.match(self.table.item(row,1).text())
            self.issues[row]=match.issue
            # Keep saved choices and edits, including partial manual selections.
            if any(box.currentData() for box in self.boxes[row]): continue
            if match.ids: self.fill_row(row,match.ids,'automatic')
        self.update_count()

    def row_changed(self,row):
        if self._updating: return
        self.sources[row]='manual'; self.update_count()

    def alias_changed(self,item):
        if self._updating or item.column()!=1: return
        # A new name must not silently keep a candidate for the previous name.
        self.fill_row(item.row(),[],''); self.refresh_leads(); self.auto_match(rows=[item.row()])

    def row_valid(self,row):
        first,second=[box.currentData() for box in self.boxes[row]]
        arity=self.row_arity(row)
        return bool(first) and first!=second and bool(second)==(arity==2)

    def row_arity(self,row):
        channel=self.result.channels[row]; name=self.table.item(row,1).text().strip()
        mode=self.result.binding_modes.get(channel)
        if name==self.result.aliases.get(channel,channel) and mode in ('monopolar','bipolar'):
            return 2 if mode=='bipolar' else 1
        return binding_arity(name,self.scene.contacts,self.name_index)

    def refresh_leads(self):
        previous={group:(box.currentData(),reverse.isChecked()) for group,(box,reverse) in getattr(self,'lead_boxes',{}).items()}
        targets=sorted({c.group for c in self.scene.contacts})
        groups=sorted({group for row in range(self.table.rowCount()) for group,_ in channel_tokens(self.table.item(row,1).text())})
        self.leads.setRowCount(len(groups)); self.lead_boxes={}
        for row,group in enumerate(groups):
            item=QTableWidgetItem(group); item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable); self.leads.setItem(row,0,item)
            box=QComboBox(); box.addItem(tr('未対応'),None)
            for name in targets: box.addItem(name,name)
            reverse=QCheckBox(); self.leads.setCellWidget(row,1,box); self.leads.setCellWidget(row,2,reverse)
            if group in previous:
                target,reversed_order=previous[group]
                box.setCurrentIndex(max(0,box.findData(target))); reverse.setChecked(reversed_order)
            else:
                matches=[name for name in targets if normalized_contact_name(name)==normalized_contact_name(group)]
                if len(matches)==1: box.setCurrentIndex(box.findData(matches[0]))
            self.lead_boxes[group]=(box,reverse)

    def suggest_aliases(self):
        self._updating=True
        suggestions=alias_candidates(self.result,self.scene.results)
        for row,channel in enumerate(self.result.channels):
            if channel in suggestions and self.table.item(row,1).text()==channel: self.table.item(row,1).setText(suggestions[channel])
        aliases={channel:self.table.item(row,1).text() for row,channel in enumerate(self.result.channels)}
        bindings=shared_binding_candidates(self.result,self.scene.results,aliases)
        for row,channel in enumerate(self.result.channels):
            if channel not in bindings or any(b.currentData() for b in self.boxes[row]): continue
            if all(self.boxes[row][0].findData(uid)>0 for uid in bindings[channel]): self.fill_row(row,bindings[channel],'shared')
        self._updating=False; self.refresh_leads(); self.auto_match()

    def apply_leads(self):
        maps={g:(box.currentData(),rev.isChecked()) for g,(box,rev) in self.lead_boxes.items() if box.currentData()}
        targets=[value[0] for value in maps.values()]
        if len(targets)!=len(set(targets)):
            self.info.setText(tr('複数の電極名に同じ画像側の電極が指定されています。確認してください。')); return
        for row,boxes in enumerate(self.boxes):
            bindings=lead_binding(self.table.item(row,1).text().strip(),self.scene.contacts,maps)
            if bindings: self.fill_row(row,bindings,'lead')
        self.tabs.setCurrentIndex(0); self.update_count()

    def update_count(self,*_):
        if self._updating or not hasattr(self,'info'): return
        valid=0; automatic=0; self._updating=True
        labels={'saved':'保存済み','automatic':'自動照合・未確定','manual':'手動で指定',
                'shared':'他の結果からの候補','lead':'電極名からの候補'}
        issues={'ambiguous':'接点名が重複・要確認','missing':'接点が不足・要確認',
                'same_contact':'同じ接点・要確認','unmatched':'対応なし'}
        for row in range(len(self.boxes)):
            mapped=self.row_valid(row); valid+=mapped; automatic+=mapped and self.sources[row]=='automatic'
            text=labels.get(self.sources[row],'対応候補') if mapped else issues.get(self.issues.get(row,''),'未対応・要確認')
            if not mapped and any(b.currentData() for b in self.boxes[row]): text='接点の組み合わせを確認'
            item=self.table.item(row,4); item.setText(tr(text)); item.setToolTip(tr(text))
            item.setForeground(QColor('#90dfbc' if mapped else '#f0ce96'))
            self.table.setRowHidden(row,self.unresolved_only.isChecked() and mapped)
        self._updating=False
        self.info.setText(tr('対応候補 {mapped}/{total}（自動照合 {automatic}・未対応 {missing}）。確認して保存すると反映します。').format(
            mapped=valid,total=len(self.boxes),automatic=automatic,missing=len(self.boxes)-valid))
        self.table.setToolTip(tr('チャンネル名をダブルクリックすると画像の位置を確認できます。'))

    def jump(self,row,column):
        if self.panel.window.scene is not self.scene: return
        ids=[b.currentData() for b in self.boxes[row] if b.currentData()]
        contacts=[c for c in self.scene.contacts if c.uid in ids]
        if contacts:
            import numpy as np
            self.panel.view.setCurrentIndex(self.panel.view.findData('brain'))
            self.panel.window.select_world(np.mean([c.position for c in contacts],axis=0))

    def commit(self):
        if self.panel.window.scene is not self.scene: self.reject(); return
        current_ids={c.uid for c in self.scene.contacts}; self.name_index=ContactNameIndex(self.scene.contacts)
        bindings={}; aliases={}
        for row,channel in enumerate(self.result.channels):
            first,second=[b.currentData() for b in self.boxes[row]]
            if any(uid and uid not in current_ids for uid in (first,second)):
                self.info.setText(tr('接点の構成が変わりました。対応画面を開き直してください。')); return
            name=self.table.item(row,1).text().strip()
            if (second and (not first or first==second)) or (first and self.row_arity(row)==2 and not second):
                self.info.setText(tr('双極チャンネルは異なる2接点を指定してください。')); return
            if first and second and self.row_arity(row)==1:
                self.info.setText(tr('単極チャンネルは接点1だけを指定してください。')); return
            if first: bindings[channel]=[first]+([second] if second else [])
            alias=self.table.item(row,1).text().strip()
            if alias and alias!=channel: aliases[channel]=alias
        self.result.bindings=bindings; self.result.aliases=aliases
        self.result.binding_modes={key:'bipolar' if len(ids)==2 else 'monopolar' for key,ids in bindings.items()}
        self.result.settings['mapping_reviewed']=True
        self.panel.refresh()
        if len(resolved_positions(self.result,self.scene.contacts)[0]):self.panel.show_on_electrodes()
        self.accept()
