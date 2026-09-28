"""Review the current contact labels before exporting the visible rows."""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QPushButton,QTableWidget,
    QTableWidgetItem,QAbstractItemView,QHeaderView,QFileDialog,QMessageBox)
from .contact_labels import contact_label_records,write_contact_labels
from .i18n import tr


class ContactLabelsDialog(QDialog):
    def __init__(self,scene,directory,parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('コンタクトの解剖ラベル一覧')); self.resize(1110,600)
        self.records=contact_label_records(scene); self.directory=Path(directory)
        layout=QVBoxLayout(self)
        hint=QLabel(tr('現在のコンタクト中心にあるFreeSurferラベルです。視床核は別の列に表示します。座標は患者固有のMRI RAS / mmです。'))
        hint.setWordWrap(True); layout.addWidget(hint)
        top=QHBoxLayout(); top.addWidget(QLabel(tr('電極')))
        self.group=QComboBox(); self.group.setObjectName('dataNames'); self.group.addItem(tr('すべての電極'),'')
        for name in dict.fromkeys(row['electrode'] for row in self.records): self.group.addItem(name,name)
        self.group.currentIndexChanged.connect(self.refresh); top.addWidget(self.group); top.addStretch()
        self.summary=QLabel(); top.addWidget(self.summary); layout.addLayout(top)
        self.table=QTableWidget(0,9); self.table.verticalHeader().hide()
        self.table.setHorizontalHeaderLabels([tr('電極'),tr('コンタクト'),tr('状態'),'FreeSurfer ID',tr('解剖ラベル'),tr('視床核'),'R mm','A mm','S mm'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4,QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(250); layout.addWidget(self.table,1)
        footer=QHBoxLayout(); self.export_button=QPushButton(tr('表示中の一覧をCSV出力')); self.export_button.setObjectName('primary')
        self.export_button.clicked.connect(self.export_csv); footer.addWidget(self.export_button); footer.addStretch()
        close=QPushButton(tr('閉じる')); close.clicked.connect(self.accept); footer.addWidget(close); layout.addLayout(footer)
        self.refresh()

    def visible_records(self):
        group=self.group.currentData()
        return [row for row in self.records if not group or row['electrode']==group]

    def refresh(self,*_):
        rows=self.visible_records(); self.table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            status=tr({'reviewed':'確認済み','uncertain':'要修正'}.get(row['review_status'],'未確認'))
            name=row['freesurfer_label'] or tr({'not_loaded':'解剖ラベル未読込','outside_volume':'撮像範囲外'}.get(row['label_status'],'背景・ラベルなし'))
            values=(row['electrode'],row['contact'],status,str(row['freesurfer_id']),name,row['thalamic_nucleus'] or '—',
                    *(f"{row['mri_'+axis+'_mm']:.2f}" for axis in 'ras'))
            for j,value in enumerate(values):
                item=QTableWidgetItem(value); item.setToolTip(value); self.table.setItem(i,j,item)
        self.summary.setText(tr('{count}個 / 位置確認済み {reviewed}個').format(count=len(rows),reviewed=sum(r['review_status']=='reviewed' for r in rows)))
        self.export_button.setEnabled(bool(rows))

    def export_csv(self):
        path,_=QFileDialog.getSaveFileName(self,tr('解剖ラベルCSVを保存'),str(self.directory/'contact_anatomy.csv'),'CSV (*.csv)')
        if not path: return
        if not path.lower().endswith('.csv'): path+='.csv'
        try:
            write_contact_labels(path,self.visible_records())
            self.summary.setText(tr('解剖ラベルCSVを保存しました。'))
        except OSError:
            QMessageBox.warning(self,tr('電極出力'),tr('保存できませんでした。保存先を確認してください。'))
