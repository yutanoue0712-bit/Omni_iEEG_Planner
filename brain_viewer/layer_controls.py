"""Reusable checked opacity rows for registered images and anatomical models."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget,QGroupBox,QVBoxLayout,QHBoxLayout,QCheckBox,QSlider,QLabel,QToolButton
from .i18n import tr


class LayerRow(QWidget):
    changed=Signal()

    def __init__(self,key,label,visible,opacity,parent=None):
        super().__init__(parent)
        self.setObjectName('layer_'+key)
        layout=QHBoxLayout(self); layout.setContentsMargins(0,3,0,3); layout.setSpacing(7)
        self.check=QCheckBox(tr(label)); self.check.setChecked(visible); self.check.setMinimumWidth(67)
        self.check.setProperty('layerCheck',True)
        self.slider=QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0,100); self.slider.setValue(opacity)
        self.slider.setToolTip(tr('不透明度：0%で透明、100%で不透明'))
        self.percent=QLabel(f'{opacity}%'); self.percent.setFixedWidth(34)
        self.percent.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.check); layout.addWidget(self.slider,1); layout.addWidget(self.percent)
        self.check.toggled.connect(self.update_row); self.slider.valueChanged.connect(self.update_row)
        self.update_row()

    def update_row(self,*_):
        self.percent.setText(f'{self.slider.value()}%')
        self.slider.setEnabled(self.check.isChecked())
        self.changed.emit()

    def state(self):
        return {'visible':self.check.isChecked(),'opacity':self.slider.value()/100}

    def allow_removal(self, callback):
        self.remove_button=QToolButton(self)
        self.remove_button.setText('×'); self.remove_button.setFixedSize(23,23)
        self.remove_button.setToolTip(tr('この画像レイヤーを削除'))
        self.remove_button.setAccessibleName(tr('この画像レイヤーを削除'))
        self.remove_button.clicked.connect(callback)
        self.layout().addWidget(self.remove_button)

    def restore(self,state):
        self.check.blockSignals(True); self.slider.blockSignals(True)
        try:
            self.check.setChecked(bool(state.get('visible',self.check.isChecked())))
            self.slider.setValue(round(float(state.get('opacity',self.slider.value()/100))*100))
        finally:
            self.check.blockSignals(False); self.slider.blockSignals(False)
        self.update_row()


class LayerList(QGroupBox):
    changed=Signal()

    def __init__(self,title,parent=None):
        super().__init__(tr(title),parent)
        self.rows={}; self.rows_layout=QVBoxLayout(self)
        self.rows_layout.setSpacing(2)
        heading=QLabel(tr('表示 / 不透明度'))
        heading.setObjectName('muted'); self.rows_layout.addWidget(heading)

    def add_layer(self,key,label,visible=True,opacity=100):
        if key in self.rows: raise ValueError('Duplicate layer key')
        row=LayerRow(key,label,visible,opacity,self)
        row.changed.connect(self.changed)
        self.rows[key]=row; self.rows_layout.addWidget(row)
        return row

    def state(self): return {key:row.state() for key,row in self.rows.items()}

    def remove_layer(self,key):
        row=self.rows.pop(key)
        self.rows_layout.removeWidget(row)
        row.hide(); row.deleteLater()

    def restore(self,state):
        for key,value in state.items():
            if key in self.rows and isinstance(value,dict): self.rows[key].restore(value)
