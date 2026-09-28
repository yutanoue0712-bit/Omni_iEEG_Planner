"""Electrode color and marker controls shared by live views and exports."""
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import QWidget,QGroupBox,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QDoubleSpinBox,QSlider,QCheckBox,QPushButton
from .i18n import tr
from .result_display import metric_display,marker_options,spatial_options


class ResultDisplayControls(QGroupBox):
    def __init__(self,panel):
        super().__init__(tr('電極のカラーマップ'));self.panel=panel;self.loading=False
        layout=QVBoxLayout(self)
        self.spatial_mode=QComboBox()
        for label,key in [('接点の色球','points'),('脳表ヒートマップ','surface'),('脳表ヒートマップ＋色球','both')]:
            self.spatial_mode.addItem(tr(label),key)
        self.row(layout,'3D表示方法',self.spatial_mode)
        self.distance=QDoubleSpinBox();self.distance.setRange(1,50);self.distance.setDecimals(1)
        self.distance.setSingleStep(1);self.distance.setKeyboardTracking(False)
        self.row(layout,'脳表への最大距離 mm',self.distance)
        self.distance_slider=QSlider(Qt.Orientation.Horizontal);self.distance_slider.setRange(10,500)
        self.distance_slider.setToolTip(tr('接点または双極中点から脳表までの距離。直径ではありません。'))
        layout.addWidget(self.distance_slider)
        self.distance_timer=QTimer(self);self.distance_timer.setSingleShot(True);self.distance_timer.setInterval(70)
        self.distance_timer.timeout.connect(self.changed)
        self.distance_slider.valueChanged.connect(self.distance_from_slider)
        self.distance_slider.sliderReleased.connect(self.changed)
        self.distance.valueChanged.connect(self.distance_from_spin)
        self.palette=QComboBox()
        for label,key in [('自動（値の範囲から）','auto'),('Inferno','inferno'),('Turbo','turbo'),('Viridis','viridis'),('青・白・赤','coolwarm'),('Jet（青→赤）','jet')]:
            self.palette.addItem(tr(label),key)
        layout.addWidget(self.palette)
        self.reverse=QCheckBox(tr('色の向きを反転'));layout.addWidget(self.reverse)
        self.threshold_mode=QComboBox()
        for label,key in [('全ての値を表示','none'),('指定値以上を表示','above'),('絶対値が指定値以上を表示','absolute')]:
            self.threshold_mode.addItem(tr(label),key)
        layout.addWidget(self.threshold_mode)
        self.threshold=QDoubleSpinBox();self.threshold.setDecimals(5);self.threshold.setRange(-1e12,1e12)
        self.threshold.setKeyboardTracking(False);self.row(layout,'表示しきい値',self.threshold)
        self.radius=QDoubleSpinBox();self.radius.setRange(.5,8);self.radius.setSingleStep(.5)
        self.radius.setDecimals(1);self.row(layout,'色球の半径 mm',self.radius)
        self.slab=QDoubleSpinBox();self.slab.setRange(.5,30);self.slab.setSingleStep(.5)
        self.slab.setDecimals(1);self.row(layout,'2D表示の厚み mm',self.slab)
        self.opacity=QSlider(Qt.Orientation.Horizontal);self.opacity.setRange(0,100)
        self.row(layout,'解析色の不透明度',self.opacity)
        self.labels=QCheckBox(tr('解析チャンネル名を表示'));layout.addWidget(self.labels)
        self.brain_opacity=QSlider(Qt.Orientation.Horizontal);self.brain_opacity.setRange(0,100)
        self.brain_opacity.setValue(panel.window.opacity_slider.value())
        self.brain_opacity.valueChanged.connect(panel.window.opacity_slider.setValue)
        panel.window.opacity_slider.valueChanged.connect(self.brain_opacity.setValue)
        self.row(layout,'脳表の不透明度',self.brain_opacity)
        hint=QLabel(tr('色の範囲は全時間で共通です。しきい値は表示だけに適用します。'))
        hint.setWordWrap(True);hint.setObjectName('muted');layout.addWidget(hint)
        guide=QLabel(tr('灰色は電極の位置、色球は解析値です。双極は2接点の中点に表示します。'))
        guide.setWordWrap(True);guide.setObjectName('muted');layout.addWidget(guide)
        surface_hint=QLabel(tr('脳表ヒートマップは表示用の空間補間です。距離は記録範囲の推定ではありません。2D断面は接点の色球で確認します。'))
        surface_hint.setWordWrap(True);surface_hint.setObjectName('muted');layout.addWidget(surface_hint)
        for widget in (self.palette,self.threshold_mode,self.spatial_mode):widget.currentIndexChanged.connect(self.changed)
        for widget in (self.threshold,self.radius,self.slab,self.opacity):widget.valueChanged.connect(self.changed)
        for widget in (self.reverse,self.labels):widget.toggled.connect(self.changed)

    @staticmethod
    def row(layout,label,widget):
        row=QHBoxLayout();row.addWidget(QLabel(tr(label)));row.addWidget(widget,1);layout.addLayout(row)

    def bind(self):
        self.distance_timer.stop()
        self.loading=True
        result=self.panel.selected();self.setEnabled(result is not None)
        metric=max(0,self.panel.metric_combo.currentIndex())
        display=metric_display(result,metric);markers=marker_options(result)
        spatial=spatial_options(result)
        self.spatial_mode.setCurrentIndex(self.spatial_mode.findData(spatial['mode']))
        self.distance.setValue(spatial['distance_mm']);self.distance_slider.setValue(round(spatial['distance_mm']*10))
        self.distance.setEnabled(spatial['mode']!='points');self.distance_slider.setEnabled(spatial['mode']!='points')
        self.palette.setCurrentIndex(self.palette.findData(display['colormap']))
        self.reverse.setChecked(display['reverse'])
        self.threshold_mode.setCurrentIndex(self.threshold_mode.findData(display['threshold_mode']))
        self.threshold.setValue(display['threshold']);self.threshold.setEnabled(display['threshold_mode']!='none')
        self.radius.setValue(markers['radius_mm']);self.slab.setValue(markers['slab_mm'])
        self.opacity.setValue(round(markers['opacity']*100));self.labels.setChecked(markers['labels'])
        self.loading=False

    def distance_from_slider(self,value):
        if self.loading:return
        blocked=self.distance.blockSignals(True);self.distance.setValue(value/10);self.distance.blockSignals(blocked)
        self.distance_timer.start()

    def distance_from_spin(self,value):
        if self.loading:return
        blocked=self.distance_slider.blockSignals(True);self.distance_slider.setValue(round(value*10));self.distance_slider.blockSignals(blocked)
        self.distance_timer.stop();self.changed()

    def changed(self,*_):
        if self.loading or self.panel.busy:return
        self.distance_timer.stop()
        result=self.panel.selected()
        if result is None:return
        metric=result.metrics[max(0,self.panel.metric_combo.currentIndex())]
        mode=self.threshold_mode.currentData();self.threshold.setEnabled(mode!='none')
        result.settings.setdefault('display',{})[metric]={
            'colormap':self.palette.currentData(),'reverse':self.reverse.isChecked(),
            'threshold_mode':mode,'threshold':self.threshold.value()}
        result.settings['markers']={'radius_mm':self.radius.value(),'slab_mm':self.slab.value(),
                                   'opacity':self.opacity.value()/100,'labels':self.labels.isChecked()}
        result.settings['spatial']={'mode':self.spatial_mode.currentData(),'distance_mm':self.distance.value()}
        self.distance.setEnabled(self.spatial_mode.currentData()!='points')
        self.distance_slider.setEnabled(self.spatial_mode.currentData()!='points')
        self.panel.refresh()


class ResultPlaybackBar(QWidget):
    """Keep time navigation next to the images when the settings panel is scrolled."""
    def __init__(self,panel):
        super().__init__();self.panel=panel
        layout=QHBoxLayout(self);layout.setContentsMargins(8,3,8,5)
        self.previous=QPushButton('◀');self.previous.setToolTip(tr('前の時点'))
        self.next=QPushButton('▶');self.next.setToolTip(tr('次の時点'))
        self.play=QPushButton(tr('再生'));self.play.setMinimumWidth(64)
        self.slider=QSlider(Qt.Orientation.Horizontal);self.label=QLabel()
        for widget in (self.previous,self.play,self.next):layout.addWidget(widget)
        layout.addWidget(self.slider,1);layout.addWidget(self.label)
        self.previous.clicked.connect(lambda:panel.slider.setValue(panel.slider.value()-1))
        self.next.clicked.connect(lambda:panel.slider.setValue(panel.slider.value()+1))
        self.play.clicked.connect(panel.toggle_play);self.slider.valueChanged.connect(panel.slider.setValue)
        self.hide()

    def sync(self):
        result=self.panel.selected()
        active=self.panel.window.workflow_tabs.currentIndex()==4 and result is not None and result.kind=='time_series'
        self.setVisible(bool(active))
        if not active:return
        self.slider.blockSignals(True);self.slider.setRange(self.panel.slider.minimum(),self.panel.slider.maximum())
        self.slider.setValue(self.panel.slider.value());self.slider.blockSignals(False)
        self.label.setText(self.panel.time_label.text());self.play.setText(self.panel.play.text())
        self.play.setEnabled(self.panel.play.isEnabled() and not self.panel.exporting)
        for widget in (self.previous,self.next,self.slider):widget.setEnabled(not self.panel.exporting)
