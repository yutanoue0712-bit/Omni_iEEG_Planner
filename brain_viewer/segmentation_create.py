"""Creation controls for all loaded FreeSurfer labels and selected image sequences."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QDoubleSpinBox,
    QPushButton,QCheckBox,QDialog,QLineEdit,QListWidget,QListWidgetItem,QDialogButtonBox)
from .i18n import tr
from .segmentation_sources import available_labels, image_sources, get_source, extract_segment


class LabelPicker(QDialog):
    def __init__(self, scene, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('FreeSurferの解剖ラベルを選択')); self.resize(560,610)
        layout=QVBoxLayout(self)
        hint=QLabel(tr('読み込み済みの解剖・視床核を選択します。各ラベルを独立した領域として追加します。'))
        hint.setWordWrap(True); layout.addWidget(hint)
        self.search=QLineEdit(); self.search.setPlaceholderText(tr('名前またはラベル番号で検索')); layout.addWidget(self.search)
        self.collection=QComboBox()
        for text,key in (('すべて','all'),('皮質・皮質下','anatomy'),('詳細視床核','nuclei')): self.collection.addItem(tr(text),key)
        layout.addWidget(self.collection)
        self.labels=QListWidget(); layout.addWidget(self.labels,1)
        for record in available_labels(scene):
            prefix='FS' if record['source']=='anatomy' else 'Nuclei'
            item=QListWidgetItem(f"{prefix} · {record['label']} · {record['name']}")
            item.setData(Qt.ItemDataRole.UserRole,record)
            item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if (record['source'],record['label']) in selected else Qt.CheckState.Unchecked)
            self.labels.addItem(item)
        self.count=QLabel(); layout.addWidget(self.count)
        clear=QPushButton(tr('選択を解除')); clear.clicked.connect(self.clear); layout.addWidget(clear)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.search.textChanged.connect(self.filter); self.collection.currentIndexChanged.connect(self.filter)
        self.labels.itemChanged.connect(self.update_count); self.update_count()

    def checked(self):
        return [self.labels.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.labels.count())
                if self.labels.item(i).checkState()==Qt.CheckState.Checked]

    def filter(self,*_):
        query=self.search.text().casefold()
        query={'視床':'thalam','脳梁':'callos','脳室':'ventric','海馬':'hippo','扁桃体':'amyg'}.get(query,query)
        group=self.collection.currentData()
        for i in range(self.labels.count()):
            item=self.labels.item(i); record=item.data(Qt.ItemDataRole.UserRole)
            item.setHidden(query not in item.text().casefold() or group not in ('all',record['source']))

    def clear(self):
        for i in range(self.labels.count()): self.labels.item(i).setCheckState(Qt.CheckState.Unchecked)

    def update_count(self,*_):
        self.count.setText(tr('{total}ラベル中 {selected}個を選択').format(total=self.labels.count(),selected=len(self.checked())))


class ImageExtractionPane(QWidget):
    def __init__(self, owner):
        super().__init__(); self.owner=owner; self.window=owner.window; self.seed=None; self.resetting=False
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(7)
        layout.addWidget(QLabel(tr('抽出元の画像')))
        self.source=QComboBox(); self.source.setObjectName('sourceImageNames'); layout.addWidget(self.source)
        self.show_button=QPushButton(tr('選んだ画像を表示')); self.show_button.clicked.connect(self.show_source); layout.addWidget(self.show_button)
        self.category=QComboBox()
        for name,key in (('任意の領域','manual'),('Lesion','lesion'),('Artery','artery'),('Vessel','vessel')): self.category.addItem(tr(name),key)
        layout.addWidget(self.category)
        row=QHBoxLayout(); self.lower=QDoubleSpinBox(); self.upper=QDoubleSpinBox()
        for text,spin in (('下限',self.lower),('上限',self.upper)):
            row.addWidget(QLabel(tr(text))); spin.setRange(-1e9,1e9); spin.setDecimals(2); row.addWidget(spin,1)
        layout.addLayout(row)
        self.units=QLabel(); self.units.setObjectName('muted'); self.units.setWordWrap(True); layout.addWidget(self.units)
        self.connected=QCheckBox(tr('起点とつながる部分だけ')); self.connected.setChecked(True); layout.addWidget(self.connected)
        self.local=QCheckBox(tr('起点の近傍に限定')); self.local.setChecked(True)
        row=QHBoxLayout(); row.addWidget(self.local)
        self.radius=QDoubleSpinBox(); self.radius.setRange(1,250); self.radius.setValue(30); self.radius.setSuffix(' mm'); row.addWidget(self.radius)
        layout.addLayout(row)
        self.seed_button=QPushButton(tr('現在の断面位置を起点にする')); self.seed_button.clicked.connect(self.capture_seed); layout.addWidget(self.seed_button)
        self.seed_info=QLabel(); self.seed_info.setWordWrap(True); self.seed_info.setObjectName('muted'); layout.addWidget(self.seed_info)
        row=QHBoxLayout()
        self.preview_button=QPushButton(tr('候補をプレビュー')); self.preview_button.clicked.connect(self.preview); row.addWidget(self.preview_button)
        self.clear_button=QPushButton(tr('候補を消す')); self.clear_button.clicked.connect(self.clear_preview); self.clear_button.setEnabled(False); row.addWidget(self.clear_button)
        layout.addLayout(row)
        self.result=QLabel(tr('画像値からの候補です。病変・血管を自動判別する処理ではありません。'))
        self.result.setWordWrap(True); self.result.setObjectName('muted'); layout.addWidget(self.result)
        self.source.currentIndexChanged.connect(self.source_changed)
        for spin in (self.lower,self.upper,self.radius): spin.valueChanged.connect(self.invalidate)
        for check in (self.connected,self.local): check.toggled.connect(self.invalidate)
        self.category.currentIndexChanged.connect(self.invalidate)

    def reset_scene(self):
        self.resetting=True; self.seed=None
        self.source.blockSignals(True); self.source.clear()
        for source in image_sources(self.window.scene): self.source.addItem(tr(source.name) if source.uid in ('mri','ct') else source.name,source.uid)
        self.source.blockSignals(False)
        self.resetting=False; self.source_changed(display=False)

    def source_changed(self,*_,display=True):
        self.invalidate()
        if self.window.scene is None: return
        source=get_source(self.window.scene,self.source.currentData())
        sample=source.data[::3,::3,::3]
        valid=source.valid[::3,::3,::3] if source.valid is not None else sample!=0
        values=sample[valid & np.isfinite(sample)]
        low,high=np.percentile(values,[60,99.5]) if len(values) else (0.,1.)
        self.resetting=True; self.lower.setValue(float(low)); self.upper.setValue(float(max(low+1.,high))); self.resetting=False
        self.units.setText(tr('元画像の値（HU）・位置合わせ後の格子') if source.units=='HU' else tr('画像固有の値・window/levelとは独立'))
        self.capture_seed()
        if display: self.show_source()

    def show_source(self):
        if self.window.scene is None: return
        uid=self.source.currentData()
        self.window._restoring=True
        try:
            for key,row in self.window.image_layers.rows.items():
                row.check.setChecked(key==uid)
                if key==uid: row.slider.setValue(100)
            if uid not in ('mri','ct'): self.window.extra_styles[uid]='full'
            self.window.window_target.setCurrentIndex(self.window.window_target.findData(uid))
        finally: self.window._restoring=False
        self.window.update_layers()

    def capture_seed(self):
        self.invalidate()
        if self.window.scene is None: self.seed_info.clear(); return
        self.seed=np.clip(self.window.ijk,0,np.array(self.window.scene.data.shape)-1).copy(); world=self.window.scene.world(self.seed)
        source=get_source(self.window.scene,self.source.currentData()); value=float(source.data[tuple(self.seed)])
        self.seed_info.setText(tr('起点 R/A/S: {position} mm\n画像値: {value}').format(position=', '.join(f'{x:.1f}' for x in world),value=f'{value:.2f}'))

    def invalidate(self,*_):
        if self.resetting: return
        self.radius.setEnabled(self.local.isChecked())
        self.clear_preview()

    def clear_preview(self):
        self.clear_button.setEnabled(False)
        if self.window.scene is not None and self.window.scene.segmentation_preview is not None:
            self.window.scene.segmentation_preview=None
            self.window.update_layers()
        self.result.setText(tr('画像値からの候補です。病変・血管を自動判別する処理ではありません。'))
        if hasattr(self.owner,'create_button'): self.owner.update_create_button()

    def preview(self):
        if self.window.scene is None or self.window.worker is not None: return
        self.owner.finish_stroke(); self.owner.mode.setCurrentIndex(0); self.clear_preview(); self.show_source()
        scene=self.window.scene
        parameters=dict(source_id=self.source.currentData(), low=self.lower.value(), high=self.upper.value(),
            category=self.category.currentData(),name=self.owner.name.text().strip(),seed=self.seed.copy() if self.seed is not None else None,
            connected=self.connected.isChecked(),radius_mm=self.radius.value() if self.local.isChecked() else 0.)
        def completed(segment):
            scene.segmentation_preview=segment
            self.clear_button.setEnabled(True)
            self.window.opacity_slider.setValue(min(25,self.window.opacity_slider.value()))
            self.window.update_layers()
            self.result.setText(tr('候補 {count} voxel・未追加\n確認して「領域として追加」を押してください。').format(count=int(segment.mask.sum())))
            self.owner.update_create_button()
        self.window.run_job(lambda _:extract_segment(scene,**parameters),completed,tr('画像から領域候補を作成しています…'))
