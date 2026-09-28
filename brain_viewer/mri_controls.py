"""Display controls for registered scalar image layers, including native PET."""
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QComboBox, QDoubleSpinBox, QLabel, QPushButton, QSizePolicy, QSlider, QCheckBox, QMessageBox
from .i18n import tr
from .image_colormaps import IMAGE_PALETTES, default_image_mode, value_units
from .image_colorbar import ImageColorBar


class MRIControls:
    def build_extra_controls(self):
        self.extra_styles = {}
        self.extra_palettes = {}
        self.extra_surfaces = {}
        self._extra_domains = {}
        self._extra_render_timer = QTimer(self)
        self._extra_render_timer.setSingleShot(True)
        self._extra_render_timer.setInterval(30)
        self._extra_render_timer.timeout.connect(self.update_layers)
        self.extra_group = QGroupBox(tr('追加画像の調整'))
        layout = QVBoxLayout(self.extra_group)
        self.extra_combo = QComboBox(); self.extra_combo.setObjectName('imageNames')
        self.extra_combo.currentIndexChanged.connect(self.select_extra)
        layout.addWidget(self.extra_combo)
        self.extra_mode = QComboBox()
        self.extra_mode.addItem(tr('グレースケールで重ねる'), 'full')
        self.extra_mode.addItem(tr('ヒートマップで重ねる'), 'heatmap')
        self.extra_mode.addItem(tr('市松模様で照合'), 'checker')
        self.extra_mode.addItem(tr('方向カラー（FA）'), 'direction')
        self.extra_mode.currentIndexChanged.connect(self.change_extra_mode)
        layout.addWidget(self.extra_mode)
        self.extra_palette = QComboBox()
        for palette in IMAGE_PALETTES: self.extra_palette.addItem(palette.title(), palette)
        self.extra_palette.currentIndexChanged.connect(self.change_extra_palette)
        layout.addWidget(self.extra_palette)
        self.extra_labels = {}
        for label, attribute in (('幅', 'extra_window'), ('中心', 'extra_level'),
                                 ('表示下限', 'extra_lower'), ('表示上限', 'extra_upper')):
            caption = QLabel(tr(label)); self.extra_labels[attribute] = caption
            layout.addWidget(caption)
            spin = QDoubleSpinBox(); spin.setRange(.001 if attribute == 'extra_window' else -1e9, 1e9)
            spin.setDecimals(2); spin.setSingleStep(25)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self.update_extra_range if attribute in ('extra_lower','extra_upper') else self.update_extra)
            setattr(self, attribute, spin); layout.addWidget(spin)
            if attribute in ('extra_lower','extra_upper'):
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0,1000); slider.setAccessibleName(tr(label))
                slider.valueChanged.connect(lambda value,key=attribute:self.slide_extra_range(key,value))
                slider.sliderReleased.connect(self.finish_extra_range_drag)
                setattr(self,attribute+'_slider',slider); layout.addWidget(slider)
        self.extra_colorbar = ImageColorBar(); layout.addWidget(self.extra_colorbar)
        self.extra_surface = QCheckBox(tr('PETを脳表3Dに重ねる'))
        self.extra_surface.setToolTip(tr('皮質内のPET値を脳表に投影します。色・範囲・不透明度は2Dと共通です。'))
        self.extra_surface.toggled.connect(self.change_extra_surface)
        layout.addWidget(self.extra_surface)
        self.extra_info = QLabel(); self.extra_info.setWordWrap(True); self.extra_info.setObjectName('muted')
        layout.addWidget(self.extra_info)
        compare = QPushButton(tr('基準MRIとこの画像を照合'))
        compare.clicked.connect(self.compare_extra); layout.addWidget(compare)
        remove = QPushButton(tr('選択した画像レイヤーを削除'))
        remove.clicked.connect(lambda:self.remove_image_layer(self.extra_combo.currentData()))
        layout.addWidget(remove)
        return self.extra_group

    def remove_image_layer(self, uid):
        if self.scene is None or self.worker is not None or not uid:return
        from .image_removal import removal_ids, remove_images
        removed=removal_ids(self.scene,uid)
        names=[layer.name for layer in self.scene.extra_mris if layer.uid in removed]
        if uid=='ct':names=['CT']
        description=tr('次の画像を現在のレイヤー一覧から削除します。元のDICOM・NIfTIと保存履歴は削除しません。')
        if len(removed)>1:description+='\n'+tr('この画像から作成した補正候補も一覧から削除します。')
        if uid=='ct' and any(c.ct_position is not None for c in self.scene.contacts):
            description+='\n'+tr('電極座標と、電極確認に必要な元CTは保持します。')
        description+='\n\n'+'\n'.join(names)
        dialog=QMessageBox(QMessageBox.Icon.Question,tr('画像レイヤーを削除'),description,
                          QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.Cancel,self)
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        answer=dialog.exec(); dialog.deleteLater()
        if answer!=QMessageBox.StandardButton.Yes:return
        self.segmentation.finish_stroke(); self.segmentation.extraction.clear_preview()
        self.postop.clear_mask_preview()
        state=self.view_state()
        self._restoring=True
        try:
            remove_images(self.scene,uid)
            self.refresh_mri_layers()
            self.extra_styles={k:v for k,v in state.get('extra_styles',{}).items() if k not in removed}
            self.extra_palettes={k:v for k,v in state.get('extra_palettes',{}).items() if k not in removed}
            self.extra_surfaces={k:v for k,v in state.get('extra_surfaces',{}).items() if k not in removed}
            self.image_layers.restore(state.get('image_layers',{}))
            if uid=='ct':self.ct_visible.setChecked(False)
            self.image_layers.rows['ct'].setEnabled(self.scene.ct is not None)
            self.window_target.model().item(1).setEnabled(self.scene.ct is not None)
            selected=state.get('extra_selected')
            if selected not in {layer.uid for layer in self.scene.extra_mris}:
                selected=next((layer.uid for layer in self.scene.extra_mris),None)
            self.extra_combo.setCurrentIndex(max(0,self.extra_combo.findData(selected)))
            target=state.get('window_target')
            self.window_target.setCurrentIndex(max(0,self.window_target.findData(target if target not in removed else 'mri')))
            if not any(row.check.isChecked() for row in self.image_layers.rows.values()):self.mri_visible.setChecked(True)
            self.select_extra(); self.postop.refresh()
            self.segmentation.extraction.reset_scene()
            self.diffusion.select_model(); self.review.refresh(); self.refresh_import_choices(); self.update_electrode_info()
            if uid=='ct':self.ct_info.setText(tr('CT未読込'))
        finally:self._restoring=False
        self.scene.view_state=self.view_state()
        self.update_layers()
        self.message(tr('画像レイヤーを削除しました。「保存」で反映します。元ファイルは保持されています。'))

    def extra_layer(self, uid=None):
        if self.scene is None: return None
        uid = uid or self.extra_combo.currentData()
        return next((layer for layer in self.scene.extra_mris if layer.uid == uid), None)

    def refresh_mri_layers(self):
        for key in list(self.image_layers.rows):
            if key not in ('mri', 'ct'): self.image_layers.remove_layer(key)
        self.extra_styles = {}
        self.extra_palettes = {}
        self.extra_surfaces = {}
        self._extra_domains = {}
        self._extra_render_timer.stop()
        self.extra_combo.blockSignals(True)
        self.extra_combo.clear()
        self.extra_combo.addItem(tr('追加画像を選択'), None)
        self.window_target.blockSignals(True)
        while self.window_target.count() > 2: self.window_target.removeItem(2)
        if self.scene:
            for layer in self.scene.extra_mris:
                row = self.image_layers.add_layer(layer.uid, layer.name[:8]+('…' if len(layer.name)>8 else ''), True, 50)
                row.check.setToolTip(layer.name)
                row.check.setProperty('userText', True)
                row.allow_removal(lambda checked=False,uid=layer.uid:self.remove_image_layer(uid))
                policy = row.check.sizePolicy(); policy.setHorizontalPolicy(QSizePolicy.Policy.Minimum); row.check.setSizePolicy(policy)
                self.extra_combo.addItem(layer.name, layer.uid)
                self.window_target.addItem(layer.name, layer.uid)
        self.window_target.blockSignals(False)
        self.extra_combo.blockSignals(False)
        self.extra_group.setVisible(bool(self.scene and self.scene.extra_mris))
        if self.extra_combo.count() > 1: self.extra_combo.setCurrentIndex(1)
        self.image_layers.rows_layout.removeWidget(self.image_layers.rows['ct'])
        self.image_layers.rows_layout.addWidget(self.image_layers.rows['ct'])

    def select_extra(self, *_):
        layer = self.extra_layer()
        if layer is None: return
        controls = (self.extra_window, self.extra_level, self.extra_lower, self.extra_upper, self.extra_mode, self.extra_palette,
                    self.extra_lower_slider, self.extra_upper_slider, self.extra_surface)
        for control in controls: control.blockSignals(True)
        self.extra_mode.model().item(self.extra_mode.findData('direction')).setEnabled(layer.sequence == 'DTI-FA')
        self.extra_window.setValue(layer.window); self.extra_level.setValue(layer.level)
        self.extra_lower.setValue(layer.level-layer.window/2); self.extra_upper.setValue(layer.level+layer.window/2)
        self.extra_mode.setCurrentIndex(max(0, self.extra_mode.findData(self.extra_styles.get(layer.uid, default_image_mode(layer)))))
        self.extra_palette.setCurrentIndex(max(0, self.extra_palette.findData(self.extra_palettes.get(layer.uid, 'hot'))))
        self.extra_surface.setChecked(self.extra_surfaces.get(layer.uid,False))
        self.extra_surface.setVisible(layer.sequence=='PET')
        self.sync_extra_sliders(layer)
        for control in controls: control.blockSignals(False)
        heatmap = self.extra_mode.currentData() == 'heatmap'
        for attribute, caption in self.extra_labels.items():
            visible = heatmap == (attribute in ('extra_lower','extra_upper'))
            caption.setVisible(visible); getattr(self, attribute).setVisible(visible)
            if attribute in ('extra_lower','extra_upper'): getattr(self,attribute+'_slider').setVisible(visible)
        colored = heatmap or (layer.sequence=='PET' and self.extra_surface.isChecked())
        self.extra_palette.setVisible(colored); self.extra_colorbar.setVisible(colored)
        self.extra_colorbar.set_range(self.extra_palette.currentData(),layer.level-layer.window/2,layer.level+layer.window/2)
        info = tr('基準MRIへ剛体位置合わせ済み・目視確認前\n表示名：{name}').format(name=layer.name)
        if layer.deformation is not None:
            info=tr('術後MRIの変形補正\n表示名：{name}').format(name=layer.name)
            info+='\n'+tr('確認済み' if layer.quality.get('review_status')=='visually_reviewed' else '目視確認前')
        if layer.sequence == 'PET':
            info += '\n' + tr('画像値の単位：{units}（SUVへの変換なし）').format(units=value_units(layer) or tr('未確認'))
        self.extra_info.setText(info)

    def sync_extra_sliders(self, layer):
        low,high=layer.level-layer.window/2,layer.level+layer.window/2
        if layer.uid not in self._extra_domains:
            self._extra_domains[layer.uid]=(min(0.,float(np.nanmin(layer.raw))),float(np.nanmax(layer.raw)))
        a,b=self._extra_domains[layer.uid]
        a,b=min(a,low),max(b,high,a+.01)
        self._extra_domains[layer.uid]=(a,b)
        for slider,value in ((self.extra_lower_slider,low),(self.extra_upper_slider,high)):
            slider.setValue(int(round((value-a)/(b-a)*1000)))

    def slide_extra_range(self, attribute, value):
        layer=self.extra_layer()
        if self._restoring or layer is None: return
        a,b=self._extra_domains[layer.uid]
        value=a+value/1000*(b-a); gap=max(.01,(b-a)/1000)
        low,high=layer.level-layer.window/2,layer.level+layer.window/2
        if attribute=='extra_lower': low=min(value,high-gap)
        else: high=max(value,low+gap)
        layer.window,layer.level=high-low,(high+low)/2
        self.select_extra()
        if not self._extra_render_timer.isActive(): self._extra_render_timer.start()

    def finish_extra_range_drag(self):
        self._extra_render_timer.stop(); self.update_layers()

    def change_extra_surface(self, enabled):
        layer=self.extra_layer()
        if self._restoring or layer is None: return
        self.extra_surfaces[layer.uid]=bool(enabled)
        self.select_extra(); self.update_layers()

    def change_extra_mode(self, *_):
        layer = self.extra_layer()
        if self._restoring or layer is None: return
        self.extra_styles[layer.uid] = self.extra_mode.currentData()
        self.select_extra(); self.update_layers()

    def change_extra_palette(self, *_):
        layer = self.extra_layer()
        if self._restoring or layer is None: return
        self.extra_palettes[layer.uid] = self.extra_palette.currentData()
        self.select_extra(); self.update_layers()

    def update_extra_range(self, *_):
        layer = self.extra_layer()
        if self._restoring or layer is None: return
        low, high = self.extra_lower.value(), self.extra_upper.value()
        if high <= low:
            if self.sender() is self.extra_lower: high = low+.01
            else: low = high-.01
        layer.window, layer.level = high-low, (high+low)/2
        self.select_extra(); self.update_layers()

    def update_extra(self, *_):
        if self._restoring: return
        layer = self.extra_layer()
        if layer is None: return
        layer.window, layer.level = self.extra_window.value(), self.extra_level.value()
        self.select_extra()
        self.update_layers()

    def extra_options(self):
        if self.scene is None: return {}
        return {layer.uid: dict(self.image_layers.rows[layer.uid].state(), window=layer.window, level=layer.level,
                               mode=self.extra_styles.get(layer.uid, default_image_mode(layer)),
                               surface_visible=self.extra_surfaces.get(layer.uid,False),
                               palette=self.extra_palettes.get(layer.uid, 'hot')) for layer in self.scene.extra_mris}

    def compare_extra(self):
        layer = self.extra_layer()
        if layer is None: return
        for key, row in self.image_layers.rows.items():
            row.check.setChecked(key in ('mri', layer.uid))
        self.mri_opacity.setValue(100)
        self.image_layers.rows[layer.uid].slider.setValue(100)
        self.extra_mode.setCurrentIndex(self.extra_mode.findData('checker'))
        self.window_target.setCurrentIndex(self.window_target.findData(layer.uid))
        self.message(tr('基準MRIと追加画像の位置関係を確認してください。'))
