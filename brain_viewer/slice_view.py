"""Qt raster slices with physical aspect ratio and shared RAS crosshair."""
from __future__ import annotations
from .i18n import tr

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .imaging import Scene, display_position, index_from_display, plane_axes, plane_pixels
from .rendering import composite_plane


PLANE_INFO = {
    2: ("Axial", tr("水平断"), "#69d4df", ("R", "L", "A", "P")),
    1: ("Coronal", tr("冠状断"), "#f1c87b", ("R", "L", "S", "I")),
    0: ("Sagittal", tr("矢状断"), "#ad9ce8", ("A", "P", "S", "I")),
}


class SliceCanvas(QWidget):
    cursor_changed = Signal(object)
    slice_step = Signal(int)
    window_delta = Signal(float, float)
    brush_requested = Signal(int, object, str)
    contour_requested = Signal(int, object)
    focus_requested = Signal()

    def __init__(self, axis: int, parent=None):
        super().__init__(parent)
        self.axis = axis
        self.scene: Scene | None = None
        self.ijk = np.zeros(3, dtype=int)
        self.low, self.high = 0., 1.
        self.zoom = 1.0
        self.pan = QPointF()
        self._last_mouse = QPointF()
        self._panning = False
        self._windowing = False
        self.brush_mode = 'navigate'
        self.brush_radius = 2.5
        self._painting = False
        self._contour = []
        self._tract_cache = {}
        self._hover = None
        self.annotation_enabled = True
        self.show_annotation = False
        self.ct_options = {}
        self.contact_options = {"visible": True, "group": "", "source": False, "tolerance": 2.0}
        self._image = QImage()
        self.setMinimumSize(180, 170)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip(tr("マウス位置：解剖ラベル / クリック：位置を移動\n右ドラッグ：横＝window幅、縦＝level\nホイール：断面を移動\nCtrl＋ホイール：拡大縮小\n中央ドラッグ：平行移動"))

    def set_scene(self, scene: Scene):
        self._contour = []; self._tract_cache = {}
        self.set_brush('navigate', self.brush_radius)
        self.scene = scene
        self._hover = None
        self.ijk = scene.initial_index()
        self.low, self.high = scene.contrast_limits()
        self.show_annotation = False
        self.reset_view()
        self._update_image()

    def clear_scene(self):
        self._contour = []; self._tract_cache = {}
        self.set_brush('navigate', self.brush_radius)
        self.scene = None; self._image = QImage(); self.show_annotation = False; self._hover = None
        self.update()

    def set_cursor(self, ijk):
        changed = int(ijk[self.axis]) != self.ijk[self.axis]
        self.ijk = np.array(ijk, dtype=int).copy()
        if changed:
            self._update_image()
        self.update()

    def set_contrast(self, low: float, high: float):
        self.low, self.high = low, max(high, low + .001)
        self._update_image()

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF()
        self.update()

    def _update_image(self):
        if self.scene is None:
            return
        raw = composite_plane(self.scene, self.axis, self.ijk[self.axis], self.low, self.high, self.ct_options)
        rgb = np.ascontiguousarray(raw.transpose(1, 0, 2)[::-1, ::-1])
        self._image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def set_layers(self, ct_options=None, contact_options=None, annotations=None):
        if ct_options is not None:
            self.ct_options = dict(ct_options)
        if contact_options is not None:
            self.contact_options = dict(contact_options)
        if annotations is not None:
            self.annotation_enabled = bool(annotations)
        self._update_image()

    def image_rect(self) -> QRectF:
        if self.scene is None:
            return QRectF()
        h, v = plane_axes(self.axis)
        width = self.scene.data.shape[h] * self.scene.spacing[h]
        height = self.scene.data.shape[v] * self.scene.spacing[v]
        scale = min(max(1, self.width() - 52) / width, max(1, self.height() - 38) / height) * self.zoom
        w, d = width * scale, height * scale
        return QRectF((self.width() - w) / 2 + self.pan.x(), (self.height() - d) / 2 + self.pan.y(), w, d)

    def position_for_index(self, ijk) -> QPointF:
        r = self.image_rect()
        h, v = plane_axes(self.axis)
        u, w = display_position(self.scene.data.shape, self.axis, ijk)
        return QPointF(r.left() + (u + .5) / self.scene.data.shape[h] * r.width(),
                       r.top() + (w + .5) / self.scene.data.shape[v] * r.height())

    def _index_at(self, position: QPointF):
        if self.scene is None:
            return
        r = self.image_rect()
        if not r.contains(position):
            return
        h, v = plane_axes(self.axis)
        uv = [(position.x() - r.left()) / r.width() * self.scene.data.shape[h] - .5,
              (position.y() - r.top()) / r.height() * self.scene.data.shape[v] - .5]
        return index_from_display(self.scene.data.shape, self.axis, uv, self.ijk)

    def _select(self, position: QPointF):
        index = self._index_at(position)
        if index is None: return
        self.show_annotation = True
        self.cursor_changed.emit(index)
        self._update_image()

    def hover_annotation(self):
        if (not self.annotation_enabled or self._hover is None or self.brush_mode!='navigate'
                or self._panning or self._windowing): return None
        index=self._index_at(self._hover)
        if index is None: return None
        label,name=self.scene.annotation(index)
        return {'label':label,'name':name,'index':index}

    def contact_is_hovered(self, center):
        return (self._hover is not None and self.brush_mode=='navigate' and not self._panning and not self._windowing
                and (center-self._hover).manhattanLength()<=12)

    def set_brush(self, mode, radius):
        self._contour = []
        self._end_stroke()
        self.brush_mode, self.brush_radius = mode, radius
        self.setCursor(Qt.CursorShape.CrossCursor if mode != 'navigate' else Qt.CursorShape.ArrowCursor)
        if mode != 'navigate': self.show_annotation = False
        self._update_image(); self.update()

    def _end_stroke(self):
        if self._painting:
            self._painting = False
            self.brush_requested.emit(self.axis, None, 'end')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080f19"))
        if self.scene is None:
            painter.setPen(QColor("#75849a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr("MRIを準備しています"))
            return
        r = self.image_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(r, self._image)
        self._paint_contacts(painter, r)
        self._paint_results(painter, r)
        from .diffusion_rendering import paint_roi_spheres, paint_tracts
        paint_tracts(self,painter,r)
        paint_roi_spheres(self,painter,r)
        center = self.position_for_index(self.ijk)
        painter.setPen(QPen(QColor(PLANE_INFO[self.axis][2]), 1.0))
        painter.save()
        painter.setClipRect(r.intersected(QRectF(self.rect())))
        gap = 4
        painter.drawLine(QPointF(r.left(), center.y()), QPointF(center.x() - gap, center.y()))
        painter.drawLine(QPointF(center.x() + gap, center.y()), QPointF(r.right(), center.y()))
        painter.drawLine(QPointF(center.x(), r.top()), QPointF(center.x(), center.y() - gap))
        painter.drawLine(QPointF(center.x(), center.y() + gap), QPointF(center.x(), r.bottom()))
        painter.restore()
        if self._contour:
            painter.setPen(QPen(QColor('#ffe48f'), 2))
            painter.setBrush(QColor(255, 220, 120, 40))
            painter.drawPolygon(QPolygonF([self.position_for_index(p) for p in self._contour]))
        if self.brush_mode in ('paint','erase') and self._hover is not None and r.contains(self._hover):
            h, v = plane_axes(self.axis)
            rx = self.brush_radius*r.width()/(self.scene.data.shape[h]*self.scene.spacing[h])
            ry = self.brush_radius*r.height()/(self.scene.data.shape[v]*self.scene.spacing[v])
            painter.setPen(QPen(QColor('#ffab9d' if self.brush_mode == 'erase' else '#e9ffff'), 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self._hover, rx, ry)
        painter.setPen(QColor("#c2cedb"))
        left, right, top, bottom = PLANE_INFO[self.axis][3]
        for text, box in ((left, QRectF(7, self.height()/2-10, 24, 20)),
                          (right, QRectF(self.width()-31, self.height()/2-10, 24, 20)),
                          (top, QRectF(self.width()/2-12, 3, 24, 20)),
                          (bottom, QRectF(self.width()/2-12, self.height()-23, 24, 20))):
            painter.fillRect(box, QColor(8, 15, 25, 190))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        annotation=self.hover_annotation()
        if annotation is not None:
            label, name = annotation['label'],annotation['name']
            source = tr("合成ラベル") if self.scene.kind == "synthetic" else "FreeSurfer"
            content = name + (f"\n{source}  ·  ID {label}" if label else "")
            width = min(self.width()-28, max(160, painter.fontMetrics().horizontalAdvance(name)+22))
            height = 49 if label else 31
            x = min(max(12, self._hover.x()+14), self.width()-width-12)
            y = min(max(12, self._hover.y()+14), self.height()-height-12)
            box = QRectF(x, y, width, height)
            painter.setBrush(QColor(19, 43, 53, 240))
            painter.setPen(QPen(QColor("#58cbb7"), 1))
            painter.drawRoundedRect(box, 5, 5)
            painter.setPen(QColor("#e2fff7"))
            painter.drawText(box.adjusted(10, 5, -8, -5), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, content)

    def _paint_contacts(self, painter, rect):
        if not self.contact_options.get("visible", True) or not self.scene.contacts:
            return
        positions = self.scene.contact_positions(self.contact_options.get("source", False))
        plane = self.scene.world(self.ijk)[self.axis]
        tolerance = self.contact_options.get("tolerance", 2.)
        indices = self.scene.index(positions)
        painter.save()
        painter.setClipRect(rect.intersected(QRectF(self.rect())))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for contact, world, index in zip(self.scene.contacts, positions, indices):
            group = self.contact_options.get("group", "")
            if group and contact.group != group or abs(world[self.axis]-plane) > tolerance:
                continue
            center = self.position_for_index(index)
            color = QColor('#909eae') if getattr(self,'result_active',False) else QColor.fromRgbF(*contact.color)
            painter.setPen(QPen(color, 1.6))
            painter.setBrush(QColor(0, 0, 0, 70))
            painter.drawEllipse(center, 3.7, 3.7)
            if contact.name == self.contact_options.get("selected", ""):
                painter.drawEllipse(center, 6.2, 6.2)
            if self.contact_is_hovered(center):
                painter.drawText(center + QPointF(8, -7), contact.name)
        painter.restore()

    def _paint_results(self,painter,rect):
        positions=getattr(self,'result_points',[])
        if not len(positions): return
        plane=self.scene.world(self.ijk)[self.axis]
        options=getattr(self,'result_options',{})
        labels=getattr(self,'result_labels',[])
        h,_=plane_axes(self.axis)
        radius=options.get('radius_mm',2.)*rect.width()/(self.scene.data.shape[h]*self.scene.spacing[h])
        painter.save(); painter.setClipRect(rect.intersected(QRectF(self.rect())))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(options.get('opacity',1.))
        for i,(world,index,rgb) in enumerate(zip(positions,self.scene.index(positions),self.result_colors)):
            if abs(world[self.axis]-plane)>options.get('slab_mm',4.)/2: continue
            color=QColor.fromRgbF(*rgb); painter.setBrush(color); painter.setPen(QPen(QColor('#e8f3ff'),.8))
            center=self.position_for_index(index);painter.drawEllipse(center,radius,radius)
            if options.get('labels') and i<len(labels):
                painter.setPen(QColor('#eff6ff'));painter.drawText(center+QPointF(radius+3,-3),labels[i])
        painter.restore()

    def mousePressEvent(self, event):
        self.setFocus()
        self._last_mouse = event.position()
        self._hover = event.position()
        self._windowing = event.button() == Qt.MouseButton.RightButton
        if self._windowing:
            event.accept()
            return
        self._panning = event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        if event.button() == Qt.MouseButton.LeftButton and not self._panning:
            if self.brush_mode in ('contour','contour_erase','tract_exclude'):
                point = self._index_at(event.position())
                if point is not None: self._contour = [np.asarray(point,float)]
                self.update()
            elif self.brush_mode != 'navigate':
                point = self._index_at(event.position())
                if point is not None:
                    self._painting = True
                    self.brush_requested.emit(self.axis, point, 'start')
            else:
                self._select(event.position())

    def mouseMoveEvent(self, event):
        self._hover = event.position()
        if self._windowing:
            delta = event.position() - self._last_mouse
            self.window_delta.emit(delta.x(), delta.y())
        elif self._panning:
            self.pan += event.position() - self._last_mouse
            self.update()
        elif self._contour and event.buttons() & Qt.MouseButton.LeftButton:
            point = self._index_at(event.position())
            if point is not None and np.linalg.norm(np.asarray(point)-self._contour[-1]) >= .5:
                self._contour.append(np.asarray(point,float))
            self.update()
        elif self._painting:
            point = self._index_at(event.position())
            if point is None or not event.buttons() & Qt.MouseButton.LeftButton: self._end_stroke()
            else: self.brush_requested.emit(self.axis, point, 'move')
        elif event.buttons() & Qt.MouseButton.LeftButton and self.brush_mode == 'navigate':
            self._select(event.position())
        self._last_mouse = event.position()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._contour:
            point = self._index_at(event.position())
            if point is not None: self._contour.append(np.asarray(point,float))
            contour, self._contour = self._contour, []
            if len(contour) >= 3: self.contour_requested.emit(self.axis, contour)
            self.update()
        self._end_stroke()
        self._panning = False
        self._windowing = False
        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._contour = []
            self._end_stroke()
            self.focus_requested.emit()
            event.accept()

    def leaveEvent(self, event):
        self._hover = None; self.show_annotation = False; self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        self._contour = []
        self._end_stroke()
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not delta:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom = float(np.clip(self.zoom * (1.15 if delta > 0 else 1/1.15), .3, 8))
            self.update()
        else:
            self.slice_step.emit(1 if delta > 0 else -1)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._contour = []; self._end_stroke(); self.update(); event.accept(); return
        self._end_stroke()
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Right):
            self.slice_step.emit(1)
        elif event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Left):
            self.slice_step.emit(-1)
        elif event.key() == Qt.Key.Key_Home:
            self.reset_view()
        else:
            super().keyPressEvent(event)


class SlicePanel(QFrame):
    def __init__(self, axis: int, parent=None):
        super().__init__(parent)
        self.axis = axis
        self.setObjectName("viewPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        title = QHBoxLayout()
        title.setContentsMargins(12, 8, 12, 6)
        label = QLabel(f"{PLANE_INFO[axis][0]}  /  {PLANE_INFO[axis][1]}")
        self.name_label = label
        label.setStyleSheet(f"color:{PLANE_INFO[axis][2]}; font-weight:600;")
        title.addWidget(label)
        title.addStretch()
        self.location = QLabel("—")
        self.location.setObjectName("muted")
        title.addWidget(self.location)
        layout.addLayout(title)
        self.canvas = SliceCanvas(axis)
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        controls.setContentsMargins(12, 5, 12, 7)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        controls.addWidget(self.slider, 1)
        self.counter = QLabel("— / —")
        self.counter.setMinimumWidth(72)
        self.counter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.counter.setObjectName("muted")
        controls.addWidget(self.counter)
        layout.addLayout(controls)

    def set_scene(self, scene: Scene):
        self.canvas.set_scene(scene)
        self.slider.blockSignals(True)
        self.slider.setRange(0, scene.data.shape[self.axis] - 1)
        self.slider.setValue(int(self.canvas.ijk[self.axis]))
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)

    def clear_scene(self):
        self.canvas.clear_scene()
        self.slider.setEnabled(False); self.location.setText('—'); self.counter.setText('— / —')

    def set_cursor(self, ijk):
        self.canvas.set_cursor(ijk)
        value = int(ijk[self.axis])
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.counter.setText(f"{value+1} / {self.slider.maximum()+1}")
        world = self.canvas.scene.world(ijk)
        self.location.setText(f"{'RAS'[self.axis]} {world[self.axis]:+.1f} mm")
