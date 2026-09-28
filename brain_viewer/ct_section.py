"""Navigable native-CT sections; viewport changes never move contact coordinates."""
import numpy as np
from PySide6.QtCore import QPointF,QRectF,Qt,Signal
from PySide6.QtGui import QColor,QImage,QPainter,QPen
from PySide6.QtWidgets import QWidget,QHBoxLayout,QLabel,QPushButton,QSizePolicy
from .i18n import tr
from .electrode_localization import sample_ct,orthogonal_basis
from .electrode_editing import contact_axis


STATUS_COLOR={'unreviewed':'#ffca72','uncertain':'#ff737d','reviewed':'#74dec1'}


class CTSection(QWidget):
    selected=Signal(int)
    moved=Signal(object)
    window_delta=Signal(float,float)
    MIN_ZOOM=.1
    MAX_ZOOM=10.

    def __init__(self,mode,parent=None):
        super().__init__(parent)
        self.mode=mode;self.scene=None;self.contacts=[];self.index=0
        self.image=QImage();self.edit=False;self.center=np.zeros(3)
        self.u=np.array([1.,0.,0.]);self.v=np.array([0.,1.,0.]);self.base_extent=np.array([60.,12.])
        self.zoom=1.;self.pan=np.zeros(2);self.width_value=4000.;self.level_value=1000.
        self._context=None;self._selected_uid=None;self._sample_key=None;self.raw=None
        self._last=QPointF();self._panning=False
        self.setMinimumSize(220,160);self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(tr('ホイール：拡大・縮小 / Shift＋左ドラッグ・中ドラッグ：表示移動 / Home：リセット'))
        self.header=QWidget(self);row=QHBoxLayout(self.header);row.setContentsMargins(10,2,10,2);row.setSpacing(4)
        title=tr(('電極に沿うCT断面 1','電極に沿うCT断面 2','選択コンタクトの横断面')[mode])
        self.title=QLabel(title);self.title.setToolTip(title);self.title.setStyleSheet('color:#90cbd5;')
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        row.addWidget(self.title,1)
        self.zoom_out=QPushButton('−');self.zoom_in=QPushButton('+');self.reset_button=QPushButton(tr('リセット'))
        self.percent=QLabel('100%');self.percent.setFixedWidth(45);self.percent.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.percent.setToolTip(tr('標準の表示範囲を100%とした倍率'))
        for button,label,width in ((self.zoom_out,'縮小',28),(self.zoom_in,'拡大',28),(self.reset_button,'表示範囲をリセット',64)):
            button.setStyleSheet('padding:0;');button.setFixedSize(width,26)
            button.setAutoDefault(False);button.setToolTip(tr(label));button.setAccessibleName(tr(label))
        row.addWidget(self.zoom_out);row.addWidget(self.percent);row.addWidget(self.zoom_in);row.addWidget(self.reset_button)
        self.zoom_out.clicked.connect(lambda:self.zoom_by(1/1.25))
        self.zoom_in.clicked.connect(lambda:self.zoom_by(1.25))
        self.reset_button.clicked.connect(self.reset_view)
        self.update_controls()

    @property
    def extent(self):
        return self.base_extent/self.zoom

    @property
    def view_center(self):
        return self.center+self.pan[0]*self.u+self.pan[1]*self.v

    def configure(self,scene,contacts,index,width,level):
        context=(id(scene.raw_ct),contacts[0].group) if contacts else None
        if context!=self._context:
            self.zoom=1.;self.pan[:]=0;self._context=context
        selected_uid=contacts[index].uid if contacts else None
        if self.mode==2 and selected_uid!=self._selected_uid:self.pan[:]=0
        self._selected_uid=selected_uid
        self.scene=scene;self.contacts=contacts;self.index=index
        self.width_value=max(float(width),1.);self.level_value=float(level)
        if not contacts:
            self.image=QImage();self.raw=None;self._sample_key=None;self.update_controls();self.update();return
        axis,_=contact_axis(contacts);axis,u,v=orthogonal_basis(axis)
        points=np.array([c.ct_position for c in contacts])
        if self.mode==2:
            self.center=points[index].copy();self.u=u;self.v=v;self.base_extent=np.array([14.,14.])
        else:
            self.center=(points[0]+points[-1])/2;self.u=axis;self.v=u if self.mode==0 else v
            self.base_extent=np.array([max(30.,np.linalg.norm(points[-1]-points[0])+16),12.])
        self.refresh_image()

    def refresh_image(self):
        if self.scene is None or not self.contacts:return
        extent=self.extent;center=self.view_center
        # Bounded native-CT sampling. Zooming out samples new surrounding anatomy;
        # window/level changes reuse the sampled plane and do not read/resample CT.
        w=max(150,min(1200,int(extent[0]/.12)));h=max(80,min(320,int(extent[1]/.12)))
        key=(id(self.scene.raw_ct),tuple(self.scene.raw_ct_affine.ravel()),tuple(center),
             tuple(self.u),tuple(self.v),tuple(extent),w,h)
        if key!=self._sample_key:
            x=(np.arange(w)+.5)/w*extent[0]-extent[0]/2
            y=extent[1]/2-(np.arange(h)+.5)/h*extent[1]
            coordinates=center+x[None,:,None]*self.u+y[:,None,None]*self.v
            self.raw=sample_ct(self.scene.raw_ct,self.scene.raw_ct_affine,coordinates.reshape(-1,3)).reshape(h,w)
            self._sample_key=key
        gray=np.ascontiguousarray(np.clip((self.raw-self.level_value+self.width_value/2)*255/self.width_value,0,255).astype(np.uint8))
        self.image=QImage(gray.data,w,h,gray.strides[0],QImage.Format.Format_Grayscale8).copy()
        self.update_controls();self.update()

    def update_controls(self):
        ready=bool(self.contacts)
        self.percent.setText(f'{self.zoom*100:.0f}%')
        self.zoom_out.setEnabled(ready and self.zoom>self.MIN_ZOOM+1e-8)
        self.zoom_in.setEnabled(ready and self.zoom<self.MAX_ZOOM-1e-8)
        self.reset_button.setEnabled(ready)

    def reset_view(self):
        self.zoom=1.;self.pan[:]=0;self.refresh_image()

    def image_rect(self):
        area=QRectF(12,34,max(1,self.width()-24),max(1,self.height()-52))
        scale=min(area.width()/self.extent[0],area.height()/self.extent[1])
        w,h=self.extent*scale
        return QRectF(area.center().x()-w/2,area.center().y()-h/2,w,h)

    def plane_at(self,position):
        r=self.image_rect()
        x=(position.x()-r.center().x())/r.width()*self.extent[0]
        y=-(position.y()-r.center().y())/r.height()*self.extent[1]
        return self.view_center+x*self.u+y*self.v

    def project(self,point):
        delta=np.asarray(point)-self.view_center;r=self.image_rect()
        return QPointF(r.center().x()+np.dot(delta,self.u)/self.extent[0]*r.width(),
                       r.center().y()-np.dot(delta,self.v)/self.extent[1]*r.height())

    def zoom_by(self,factor,anchor=None):
        if not self.contacts or not np.isfinite(factor) or factor<=0:return
        target=float(np.clip(self.zoom*factor,self.MIN_ZOOM,self.MAX_ZOOM))
        if abs(target-self.zoom)<1e-10:return
        r=self.image_rect()
        anchor=anchor if anchor is not None and r.contains(anchor) else r.center()
        before=self.plane_at(anchor);self.zoom=target;delta=before-self.plane_at(anchor)
        self.pan+=np.array([np.dot(delta,self.u),np.dot(delta,self.v)])
        self.refresh_image()

    def pan_by(self,delta):
        if not self.contacts:return
        r=self.image_rect()
        self.pan+=np.array([-delta.x()/r.width()*self.extent[0],delta.y()/r.height()*self.extent[1]])
        self.refresh_image()

    def resizeEvent(self,event):
        super().resizeEvent(event);self.header.setGeometry(0,0,self.width(),30)

    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#080f19'))
        if self.image.isNull():return
        r=self.image_rect();p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform);p.drawImage(r,self.image)
        p.save();p.setClipRect(r);p.setRenderHint(QPainter.RenderHint.Antialiasing)
        normal=np.cross(self.u,self.v)
        for i,c in enumerate(self.contacts):
            if self.mode==2 and abs(np.dot(c.ct_position-self.center,normal))>2:continue
            point=self.project(c.ct_position)
            p.setPen(QPen(QColor(STATUS_COLOR.get(c.status,'#ffca72')),2 if i==self.index else 1))
            p.setBrush(Qt.BrushStyle.NoBrush);p.drawEllipse(point,5,5)
            if i==self.index:p.drawEllipse(point,8,8)
            if self.mode!=2:p.drawText(point+QPointF(-4,-11),str(i+1))
        p.restore();p.setPen(QColor('#a0b4c3'))
        text=tr('元CT・表示範囲 {width} × {height} mm').format(width=f'{self.extent[0]:.1f}',height=f'{self.extent[1]:.1f}')
        p.drawText(12,self.height()-4,p.fontMetrics().elidedText(text,Qt.TextElideMode.ElideRight,max(1,self.width()-24)))

    def mousePressEvent(self,event):
        self._last=event.position();self.setFocus()
        if not self.contacts or not self.image_rect().contains(event.position()):return
        if event.button()==Qt.MouseButton.MiddleButton or (event.button()==Qt.MouseButton.LeftButton and event.modifiers()&Qt.KeyboardModifier.ShiftModifier):
            self._panning=True;self.setCursor(Qt.CursorShape.ClosedHandCursor);event.accept();return
        if event.button()!=Qt.MouseButton.LeftButton:return
        visible=[i for i,c in enumerate(self.contacts) if self.image_rect().contains(self.project(c.ct_position))
                 and (self.mode!=2 or abs(np.dot(c.ct_position-self.center,np.cross(self.u,self.v)))<=2)]
        nearest=min(visible,key=lambda i:(self.project(self.contacts[i].ct_position)-event.position()).manhattanLength()) if visible else None
        # Clicking another marker always selects it, including in move mode.
        # A nearby click around the selected double ring remains a precise edit.
        if nearest is not None and nearest!=self.index and (self.project(self.contacts[nearest].ct_position)-event.position()).manhattanLength()<=10:
            self.selected.emit(nearest); event.accept(); return
        if self.edit:
            if self.project(self.contacts[self.index].ct_position).toPoint()==event.position().toPoint():
                event.accept(); return
            # Preserve depth relative to the displayed plane even after zoom/pan.
            normal=np.cross(self.u,self.v)
            offset=np.dot(self.contacts[self.index].ct_position-self.view_center,normal)
            self.moved.emit(self.plane_at(event.position())+offset*normal)
        else:
            if nearest is not None: self.selected.emit(nearest)
        event.accept()

    def mouseMoveEvent(self,event):
        delta=event.position()-self._last
        if self._panning:self.pan_by(delta)
        elif event.buttons()&Qt.MouseButton.RightButton:self.window_delta.emit(delta.x(),delta.y())
        self._last=event.position()

    def mouseReleaseEvent(self,event):
        if self._panning:
            self._panning=False;self.unsetCursor();event.accept()
        else:super().mouseReleaseEvent(event)

    def wheelEvent(self,event):
        if self.contacts and not self.image.isNull():
            delta=event.angleDelta().y()/120 if event.angleDelta().y() else event.pixelDelta().y()/40
            if delta:self.zoom_by(1.25**float(np.clip(delta,-10,10)),event.position())
        event.accept()

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Home:self.reset_view();event.accept()
        else:super().keyPressEvent(event)
