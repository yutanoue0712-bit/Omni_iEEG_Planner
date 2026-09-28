"""Patient-RAS oblique views along an existing SEEG lead; no inferred surgical plan."""
import numpy as np
from scipy.ndimage import map_coordinates
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QLabel
from .i18n import tr
from .electrode_localization import orthogonal_basis


def lead_frame(contacts):
    """Fit the RAS axis, oriented from the first listed contact to the last."""
    points=np.array([c.position for c in contacts],float)
    if len(points)<2 or np.linalg.norm(points[-1]-points[0])<.01:
        raise ValueError('Two distinct contacts are required')
    _,_,basis=np.linalg.svd(points-points.mean(0),full_matrices=False)
    axis=basis[0]
    if np.dot(axis,points[-1]-points[0])<0:axis=-axis
    axis,u,v=orthogonal_basis(axis)
    return points[0],axis,u,v,float(np.dot(points[-1]-points[0],axis))


def sample_reference(data, scene, coordinates, fill=0., order=1):
    return map_coordinates(data,scene.index(coordinates.reshape(-1,3)).T,
                           order=order,mode='constant',cval=fill,prefilter=False).reshape(coordinates.shape[:2])


class TrajectoryCanvas(QWidget):
    focus_requested=Signal()
    step_requested=Signal(int)

    def __init__(self,panel):
        super().__init__(panel);self.panel=panel;self.image=QImage();self._last=QPointF()
        self.setMinimumSize(180,160)
        self.center=np.zeros(3);self.u=np.eye(3)[0];self.v=np.eye(3)[1];self.extent=(80.,80.)

    def image_rect(self):
        scale=min(max(1,self.width()-22)/self.extent[0],max(1,self.height()-30)/self.extent[1])
        w,h=np.array(self.extent)*scale
        return QRectF((self.width()-w)/2,(self.height()-h)/2,w,h)

    def project(self,point):
        r=self.image_rect();d=np.asarray(point)-self.center
        return QPointF(r.center().x()+np.dot(d,self.u)*r.width()/self.extent[0],
                       r.center().y()-np.dot(d,self.v)*r.height()/self.extent[1])

    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#080f19'))
        if self.image.isNull():
            p.setPen(QColor('#97b2c8'))
            p.drawText(self.rect().adjusted(12,12,-12,-12),Qt.AlignmentFlag.AlignCenter|Qt.TextFlag.TextWordWrap,
                       tr('SEEG電極を読み込むと、電極に沿う断面を表示できます。'))
            return
        r=self.image_rect();p.drawImage(r,self.image);p.save();p.setClipRect(r)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor('#ffcd78'),1.3));p.setBrush(Qt.BrushStyle.NoBrush)
        normal=np.cross(self.u,self.v)
        for contact in self.panel.contacts:
            if self.panel.plane.currentData()=='cross' and abs(np.dot(contact.position-self.center,normal))>1.5:continue
            pos=self.project(contact.position);p.drawEllipse(pos,4,4);p.drawText(pos+QPointF(6,-5),contact.name)
        if self.panel.frame is not None:
            origin,axis,_,_,_=self.panel.frame
            p.setPen(QPen(QColor('#6bdae5'),1,Qt.PenStyle.DashLine))
            pos=self.project(origin+axis*self.panel.distance())
            p.drawLine(QPointF(pos.x(),r.top()),QPointF(pos.x(),r.bottom()))
            p.drawLine(QPointF(r.left(),pos.y()),QPointF(r.right(),pos.y()))
        p.restore()
        # Explicit physical orientation for oblique sections.
        def label(vector):
            a=int(np.argmax(np.abs(vector)));return ('R','A','S')[a] if vector[a]>=0 else ('L','P','I')[a]
        p.setPen(QColor('#aac6d4'))
        p.drawText(4,self.height()//2,label(-self.u))
        p.drawText(self.width()-14,self.height()//2,label(self.u))
        p.drawText(self.width()//2,13,label(self.v))
        p.drawText(self.width()//2,self.height()-3,label(-self.v))

    def mouseDoubleClickEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:self.focus_requested.emit();event.accept()

    def mousePressEvent(self,event):
        self._last=event.position()
        if event.button()==Qt.MouseButton.LeftButton and not self.image.isNull():
            r=self.image_rect()
            if r.contains(event.position()):
                x=(event.position().x()-r.center().x())/r.width()*self.extent[0]
                y=-(event.position().y()-r.center().y())/r.height()*self.extent[1]
                self.panel.window.select_world(self.center+x*self.u+y*self.v)

    def mouseMoveEvent(self,event):
        if event.buttons() & Qt.MouseButton.RightButton:
            delta=event.position()-self._last;self.panel.window.drag_window(delta.x(),delta.y())
        self._last=event.position()

    def wheelEvent(self,event):
        delta=event.angleDelta().y() or event.pixelDelta().y()
        if delta:self.step_requested.emit(1 if delta>0 else -1)
        event.accept()


class TrajectoryPanel(QFrame):
    def __init__(self,window):
        super().__init__();self.window=window;self.contacts=[];self.frame=None;self.signature=None
        self.setObjectName('viewPanel');layout=QVBoxLayout(self);layout.setContentsMargins(4,3,4,3)
        row=QHBoxLayout();self.groups=QComboBox();self.groups.setObjectName('dataNames');row.addWidget(self.groups,1)
        self.plane=QComboBox()
        for text,key in [('横断面','cross'),('電極に沿う断面','along')]:self.plane.addItem(tr(text),key)
        row.addWidget(self.plane);layout.addLayout(row)
        self.source=QComboBox()
        for text,key in [('T1 + CT','both'),('T1','mri'),('CT','ct')]:self.source.addItem(text,key)
        self.source.setToolTip(tr('Trajectory断面に表示する画像'));layout.addWidget(self.source)
        self.canvas=TrajectoryCanvas(self);layout.addWidget(self.canvas,1)
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setRange(-20,200);layout.addWidget(self.slider)
        self.info=QLabel();self.info.setWordWrap(True);self.info.setObjectName('muted');layout.addWidget(self.info)
        self.groups.currentIndexChanged.connect(self.select_group)
        self.plane.currentIndexChanged.connect(self.refresh_image);self.source.currentIndexChanged.connect(self.refresh_image)
        self.slider.valueChanged.connect(self.move)
        self.canvas.step_requested.connect(lambda n:self.slider.setValue(self.slider.value()+n))

    def refresh(self):
        scene=self.window.scene
        contacts=scene.contacts if scene else []
        sig=(id(scene),tuple((c.uid,c.group,tuple(c.position)) for c in contacts if c.kind=='SEEG'))
        if sig!=self.signature:
            self.signature=sig;current=self.groups.currentData()
            self.groups.blockSignals(True);self.groups.clear()
            for name in dict.fromkeys(c.group for c in contacts if c.kind=='SEEG'):
                if sum(c.group==name and c.kind=='SEEG' for c in contacts)>=2:self.groups.addItem(name,name)
            self.groups.setCurrentIndex(max(0,self.groups.findData(current)));self.groups.blockSignals(False)
            self.select_group()
        else:self.refresh_image()

    def select_group(self,*_):
        scene=self.window.scene
        self.contacts=[c for c in scene.contacts if c.group==self.groups.currentData() and c.kind=='SEEG'] if scene else []
        try:self.frame=lead_frame(self.contacts)
        except ValueError:self.frame=None
        self.slider.setEnabled(self.frame is not None)
        if self.frame is not None:
            self.slider.blockSignals(True);self.slider.setRange(-20,int(np.ceil(self.frame[4]*2))+20)
            self.slider.setValue(0);self.slider.blockSignals(False)
        self.refresh_image()

    def distance(self):return self.slider.value()/2

    def move(self,*_):
        self.refresh_image()
        if self.frame is not None:self.window.select_world(self.frame[0]+self.frame[1]*self.distance())

    def refresh_image(self,*_):
        scene=self.window.scene;canvas=self.canvas
        if self.frame is None or scene is None:
            canvas.image=QImage();canvas.update();self.info.setText(tr('既存SEEGの確認用。計画ルートは今後追加。'));return
        origin,axis,u,v,length=self.frame
        if self.plane.currentData()=='cross':
            canvas.center=origin+axis*self.distance();canvas.u=u;canvas.v=v;canvas.extent=(80.,80.)
        else:
            canvas.center=origin+axis*length/2;canvas.u=axis;canvas.v=u;canvas.extent=(max(50.,length+20),60.)
        # Bounded raster, independent of widget enlargement.
        width,height=[min(400,max(100,int(size/.5))) for size in canvas.extent]
        x=(np.arange(width)+.5)/width*canvas.extent[0]-canvas.extent[0]/2
        y=canvas.extent[1]/2-(np.arange(height)+.5)/height*canvas.extent[1]
        coords=canvas.center+x[None,:,None]*canvas.u+y[:,None,None]*canvas.v
        options=self.window.ct_options();source=self.source.currentData()
        width_mri=self.window.window_spin.value();level=self.window.level_spin.value()
        raw=sample_reference(scene.data,scene,coords)
        grey=np.clip((raw-level+width_mri/2)*255/max(.01,width_mri),0,255)
        rgb=np.repeat(grey[...,None],3,axis=2) if source!='ct' else np.zeros(grey.shape+(3,))
        if scene.ct is not None and source!='mri':
            ct=sample_reference(scene.ct,scene,coords,-1024)
            valid=sample_reference(scene.ct_valid,scene,coords,order=0)>0
            value=np.clip((ct-options['level']+options['window']/2)*255/max(.1,options['window']),0,255)
            alpha=valid.astype(float)
            if source=='both':alpha*=np.clip((ct-150)/200,0,1)*options.get('opacity',.65)
            color=value[...,None]*(np.array([1.,.76,.38]) if source=='both' else np.ones(3))
            rgb=rgb*(1-alpha[...,None])+color*alpha[...,None]
        pixels=np.ascontiguousarray(np.clip(rgb,0,255).astype(np.uint8))
        canvas.image=QImage(pixels.data,width,height,pixels.strides[0],QImage.Format.Format_RGB888).copy();canvas.update()
        self.info.setText(tr('{first} → {last} / {distance} mm（最初の接点から）\nホイール：電極軸に沿って0.5 mmずつ移動').format(
            first=self.contacts[0].name,last=self.contacts[-1].name,distance=f'{self.distance():.1f}'))

    def state(self):
        return {'group':self.groups.currentData(),'plane':self.plane.currentData(),'source':self.source.currentData(),'distance':self.slider.value()}

    def restore(self,state):
        for widget,key in [(self.groups,'group'),(self.plane,'plane'),(self.source,'source')]:
            found=widget.findData(state.get(key))
            if found>=0:widget.setCurrentIndex(found)
        self.slider.blockSignals(True);self.slider.setValue(int(state.get('distance',0)));self.slider.blockSignals(False)
        self.refresh_image()
