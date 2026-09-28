"""Whole-brain 3D and native CT/reference MRI context for contact review."""
from dataclasses import replace
from itertools import product
import numpy as np
import nibabel as nib
import vtk
from PySide6.QtCore import QPointF,QRectF,Qt,Signal
from PySide6.QtGui import QColor,QImage,QPainter,QPen
from PySide6.QtWidgets import QWidget,QVBoxLayout,QLabel,QComboBox,QStackedWidget
from .i18n import tr
from .imaging import plane_axes
from .surface_view import SurfacePanel
from scipy.ndimage import map_coordinates


def overview_source(scene, contacts, modality):
    if modality == 'MRI':
        data = scene.raw_mri if scene.raw_mri is not None else scene.data
        affine = scene.source_affine if scene.raw_mri is not None else scene.affine
        positions = np.array([c.position for c in contacts])
    else:
        data,affine = scene.raw_ct,scene.raw_ct_affine
        positions = np.array([c.ct_position for c in contacts])
    return data,affine,positions


def overview_plane(shape,affine,point,axis):
    """A patient-axis plane through the contact, sized to contain the entire native CT FOV."""
    corners=nib.affines.apply_affine(affine,np.array(list(product(*[(-.5,n-.5) for n in shape]))))
    lower,upper=corners.min(0),corners.max(0)
    h,v=plane_axes(axis)
    center=np.array(point,dtype=float).copy()
    center[h]=(lower[h]+upper[h])/2; center[v]=(lower[v]+upper[v])/2
    return center,-np.eye(3)[h],np.eye(3)[v],(upper[[h,v]]-lower[[h,v]]+8.)


class HeadCT(QWidget):
    selected=Signal(int)
    window_delta=Signal(float,float)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.contacts=[]; self.index=0; self.axis=2; self.image=QImage(); self.setMinimumSize(240,230)
        self.center=np.zeros(3); self.u=np.array([-1.,0.,0.]); self.v=np.array([0.,1.,0.]); self.extent=np.array([240.,240.])
        self.positions=np.empty((0,3)); self.modality='CT'; self._sample_key=None; self.raw=None; self._last=QPointF()
        self.setToolTip(tr('右ドラッグ：表示中の画像のwindow / level'))

    def configure(self,scene,contacts,index,axis,width,level,modality='CT'):
        self.contacts=contacts; self.index=index; self.axis=axis
        self.modality=modality
        if not contacts:
            self.image=QImage(); self.raw=None; self._sample_key=None; self.positions=np.empty((0,3)); self.update(); return
        data,affine,self.positions=overview_source(scene,contacts,modality)
        self.center,self.u,self.v,self.extent=overview_plane(data.shape,affine,self.positions[index],axis)
        w,h=np.clip(np.ceil(self.extent/.7).astype(int),100,600)
        key=(modality,id(data),tuple(affine.ravel()),tuple(self.center),axis,w,h)
        if key!=self._sample_key:
            x=((np.arange(w)+.5)/w-.5)*self.extent[0]
            y=(.5-(np.arange(h)+.5)/h)*self.extent[1]
            points=self.center+x[None,:,None]*self.u+y[:,None,None]*self.v
            indices=nib.affines.apply_affine(np.linalg.inv(affine),points.reshape(-1,3))
            self.raw=map_coordinates(data,indices.T,order=1,mode='constant',cval=0 if modality=='MRI' else -1024.,prefilter=False).reshape(h,w)
            self._sample_key=key
        grey=np.ascontiguousarray(np.clip((self.raw-level+width/2)*255/max(.001,width),0,255).astype(np.uint8))
        self.image=QImage(grey.data,w,h,grey.strides[0],QImage.Format.Format_Grayscale8).copy()
        self.update()

    def image_rect(self):
        area=QRectF(24,24,max(1,self.width()-48),max(1,self.height()-54))
        scale=min(area.width()/self.extent[0],area.height()/self.extent[1])
        w,h=self.extent*scale
        return QRectF(area.center().x()-w/2,area.center().y()-h/2,w,h)

    def project(self,point):
        delta=point-self.center; rect=self.image_rect()
        return QPointF(rect.center().x()+np.dot(delta,self.u)/self.extent[0]*rect.width(),
                       rect.center().y()-np.dot(delta,self.v)/self.extent[1]*rect.height())

    def paintEvent(self,event):
        p=QPainter(self); p.fillRect(self.rect(),QColor('#080f19'))
        if self.image.isNull(): return
        p.drawImage(self.image_rect(),self.image); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        count=len(self.contacts)
        for i,position in enumerate(self.positions):
            center=self.project(position)
            color=QColor('#79ead1' if i==0 else '#f799c3' if i==count-1 else '#ffc86c')
            outside=abs(position[self.axis]-self.center[self.axis])>2
            pen=QPen(color,2 if i==self.index else 1)
            if outside: pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(center,6 if i in (0,count-1) else 3.5,6 if i in (0,count-1) else 3.5)
            if i==self.index: p.drawEllipse(center,9,9)
            if i in (0,count-1) or i==self.index: p.drawText(center+QPointF(7,-7),str(i+1))
        directions={2:('R','L','A','P'),1:('R','L','S','I'),0:('A','P','S','I')}[self.axis]
        for label,rect in zip(directions,(QRectF(2,self.height()/2-10,20,20),QRectF(self.width()-22,self.height()/2-10,20,20),
                                           QRectF(self.width()/2-10,2,20,20),QRectF(self.width()/2-10,self.height()-26,20,20))):
            p.setPen(QColor('#c8dbe6')); p.drawText(rect,Qt.AlignmentFlag.AlignCenter,label)

    def mousePressEvent(self,event):
        self._last=event.position()
        if event.button()!=Qt.MouseButton.LeftButton or not self.contacts: return
        distances=[(self.project(position)-event.position()).manhattanLength() for position in self.positions]
        nearest=int(np.argmin(distances))
        if distances[nearest]<20: self.selected.emit(nearest)

    def mouseMoveEvent(self,event):
        if event.buttons()&Qt.MouseButton.RightButton:
            delta=event.position()-self._last; self.window_delta.emit(delta.x(),delta.y())
        self._last=event.position()


class ContactOverview(QWidget):
    selected=Signal(int)
    window_delta=Signal(float,float)

    def __init__(self,scene,parent=None):
        super().__init__(parent)
        self.contacts=[]; self.index=0; self.width_value=4000; self.level_value=1000
        self.initialized=False; self.closed=False; self.signature=None; self.scene=scene
        self.setMinimumWidth(300)
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(5)
        self.selector=QComboBox()
        for label,key in (('脳全体 3D','3d'),('頭部CT：水平断',2),('頭部CT：冠状断',1),('頭部CT：矢状断',0)):
            self.selector.addItem(tr(label),key)
        for label,axis in (('T1 MRI：水平断',2),('T1 MRI：冠状断',1),('T1 MRI：矢状断',0)):
            self.selector.addItem(tr(label),'mri_'+str(axis))
        self.selector.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self.selector)
        from PySide6.QtWidgets import QHBoxLayout,QDoubleSpinBox
        self.mri_controls=QWidget(); row=QHBoxLayout(self.mri_controls); row.setContentsMargins(0,0,0,0)
        low,high=scene.contrast_limits()
        self.mri_width=QDoubleSpinBox(); self.mri_level=QDoubleSpinBox()
        for label,spin,value,minimum in (('MRI 幅',self.mri_width,high-low,.01),('MRI 中心',self.mri_level,(high+low)/2,-1e9)):
            row.addWidget(QLabel(tr(label))); spin.setRange(minimum,1e9); spin.setDecimals(1); spin.setValue(value)
            spin.setKeyboardTracking(False); spin.valueChanged.connect(self.refresh); row.addWidget(spin)
        layout.addWidget(self.mri_controls); self.mri_controls.hide()
        self.stack=QStackedWidget(); layout.addWidget(self.stack,1)
        self.surface=SurfacePanel(self.stack); self.stack.addWidget(self.surface)
        self.head_ct=HeadCT(); self.stack.addWidget(self.head_ct)
        self.head_ct.selected.connect(self.selected)
        self.head_ct.window_delta.connect(self.image_windowing)
        self.context_scene=replace(scene,surfaces={k:v for k,v in scene.surfaces.items() if k.startswith('pial')},
            contacts=[],nuclei=None,nuclei_display=None,nuclei_names={},segmentations=[],segmentation_preview=None)
        self.surface.set_scene(self.context_scene)
        self.surface.set_surface_options(opacity=.16,show_planes=False)
        self.surface.widget.setToolTip(tr('左ドラッグ：回転・ホイール：拡大・コンタクトのダブルクリック：選択'))
        self.surface.point_picked.connect(self.pick)
        self.surface.widget.window_delta.connect(self.window_delta)
        self.markers=[]
        for color in ((.47,.92,.82),(.97,.60,.76)):
            text=vtk.vtkBillboardTextActor3D(); text.GetTextProperty().SetColor(*color)
            text.GetTextProperty().SetFontSize(19); text.GetTextProperty().BoldOn(); text.PickableOff()
            self.surface.renderer.AddActor(text)
            sphere=vtk.vtkSphereSource(); sphere.SetRadius(1.6); sphere.SetThetaResolution(16); sphere.SetPhiResolution(12)
            mapper=vtk.vtkPolyDataMapper(); mapper.SetInputConnection(sphere.GetOutputPort())
            actor=vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(*color); actor.PickableOff()
            self.surface.renderer.AddActor(actor); self.markers.append((text,sphere,actor))
        self.legend=QLabel(); self.legend.setWordWrap(True); layout.addWidget(self.legend)
        self.hint=QLabel(); self.hint.setWordWrap(True); self.hint.setObjectName('muted'); layout.addWidget(self.hint)

    def showEvent(self,event):
        super().showEvent(event)
        if not self.initialized and not self.closed:
            self.surface.initialize(); self.initialized=True
        self.refresh()

    def configure(self,scene,contacts,index,width,level):
        self.scene=scene; self.contacts=contacts; self.index=index; self.width_value=width; self.level_value=level
        self.refresh()

    def refresh(self,*_):
        if not hasattr(self,'surface') or self.closed: return
        if not self.contacts:
            self.context_scene.contacts=[]; self.surface.refresh_contacts(self.context_scene)
            self.surface.cursor_actor.SetVisibility(False)
            for text,_,actor in self.markers: text.SetVisibility(False); actor.SetVisibility(False)
            self.head_ct.configure(self.scene,[],0,2,self.width_value,self.level_value)
            self.legend.setText(''); self.hint.setText(''); self.signature=None
            self.surface.render(); return
        mode=self.selector.currentData()
        mri=isinstance(mode,str) and mode.startswith('mri_')
        self.mri_controls.setVisible(mri)
        self.stack.setCurrentIndex(0 if mode=='3d' else 1)
        signature=tuple((c.uid,c.name,*c.position) for c in self.contacts)
        if signature!=self.signature:
            self.context_scene.contacts=list(self.contacts)
            self.surface.refresh_contacts(self.context_scene); self.signature=signature
        self.surface.cursor_source.SetCenter(*self.contacts[self.index].position)
        self.surface.cursor_actor.SetVisibility(True)
        for (text,sphere,actor),i in zip(self.markers,(0,len(self.contacts)-1)):
            point=self.contacts[i].position
            text.SetInput(str(i+1)); text.SetPosition(*(point+[2,2,2])); text.SetVisibility(True)
            sphere.SetCenter(*point); actor.SetVisibility(True)
        confirmed=all(c.status=='reviewed' for c in self.contacts)
        self.legend.setText(tr('1：先端 / {count}：最後端').format(count=len(self.contacts)) if confirmed else
                            tr('1：先端候補 / {count}：最後端候補').format(count=len(self.contacts)))
        self.legend.setStyleSheet('color:#a2e8d7;')
        self.hint.setText(tr('選択した電極を脳全体に表示。左ドラッグで回転できます。') if mode=='3d' else
                         tr('全接点を投影表示。破線の丸は選択断面から離れた点です。'))
        if mode=='3d': self.surface.render()
        else:
            self.head_ct.configure(self.scene,self.contacts,self.index,int(mode[-1]) if mri else int(mode),
                self.mri_width.value() if mri else self.width_value,self.mri_level.value() if mri else self.level_value,
                modality='MRI' if mri else 'CT')

    def image_windowing(self,dx,dy):
        if self.head_ct.modality!='MRI': self.window_delta.emit(dx,dy); return
        width=self.mri_width.value()
        for spin in (self.mri_width,self.mri_level): spin.blockSignals(True)
        self.mri_width.setValue(width*np.exp(np.clip(dx/160,-2,2)))
        self.mri_level.setValue(self.mri_level.value()+dy*width/180)
        for spin in (self.mri_width,self.mri_level): spin.blockSignals(False)
        self.refresh()

    def pick(self,point):
        if not self.contacts: return
        distance=np.linalg.norm(np.array([c.position for c in self.contacts])-point,axis=1)
        if distance.min()<3: self.selected.emit(int(distance.argmin()))

    def shutdown(self):
        if not self.closed:
            self.closed=True
            self.surface.shutdown()
