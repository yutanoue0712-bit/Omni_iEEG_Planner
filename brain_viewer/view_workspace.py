"""Reusable 1/4/6-panel workspace. Layout changes never reload image volumes."""
import numpy as np
from PySide6.QtCore import Qt, QEvent, QPointF
from PySide6.QtWidgets import (QWidget,QFrame,QHBoxLayout,QVBoxLayout,QGridLayout,
                              QComboBox,QToolButton,QLabel,QTabBar)
from .i18n import tr, translate_widgets
from .slice_view import SlicePanel
from .surface_view import SurfacePanel
from .trajectory_view import TrajectoryPanel

KINDS=[('Axial','axial'),('Coronal','coronal'),('Sagittal','sagittal'),('3D','surface'),('Trajectory','trajectory')]
AXES={'axial':2,'coronal':1,'sagittal':0}
DEFAULTS=['axial','coronal','sagittal','surface','axial','trajectory']


class SurfaceMirror(SurfacePanel):
    """Separate camera/window, shared immutable meshes and global display properties."""
    def __init__(self,source):
        super().__init__()
        self.source=source;self._props=()
        source.updated.connect(self.sync)
        self.sync()

    def sync(self):
        new_scene=self.scene is not self.source.scene
        self.scene=self.source.scene;self.center=self.source.center.copy()
        props=list(self.source.renderer.GetViewProps())
        signature=tuple(p.GetAddressAsString('') for p in props)
        if signature!=self._props:
            self.renderer.RemoveAllViewProps()
            for prop in props:self.renderer.AddViewProp(prop)
            self.widget.picker.InitializePickList()
            for prop in self.source.widget.picker.GetPickList():self.widget.picker.AddPickList(prop)
            self._props=signature
        if new_scene and self.scene is not None:
            self.restore_camera(self.source.camera_state())
        self.render()


class PanelHost(QFrame):
    def __init__(self,workspace,index):
        super().__init__();self.workspace=workspace;self.index=index
        self.content=None;self.remembered=None;self.assigned_kind=None;self.cached={}
        self.setObjectName('viewPanel')
        self.box=QVBoxLayout(self);self.box.setContentsMargins(0,0,0,0);self.box.setSpacing(0)
        row=QHBoxLayout();row.setContentsMargins(5,2,5,2);row.setSpacing(4)
        number=QLabel(str(index+1));number.setObjectName('muted');row.addWidget(number)
        self.choice=QComboBox()
        for label,key in KINDS:self.choice.addItem(label,key)
        self.choice.setCurrentIndex(self.choice.findData(DEFAULTS[index]));row.addWidget(self.choice,1)
        self.focus=QToolButton();self.focus.setText('⛶');self.focus.setToolTip(tr('1画面に拡大 / 元の画面数へ戻す'))
        self.focus.clicked.connect(lambda:workspace.toggle_focus(index));row.addWidget(self.focus)
        self.box.addLayout(row)
        self.choice.currentIndexChanged.connect(workspace.arrange)


class ViewWorkspace(QWidget):
    def __init__(self,window):
        super().__init__();self.window=window;self.count=4;self.previous_count=4;self.focused=0
        self.extra_slices=[];self.mirrors=[];self.trajectories=[];self._arranging=False;self._restoring_layout=False
        outer=QHBoxLayout(self);outer.setContentsMargins(0,0,0,0);outer.setSpacing(5)
        self.grid_widget=QWidget();self.grid=QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0,0,0,0);self.grid.setSpacing(7);outer.addWidget(self.grid_widget,1)
        self.tabs=QTabBar();self.tabs.setShape(QTabBar.Shape.RoundedEast);self.tabs.setExpanding(False)
        for label,value in [('4画面',4),('1画面',1),('6画面',6)]:
            idx=self.tabs.addTab(tr(label));self.tabs.setTabData(idx,value)
        self.tabs.setToolTip(tr('画面数を切り替える'));outer.addWidget(self.tabs,0,Qt.AlignmentFlag.AlignTop)
        self.tabs.currentChanged.connect(lambda i:self.set_count(self.tabs.tabData(i)))
        self.parking=QWidget(self);self.parking.hide()
        self.hosts=[PanelHost(self,i) for i in range(6)]
        for panel in window.slices.values():self.connect_slice(panel,existing=True)
        self.connect_surface(window.surface)
        self.arrange()

    def slice_panels(self):
        return [*self.window.slices.values(),*self.extra_slices]

    def surface_panels(self):
        return [self.window.surface,*self.mirrors]

    def connect_slice(self,panel,existing=False):
        w=self.window;canvas=panel.canvas
        panel.name_label.hide()
        if not existing:
            canvas.cursor_changed.connect(w.set_cursor)
            canvas.slice_step.connect(lambda n,axis=panel.axis:w.step_slice(axis,n))
            canvas.window_delta.connect(w.drag_window)
            panel.slider.valueChanged.connect(lambda value,axis=panel.axis:w.move_slice(axis,value))
            canvas.brush_requested.connect(w.segmentation.draw)
        canvas.contour_requested.connect(w.segmentation.enclose)
        canvas.contour_requested.connect(w.diffusion.exclusion.select_2d)
        canvas.focus_requested.connect(lambda:self.focus_widget(panel))
        canvas.installEventFilter(self)

    def connect_surface(self,panel):
        panel.name_label.hide()
        panel.widget.panel_focus_enabled=True
        panel.widget.focus_requested.connect(lambda:self.focus_widget(panel))
        panel.widget.installEventFilter(self)
        panel.widget.tract_lasso_requested.connect(lambda polygon:self.window.diffusion.exclusion.select_3d(panel,polygon))
        panel.widget.setToolTip(tr('ドラッグ：回転 / ダブルクリック：1画面と復帰\nCtrl＋ダブルクリック：選択位置の断面へ'))
        # Keep contact editor's double-click selection unchanged.
        for label in panel.findChildren(QLabel):
            if label.objectName()=='viewHint':label.setText(tr('ダブルクリック：拡大・復帰 / Ctrl＋ダブルクリック：断面を連動'))

    def eventFilter(self,obj,event):
        if event.type()==QEvent.Type.MouseButtonPress:
            for host in self.hosts:
                if host.content and (host.content is obj or host.content.isAncestorOf(obj)):
                    self.focused=host.index;break
        return super().eventFilter(obj,event)

    def focus_widget(self,panel):
        host=next((h for h in self.hosts if h.content is panel),None)
        if host:self.toggle_focus(host.index)

    def set_count(self,count):
        if count not in (1,4,6):return
        if count!=1:self.previous_count=count
        self.count=count
        if count!=1 and self.focused>=count:self.focused=0
        self.arrange()

    def toggle_focus(self,index):
        if self.count==1:
            self.count=self.previous_count
        else:
            self.previous_count=self.count;self.count=1
        self.focused=index
        self.arrange()

    def acquire(self,kind,used,preferred):
        w=self.window
        if kind in AXES:
            pool=[w.slices[AXES[kind]],*[p for p in self.extra_slices if p.axis==AXES[kind]]]
        elif kind=='surface':pool=self.surface_panels()
        else:pool=self.trajectories
        if preferred in pool and preferred not in used:return preferred
        for panel in pool:
            if panel not in used:return panel
        if kind in AXES:
            panel=SlicePanel(AXES[kind]);self.extra_slices.append(panel);self.connect_slice(panel)
            if w.scene:
                panel.set_scene(w.scene);panel.set_cursor(w.ijk)
                panel.canvas.set_contrast(w.window_spin.value()*-.5+w.level_spin.value(),
                                          w.window_spin.value()*.5+w.level_spin.value())
                panel.canvas.set_layers(w.ct_options(),w.contact_options(),w.annotation_check.isChecked())
                panel.canvas.result_points=getattr(w.slices[2].canvas,'result_points',[])
                panel.canvas.result_colors=getattr(w.slices[2].canvas,'result_colors',[])
                panel.canvas.result_labels=getattr(w.slices[2].canvas,'result_labels',[])
                panel.canvas.result_options=getattr(w.slices[2].canvas,'result_options',{})
                panel.canvas.result_active=getattr(w.slices[2].canvas,'result_active',False)
        elif kind=='surface':
            panel=SurfaceMirror(w.surface);self.mirrors.append(panel);self.connect_surface(panel)
            panel.point_picked.connect(w.select_world);panel.widget.window_delta.connect(w.drag_window)
            panel.initialize()
        else:
            panel=TrajectoryPanel(w);self.trajectories.append(panel)
            panel.canvas.focus_requested.connect(lambda:self.focus_widget(panel))
            panel.canvas.installEventFilter(self);panel.refresh()
        translate_widgets(panel)
        return panel

    def arrange(self,*_):
        if self._arranging:return
        self._arranging=True
        try:
            if not self._restoring_layout:
                for host in self.hosts:
                    if host.content:
                        host.cached[host.assigned_kind]=self.content_state(host.content)
            indices=[self.focused] if self.count==1 else list(range(self.count))
            used=[];assignments=[]
            for index in indices:
                host=self.hosts[index]
                panel=self.acquire(host.choice.currentData(),used,host.content or host.remembered)
                used.append(panel);assignments.append((host,panel))
            for host in self.hosts:
                if host.content:
                    host.remembered=host.content
                    host.box.removeWidget(host.content);host.content.setParent(self.parking)
                    host.content=None
                self.grid.removeWidget(host);host.hide()
            for row in range(3):self.grid.setRowStretch(row,0)
            for col in range(3):self.grid.setColumnStretch(col,0)
            cols=3 if self.count==6 else (2 if self.count==4 else 1)
            for position,(host,panel) in enumerate(assignments):
                host.content=panel;host.assigned_kind=host.choice.currentData()
                host.box.addWidget(panel,1);panel.show()
                row,col=divmod(position,cols);self.grid.addWidget(host,row,col);host.show()
                self.grid.setRowStretch(row,1);self.grid.setColumnStretch(col,1)
                if isinstance(panel,TrajectoryPanel):panel.refresh()
                self.restore_content(panel,host.cached.get(host.assigned_kind,{}))
                translate_widgets(host)
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex({4:0,1:1,6:2}[self.count]);self.tabs.blockSignals(False)
        finally:self._arranging=False
        if hasattr(self.window,'diffusion'):
            self.window.diffusion.update_roi_preview();self.window.segmentation.update_tools()
            self.window.diffusion.exclusion.update_tools()
        self.refresh_trajectories()

    def refresh_trajectories(self):
        for panel in self.trajectories:
            if panel.isVisible():panel.refresh()

    def content_state(self,panel):
        item={}
        if isinstance(panel,SlicePanel):
            item.update(zoom=panel.canvas.zoom,pan=[panel.canvas.pan.x(),panel.canvas.pan.y()])
        elif isinstance(panel,SurfacePanel):item['camera']=panel.camera_state()
        elif isinstance(panel,TrajectoryPanel):item['trajectory']=panel.state()
        return item

    def panel_state(self,host):
        kind=host.choice.currentData()
        item=(self.content_state(host.content) if host.content and host.assigned_kind==kind
              else dict(host.cached.get(kind,{})))
        return dict(item,kind=kind)

    def state(self):
        panels=[self.panel_state(host) for host in self.hosts]
        return {'count':self.count,'previous_count':self.previous_count,'focused':self.focused,'panels':panels}

    def restore(self,state):
        count=state.get('count',4);self.count=count if count in (1,4,6) else 4
        self.previous_count=6 if state.get('previous_count')==6 else 4
        self.focused=max(0,min(5,int(state.get('focused',0))))
        for i,host in enumerate(self.hosts):
            records=state.get('panels',[]);item=records[i] if i<len(records) else {}
            host.cached={item.get('kind',DEFAULTS[i]):{k:v for k,v in item.items() if k!='kind'}}
            host.choice.blockSignals(True)
            host.choice.setCurrentIndex(max(0,host.choice.findData(item.get('kind',DEFAULTS[i]))))
            host.choice.blockSignals(False)
        self._restoring_layout=True
        try:self.arrange()
        finally:self._restoring_layout=False

    def restore_content(self,panel,item):
        if not item:return
        if isinstance(panel,SlicePanel):
            zoom=float(item.get('zoom',panel.canvas.zoom));pan=np.asarray(item.get('pan',[panel.canvas.pan.x(),panel.canvas.pan.y()]),float)
            if np.isfinite(zoom) and pan.shape==(2,) and np.isfinite(pan).all():
                panel.canvas.zoom=float(np.clip(zoom,.3,8));panel.canvas.pan=QPointF(*np.clip(pan,-10000,10000));panel.canvas.update()
        elif isinstance(panel,SurfacePanel):panel.restore_camera(item.get('camera',{}))
        elif isinstance(panel,TrajectoryPanel):panel.restore(item.get('trajectory',{}))

    def shutdown(self):
        for panel in self.mirrors:panel.shutdown()
