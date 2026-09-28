"""Shared result colors and annotated local exports."""
import math
import numpy as np
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget
from .analysis_results import colors, color_limits, resolved_positions
from .result_display import metric_display,spatial_options
from .i18n import tr


def time_text(result, frame):
    if result.kind=='static': return tr('静的な集計値')
    unit=result.time_unit or tr('単位未確認')
    return f't = {result.times[frame]:g} {unit}   ({frame+1}/{len(result.times)})'


def draw_header(painter,rect,result,metric,frame,contacts):
    painter.fillRect(rect,QColor('#101a29'))
    painter.setPen(QColor('#eff6ff')); painter.setFont(QFont('Yu Gothic UI',13,QFont.Weight.Bold))
    title=result.name+'  /  '+result.metrics[metric]
    title=painter.fontMetrics().elidedText(title,Qt.TextElideMode.ElideMiddle,int(rect.width()-32))
    painter.drawText(QRectF(rect.x()+16,rect.y()+8,rect.width()-32,28),title)
    painter.setFont(QFont('Yu Gothic UI',10)); painter.setPen(QColor('#c5d5e5'))
    painter.drawText(QRectF(rect.x()+16,rect.y()+40,rect.width()-32,24),time_text(result,frame))
    lo,hi=color_limits(result,metric); left=rect.right()-min(270,rect.width()*.40); width=rect.right()-left-20
    options=metric_display(result,metric)
    rgb=colors(np.linspace(lo,hi,128),(lo,hi),options['colormap'],options['reverse'])
    for i,c in enumerate(rgb): painter.fillRect(QRectF(left+i*width/128,rect.y()+45,width/128+1,9),QColor.fromRgbF(*c))
    painter.drawText(QRectF(left,rect.y()+60,width,22),Qt.AlignmentFlag.AlignLeft,f'{lo:g}')
    painter.drawText(QRectF(left,rect.y()+60,width,22),Qt.AlignmentFlag.AlignRight,f'{hi:g}')
    mapped=len(resolved_positions(result,contacts)[0])
    units=result.units[metric] or tr('値の単位未確認')
    painter.drawText(QRectF(rect.x()+16,rect.y()+68,max(160,left-rect.x()-30),24),
                     tr('位置対応 {mapped}/{total} · {unit}').format(mapped=mapped,total=len(result.channels),unit=units))


class ResultLegend(QWidget):
    """The same numeric/color key is visible in the viewer and the exported image."""
    def __init__(self):
        super().__init__();self.result=None;self.contacts=[];self.metric_index=0;self.frame=0
        self.setFixedHeight(102)

    def set_result(self,result,contacts,metric,frame):
        self.result=result;self.contacts=contacts;self.metric_index=metric;self.frame=frame;self.update()

    def paintEvent(self,event):
        if self.result is None:return
        painter=QPainter(self)
        draw_header(painter,QRectF(self.rect()),self.result,self.metric_index,self.frame,self.contacts)
        painter.end()


class ResultCanvas(QWidget):
    channel_clicked=Signal(int)

    def __init__(self):
        super().__init__(); self.result=None; self.contacts=[]; self.metric_index=0; self.frame=0; self.hit_boxes=[]
        self.setMinimumSize(450,350)

    def set_result(self,result,contacts=(),metric=0,frame=0):
        self.result=result; self.contacts=contacts; self.metric_index=metric; self.frame=frame; self.update()

    def paintEvent(self,event):
        painter=QPainter(self); self.draw(painter,QRectF(self.rect())); painter.end()

    def draw(self,painter,rect):
        painter.fillRect(rect,QColor('#080f19')); self.hit_boxes=[]
        if self.result is None:
            painter.setPen(QColor('#c5d5e5'))
            painter.drawText(rect,Qt.AlignmentFlag.AlignCenter,tr('左側から解析Excelを追加してください'))
            return
        r=self.result; draw_header(painter,QRectF(rect.x(),rect.y(),rect.width(),102),r,self.metric_index,self.frame,self.contacts)
        n=len(r.channels); columns=max(1,math.ceil(n/36)); row_count=math.ceil(n/columns)
        area=rect.adjusted(14,110,-14,-38); col_width=area.width()/columns; row_height=min(32.,area.height()/max(1,row_count))
        font=QFont('Yu Gothic UI'); font.setPixelSize(max(9,min(14,int(row_height*.63)))); painter.setFont(font)
        lo,hi=color_limits(r,self.metric_index); values=r.values[:,self.frame,self.metric_index]
        options=metric_display(r,self.metric_index);palette=colors(values,(lo,hi),options['colormap'],options['reverse'])
        for i,(channel,value,color) in enumerate(zip(r.channels,values,palette)):
            col,row=divmod(i,row_count); box=QRectF(area.left()+col*col_width,area.top()+row*row_height,col_width-10,row_height)
            self.hit_boxes.append((box,i))
            if row%2==0: painter.fillRect(box,QColor('#132033'))
            if np.isfinite(value):
                zero=float(np.clip((0-lo)/(hi-lo),0,1)); fraction=float(np.clip((value-lo)/(hi-lo),0,1))
                tint=QColor.fromRgbF(*color,.28)
                painter.fillRect(QRectF(box.x()+8+min(zero,fraction)*(box.width()-12),box.y()+2,
                    abs(fraction-zero)*(box.width()-12),max(1,box.height()-4)),tint)
            painter.fillRect(QRectF(box.x()+2,box.y()+row_height*.2,4,row_height*.6),QColor.fromRgbF(*color))
            painter.setPen(QColor('#d4e0ec'))
            # Full source names remain available on hover and at export resolution.
            text=painter.fontMetrics().elidedText(channel,Qt.TextElideMode.ElideRight,int(box.width()*.68-10))
            painter.drawText(box.adjusted(10,0,-box.width()*.30,0),Qt.AlignmentFlag.AlignVCenter,text)
            painter.setPen(QColor('#e5edf6') if np.isfinite(value) else QColor('#9da8b5'))
            painter.drawText(box.adjusted(box.width()*.70,0,-4,0),Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignRight,
                             f'{value:.4g}' if np.isfinite(value) else 'n/a')
        painter.setFont(QFont('Yu Gothic UI',9)); painter.setPen(QColor('#9ab0c5'))
        painter.drawText(rect.adjusted(16,rect.height()-29,-16,-3),tr('チャンネル別表示 · 灰色のn/a＝欠損 · 色尺度は全時間で固定'))

    def mousePressEvent(self,event):
        for rect,index in self.hit_boxes:
            if rect.contains(event.position()): self.channel_clicked.emit(index); break

    def mouseMoveEvent(self,event):
        if self.result:
            for rect,index in self.hit_boxes:
                if rect.contains(event.position()): self.setToolTip(self.result.channels[index]); break


def export_frame(panel,width=1800,height=1120):
    result=panel.selected(); image=QImage(width,height,QImage.Format.Format_RGB888); image.fill(QColor('#080f19'))
    painter=QPainter(image); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if panel.view.currentData()=='channels':
        panel.chart.draw(painter,QRectF(0,0,width,height))
    else:
        window=panel.window; window.surface.render()
        source=(window.surface.capture_image() if panel.view.currentData()=='surface'
                else window.capture_widget(window.views).toImage())
        available=QRectF(0,110,width,height-170); size=source.size(); size.scale(available.size().toSize(),Qt.AspectRatioMode.KeepAspectRatio)
        target=QRectF(available.center().x()-size.width()/2,available.center().y()-size.height()/2,size.width(),size.height())
        painter.drawImage(target,source)
        draw_header(painter,QRectF(0,0,width,102),result,panel.metric_combo.currentIndex(),panel.slider.value(),window.scene.contacts)
        painter.setPen(QColor('#bfcedd')); painter.setFont(QFont('Yu Gothic UI',11))
        spatial=spatial_options(result)
        note=(tr('脳表への補間：最大距離 {distance} mm。双極は2接点の中点が基準です。表示用の補間で、記録範囲の推定ではありません。').format(distance=f"{spatial['distance_mm']:g}")
              if spatial['mode']!='points' else tr('双極チャンネルは2接点の中点に表示。色球は表示用で、記録範囲の推定ではありません。'))
        painter.drawText(QRectF(16,height-52,width-32,48),Qt.TextFlag.TextWordWrap,note)
    painter.end(); return image


class ResultGlyphs:
    def __init__(self,renderer):
        import vtk
        self.renderer=renderer; self.data=vtk.vtkPolyData(); self.sphere=vtk.vtkSphereSource()
        self.sphere.SetRadius(2.); self.sphere.SetThetaResolution(16); self.sphere.SetPhiResolution(12)
        self.mapper=vtk.vtkGlyph3DMapper(); self.mapper.SetInputData(self.data)
        self.mapper.SetSourceConnection(self.sphere.GetOutputPort()); self.mapper.ScalingOff()
        self.mapper.SetColorModeToDirectScalars(); self.mapper.SetScalarModeToUsePointData()
        self.actor=vtk.vtkActor(); self.actor.SetMapper(self.mapper); self.actor.PickableOff()
        self.actor.GetProperty().SetAmbient(.6); self.actor.GetProperty().SetDiffuse(.4)
        self.actor.GetProperty().LightingOff()
        self.actor.SetVisibility(False); renderer.AddActor(self.actor)
        self.label_mapper=vtk.vtkLabeledDataMapper();self.label_mapper.SetInputData(self.data)
        self.label_mapper.SetLabelModeToLabelFieldData();self.label_mapper.SetFieldDataName('channel_labels')
        text=self.label_mapper.GetLabelTextProperty();text.SetFontSize(12);text.SetColor(.94,.97,1)
        text.SetBold(False);text.SetShadow(False)
        self.label_actor=vtk.vtkActor2D();self.label_actor.SetMapper(self.label_mapper)
        self.label_actor.PickableOff();self.label_actor.SetVisibility(False);renderer.AddActor(self.label_actor)

    def set_points(self,positions,rgb,radius=2.,opacity=1.,labels=()):
        import vtk
        from vtkmodules.util.numpy_support import numpy_to_vtk
        points=vtk.vtkPoints(); points.SetData(numpy_to_vtk(np.ascontiguousarray(positions,dtype=float).reshape(-1,3),deep=True))
        self.data.SetPoints(points)
        self.data.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(np.asarray(rgb)*255,dtype=np.uint8).reshape(-1,3),deep=True))
        self.sphere.SetRadius(radius);self.actor.GetProperty().SetOpacity(opacity)
        names=vtk.vtkStringArray();names.SetName('channel_labels')
        for label in labels:names.InsertNextValue(label)
        self.data.GetPointData().RemoveArray('channel_labels');self.data.GetPointData().AddArray(names)
        self.data.Modified(); self.actor.SetVisibility(bool(len(positions)) and opacity>0)
        self.label_actor.SetVisibility(bool(len(positions)) and len(labels)==len(positions) and opacity>0)
