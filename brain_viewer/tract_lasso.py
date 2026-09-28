"""A VTK overlay for freehand selection that works in native OpenGL panels."""
import vtk
from PySide6.QtCore import Qt


class TractLasso:
    def __init__(self,widget):
        self.widget=widget;self.enabled=False;self.points=[];self.actor=None

    def clear(self,render=True):
        self.points=[]
        if self.actor is not None and self.widget.scene_renderer is not None:
            self.widget.scene_renderer.RemoveViewProp(self.actor);self.actor=None
            if render and self.widget.isVisible():self.widget.GetRenderWindow().Render()

    def set_enabled(self,enabled):
        self.clear()
        self.enabled=bool(enabled)
        self.widget.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def press(self,event):
        if self.enabled and event.button()==Qt.MouseButton.LeftButton:
            self.widget.setFocus();self.points=[(event.position().x(),event.position().y())]
            event.accept();return True
        return False

    def move(self,event):
        if not self.points:return False
        if not event.buttons() & Qt.MouseButton.LeftButton:self.clear();return True
        point=(event.position().x(),event.position().y())
        if sum((a-b)**2 for a,b in zip(point,self.points[-1]))>=4:
            self.points.append(point);self.draw()
        event.accept();return True

    def release(self,event):
        if event.button()!=Qt.MouseButton.LeftButton or not self.points:return False
        self.points.append((event.position().x(),event.position().y()))
        normalized=[(2*x/max(1,self.widget.width())-1,1-2*y/max(1,self.widget.height())) for x,y in self.points]
        self.clear()
        if len(normalized)>=3:self.widget.tract_lasso_requested.emit(normalized)
        event.accept();return True

    def draw(self):
        renderer=self.widget.scene_renderer
        if renderer is None:return
        size=self.widget.GetRenderWindow().GetSize()
        points=vtk.vtkPoints()
        for x,y in [*self.points,self.points[0]]:
            points.InsertNextPoint(x*size[0]/max(1,self.widget.width()),
                                   (self.widget.height()-y)*size[1]/max(1,self.widget.height()),0)
        cells=vtk.vtkCellArray();cells.InsertNextCell(points.GetNumberOfPoints())
        for i in range(points.GetNumberOfPoints()):cells.InsertCellPoint(i)
        poly=vtk.vtkPolyData();poly.SetPoints(points);poly.SetLines(cells)
        if self.actor is None:
            self.actor=vtk.vtkActor2D();mapper=vtk.vtkPolyDataMapper2D();self.actor.SetMapper(mapper)
            self.actor.GetProperty().SetColor(1.,.85,.3);self.actor.GetProperty().SetLineWidth(2)
            self.actor.PickableOff();renderer.AddViewProp(self.actor)
        self.actor.GetMapper().SetInputData(poly)
        self.widget.GetRenderWindow().Render()
