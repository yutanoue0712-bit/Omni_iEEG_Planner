"""VTK cortical surfaces and MRI planes in the same scanner RAS space."""
from __future__ import annotations
from .i18n import tr

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import vtk

from .imaging import Scene, plane_axes
from .rendering import composite_plane
from .anatomy import nucleus_color
from .result_rendering import ResultGlyphs


class BrainInteractor(QVTKRenderWindowInteractor):
    point_picked = Signal(object)
    window_delta = Signal(float, float)
    focus_requested = Signal()
    tract_lasso_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(.008)
        self.picker.PickFromListOn()
        self.scene_renderer = None
        self._windowing = False
        self._last_window_position = None
        self.panel_focus_enabled = False
        from .tract_lasso import TractLasso
        self.tract_lasso=TractLasso(self)

    def set_tract_lasso(self,enabled):
        self.tract_lasso.set_enabled(enabled)

    def mousePressEvent(self, event):
        if self.tract_lasso.press(event):return
        if event.button() == Qt.MouseButton.RightButton:
            self._windowing = True
            self._last_window_position = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.tract_lasso.move(event):return
        if self._windowing:
            delta = event.position() - self._last_window_position
            self.window_delta.emit(delta.x(), delta.y())
            self._last_window_position = event.position()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.tract_lasso.release(event):return
        if event.button() == Qt.MouseButton.RightButton and self._windowing:
            self._windowing = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.tract_lasso.clear()
        if (self.panel_focus_enabled and event.button() == Qt.MouseButton.LeftButton
                and not event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.focus_requested.emit()
            event.accept()
            return
        if self.scene_renderer is not None and event.button() == Qt.MouseButton.LeftButton:
            width, height = self.GetRenderWindow().GetSize()
            x = event.position().x() * width / max(1, self.width())
            y = (self.height() - event.position().y() - 1) * height / max(1, self.height())
            if self.picker.Pick(x, y, 0, self.scene_renderer):
                self.point_picked.emit(np.array(self.picker.GetPickPosition()))
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Escape and self.tract_lasso.points:
            self.tract_lasso.clear();event.accept();return
        super().keyPressEvent(event)

    def wheelEvent(self,event):
        self.tract_lasso.clear()
        super().wheelEvent(event)

    def resizeEvent(self,event):
        if hasattr(self,'tract_lasso'):self.tract_lasso.clear(render=False)
        super().resizeEvent(event)


def mesh_polydata(mesh):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices), deep=True))
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


class SurfacePanel(QFrame):
    point_picked = Signal(object)
    updated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewPanel")
        self.scene: Scene | None = None
        self.actors = {}
        self.polys = {}
        self.planes = {}
        self.nucleus_actors = {}
        self.segment_actors = {}
        self.tract_actors = {}
        self.roi_actors = []
        self._segment_revisions = {}
        self.contact_actors = []
        self.contact_source = False
        self.contact_options = {"visible": True, "group": "", "source": False}
        self.ct_options = {}
        from .pet_surface import PETSurfaceColors
        self.pet_colors = PETSurfaceColors()
        from .result_surface import ResultSurfaceColors
        self.result_surface = ResultSurfaceColors()
        self.mode = "pial"
        self.opacity = 1.
        self.surface_visible = True
        self.hemispheres = {"lh": True, "rh": True}
        self.show_planes = False
        self.low, self.high = 0., 1.
        self.ijk = np.zeros(3, dtype=int)
        self.center = np.zeros(3)
        self.render_count = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(12, 5, 10, 5)
        name = QLabel(tr("Surface  /  脳表 3D"))
        self.name_label = name
        name.setStyleSheet("color:#91dfbc; font-weight:600;")
        header.addWidget(name)
        header.addStretch()
        self.presets = QComboBox()
        self.presets.addItems([tr("斜め"), tr("前方"), tr("左側"), tr("右側"), tr("上方")])
        self.presets.setFixedWidth(82)
        self.presets.currentIndexChanged.connect(self.set_camera_preset)
        header.addWidget(self.presets)
        reset = QPushButton(tr("リセット"))
        reset.clicked.connect(lambda: self.set_camera_preset(self.presets.currentIndex()))
        header.addWidget(reset)
        layout.addLayout(header)
        self.widget = BrainInteractor(self)
        self.widget.setMinimumSize(200, 170)
        self.widget.setToolTip(tr("左ドラッグ：回転\nホイール：拡大縮小\n右ドラッグ：MRIまたはCTのwindow/level\n中央ドラッグ：平行移動\n脳表・電極をダブルクリック：その位置の断面へ"))
        layout.addWidget(self.widget, 1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(.035, .059, .093)
        self.renderer.SetBackground2(.075, .115, .17)
        self.renderer.GradientBackgroundOn()
        self.widget.scene_renderer = self.renderer
        window = self.widget.GetRenderWindow()
        window.SetMultiSamples(0)
        window.SetAlphaBitPlanes(1)
        window.AddRenderer(self.renderer)
        self.result_glyphs = ResultGlyphs(self.renderer)
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(60)
        self.renderer.SetOcclusionRatio(.1)
        window.AddObserver(vtk.vtkCommand.EndEvent, self._render_finished)
        self.interactor = window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.widget.point_picked.connect(self.point_picked)
        marker = vtk.vtkAnnotatedCubeActor()
        for axis, positive, negative in (("X", "R", "L"), ("Y", "A", "P"), ("Z", "S", "I")):
            getattr(marker, f"Set{axis}PlusFaceText")(positive)
            getattr(marker, f"Set{axis}MinusFaceText")(negative)
        marker.GetCubeProperty().SetColor(.22, .32, .42)
        marker.GetTextEdgesProperty().SetColor(.82, .89, .96)
        marker.GetTextEdgesProperty().SetLineWidth(1)
        self.orientation = vtk.vtkOrientationMarkerWidget()
        self.orientation.SetOrientationMarker(marker)
        self.orientation.SetInteractor(self.interactor)
        self.orientation.SetViewport(.0, .0, .20, .23)
        self.cursor_source = vtk.vtkSphereSource()
        self.cursor_source.SetRadius(1.6)
        self.cursor_source.SetThetaResolution(20)
        self.cursor_source.SetPhiResolution(16)
        cursor_mapper = vtk.vtkPolyDataMapper()
        cursor_mapper.SetInputConnection(self.cursor_source.GetOutputPort())
        self.cursor_actor = vtk.vtkActor()
        self.cursor_actor.SetMapper(cursor_mapper)
        self.cursor_actor.GetProperty().SetColor(1., .46, .31)
        self.cursor_actor.GetProperty().SetAmbient(.7)
        self.cursor_actor.PickableOff()
        self.cursor_actor.SetVisibility(False)
        self.renderer.AddActor(self.cursor_actor)
        footer = QLabel(tr("ドラッグで回転  ·  ホイールで拡大  ·  脳表をダブルクリックで断面を連動"))
        footer.setObjectName("viewHint")
        layout.addWidget(footer)

    def initialize(self):
        self.widget.Initialize()
        self.orientation.SetEnabled(1)
        self.orientation.InteractiveOff()

    def _render_finished(self, *_):
        self.render_count += 1

    def render(self):
        self.updated.emit()
        if self.isVisible():
            self.widget.GetRenderWindow().Render()

    def set_scene(self, scene: Scene):
        self.result_surface.clear()
        self.pet_colors.clear()
        self.set_diffusion_rois([])
        for actor in self.tract_actors.values(): self.renderer.RemoveActor(actor)
        self.tract_actors.clear()
        self.result_glyphs.set_points([],[])
        for actor in self.segment_actors.values():
            self.renderer.RemoveActor(actor)
        self.segment_actors.clear(); self._segment_revisions.clear()
        for actor in self.nucleus_actors.values():
            self.renderer.RemoveActor(actor)
        self.nucleus_actors.clear()
        for actor in self.actors.values():
            self.renderer.RemoveActor(actor)
        for plane in self.planes.values():
            self.renderer.RemoveActor(plane[0])
        self.actors.clear()
        self.polys.clear()
        self.planes.clear()
        self.widget.picker.InitializePickList()
        self.scene = scene
        self.low, self.high = scene.contrast_limits()
        for key, mesh in scene.surfaces.items():
            poly = mesh_polydata(mesh)
            self.polys[key] = poly
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(poly)
            normals.SplittingOff()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(.80, .78, .75)
            actor.GetProperty().SetInterpolationToPhong()
            actor.GetProperty().SetAmbient(.22)
            actor.GetProperty().SetDiffuse(.78)
            actor.GetProperty().SetSpecular(.18)
            actor.GetProperty().SetSpecularPower(22)
            self.actors[key] = actor
            self.renderer.AddActor(actor)
            self.widget.picker.AddPickList(actor)
        vertices = np.concatenate([m.vertices for k, m in scene.surfaces.items() if k.startswith("pial")])
        self.center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        for axis in (0, 1, 2):
            plane = vtk.vtkPlaneSource()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            texture = vtk.vtkTexture()
            texture.InterpolateOn()
            texture.RepeatOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetTexture(texture)
            actor.GetProperty().SetLighting(False)
            actor.GetProperty().SetOpacity(.80)
            actor.PickableOff()
            self.renderer.AddActor(actor)
            self.planes[axis] = (actor, plane, texture)
        self._build_contacts()
        self._build_nuclei()
        self.refresh_segmentations(scene, render=False)
        self.refresh_tracts(render=False)
        self.set_surface_options(render=False)
        self.set_cursor(scene.initial_index(), render=False)
        self.set_camera_preset(0)

    def clear_scene(self):
        self.result_surface.clear()
        self.pet_colors.clear()
        self.set_diffusion_rois([])
        for actor in self.tract_actors.values(): self.renderer.RemoveActor(actor)
        self.tract_actors.clear()
        self.result_glyphs.set_points([],[])
        for actor in [*self.actors.values(), *self.nucleus_actors.values(),
                      *self.segment_actors.values(),
                      *[item[0] for item in self.planes.values()], *[item[1] for item in self.contact_actors]]:
            self.renderer.RemoveActor(actor)
        self.actors.clear(); self.polys.clear(); self.nucleus_actors.clear(); self.planes.clear()
        self.contact_actors.clear(); self.widget.picker.InitializePickList()
        self.segment_actors.clear(); self._segment_revisions.clear()
        self.cursor_actor.SetVisibility(False); self.scene = None
        self.render()

    def refresh_tracts(self, render=True):
        from .diffusion_storage import tract_polydata
        bundles=self.scene.tracts if self.scene else []
        ids={b.uid for b in bundles}
        for uid in list(self.tract_actors):
            if uid not in ids:self.renderer.RemoveActor(self.tract_actors.pop(uid))
        for bundle in bundles:
            signature=(id(bundle.points),bundle.edit_revision,getattr(bundle,'_highlight_revision',0))
            if bundle.uid not in self.tract_actors:
                mapper=vtk.vtkPolyDataMapper();mapper.SetInputData(tract_polydata(bundle))
                mapper.SetColorModeToDirectScalars()
                actor=vtk.vtkActor();actor.SetMapper(mapper)
                actor.GetProperty().SetLineWidth(2.);actor.GetProperty().LightingOff()
                actor.PickableOff();self.renderer.AddActor(actor)
                self.tract_actors[bundle.uid]=actor
                actor._tract_signature=signature
            actor=self.tract_actors[bundle.uid]
            if actor._tract_signature!=signature:
                actor.GetMapper().SetInputData(tract_polydata(bundle));actor._tract_signature=signature
            actor.SetVisibility(bundle.visible and bundle.opacity>0 and bundle.count>0)
            actor.GetProperty().SetOpacity(bundle.opacity)
        if render:self.render()

    def set_diffusion_rois(self,rois):
        for actor in self.roi_actors:self.renderer.RemoveActor(actor)
        self.roi_actors.clear()
        for index,roi in enumerate(rois):
            if not roi or roi.get('kind')!='sphere':continue
            source=vtk.vtkSphereSource();source.SetCenter(*roi['center']);source.SetRadius(roi['radius'])
            source.SetThetaResolution(20);source.SetPhiResolution(16)
            mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(source.GetOutputPort())
            actor=vtk.vtkActor();actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*((.98,.67,.44) if index==0 else (.49,.85,.94)))
            actor.GetProperty().SetOpacity(.25);actor.GetProperty().SetRepresentationToWireframe()
            actor.PickableOff();self.renderer.AddActor(actor);self.roi_actors.append(actor)
        self.render()

    def _build_nuclei(self):
        if self.scene.nuclei is None:
            return
        data = self.scene.nuclei
        for label in self.scene.nuclei_names:
            voxels = np.argwhere(data == label)
            if not len(voxels):
                continue
            lower = np.maximum(voxels.min(0)-1, 0)
            upper = np.minimum(voxels.max(0)+2, data.shape)
            crop = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
            mask = np.pad((data[crop] == label).astype(np.uint8), 1)
            volume = vtk.vtkImageData()
            volume.SetDimensions(*mask.shape)
            volume.GetPointData().SetScalars(numpy_to_vtk(mask.ravel(order="F"), deep=True))
            contour = vtk.vtkFlyingEdges3D()
            contour.SetInputData(volume)
            contour.SetValue(0, .5)
            affine = self.scene.nuclei_affine.copy()
            affine[:3, 3] = self.scene.nuclei_affine[:3,:3] @ (lower-1) + self.scene.nuclei_affine[:3,3]
            transform = vtk.vtkTransform()
            transform.SetMatrix(affine.ravel())
            physical = vtk.vtkTransformPolyDataFilter()
            physical.SetInputConnection(contour.GetOutputPort())
            physical.SetTransform(transform)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(physical.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*nucleus_color(label))
            actor.GetProperty().SetOpacity(.9)
            actor.SetVisibility(False)
            self.renderer.AddActor(actor)
            self.widget.picker.AddPickList(actor)
            self.nucleus_actors[label] = actor

    def set_nuclei_options(self, visible=False, selected=0, opacity=.9):
        for label, actor in self.nucleus_actors.items():
            actor.GetProperty().SetOpacity(float(np.clip(opacity,0,1)))
            actor.SetVisibility(bool(visible and opacity>0 and (not selected or label == selected)))
        self.render()

    def set_surface_options(self, mode=None, opacity=None, hemispheres=None, show_planes=None, render=True, visible=None):
        if visible is not None:
            self.surface_visible = bool(visible)
        if mode is not None:
            self.mode = mode
        if opacity is not None:
            self.opacity = float(opacity)
        if hemispheres is not None:
            self.hemispheres = dict(hemispheres)
        if show_planes is not None:
            self.show_planes = bool(show_planes)
        for key, actor in self.actors.items():
            kind, hemisphere = key.split("_")
            actor.SetVisibility(self.surface_visible and self.opacity>0 and kind == self.mode and self.hemispheres[hemisphere])
            actor.GetProperty().SetOpacity(self.opacity)
        for actor, _, _ in self.planes.values():
            actor.SetVisibility(self.show_planes)
        self.cursor_actor.SetVisibility(self.scene is not None)
        if render:
            self.render()

    def set_cursor(self, ijk, render=True):
        self.ijk = np.array(ijk, dtype=int)
        if self.scene is None:
            return
        self.cursor_source.SetCenter(*self.scene.world(ijk))
        self._update_planes()
        if render:
            self.render()

    def set_contrast(self, low, high):
        self.low, self.high = low, high
        self._update_planes()
        self.render()

    def _update_planes(self):
        if self.scene is None:
            return
        for axis, (_, plane, texture) in self.planes.items():
            h, v = plane_axes(axis)
            base = np.full(3, -.5)
            base[axis] = self.ijk[axis]
            point1 = base.copy()
            point2 = base.copy()
            point1[h] = self.scene.data.shape[h] - .5
            point2[v] = self.scene.data.shape[v] - .5
            plane.SetOrigin(*self.scene.world(base))
            plane.SetPoint1(*self.scene.world(point1))
            plane.SetPoint2(*self.scene.world(point2))
            pixels = composite_plane(self.scene, axis, self.ijk[axis], self.low, self.high, self.ct_options)
            image = vtk.vtkImageData()
            image.SetDimensions(pixels.shape[0], pixels.shape[1], 1)
            image.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(pixels.transpose(1, 0, 2)).reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
            texture.SetInputData(image)

    def set_ct_options(self, options):
        self.ct_options = dict(options)
        self.pet_colors.apply(self)
        self.result_surface.apply(self)
        self._update_planes()
        self.render()

    def refresh_contacts(self, scene):
        self.scene = scene
        self.contact_source = False
        self._build_contacts()
        self.render()

    def refresh_segmentations(self, scene, render=True):
        from .segmentation import mask_surface
        from .segmentation_sources import display_segments
        segments = display_segments(scene, self.ct_options.get('segmentation_context'))
        current = {segment.uid for segment in segments}
        for uid in set(self.segment_actors) - current:
            actor = self.segment_actors.pop(uid)
            self.renderer.RemoveActor(actor); self.widget.picker.DeletePickList(actor)
            self._segment_revisions.pop(uid, None)
        for segment in segments:
            signature = (id(segment.mask), segment.revision)
            actor = self.segment_actors.get(segment.uid)
            if actor is None:
                actor = vtk.vtkActor(); mapper = vtk.vtkPolyDataMapper()
                mapper.ScalarVisibilityOff(); actor.SetMapper(mapper)
                actor.GetProperty().SetInterpolationToPhong()
                actor.GetProperty().SetAmbient(.25)
                self.segment_actors[segment.uid] = actor
                self.renderer.AddActor(actor); self.widget.picker.AddPickList(actor)
            if self._segment_revisions.get(segment.uid) != signature:
                actor.GetMapper().SetInputData(mask_surface(segment.mask, scene.affine))
                self._segment_revisions[segment.uid] = signature
            actor.GetProperty().SetColor(*segment.color)
            actor.GetProperty().SetOpacity(segment.opacity)
            actor.SetVisibility(segment.visible_3d and segment.opacity > 0 and actor.GetMapper().GetInput().GetNumberOfPolys() > 0)
        if render: self.render()

    def _build_contacts(self):
        for _, actor in self.contact_actors:
            self.renderer.RemoveActor(actor)
            self.widget.picker.DeletePickList(actor)
        self.contact_actors = []
        if self.scene is None:
            return
        positions = self.scene.contact_positions(self.contact_source)
        groups = list(dict.fromkeys(c.group for c in self.scene.contacts))
        for group in groups:
            selected = [i for i, c in enumerate(self.scene.contacts) if c.group == group]
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(np.ascontiguousarray(positions[selected]), deep=True))
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(1.0)  # A visible marker of the contact centre, not physical contact size.
            sphere.SetThetaResolution(12)
            sphere.SetPhiResolution(10)
            glyph = vtk.vtkGlyph3D()
            glyph.SetInputData(poly)
            glyph.SetSourceConnection(sphere.GetOutputPort())
            glyph.ScalingOff()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            color = self.scene.contacts[selected[0]].color
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetAmbient(.5)
            self.renderer.AddActor(actor)
            self.widget.picker.AddPickList(actor)
            self.contact_actors.append((group, actor))
            if len(selected) > 1 and all(self.scene.contacts[i].kind == "SEEG" for i in selected):
                # Connect contact centres in their imported channel order.
                line = vtk.vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(len(selected))
                for index in range(len(selected)):
                    line.GetPointIds().SetId(index, index)
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(line)
                line_poly = vtk.vtkPolyData()
                line_poly.SetPoints(points)
                line_poly.SetLines(cells)
                tube = vtk.vtkTubeFilter()
                tube.SetInputData(line_poly)
                tube.SetRadius(.18)
                tube.SetNumberOfSides(8)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(tube.GetOutputPort())
                connector = vtk.vtkActor()
                connector.SetMapper(mapper)
                connector.GetProperty().SetColor(*color)
                connector.GetProperty().SetOpacity(.65)
                connector.PickableOff()
                self.renderer.AddActor(connector)
                self.contact_actors.append((group, connector))
        self.set_contact_options(self.contact_options, render=False)

    def set_contact_options(self, options, render=True):
        self.contact_options = dict(options)
        source = bool(options.get("source", False))
        if source != self.contact_source:
            self.contact_source = source
            self._build_contacts()
        group = options.get("group", "")
        for name, actor in self.contact_actors:
            actor.SetVisibility(options.get("visible", True) and (not group or group == name))
        self.set_result_context(getattr(self,'result_active',False))
        if render:
            self.render()

    def set_result_context(self,active):
        """Neutral anatomy markers keep group colors distinct from measured values."""
        self.result_active=active
        for _,actor in self.contact_actors:
            if not hasattr(actor,'_result_base_color'):
                actor._result_base_color=actor.GetProperty().GetColor()
            actor.GetProperty().SetColor(*((.56,.62,.68) if active else actor._result_base_color))

    def set_camera_preset(self, preset: int):
        if self.scene is None:
            return
        camera = self.renderer.GetActiveCamera()
        direction = [(.9, 1.35, .65), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1)][preset]
        camera.SetFocalPoint(*self.center)
        camera.SetPosition(*(self.center + 450 * np.array(direction)))
        camera.SetViewUp(*((0, 1, 0) if preset == 4 else (0, 0, 1)))
        camera.ParallelProjectionOn()
        # Fit the cortex, independently of optional MRI planes extending beyond it.
        vertices = np.concatenate([m.vertices for k, m in self.scene.surfaces.items() if k.startswith("pial")])
        mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
        self.renderer.ResetCamera(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
        camera.Zoom(1.1)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def camera_state(self) -> dict:
        camera = self.renderer.GetActiveCamera()
        return {"position": list(camera.GetPosition()), "focal_point": list(camera.GetFocalPoint()),
                "view_up": list(camera.GetViewUp()), "parallel_scale": camera.GetParallelScale()}

    def restore_camera(self, state: dict):
        try:
            vectors = [np.asarray(state[key], dtype=float) for key in ("position", "focal_point", "view_up")]
            scale = float(state["parallel_scale"])
            if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors) or not 0.001 < scale < 1e6:
                return
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(*vectors[0])
            camera.SetFocalPoint(*vectors[1])
            camera.SetViewUp(*vectors[2])
            camera.SetParallelScale(scale)
            self.renderer.ResetCameraClippingRange()
            self.render()
        except (KeyError, ValueError, TypeError):
            return

    def export_surface(self, path):
        append = vtk.vtkAppendPolyData()
        for key, poly in self.polys.items():
            kind, hemi = key.split("_")
            if kind == self.mode and self.hemispheres[hemi]:
                append.AddInputData(poly)
        append.Update()
        output = vtk.vtkPolyData()
        output.ShallowCopy(append.GetOutput())
        coordinate = vtk.vtkStringArray()
        coordinate.SetName("CoordinateSystem")
        coordinate.InsertNextValue("scanner_RAS_mm")
        output.GetFieldData().AddArray(coordinate)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(output)
        if not writer.Write():
            raise OSError(tr("3Dモデルを書き出せませんでした。"))

    def capture_image(self) -> QImage:
        # Native OpenGL children are not reliably included by QWidget.grab().
        self.widget.GetRenderWindow().Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.widget.GetRenderWindow())
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        data = capture.GetOutput()
        width, height, _ = data.GetDimensions()
        pixels = vtk_to_numpy(data.GetPointData().GetScalars()).reshape(height, width, 3)[::-1].copy()
        return QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888).copy()

    def shutdown(self):
        self.orientation.SetEnabled(0)
        self.widget.Finalize()
