"""CT/MRI physical coordinates and non-mutating hover labels."""
from copy import deepcopy
import unittest
import numpy as np
import nibabel as nib
from PySide6.QtCore import Qt,QPointF,QPoint,QEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from brain_viewer.contact_overview import HeadCT
from brain_viewer.slice_view import SliceCanvas
from brain_viewer.ct_section import CTSection
from brain_viewer.alignment_review import rigid_delta
from test_ct_localization import native_scene


class ReviewInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def setUp(self): self.widgets=[]
    def tearDown(self):
        for widget in self.widgets: widget.close(); widget.deleteLater()
        QApplication.processEvents()

    def test_mri_overview_uses_transformed_contact_and_native_mri_sampling(self):
        scene=native_scene()
        scene.ct_to_mri=rigid_delta([13,-7,19],[5,-3,2],[0,0,0])
        contact=scene.contacts[0]
        contact.ct_position=nib.affines.apply_affine(np.linalg.inv(scene.ct_to_mri),contact.position)
        shape=(25,26,27); affine=np.diag([.7,.8,.9,1.]); affine[:3,3]=contact.position-np.array(shape)/2*np.diag(affine)[:3]
        grid=nib.affines.apply_affine(affine,np.indices(shape).reshape(3,-1).T)
        gradient=np.array([.8,.3,.6]); scene.raw_mri=(50+grid@gradient).reshape(shape).astype(np.float32)
        scene.source_affine=affine
        view=HeadCT(); self.widgets.append(view); view.resize(420,500)
        for axis in (0,1,2):
            view.configure(scene,scene.contacts,0,axis,300,150,modality='MRI')
            np.testing.assert_array_equal(view.positions[0],contact.position)
            self.assertEqual(view.center[axis],contact.position[axis])
            h,w=view.raw.shape; r,c=h//2,w//2
            world=view.center+((c+.5)/w-.5)*view.extent[0]*view.u+(.5-(r+.5)/h)*view.extent[1]*view.v
            self.assertAlmostEqual(float(view.raw[r,c]),50+world@gradient,places=4)
            raw=view.raw; view.configure(scene,scene.contacts,0,axis,600,200,modality='MRI')
            self.assertIs(view.raw,raw)
        view.configure(scene,scene.contacts,0,2,4000,1000)
        np.testing.assert_array_equal(view.positions[0],contact.ct_position)

    def test_hover_label_follows_mouse_without_moving_crosshair_or_rebuilding_image(self):
        scene=native_scene(); scene.label_volume=np.full(scene.data.shape,10,np.uint16)
        scene.label_volume[8:]=49; scene.label_names={10:'Left-Thalamus',49:'Right-Thalamus'}
        canvas=SliceCanvas(2); self.widgets.append(canvas); canvas.resize(500,500); canvas.set_scene(scene)
        canvas.show(); QApplication.processEvents(); current=canvas.ijk.copy(); image=canvas._image
        target=current.copy(); target[0]=10; position=canvas.position_for_index(target)
        event=QMouseEvent(QEvent.Type.MouseMove,position,canvas.mapToGlobal(position.toPoint()),
            Qt.MouseButton.NoButton,Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(canvas,event)
        self.assertEqual(canvas.hover_annotation()['name'],'Right-Thalamus')
        np.testing.assert_array_equal(canvas.ijk,current); self.assertIs(canvas._image,image)
        self.assertTrue(canvas.contact_is_hovered(position))
        QApplication.sendEvent(canvas,QEvent(QEvent.Type.Leave))
        self.assertIsNone(canvas.hover_annotation()); self.assertFalse(canvas.contact_is_hovered(position))
        canvas._hover=position; canvas.set_brush('paint',2.)
        self.assertIsNone(canvas.hover_annotation())

    def test_move_mode_selects_other_marker_then_allows_small_adjustments(self):
        scene=native_scene(); lead=scene.contacts
        other=deepcopy(lead[0]); other.uid='second'; other.ct_position=other.ct_position+[4,0,0]; other.position=other.ct_position.copy(); lead.append(other)
        section=CTSection(0); self.widgets.append(section); section.resize(700,250)
        section.configure(scene,lead,0,4000,1000); section.edit=True; section.show(); QApplication.processEvents()
        moved=[]; selected=[]; section.moved.connect(moved.append); section.selected.connect(selected.append)
        QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,section.project(lead[0].ct_position).toPoint())
        self.assertFalse(moved)
        QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,section.project(other.ct_position).toPoint())
        self.assertEqual(selected,[1]); self.assertFalse(moved)
        point=section.project(lead[0].ct_position).toPoint()+QPoint(3,-2)
        QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
        self.assertEqual(len(moved),1)
        self.assertGreater(np.linalg.norm(moved[0]-lead[0].ct_position),.01)
        self.assertLess(np.linalg.norm(moved[0]-lead[0].ct_position),1.)
