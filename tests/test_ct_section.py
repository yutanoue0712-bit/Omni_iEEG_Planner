"""Viewport navigation against physical native-CT coordinates, using synthetic data."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtCore import Qt,QPoint,QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from brain_viewer.ct_section import CTSection
from brain_viewer.imaging import Contact


def contacts(points):
    return [Contact(f'A{i+1}','A',np.asarray(p,float),(1.,.5,.2),ct_position=np.asarray(p,float)) for i,p in enumerate(points)]


class CTSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.sections=[]

    def tearDown(self):
        for section in self.sections:section.close();section.deleteLater()
        QApplication.processEvents()

    def section(self,mode,scene,lead,index=1):
        section=CTSection(mode);self.sections.append(section);section.resize(700,270)
        section.configure(scene,lead,index,4000,1000);section.show();QApplication.processEvents()
        return section

    def simple(self):
        affine=np.diag([.5,.5,.5,1]);affine[:3,3]=-25
        data=np.zeros((101,101,101),np.float32)
        return SimpleNamespace(raw_ct=data,raw_ct_affine=affine),contacts([[-10,0,0],[0,0,0],[10,0,0]])

    def test_zoom_out_samples_ct_outside_original_crop_and_pan_follows_anatomy(self):
        scene,lead=self.simple();scene.raw_ct[48:53,48:53,69:72]=3000  # RAS (0,0,+10), outside original 12 mm slab.
        section=self.section(0,scene,lead)
        self.assertEqual(float(section.raw.max()),0.)
        section.zoom_by(.5)
        self.assertGreater(float(section.raw.max()),2900.)
        self.assertTrue(section.image_rect().contains(section.project([0,0,10])))
        self.assertTrue(np.allclose(section.extent,[72,24]))
        section.pan_by(QPointF(0,section.image_rect().height()/2))
        self.assertGreater(float(section.raw.max()),2900.)
        section.reset_view();self.assertEqual(float(section.raw.max()),0.)
        self.assertLessEqual(section.raw.size,1200*320)

    def test_oblique_ct_sampling_and_contact_edit_after_zoom_and_pan(self):
        affine=np.eye(4);affine[:3,:3]=Rotation.from_euler('xyz',[17,-23,11],degrees=True).as_matrix()@np.diag([.5,.7,.9])
        origin=np.array([15.,-22.,8.]);affine[:3,3]=origin-affine[:3,:3]@np.array([60.,60.,60.])
        gradient=np.array([.7,1.1,.3]);grid=np.indices((121,121,121),dtype=np.float32)
        raw=(200+np.dot(affine[:3,3],gradient)+np.einsum('i,ijkl->jkl',gradient@affine[:3,:3],grid)).astype(np.float32)
        scene=SimpleNamespace(raw_ct=raw,raw_ct_affine=affine)
        direction=np.array([.63,-.44,.73]);direction/=np.linalg.norm(direction)
        lead=contacts([origin-direction*10,origin+np.array([.2,.3,.4]),origin+direction*10])
        for mode in range(3):
            section=self.section(mode,scene,lead);section.zoom_by(2.5);section.pan_by(QPointF(19,-11))
            h,w=section.raw.shape;r,c=h//3,w//3
            world=section.view_center+((c+.5)/w-.5)*section.extent[0]*section.u+(.5-(r+.5)/h)*section.extent[1]*section.v
            self.assertAlmostEqual(float(section.raw[r,c]),200+np.dot(world,gradient),places=3)
            normal=np.cross(section.u,section.v);depth=np.dot(lead[1].ct_position-section.view_center,normal)
            screen=section.image_rect().center().toPoint()+QPoint(22,-9)
            target=section.view_center+22/section.image_rect().width()*section.extent[0]*section.u+9/section.image_rect().height()*section.extent[1]*section.v
            # Account only for the integer QTest pixel center, independently of the inverse mapping.
            rounding=QPointF(section.image_rect().center().toPoint())-section.image_rect().center()
            target+=rounding.x()/section.image_rect().width()*section.extent[0]*section.u-rounding.y()/section.image_rect().height()*section.extent[1]*section.v
            moved=[];section.moved.connect(moved.append);section.edit=True
            QTest.mouseClick(section,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,screen)
            np.testing.assert_allclose(moved[-1],target+depth*normal,atol=1e-7)
            self.assertAlmostEqual(np.dot(moved[-1]-lead[1].ct_position,normal),0.,places=8)

    def test_wheel_keeps_cursor_anchor_and_pan_never_edits_a_contact(self):
        scene,lead=self.simple();section=self.section(0,scene,lead);section.edit=True
        moved=[];selected=[];window=[];section.moved.connect(moved.append);section.selected.connect(selected.append)
        section.window_delta.connect(lambda x,y:window.append((x,y)))
        original=np.array([c.ct_position for c in lead]);point=section.image_rect().center().toPoint()+QPoint(75,13)
        anchor=section.plane_at(point).copy()
        wheel=QWheelEvent(QPointF(point),QPointF(section.mapToGlobal(point)),QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        QApplication.sendEvent(section,wheel);self.assertGreater(section.zoom,1.)
        np.testing.assert_allclose(section.plane_at(point),anchor,atol=1e-9)
        for button,modifier in ((Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ShiftModifier),(Qt.MouseButton.MiddleButton,Qt.KeyboardModifier.NoModifier)):
            old=section.pan.copy()
            QTest.mousePress(section,button,modifier,point)
            QTest.mouseMove(section,point+QPoint(18,7),20)
            QTest.mouseRelease(section,button,modifier,point+QPoint(18,7))
            self.assertFalse(np.array_equal(old,section.pan))
        self.assertFalse(moved);self.assertFalse(selected)
        QTest.mousePress(section,Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier,point)
        QTest.mouseMove(section,point+QPoint(12,5),20)
        QTest.mouseRelease(section,Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier,point+QPoint(12,5))
        self.assertTrue(window);np.testing.assert_array_equal(original,[c.ct_position for c in lead])

    def test_views_independent_contrast_cache_and_selection_preserves_zoom(self):
        scene,lead=self.simple();panels=[self.section(i,scene,lead) for i in range(3)]
        section=panels[0];section.zoom_by(.5);section.pan_by(QPointF(24,-15))
        state=section.pan.copy();raw=section.raw;image=section.image.copy()
        section.configure(scene,lead,2,2000,300)
        self.assertIs(section.raw,raw);self.assertNotEqual(section.image,image)
        self.assertEqual(section.zoom,.5);np.testing.assert_array_equal(section.pan,state)
        self.assertTrue(all(p.zoom==1. and not p.pan.any() for p in panels[1:]))
        cross=panels[2];cross.zoom_by(3);cross.pan_by(QPointF(20,30));cross.configure(scene,lead,2,4000,1000)
        self.assertEqual(cross.zoom,3.);np.testing.assert_array_equal(cross.pan,[0,0])
        np.testing.assert_allclose(cross.plane_at(cross.image_rect().center()),lead[2].ct_position)
        other=deepcopy(lead)
        for c in other:c.group='B'
        section.configure(scene,other,0,4000,1000)
        self.assertEqual(section.zoom,1.);np.testing.assert_array_equal(section.pan,[0,0])

    def test_buttons_home_limits_and_empty_group(self):
        scene,lead=self.simple();section=self.section(0,scene,lead)
        section.zoom_in.click();self.assertGreater(section.zoom,1.)
        section.zoom_out.click();self.assertEqual(section.zoom,1.)
        section.zoom_by(1e9);self.assertEqual(section.zoom,10.);self.assertFalse(section.zoom_in.isEnabled())
        section.zoom_by(1e-12);self.assertEqual(section.zoom,.1);self.assertFalse(section.zoom_out.isEnabled())
        section.reset_button.click();self.assertEqual(section.zoom,1.)
        section.pan_by(QPointF(60,10));QTest.keyClick(section,Qt.Key.Key_Home)
        np.testing.assert_array_equal(section.pan,[0,0])
        section.configure(scene,[],0,4000,1000)
        self.assertTrue(section.image.isNull());self.assertFalse(section.zoom_in.isEnabled())
        section.zoom_by(2);self.assertEqual(section.zoom,1.)


if __name__=='__main__':unittest.main()
