"""Route clicks through actual child widgets to catch native button/line-edit overlap."""
import unittest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication,QWidget,QDialog,QVBoxLayout,QDoubleSpinBox,QSpinBox,
                               QStyle,QStyleFactory,QStyleOptionSpinBox)
from brain_viewer.window import STYLE


def button_rect(spin,up):
    option=QStyleOptionSpinBox();spin.initStyleOption(option)
    sub=QStyle.SubControl.SC_SpinBoxUp if up else QStyle.SubControl.SC_SpinBoxDown
    return spin.style().subControlRect(QStyle.ComplexControl.CC_SpinBox,option,sub,spin)


def click_spin(spin,up):
    point=button_rect(spin,up).center()
    # Sending directly to the spin box bypasses the child line edit and hid this bug.
    target=spin.childAt(point) or spin
    QTest.mouseClick(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,target.mapFrom(spin,point))
    QApplication.processEvents()
    return target


class SpinButtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.old_style=self.app.style().objectName();self.roots=[]

    def tearDown(self):
        for root in self.roots:root.close();root.deleteLater()
        QApplication.processEvents();self.app.setStyle(self.old_style)

    def box(self,cls=QDoubleSpinBox,width=150,height=None,parent=None):
        if parent is None:
            parent=QWidget();parent.setStyleSheet(STYLE);self.roots.append(parent);QVBoxLayout(parent)
        spin=cls();parent.layout().addWidget(spin)
        spin.setRange(-1000,1000);spin.setSingleStep(.25 if cls==QDoubleSpinBox else 5)
        spin.setValue(20);spin.setFixedWidth(width)
        if height is not None:spin.setFixedHeight(height)
        parent.show();QApplication.processEvents();return spin

    def test_up_and_down_use_uncovered_buttons_across_platform_styles_and_sizes(self):
        for style in QStyleFactory.keys():
            self.app.setStyle(style)
            for cls in (QDoubleSpinBox,QSpinBox):
                for width,height in ((100,28),(140,32),(280,44),(180,70)):
                    with self.subTest(style=style,cls=cls.__name__,width=width,height=height):
                        spin=self.box(cls,width,height)
                        for up in (True,False):
                            rect=button_rect(spin,up)
                            self.assertFalse(rect.isEmpty());self.assertTrue(spin.rect().contains(rect))
                            self.assertFalse(spin.lineEdit().geometry().intersects(rect))
                            before=spin.value();target=click_spin(spin,up)
                            self.assertIs(target,spin)
                            self.assertAlmostEqual(spin.value(),before+(spin.singleStep() if up else -spin.singleStep()))
                        spin.parentWidget().hide()

    def test_fractional_negative_steps_keyboard_and_direct_input(self):
        spin=self.box();spin.setValue(-16.5);spin.setSingleStep(.1)
        click_spin(spin,True);self.assertAlmostEqual(spin.value(),-16.4)
        click_spin(spin,False);self.assertAlmostEqual(spin.value(),-16.5)
        QTest.keyClick(spin,Qt.Key.Key_Up);self.assertAlmostEqual(spin.value(),-16.4)
        QTest.keyClick(spin,Qt.Key.Key_Down);self.assertAlmostEqual(spin.value(),-16.5)
        spin.lineEdit().selectAll();QTest.keyClicks(spin.lineEdit(),'36.75');QTest.keyClick(spin,Qt.Key.Key_Return)
        self.assertAlmostEqual(spin.value(),36.75)
        click_spin(spin,True);self.assertAlmostEqual(spin.value(),36.85)

    def test_limits_disabled_and_read_only_still_apply(self):
        for cls in (QDoubleSpinBox,QSpinBox):
            spin=self.box(cls);spin.setRange(0,10);spin.setSingleStep(1)
            spin.setValue(10);click_spin(spin,True);self.assertEqual(spin.value(),10)
            click_spin(spin,False);self.assertEqual(spin.value(),9)
            spin.setValue(0);click_spin(spin,False);self.assertEqual(spin.value(),0)
            click_spin(spin,True);self.assertEqual(spin.value(),1)
            spin.setReadOnly(True);click_spin(spin,True);click_spin(spin,False);self.assertEqual(spin.value(),1)
            spin.setReadOnly(False);spin.setEnabled(False)
            click_spin(spin,True);click_spin(spin,False);self.assertEqual(spin.value(),1)

    def test_dialog_inherits_shared_fix(self):
        root=QWidget();root.setStyleSheet(STYLE);self.roots.append(root)
        dialog=QDialog(root);QVBoxLayout(dialog);self.roots.append(dialog)
        for cls in (QDoubleSpinBox,QSpinBox):
            spin=self.box(cls,parent=dialog)
            self.assertIs(click_spin(spin,True),spin)
            self.assertAlmostEqual(spin.value(),20+spin.singleStep())
            click_spin(spin,False);self.assertEqual(spin.value(),20)


if __name__=='__main__':unittest.main()
