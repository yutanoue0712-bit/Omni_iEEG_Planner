"""Click numeric buttons in real workflows using synthetic images only."""
import unittest
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QScrollArea
from brain_viewer.window import ViewerWindow
from brain_viewer.imaging import make_demo,MRILayer
from brain_viewer.contact_editor import ContactEditor
from brain_viewer.analysis_results import AnalysisResult
from brain_viewer.segmentation import create_segment
from tests.test_diffusion import synthetic_model
from tests.test_spin_buttons import button_rect


class WorkflowSpinButtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.old_font=cls.app.font();cls.app.setFont(QFont('Yu Gothic UI',10))

    @classmethod
    def tearDownClass(cls):cls.app.setFont(cls.old_font)

    def check_buttons(self,spin):
        QTest.qWait(20)
        self.assertTrue(spin.isVisible() and spin.isEnabled())
        before=spin.value();step=spin.singleStep()
        # At an upper bound, start with Down so both directions can be exercised.
        directions=(True,False) if before+step<=spin.maximum() else (False,True)
        for up in directions:
            parent=spin.parentWidget()
            while parent is not None:
                if isinstance(parent,QScrollArea):parent.ensureWidgetVisible(spin,10,20)
                parent=parent.parentWidget()
            QTest.qWait(20)
            root=spin.window();point=spin.mapTo(root,button_rect(spin,up).center())
            # Use this window's hit test; other desktop apps can cover the test window.
            target=root.childAt(point)
            self.assertIs(target,spin,'Another widget intercepts the visible arrow')
            value=spin.value()
            QTest.mouseClick(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,target.mapFrom(root,point))
            QApplication.processEvents()
            self.assertAlmostEqual(spin.value(),value+(step if up else -step))
        self.assertAlmostEqual(spin.value(),before)

    def test_image_segmentation_dti_results_and_contact_dialog(self):
        scene=make_demo();self.assertEqual(scene.kind,'synthetic')
        scene.extra_mris.append(MRILayer('mri_spincheck','Synthetic b0','DTI-b0',scene.data,
            np.ones(scene.data.shape,bool),scene.data,scene.affine,np.eye(4),window=512,level=256))
        synthetic_model(scene)
        scene.results.append(AnalysisResult('Synthetic result','time_series',['E01-1'],['Power'],
            np.array([[[0.],[10.],[20.]]]),np.array([0.,1.,2.]),['a.u.'],time_unit='s'))
        window=ViewerWindow(smoke=True,demo=True)
        try:
            window.show();window.initialize();window.install_scene(scene);QTest.qWait(250)
            window.workflow_tabs.setCurrentIndex(0);window.side_tabs.setCurrentIndex(0)
            for name in ('extra_window','extra_level','window_spin','level_spin','ct_window','ct_level'):
                with self.subTest(control=name):self.check_buttons(getattr(window,name))
            self.assertEqual(scene.extra_mris[0].window,512)
            self.assertEqual(scene.extra_mris[0].level,256)
            window.findChild(QScrollArea).ensureWidgetVisible(window.extra_window)
            QApplication.processEvents()
            window.grab().save('private_reports/spin_controls_synthetic.png')
            window.workflow_tabs.setCurrentIndex(1)
            window.segmentation.add_segments([create_segment(scene,'manual','Synthetic ROI')])
            self.check_buttons(window.segmentation.radius)
            window.workflow_tabs.setCurrentIndex(5)
            for spin in (window.diffusion.roi1.radius,window.diffusion.roi_thickness,
                         *window.diffusion.parameters.values(),window.diffusion.max_seeds):
                self.check_buttons(spin)
            window.workflow_tabs.setCurrentIndex(4)
            for name in ('minimum','maximum','fps','start','end'):
                with self.subTest(control='results.'+name):self.check_buttons(getattr(window.results,name))
            original=np.array([c.ct_position for c in scene.contacts])
            editor=ContactEditor(scene,window)
            try:
                editor.show();QApplication.processEvents()
                for spin in (editor.expected,*editor.position,editor.width_spin,editor.level_spin):
                    self.check_buttons(spin)
            finally:editor.reject()
            np.testing.assert_array_equal(original,[c.ct_position for c in scene.contacts])
        finally:window.close();QApplication.processEvents()


if __name__=='__main__':unittest.main()
