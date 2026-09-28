"""Name ambiguity, review transactions and bipolar spatial identity."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication,QComboBox,QWidget
from brain_viewer.analysis_results import AnalysisResult,resolved_positions,binding_arity
from brain_viewer.contact_matching import ContactNameIndex
from brain_viewer.imaging import Contact
from brain_viewer.result_mapping import MappingDialog


def contacts(group='LPH',numbers=(1,2,3,10)):
    return [Contact(f'{group}-{i:02}',group,np.array([float(i),2.,3.]),(.5,.5,.5),uid=f'{group}_{i}') for i in numbers]


def result(channels):
    return AnalysisResult('Matching test','static',channels,['value'],np.ones((len(channels),1,1)),np.array([0.]),[''])


class NameMatchingTests(unittest.TestCase):
    def test_bipolar_formats_preserve_endpoint_order_and_numbers(self):
        index=ContactNameIndex(contacts())
        for name in ('LPH1-LPH2','LPH01-LPH02','LPH-01-LPH-02',' lph1 – LPH2 ','ＬＰＨ１−ＬＰＨ２'):
            with self.subTest(name=name): self.assertEqual(index.match(name).ids,('LPH_1','LPH_2'))
        self.assertEqual(index.match('LPH10-LPH2').ids,('LPH_10','LPH_2'))
        self.assertEqual(index.match('LPH2-LPH1').ids,('LPH_2','LPH_1'))
        self.assertEqual(index.match('LPH-1').ids,('LPH_1',))
        self.assertEqual(binding_arity('LPH-1',contacts()),1)

    def test_hyphens_and_digits_in_electrode_name_are_not_stripped(self):
        index=ContactNameIndex(contacts('LA-PH')+contacts('E01'))
        self.assertEqual(index.match('LA-PH1-LA-PH2').ids,('LA-PH_1','LA-PH_2'))
        self.assertEqual(index.match('E011-E012').ids,('E01_1','E01_2'))
        self.assertEqual(index.match('E01-01-E01-02').ids,('E01_1','E01_2'))

    def test_missing_truncated_same_contact_and_unknown_electrodes_are_not_guessed(self):
        index=ContactNameIndex(contacts())
        for name in ('LPH3-LPH4','LPH1-LPH','LPH1-LPH1','OTHER1-OTHER2','LPH1-2'):
            with self.subTest(name=name): self.assertFalse(index.match(name).ids)
        self.assertEqual(index.match('LPH3-LPH4').issue,'missing')
        self.assertEqual(index.match('LPH1-LPH1').issue,'same_contact')

    def test_normalized_duplicates_and_multiple_parses_remain_ambiguous(self):
        c=contacts(); duplicate=deepcopy(c[0]); duplicate.name='lph1'; duplicate.uid='duplicate'; c.append(duplicate)
        self.assertEqual(ContactNameIndex(c).match('LPH1-LPH2').issue,'ambiguous')
        # One ambiguous endpoint cannot be resolved by eliminating a same-UID pair.
        c[1].name='LPH1'; self.assertEqual(ContactNameIndex(c).match('LPH1-LPH1').issue,'ambiguous')
        c=contacts('E01'); c+=contacts('E0',numbers=(11,12))
        self.assertEqual(ContactNameIndex(c).match('E011-E012').issue,'ambiguous')

    def test_cross_electrode_pairs_use_named_endpoints(self):
        index=ContactNameIndex(contacts()+contacts('RPH'))
        self.assertEqual(index.match('LPH2-RPH10').ids,('LPH_2','RPH_10'))


class MappingReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.result=result(['LPH1-LPH2','LPH2-LPH3','LPH3-LPH4'])
        self.window=QWidget(); self.window.scene=SimpleNamespace(contacts=contacts(),results=[self.result])
        self.window.select_world=Mock()
        view=QComboBox(self.window); view.addItem('Channels','channels'); view.addItem('Brain','brain')
        self.panel=SimpleNamespace(window=self.window,selected=lambda:self.result,view=view,
                                   refresh=Mock(),show_on_electrodes=Mock())
        self.dialog=None

    def tearDown(self):
        if self.dialog: self.dialog.close(); self.dialog.deleteLater()
        self.window.close(); self.window.deleteLater(); QApplication.processEvents()

    def open_dialog(self):
        self.dialog=MappingDialog(self.panel); return self.dialog

    def test_candidates_cancel_and_filter_do_not_change_patient_bindings(self):
        dialog=self.open_dialog()
        self.assertEqual([b.currentData() for b in dialog.boxes[0]],['LPH_1','LPH_2'])
        self.assertEqual(dialog.lead_boxes['LPH'][0].currentData(),'LPH')
        self.assertEqual(self.result.bindings,{})
        self.assertEqual(len(resolved_positions(self.result,self.window.scene.contacts)[0]),0)
        dialog.unresolved_only.setChecked(True)
        self.assertTrue(dialog.table.isRowHidden(0)); self.assertFalse(dialog.table.isRowHidden(2))
        dialog.reject(); self.assertEqual(self.result.bindings,{})
        self.assertNotIn('mapping_reviewed',self.result.settings)

    def test_confirm_maps_midpoints_and_follows_uid_after_edit(self):
        dialog=self.open_dialog(); dialog.jump(0,0)
        np.testing.assert_array_equal(self.window.select_world.call_args.args[0],[1.5,2,3])
        dialog.commit()
        self.assertEqual(self.result.bindings,{'LPH1-LPH2':['LPH_1','LPH_2'],'LPH2-LPH3':['LPH_2','LPH_3']})
        self.assertEqual(self.result.binding_modes['LPH1-LPH2'],'bipolar')
        self.assertTrue(self.result.settings['mapping_reviewed']); self.panel.show_on_electrodes.assert_called_once()
        self.window.scene.contacts[0].name='Renamed'; self.window.scene.contacts[0].position[0]=5.
        indices,points=resolved_positions(self.result,self.window.scene.contacts)
        self.assertEqual(indices.tolist(),[0,1]); np.testing.assert_allclose(points[0],[3.5,2,3])

    def test_automatic_matching_keeps_saved_and_manual_choices(self):
        self.result.bindings={'LPH1-LPH2':['LPH_2','LPH_3']}
        dialog=self.open_dialog()
        self.assertEqual([b.currentData() for b in dialog.boxes[0]],['LPH_2','LPH_3'])
        first=dialog.boxes[1][0]; first.setCurrentIndex(first.findData('LPH_10'))
        dialog.auto_button.click()
        self.assertEqual([b.currentData() for b in dialog.boxes[1]],['LPH_10','LPH_3'])
        self.assertEqual(self.result.bindings,{'LPH1-LPH2':['LPH_2','LPH_3']})

    def test_alias_change_replaces_previous_candidate_and_preserves_lead_choices(self):
        dialog=self.open_dialog(); dialog.lead_boxes['LPH'][1].setChecked(True)
        dialog.table.item(0,1).setText('LPH2-LPH10')
        self.assertEqual([b.currentData() for b in dialog.boxes[0]],['LPH_2','LPH_10'])
        self.assertTrue(dialog.lead_boxes['LPH'][1].isChecked())
        dialog.table.item(0,1).setText('Unknown1-Unknown2')
        self.assertFalse(any(b.currentData() for b in dialog.boxes[0]))

    def test_incomplete_pair_or_deleted_contact_cannot_be_saved(self):
        dialog=self.open_dialog(); dialog.boxes[0][1].setCurrentIndex(0); dialog.commit()
        self.assertEqual(self.result.bindings,{})
        box=dialog.boxes[0][1]; box.setCurrentIndex(box.findData('LPH_2'))
        self.window.scene.contacts.pop(0); dialog.commit(); self.assertEqual(self.result.bindings,{})

    def test_saved_monopolar_mode_is_preserved_after_contact_rename(self):
        self.result=result(['LPH-01']);self.window.scene.results=[self.result]
        self.result.bindings={'LPH-01':['LPH_1']};self.result.binding_modes={'LPH-01':'monopolar'}
        self.window.scene.contacts[0].name='Renamed'
        dialog=self.open_dialog();self.assertTrue(dialog.row_valid(0));dialog.commit()
        self.assertEqual(self.result.binding_modes,{'LPH-01':'monopolar'})


if __name__=='__main__': unittest.main()
