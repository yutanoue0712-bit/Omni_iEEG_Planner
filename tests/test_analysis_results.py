from pathlib import Path
import tempfile
import unittest
import json
import numpy as np
from brain_viewer.analysis_results import (AnalysisResult,read_workbook,time_issue,exact_bindings,
    alias_candidates,resolved_positions,color_limits,colors,lead_binding,export_record,overlay_points,shared_binding_candidates)
from brain_viewer.imaging import Contact,InputError,save_scene,load_scene
from tests.test_multimodal import small_scene
from tests.xlsx_fixture import write_xlsx


def result(dynamic=False):
    return AnalysisResult('Test','time_series' if dynamic else 'static',['A1-A2','A2-A3'],['value'],
        np.array([[[0.],[4.],[-2.]],[[np.nan],[2.],[1.]]]) if dynamic else np.array([[[0.]],[[np.nan]]]),
        np.array([-10.,0.,10.]) if dynamic else np.array([0.]),[''])


def contacts():
    return [Contact(f'A{i}','A',np.array([float(i),2.,3.]),(.5,.5,.5),uid=f'id{i}') for i in range(1,4)]


class AnalysisResultTests(unittest.TestCase):
    def read(self,rows,sheet='Data',name='input.xlsx'):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/name; write_xlsx(p,{sheet:rows}); return read_workbook(p)

    def test_hfo_table_boundary_and_missing(self):
        r=self.read([['channel_name','total_events','spike_associated',None,'channel_name','spike_associated'],
                     ['A1-A2',0,None,None,'B1-B2',999],['A2-A3',4,2]],'Channel Ranking')
        self.assertEqual(r.metrics,['total_events','spike_associated']); self.assertEqual(r.values.shape,(2,1,2))
        self.assertEqual(r.values[0,0,0],0); self.assertTrue(np.isnan(r.values[0,0,1])); self.assertEqual(r.units,['events']*2)

    def test_time_numeric_strings_preserved_no_units_inferred(self):
        r=self.read([[None,'-20','-10','0'],['A1-A2','1.25','0','-2']],name='gamma.xlsx')
        np.testing.assert_array_equal(r.times,[-20,-10,0]); self.assertEqual(r.values.shape,(1,3,1))
        self.assertEqual(r.time_unit,''); self.assertEqual(r.units,['']); self.assertFalse(time_issue(r))

    def test_duplicate_times_and_range_no_silent_sort(self):
        r=self.read([[None,-10,0,10,0],['A1-A2',1,2,3,4]])
        self.assertTrue(time_issue(r)); np.testing.assert_array_equal(r.values[0,:,0],[1,2,3,4])
        r.settings.update(start=0,end=2); self.assertFalse(time_issue(r)); self.assertEqual(len(r.times),4)
        r.settings.update(start=1,end=1); self.assertTrue(time_issue(r))

    def test_duplicate_channel_rejected(self):
        with self.assertRaises(InputError): self.read([['channel','value'],['A1',1],['a1',2]])

    def test_invalid_values_and_missing_name_rejected(self):
        for row in (['A1','bad'],[None,5],['A1','inf']):
            with self.subTest(row=row),self.assertRaises(InputError): self.read([['channel','value'],row])

    def test_formula_cache_required(self):
        with self.assertRaises(InputError): self.read([['channel','value'],['A1',('1+1',None)]])
        r=self.read([['channel','value'],['A1',('1+1',2)]]); self.assertEqual(r.values[0,0,0],2.)

    def test_exact_names_only_and_ambiguous_duplicates(self):
        r=result(); c=contacts(); r.bindings=exact_bindings(r,c)
        self.assertEqual(r.bindings['A1-A2'],['id1','id2'])
        c[0].name='E01-01'; self.assertNotIn('A1-A2',exact_bindings(r,c))
        c=contacts(); c.append(Contact('a1','A',np.zeros(3),(1,1,1),uid='other'))
        self.assertNotIn('A1-A2',exact_bindings(r,c))

    def test_hyphens_in_contact_names(self):
        r=result(); c=contacts(); c[0].name='E01-01'; c[1].name='E01-02'
        r.channels=['E01-01','E01-01-E01-02']; r.bindings=exact_bindings(r,c)
        self.assertEqual(r.bindings,{'E01-01':['id1'],'E01-01-E01-02':['id1','id2']})

    def test_prefix_is_only_a_suggestion(self):
        full=result(); short=result(); short.channels=['A1-A','A2-A']
        self.assertEqual(alias_candidates(short,[full]),{'A1-A':'A1-A2','A2-A':'A2-A3'})
        self.assertEqual(exact_bindings(short,contacts()),{}); self.assertEqual(short.bindings,{})

    def test_shared_candidates_require_unique_full_name_and_do_not_apply(self):
        a=result(); a.bindings=exact_bindings(a,contacts()); b=result(); b.channels=['A1-A','A2-A']
        aliases=alias_candidates(b,[a]); proposed=shared_binding_candidates(b,[a],aliases)
        self.assertEqual(proposed['A1-A'],['id1','id2']); self.assertEqual(b.bindings,{})
        other=result(); other.bindings={'A1-A2':['id2','id3']}
        self.assertNotIn('A1-A',shared_binding_candidates(b,[a,other],aliases))

    def test_repeated_truncated_names_do_not_block_static_reference(self):
        full=result(); short=result(True); short.channels=['A1-A','A2-A']
        repeated=result(True); repeated.channels=short.channels.copy()
        self.assertEqual(alias_candidates(short,[full,repeated]),{'A1-A':'A1-A2','A2-A':'A2-A3'})
        full.channels[1]='A1-A20'
        self.assertNotIn('A1-A2',alias_candidates(result(),[full]))

    def test_monopolar_mode_survives_contact_rename(self):
        r=result(); c=contacts(); c[0].name='E01-01'; r.channels[0]='E01-01'
        r.bindings={'E01-01':['id1']}; r.binding_modes={'E01-01':'monopolar'}
        c[0].name='Renamed'; self.assertEqual(resolved_positions(r,c)[0].tolist(),[0])

    def test_uid_mapping_follows_moves_and_excludes_partial_bipolar(self):
        r=result(); c=contacts(); r.bindings=exact_bindings(r,c); c[0].name='Renamed'; c[0].position[0]=5
        indices,points=resolved_positions(r,c); np.testing.assert_allclose(points[0],[3.5,2,3])
        r.bindings['A1-A2']=['id1']; self.assertEqual(resolved_positions(r,c)[0].tolist(),[1])
        self.assertEqual(len(resolved_positions(r,c[:1])[0]),0)

    def test_lead_assistance_explicit_number_direction(self):
        c=contacts()
        for i,x in enumerate(c,1): x.group='E01'; x.name=f'E01-{i:02}'
        self.assertEqual(lead_binding('X1-X2',c,{'X':('E01',False)}),['id1','id2'])
        self.assertEqual(lead_binding('X1-X2',c,{'X':('E01',True)}),['id3','id2'])
        self.assertEqual(lead_binding('X3-X4',c,{'X':('E01',False)}),[])

    def test_fixed_color_scale_and_nan_not_at_zero(self):
        r=result(True); self.assertEqual(color_limits(r,0),(-4.,4.))
        np.testing.assert_allclose(colors(np.array([2.]),(-4,4)),colors(np.array([0.,2.]),(-4,4))[1:])
        r.bindings=exact_bindings(r,contacts()); points,_,idx=overlay_points(r,contacts(),0,0)
        self.assertEqual(idx.tolist(),[0]); self.assertEqual(len(points),1)

    def test_roundtrip_self_contained_and_export_metadata(self):
        s=small_scene(); s.contacts=contacts(); r=result(True); r.bindings=exact_bindings(r,s.contacts)
        r.time_unit='ms'; r.settings={'limits':{'value':[-3.,5.]},'start':0,'end':2}; s.results=[r]
        with tempfile.TemporaryDirectory() as d:
            folder=save_scene(s,Path(d),{}); restored=load_scene(folder/'scene.json')
            saved=restored.results[0]; np.testing.assert_array_equal(saved.values,r.values)
            self.assertEqual(saved.bindings,r.bindings); self.assertEqual(saved.settings,r.settings); self.assertEqual(saved.time_unit,'ms')
        record=export_record(r,s.contacts,0,[0,1,2],fps=20)
        self.assertIsNone(record['values'][1][0]); self.assertEqual(record['frames'][0]['time'],-10.)
        json.dumps(record,allow_nan=False)


if __name__=='__main__': unittest.main()
