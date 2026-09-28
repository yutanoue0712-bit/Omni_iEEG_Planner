from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from brain_viewer.analysis_results import read_workbook,colors,color_limits,overlay_points,exact_bindings,export_record
from brain_viewer.result_display import metric_display,marker_options,spatial_options,visible_values
from brain_viewer.imaging import save_scene,load_scene
from tests.test_analysis_results import result,contacts
from tests.test_multimodal import small_scene
from tests.xlsx_fixture import write_xlsx


class ResultDisplayTests(unittest.TestCase):
    def test_explicit_ms_preserves_besa_values_and_unknown_value_unit(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'audio_res_2.xlsx'
            write_xlsx(path,{'Sheet1':[[None,'-10','0','10'],['A1-A2','-.01','0','.023']]})
            r=read_workbook(path,time_unit='ms')
        self.assertEqual(r.time_unit,'ms');self.assertEqual(r.units,[''])
        np.testing.assert_allclose(r.values[0,:,0],[-.01,0,.023])
        np.testing.assert_array_equal(r.times,[-10,0,10])
        self.assertEqual(r.provenance['time_unit_source'],'specified_on_import')

    def test_index_metadata_and_independent_ranking_table(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'hfo_total.xlsx'
            write_xlsx(path,{'Sheet1':[[None,'channel_name',None,None,'total','before','after',None,'channel_name','total'],
                [1,'A1-A2','SEEG','A',3,2,1,None,'A2-A3',30],
                [2,'A2-A3','SEEG','A',30,12,18,None,'A1-A2',3]]})
            r=read_workbook(path)
        self.assertEqual(r.channels,['A1-A2','A2-A3']);self.assertEqual(r.metrics,['total','before','after'])
        np.testing.assert_array_equal(r.values[:,0,:],[[3,2,1],[30,12,18]])
        self.assertEqual(r.provenance['columns'],[5,6,7]);self.assertEqual(r.provenance['channel_column'],2)

    def test_palette_reverse_endpoints_and_missing(self):
        values=np.array([-4.,0.,4.,np.nan])
        a=colors(values,(-4,4),'turbo');b=colors(values,(-4,4),'turbo',True)
        np.testing.assert_allclose(a[0],b[2]);np.testing.assert_allclose(a[2],b[0])
        np.testing.assert_allclose(a[3],b[3]);self.assertFalse(np.allclose(a[:3],colors(values,(-4,4),'inferno')[:3]))

    def test_threshold_boundaries_and_missing_are_distinct_from_zero(self):
        r=result(True);v=np.array([-2.,-1.,0.,1.,2.,np.nan])
        options=metric_display(r,0)
        self.assertEqual(visible_values(v,options).tolist(),[True]*5+[False])
        options.update(threshold_mode='above',threshold=1.)
        self.assertEqual(visible_values(v,options).tolist(),[False,False,False,True,True,False])
        options.update(threshold_mode='absolute',threshold=2.)
        self.assertEqual(visible_values(v,options).tolist(),[True,False,False,False,True,False])

    def test_filtered_positions_keep_uids_and_full_time_scale(self):
        r=result(True);c=contacts();r.bindings=exact_bindings(r,c);original=r.values.copy()
        r.settings['display']={'value':{'colormap':'viridis','threshold_mode':'above','threshold':2.}}
        points,rgb,idx=overlay_points(r,c,0,1)
        self.assertEqual(idx.tolist(),[0,1]);np.testing.assert_allclose(points[0],[1.5,2,3])
        self.assertEqual(color_limits(r,0),(-4,4));self.assertEqual(len(overlay_points(r,c,0,0)[0]),0)
        np.testing.assert_array_equal(r.values,original)
        record=export_record(r,c,0,[0,1,2]);self.assertEqual(record['visible_channel_indices'],{'0':[],'1':[0,1],'2':[]})
        self.assertEqual(record['values'][0],[0,4,-2])

    def test_display_and_marker_options_survive_patient_save(self):
        r=result(True);r.settings.update(display={'value':{'colormap':'turbo','reverse':True,'threshold_mode':'absolute','threshold':1.5}},
            markers={'radius_mm':3.,'slab_mm':6.,'opacity':.7,'labels':True},spatial={'mode':'both','distance_mm':17.5})
        scene=small_scene();scene.contacts=contacts();r.bindings=exact_bindings(r,scene.contacts);scene.results=[r]
        with TemporaryDirectory() as d:
            folder=save_scene(scene,Path(d),{});saved=load_scene(folder/'scene.json').results[0]
        self.assertEqual(metric_display(r,0),metric_display(saved,0));self.assertEqual(marker_options(r),marker_options(saved))
        self.assertEqual(spatial_options(r),spatial_options(saved))
        self.assertEqual(color_limits(r,0),color_limits(saved,0))


if __name__=='__main__':unittest.main()
