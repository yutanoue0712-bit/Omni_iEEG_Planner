"""ANTs native worker, held-out geometry, portable warp and exclusion contracts."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest

import numpy as np
import SimpleITK as sitk

from test_postop_registration import phantom
from brain_viewer.imaging import InputError, save_scene, load_scene
from brain_viewer.registration import to_sitk, RAS_TO_LPS
from brain_viewer.postop_registration import pull_transform, resample_postop, export_postop, exclusion_preview
from brain_viewer.postop_deformation import field_record, validate_field, load_deformation, metadata
from brain_viewer.postop_ants import correct_postop_ants, ants_available, CorrectionCancelled


class PostopANTsTests(unittest.TestCase):
    @unittest.skipUnless(ants_available(),'Separate ANTs runtime is not installed')
    def test_real_syn_worker_corrects_known_shift_and_saves_portably(self):
        scene,base,cavity,offset=phantom()
        candidate=correct_postop_ants(scene,base,cavity,n4=False,iterations=(60,40,20),max_shift_mm=8)
        points=np.array([[x,y,z] for x in (-12,0,12) for y in (-12,0,12) for z in (-12,0,12)],float)
        truth=points.copy()
        for _ in range(30):truth[:,0]=points[:,0]-offset(truth.T)
        tx=pull_transform(candidate)
        actual=np.array([tx.TransformPoint(tuple(p*[-1,-1,1])) for p in points])*[-1,-1,1]
        before=np.linalg.norm(truth-points,axis=1); after=np.linalg.norm(truth-actual,axis=1)
        print('ANTs held-out landmarks: median before/after, p95 (mm)',
              float(np.median(before)),float(np.median(after)),float(np.percentile(after,95)),flush=True)
        self.assertLess(np.median(after),1.)
        self.assertLess(np.median(after),np.median(before)*.5)
        self.assertLess(np.percentile(after,95),1.8)
        self.assertIs(candidate.raw,base.raw)
        np.testing.assert_array_equal(base.data,base.raw)
        self.assertEqual(candidate.quality['numerical_qc']['folding_voxels'],0)
        self.assertIsNone(candidate.quality['landmark_accuracy_mm'])
        self.assertEqual(candidate.quality['ants']['ants_version'],'0.6.2')
        with tempfile.TemporaryDirectory() as tmp:
            scene.extra_mris.append(candidate)
            from brain_viewer.segmentation_sources import extract_segment
            extracted=extract_segment(scene,candidate.uid,100,900,'Postop tissue',connected=False,radius_mm=0)
            scene.segmentations.append(extracted)
            self.assertNotIn('vectors',extracted.provenance['source_deformation'])
            self.assertEqual(extracted.provenance['source_deformation']['sha256'],candidate.deformation['sha256'])
            saved=save_scene(scene,Path(tmp)/'work',{})
            obj=json.loads((saved/'scene.json').read_text(encoding='utf-8'))
            self.assertEqual(obj['schema'],'cortex-viewer/11')
            self.assertNotIn('vectors',obj['extra_mris'][-1]['deformation'])
            loaded=load_scene(saved/'scene.json'); restored=loaded.extra_mris[-1]
            self.assertEqual(loaded.segmentations[-1].provenance,extracted.provenance)
            np.testing.assert_array_equal(restored.deformation['vectors'],candidate.deformation['vectors'])
            output,valid=resample_postop(loaded,restored,restored.deformation)
            np.testing.assert_allclose(output,candidate.data,atol=1e-4)
            np.testing.assert_array_equal(valid,candidate.valid)
            out=export_postop(loaded,restored,tmp)
            exported=sitk.ReadTransform(str(out/'reference_to_native_postop_LPS.h5'))
            np.testing.assert_allclose(exported.TransformPoint((2.,4.,7.)),tx.TransformPoint((2.,4.,7.)),atol=1e-6)
            export_record=json.loads((out/'registration.json').read_text(encoding='utf-8'))
            warp=load_deformation(export_record['deformation'],out)
            np.testing.assert_array_equal(warp['vectors'],candidate.deformation['vectors'])
            self.assertFalse(export_record['point_forward_inverse_available'])

    def test_dense_lps_pull_composes_with_oblique_rigid_and_validates(self):
        scene,base,_,_=phantom()
        from brain_viewer.alignment_review import rigid_delta
        base.to_reference=rigid_delta([8,-13,6],[4,-7,2],[0,0,0])
        ref=to_sitk(scene.data,scene.affine)
        constant=sitk.TranslationTransform(3,(2.,-1.,.5))
        field=sitk.TransformToDisplacementField(constant,sitk.sitkVectorFloat64,ref.GetSize(),
                    ref.GetOrigin(),ref.GetSpacing(),ref.GetDirection())
        record=field_record(field); candidate=replace(base,deformation=record)
        point=np.array([2.,4.,7.,1.]); shifted=point.copy(); shifted[:3]+=[2.,-1.,.5]
        expected=RAS_TO_LPS@np.linalg.inv(base.to_reference)@RAS_TO_LPS@shifted
        np.testing.assert_allclose(pull_transform(candidate).TransformPoint(tuple(point[:3])),expected[:3],atol=1e-6)
        self.assertNotIn('vectors',metadata(record))
        invalid={**record,'space':'RAS'}
        with self.assertRaises(InputError):validate_field(invalid)
        with self.assertRaises(InputError):validate_field({**record,'sha256':'tampered'})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InputError):load_deformation({**metadata(record),'file':'../escape.npz'},tmp)

    def test_preview_and_cancel_leave_input_untouched(self):
        scene,base,cavity,_=phantom(); old=cavity.mask.copy()
        preview=exclusion_preview(scene,cavity,3.)
        self.assertGreater(preview.mask.sum(),old.sum())
        self.assertFalse(preview.visible_3d)
        np.testing.assert_array_equal(cavity.mask,old)
        self.assertEqual(len(scene.segmentations),1)
        if ants_available():
            stop=Event(); stop.set()
            with self.assertRaises(CorrectionCancelled):correct_postop_ants(scene,base,cavity,cancel=stop)
        with self.assertRaises(InputError):correct_postop_ants(scene,base,cavity,metric='unsupported')


if __name__=='__main__':unittest.main()
