"""Known deformation, missing tissue, transform direction and portable scene checks."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import SimpleITK as sitk

from brain_viewer.imaging import Scene, Mesh, MRILayer, Segment, InputError, save_scene, load_scene
from brain_viewer.registration import to_sitk, RAS_TO_LPS
from brain_viewer.postop_registration import (correct_postop, read_deformation, deformation_record,
    pull_transform, resample_postop, metric_masks, deformation_qc, export_postop)


def phantom():
    shape=(64,72,60); spacing=1.5
    affine=np.diag([spacing,spacing,spacing,1.]); affine[:3,3]=-np.array(shape)*spacing/2
    xyz=np.stack(np.meshgrid(*[np.arange(n) for n in shape],indexing='ij')).astype(float)
    x,y,z=(xyz-np.array(shape)[:,None,None,None]/2)*spacing
    brain=(x/38)**2+(y/44)**2+(z/35)**2<1
    fixed=np.zeros(shape,np.float32)
    fixed[brain]=650+160*np.sin(x[brain]/6)*np.cos(y[brain]/9)+90*np.cos(z[brain]/5)
    for cx,cy,cz,r,value in ((-12,6,1,8,100),(10,1,6,7,90),(-10,-25,-9,9,950),(19,23,-5,6,320)):
        fixed[(x-cx)**2+(y-cy)**2+(z-cz)**2<r*r]=value
    # Native postop u contains preop intensities at u+d(u). Truth pull solves u+d(u)=x.
    def offset(points):
        return 3.5*np.exp(-((points[0]+10)**2/24**2+(points[1]-6)**2/40**2+points[2]**2/38**2))
    raw_indices=xyz.copy(); raw_indices[0]+=offset(np.stack((x,y,z)))/spacing
    moving=map_coordinates(fixed,raw_indices,order=1,mode='constant').astype(np.float32)
    cavity=((x+23)/9)**2+((y+13)/10)**2+(z/14)**2<1
    moving[cavity]=20
    mesh=Mesh(np.array([[-30,-30,0],[30,30,0],[-30,30,25]],np.float32),np.array([[0,1,2]],np.int32))
    scene=Scene(fixed,affine,affine.copy(),shape,{'pial_lh':mesh,'pial_rh':mesh},kind='synthetic')
    scene.label_volume=np.where(brain,2,0).astype(np.uint16)
    layer=MRILayer('mr_postop','Synthetic postop','postop-T1',moving.copy(),np.ones(shape,bool),
                    moving,affine.copy(),np.eye(4),{'review_status':'visually_reviewed'},1000,500)
    exclusion=Segment('seg_cavity','Cavity','manual',cavity,(1.,.3,.2))
    scene.extra_mris=[layer]; scene.segmentations=[exclusion]
    return scene,layer,exclusion,offset


class PostopRegistrationTests(unittest.TestCase):
    def test_known_smooth_shift_with_missing_tissue_and_portable_save(self):
        scene,layer,exclusion,offset=phantom()
        candidate=correct_postop(scene,layer,exclusion,iterations=55,max_shift_mm=6,grid_mm=30)
        core=(scene.label_volume>0)&~exclusion.mask
        # Validate on held-out landmarks, independently of the optimized image metric.
        points=np.array([[a,b,c] for a in (-12,0,12) for b in (-12,0,12) for c in (-12,0,12)],float)
        truth=points.copy()
        for _ in range(30): truth[:,0]=points[:,0]-offset(truth.T)
        tx=pull_transform(candidate)
        actual=np.array([tx.TransformPoint(tuple(p*[-1,-1,1])) for p in points])*[-1,-1,1]
        before=np.linalg.norm(truth-points,axis=1); after=np.linalg.norm(truth-actual,axis=1)
        print('Postop phantom: landmark median before/after (mm)',float(np.median(before)),float(np.median(after)),flush=True)
        self.assertLess(np.median(after),.8)
        self.assertLess(np.median(after),np.median(before)*.5)
        self.assertLess(np.mean((candidate.data[core]-scene.data[core])**2),np.mean((layer.data[core]-scene.data[core])**2))
        self.assertEqual(candidate.quality['numerical_qc']['folding_voxels'],0)
        self.assertLessEqual(candidate.quality['numerical_qc']['displacement_max_mm'],6.01)
        self.assertIsNone(candidate.quality['landmark_accuracy_mm'])
        self.assertIs(candidate.raw,layer.raw)
        np.testing.assert_array_equal(layer.data,layer.raw)
        with tempfile.TemporaryDirectory() as tmp:
            scene.extra_mris.append(candidate)
            saved=save_scene(scene,Path(tmp)/'work',{'postop':{'image':layer.uid,'stage':candidate.uid}})
            restored=load_scene(saved/'scene.json'); result=restored.extra_mris[-1]
            np.testing.assert_array_equal(result.data,candidate.data)
            image,valid=resample_postop(restored,result,result.deformation)
            np.testing.assert_allclose(image,result.data,atol=1e-4)
            np.testing.assert_array_equal(valid,result.valid)
            out=export_postop(restored,result,tmp)
            imported=sitk.ReadTransform(str(out/'reference_to_native_postop_LPS.tfm'))
            np.testing.assert_allclose(imported.TransformPoint((5.,6.,7.)),tx.TransformPoint((5.,6.,7.)),atol=1e-7)
            metadata=json.loads((out/'registration.json').read_text(encoding='utf-8'))
            self.assertFalse(metadata['point_forward_inverse_available'])
            self.assertEqual(metadata['quality']['exclusion']['uid'],exclusion.uid)
            self.assertEqual(json.loads((saved/'scene.json').read_text())['schema'],'cortex-viewer/10')

    def test_cavity_is_excluded_in_both_images_and_empty_regions_rejected(self):
        scene,layer,excluded,_=phantom(); reference=to_sitk(scene.data,scene.affine)
        fixed,moving,count=metric_masks(scene,layer,reference,excluded,None,3.)
        for mask in (fixed,moving):
            xyz=sitk.GetArrayFromImage(mask).transpose(2,1,0)
            self.assertFalse(xyz[excluded.mask].any())
        self.assertGreater(count,1000)
        with self.assertRaises(InputError): correct_postop(scene,layer)
        empty=replace(excluded,mask=np.zeros(scene.data.shape,bool))
        with self.assertRaises(InputError):metric_masks(scene,layer,reference,empty,None,3.)
        all_brain=replace(excluded,mask=np.ones(scene.data.shape,bool))
        with self.assertRaises(InputError):metric_masks(scene,layer,reference,all_brain,None,3.)

    def test_pull_composition_oblique_native_geometry_and_identity(self):
        scene,layer,_,_=phantom()
        from brain_viewer.alignment_review import rigid_delta
        layer.to_reference=rigid_delta([8,-13,6],[4,-7,2],[0,0,0])
        ref=to_sitk(scene.data,scene.affine)
        bs=sitk.BSplineTransformInitializer(ref,[2,2,2],3)
        parameters=np.zeros(bs.GetNumberOfParameters()); parameters[:len(parameters)//3]=2
        bs.SetParameters(parameters.tolist()); record=deformation_record(bs)
        deformed=replace(layer,deformation=record)
        point=np.array([2.,4.,7.,1.])
        warped=np.array([*bs.TransformPoint(tuple(point[:3])),1.])
        expected=RAS_TO_LPS @ np.linalg.inv(layer.to_reference) @ RAS_TO_LPS @ warped
        np.testing.assert_allclose(pull_transform(deformed).TransformPoint(tuple(point[:3])),expected[:3])
        self.assertTrue(deformation_qc(bs,ref)['finite'])
        record['parameters'][0]=float('nan')
        with self.assertRaises(InputError):read_deformation(record)

    def test_numerical_fold_screen_detects_fold(self):
        scene,_,_,_=phantom(); ref=to_sitk(scene.data,scene.affine)
        reflection=sitk.AffineTransform(3); reflection.SetMatrix([-1.,0,0,0,1.,0,0,0,1.])
        self.assertGreater(deformation_qc(reflection,ref)['folding_voxels'],0)


if __name__=='__main__':unittest.main()
