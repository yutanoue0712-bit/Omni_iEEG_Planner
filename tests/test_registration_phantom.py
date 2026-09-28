"""Known-transform recovery; no patient images or imported electrode positions."""
import unittest
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from brain_viewer.imaging import Scene, Mesh
from brain_viewer.registration import register_ct
from brain_viewer.alignment_review import rigid_delta


class RegistrationPhantomTest(unittest.TestCase):
    def test_recovers_large_sagittal_rotation_without_external_seed(self):
        shape=(128,148,128)
        affine=np.eye(4); affine[:3,3]=[-64,-74,-64]
        x,y,z=np.meshgrid(np.arange(shape[0])-64,np.arange(shape[1])-74,np.arange(shape[2])-64,indexing='ij')
        radius=(x/52)**2+(y/62)**2+(z/52)**2
        mri=np.zeros(shape,np.float32); ct=np.full(shape,-1000,np.float32)
        mri[radius<1]=200; ct[radius<1]=35
        mri[radius<.80]=850; ct[radius<.8]=50
        ct[(radius>.82)&(radius<.94)]=1000
        for a,b,c,rx,ry,rz,mi,ci in ((-18,-10,8,12,22,10,75,8),(13,3,12,9,18,8,100,12),
                (-6,-35,-25,20,14,16,450,30),(22,28,-9,9,12,14,600,65)):
            mask=((x-a)/rx)**2+((y-b)/ry)**2+((z-c)/rz)**2<1
            mri[mask]=mi; ct[mask]=ci
        vertices=np.array([[-45,-55,-40],[45,55,45],[-45,55,45],[45,-55,-40]],np.float32)
        mesh=Mesh(vertices,np.array([[0,1,2],[1,2,3]],np.int32))
        scene=Scene(mri,affine,affine.copy(),shape,{'pial_lh':mesh,'pial_rh':mesh})
        true=rigid_delta([-24,5,-7],[4,-6,8],[0,0,0])
        ct_affine=affine.copy()
        ct_shape=(192,212,192)
        ct_affine[:3,3]=[-96,-106,-96]
        raw=resample_from_to(nib.Nifti1Image(ct,affine),(ct_shape,true @ ct_affine),order=1,cval=-1000).get_fdata(dtype=np.float32)
        estimated,quality=register_ct(scene,raw,ct_affine)
        points=np.array([[a,b,c] for a in (-25,0,25) for b in (-30,0,30) for c in (-25,0,25)])
        error=np.linalg.norm(nib.affines.apply_affine(estimated @ np.linalg.inv(true),points)-points,axis=1)
        print(f'Phantom landmark p95 error: {np.percentile(error,95):.3f} mm',flush=True)
        self.assertLess(float(np.percentile(error,95)),1.,msg=f'Known-transform phantom p95 error: {np.percentile(error,95):.3f} mm')
        self.assertGreaterEqual(quality['successful_starts'],6)
        self.assertIsNone(quality['landmark_accuracy_mm'])
