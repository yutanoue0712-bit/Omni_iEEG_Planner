from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
from brain_viewer.diffusion import DiffusionModel, fit_tensor, track_roi, roi_contains_streamline, motion_correct
from brain_viewer.diffusion_io import inspect_diffusion, read_gradients
from brain_viewer.imaging import make_demo, save_scene, load_scene, InputError, Segment
from brain_viewer.diffusion_storage import export_tract


def phantom(affine=None):
    shape=(25,19,17)
    affine=np.diag([2.,2.,2.,1.]) if affine is None else affine
    affine=affine.copy(); affine[:3,3]=-affine[:3,:3]@((np.array(shape)-1)/2)
    v=np.random.default_rng(21).normal(size=(24,3))
    v/=np.linalg.norm(v,axis=1)[:,None]
    b=np.r_[0,np.full(24,1000.)]; v=np.vstack(([0,0,0],v))
    tensor=np.diag([.0017,.0003,.0003])
    signal=1200*np.exp(-b*np.einsum('ni,ij,nj->n',v,tensor,v))
    data=np.broadcast_to(signal,shape+(len(b),)).astype(np.float32).copy()
    mask=np.ones(shape,bool); mask[[0,-1],:,:]=False; mask[:,[0,-1],:]=False; mask[:,:,[0,-1]]=False
    scale=.8+.2*np.indices(shape)[0]/(shape[0]-1)
    data *= (scale*mask)[...,None]
    return data,affine,b,v,mask


def synthetic_model(scene=None):
    data,affine,b,v,mask=phantom()
    fa,md,direction=fit_tensor(data,b,v,mask)
    model=DiffusionModel('dti_synthetic','Synthetic DTI',affine,np.eye(4),fa,direction,mask,b,v,
                         quality={'fingerprint':'synthetic'})
    scene=make_demo() if scene is None else scene
    scene.diffusions=[model]
    return scene,model


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'private_reports')
        self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)

    def write_input(self,affine=None):
        data,a,b,v,mask=phantom(affine)
        image=nib.Nifti1Image(data,a); image.header.set_xyzt_units('mm')
        image.set_qform(a,1);image.set_sform(a,1)
        path=self.folder/'dwi.nii.gz'; nib.save(image,path)
        fsl=v.copy()
        if np.linalg.det(a[:3,:3])>0:fsl[:,0]*=-1
        np.savetxt(self.folder/'dwi.bval',b[None])
        np.savetxt(self.folder/'dwi.bvec',fsl.T)
        return path,data,a,b,v,mask

    def test_fsl_handedness_and_oblique_tensor(self):
        angle=.3; rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        for sign in (1,-1):
            a=np.eye(4);a[:3,:3]=rotation@np.diag([2.*sign,2.,2.])
            path,data,a,b,v,mask=self.write_input(a)
            image,actual_b,actual_v=inspect_diffusion(path)
            np.testing.assert_allclose(actual_v,v,atol=1e-6)
            fa,md,directions=fit_tensor(data,actual_b,actual_v,mask)
            self.assertAlmostEqual(float(md[10,10,10]),.0023/3,places=7)
            eigenvalues=np.array([.0017,.0003,.0003])
            expected_fa=np.sqrt(1.5*np.sum((eigenvalues-eigenvalues.mean())**2)/np.sum(eigenvalues**2))
            self.assertAlmostEqual(float(fa[10,10,10]),expected_fa,places=5)
            physical=(a[:3,:3]/2)@directions[10,10,10]
            self.assertGreater(abs(float(physical @ (rotation[:,0]*sign))),.9999)

    def test_missing_corrupt_and_rank_deficient_gradients_rejected(self):
        path,data,a,b,v,mask=self.write_input()
        (self.folder/'dwi.bvec').unlink()
        with self.assertRaises(InputError):inspect_diffusion(path)
        np.savetxt(self.folder/'dwi.bvec',np.tile([1.,0,0],(len(b),1)).T)
        with self.assertRaises(InputError):inspect_diffusion(path)
        np.savetxt(self.folder/'dwi.bvec',np.full((3,len(b)),np.nan))
        with self.assertRaises(InputError):inspect_diffusion(path)
        np.savetxt(self.folder/'dwi.bvec',v[:-1].T)
        with self.assertRaises(InputError):inspect_diffusion(path)

    def test_scalar_3d_and_inconsistent_geometry_rejected(self):
        path,data,a,b,v,mask=self.write_input()
        image=nib.load(path)
        shifted=a.copy();shifted[0,3]+=2;image.set_qform(shifted,1);nib.save(image,path)
        with self.assertRaises(InputError):inspect_diffusion(path)
        image=nib.Nifti1Image(data[...,0],a);image.header.set_xyzt_units('mm');nib.save(image,path)
        with self.assertRaises(InputError):inspect_diffusion(path)

    def test_one_two_roi_fa_and_physical_transform(self):
        scene,model=synthetic_model()
        # Track in native space, then independently check a rotated/translated reference frame.
        transform=np.array([[0.,-1,0,10],[1,0,0,5],[0,0,1,-3],[0,0,0,1]])
        model.to_reference=transform
        roi1={'kind':'sphere','center':[10.,5.,-3.],'radius':4.}
        roi2={'kind':'sphere','center':[10.,20.,-3.],'radius':4.}
        one=track_roi(scene,model,roi1,max_seeds=50)
        two=track_roi(scene,model,roi1,roi2,max_seeds=50)
        self.assertGreater(one.count,0);self.assertGreater(two.count,0)
        self.assertLessEqual(two.count,one.count)
        for line in two.streamlines():
            self.assertTrue(roi_contains_streamline(scene,roi2,line))
            self.assertGreater(np.ptp(line[:,1]),25)
            self.assertLess(np.ptp(line[:,0]),5)
        absent={'kind':'sphere','center':[35.,20.,-3.],'radius':2.}
        self.assertEqual(track_roi(scene,model,roi1,absent,max_seeds=50).count,0)
        with self.assertRaises(InputError):track_roi(scene,model,roi1,fa_threshold=.95)

    def test_snapshot_and_tck_trk_roundtrip(self):
        scene,model=synthetic_model()
        bundle=track_roi(scene,model,{'kind':'sphere','center':[0.,0.,0.],'radius':4.},max_seeds=20)
        scene.tracts=[bundle]
        saved=save_scene(scene,self.folder/'work',{})
        restored=load_scene(saved/'scene.json')
        np.testing.assert_array_equal(restored.diffusions[0].directions,model.directions)
        np.testing.assert_array_equal(restored.tracts[0].points,bundle.points)
        for extension in ('tck','trk'):
            path=self.folder/('tracks.'+extension)
            export_tract(bundle,path,scene)
            actual=list(nib.streamlines.load(path).streamlines)
            for a,b in zip(actual,bundle.streamlines()):np.testing.assert_allclose(a,b,atol=1e-5)
            self.assertEqual(len(actual),bundle.count)

    def test_segmentation_roi_and_bounded_preview(self):
        scene,model=synthetic_model()
        mask=np.zeros(scene.data.shape,bool)
        corner=np.rint(scene.index([9.,0.,0.])).astype(int)
        mask[corner[0],corner[1]-3:corner[1]+4,corner[2]-3:corner[2]+4]=True
        scene.segmentations=[Segment('seg_roi','Thin ROI','manual',mask,(1.,.5,.2))]
        roi1={'kind':'sphere','center':[0.,0.,0.],'radius':4.}
        roi2={'kind':'segment','uid':'seg_roi'}
        bundle=track_roi(scene,model,roi1,roi2,max_seeds=50)
        self.assertGreater(bundle.count,0)
        for line in bundle.streamlines():self.assertTrue(roi_contains_streamline(scene,roi2,line))
        self.assertIn('mask_sha256',bundle.settings['roi2'])
        preview=track_roi(scene,model,{'kind':'brain'},max_seeds=50)
        self.assertEqual(preview.settings['used_seeds'],50)
        self.assertLessEqual(preview.count,50)

    def test_gradient_rotation_follows_motion_transform(self):
        from brain_viewer.registration import sampling_transform
        data,a,b,v,mask=phantom()
        angle=.09; rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        matrix=np.eye(4);matrix[:3,:3]=rotation
        with patch('SimpleITK.ImageRegistrationMethod.Execute',return_value=sampling_transform(matrix)), \
             patch('SimpleITK.ImageRegistrationMethod.GetMetricValue',return_value=-1.):
            _,rotated,motion=motion_correct(data,a,b,v)
        np.testing.assert_allclose(rotated[1:],v[1:]@rotation.T,atol=1e-6)
        np.testing.assert_allclose(np.asarray(motion)[0],np.eye(4))


if __name__=='__main__':unittest.main()
