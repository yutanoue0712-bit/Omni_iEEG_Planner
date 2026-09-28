"""PET calibration, geometry, registration and independent scalar-layer checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter
from brain_viewer.imaging import InputError, Contact, MRILayer, Scene, Mesh, save_scene, load_scene
from brain_viewer.sequences import add_image, register_mri, find_image_sources, source_modality
from brain_viewer.volume_io import read_volume
from brain_viewer.rendering import composite_plane
from brain_viewer.alignment_review import rigid_delta
from brain_viewer.segmentation_sources import image_sources
from test_dicom_input import write_dicom
from test_multimodal import small_scene


def write_pet(folder, frame=1000, slope_offset=0):
    folder.mkdir(exist_ok=True)
    original, _ = write_dicom(folder)
    expected=[]
    for z,path in enumerate(sorted(folder.glob('*.dcm'))):
        ds=pydicom.dcmread(path)
        pixels=np.arange(72,dtype=np.int16).reshape(8,9)+z*80
        ds.Modality='PT'; ds.SOPClassUID=pydicom.uid.PositronEmissionTomographyImageStorage
        ds.file_meta.MediaStorageSOPClassUID=ds.SOPClassUID
        ds.Units='BQML'; ds.FrameReferenceTime=str(frame)
        ds.RescaleSlope=str(.25+z*.125+slope_offset); ds.RescaleIntercept='1.75'
        ds.PixelData=pixels.tobytes(); ds.save_as(path,enforce_file_format=True)
        expected.append(pixels*float(ds.RescaleSlope)+1.75)
    return original,np.array(expected).transpose(2,1,0)


class PETImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)

    def test_native_pet_fractional_per_slice_calibration_and_ras(self):
        original,expected=write_pet(self.root)
        sources=find_image_sources(self.root,'PT',dicom_only=True)
        self.assertEqual(len(sources),1)
        source=sources[0][1]
        # File order is deliberately reversed; physical positions must win.
        source=dict(source,files=tuple(reversed(source['files'])))
        image=read_volume(source,modality='PT')
        np.testing.assert_allclose(image.get_fdata(),expected,atol=1e-5)
        point=(3,5,4)
        np.testing.assert_allclose(nib.affines.apply_affine(image.affine,point),
            np.array(original.TransformIndexToPhysicalPoint(point))*[-1,-1,1],atol=1e-5)
        self.assertEqual(image.extra['value_units'],'BQML')
        self.assertFalse(image.extra['suv_conversion'])
        with self.assertRaises(InputError): read_volume(source,modality='CT')

    def test_pet_and_ct_and_time_frames_are_separate_candidates(self):
        write_pet(self.root/'pet')
        ct=self.root/'ct'; ct.mkdir(); write_dicom(ct)
        second=self.root/'second'; write_pet(second,frame=2000,slope_offset=1)
        # Two time frames with the same Series UID and positions in one directory.
        for path in second.glob('*.dcm'): path.rename(self.root/'pet'/('frame2_'+path.name))
        pet=find_image_sources(self.root,'PT',dicom_only=True)
        bone=find_image_sources(self.root,'CT',dicom_only=True)
        self.assertEqual([v['count'] for _,v in pet],[6,6])
        self.assertEqual(len(bone),1)
        self.assertEqual(source_modality('PET'),'PT'); self.assertEqual(source_modality('CT-bone'),'CT')

    def test_heatmap_transparency_visibility_and_palette_do_not_mutate_values(self):
        scene=small_scene(); values=np.full(scene.data.shape,40.,np.float32)
        values[0]=0; values[1]=5; valid=np.ones(values.shape,bool); valid[2]=False
        layer=MRILayer('mr_pet','PET','PET',values,valid,values.copy(),scene.affine,np.eye(4),{},100,50)
        scene.extra_mris=[layer]; before=values.copy()
        options={'extra_mris':{layer.uid:{'opacity':1.}}}
        rgb=composite_plane(scene,2,5,0,200,options)
        self.assertFalse(np.array_equal(rgb[3,:,0],rgb[3,:,1]))
        np.testing.assert_array_equal(rgb[0],127); np.testing.assert_array_equal(rgb[2],127)
        options['extra_mris'][layer.uid]['palette']='turbo'
        self.assertFalse(np.array_equal(rgb,composite_plane(scene,2,5,0,200,options)))
        for key,value in (('visible',False),('opacity',0.)):
            options['extra_mris'][layer.uid]={key:value}
            np.testing.assert_array_equal(composite_plane(scene,2,5,0,200,options),127)
        options['extra_mris'][layer.uid]={'window':60,'level':50,'opacity':1.}
        np.testing.assert_array_equal(composite_plane(scene,2,5,0,200,options)[1],127)
        np.testing.assert_array_equal(layer.data,before); np.testing.assert_array_equal(layer.raw,before)

    def test_pet_import_cache_and_bone_ct_preserve_postoperative_ct_contacts_and_save(self):
        scene=small_scene()
        scene.ct=np.linspace(-1000,3000,scene.data.size,dtype=np.float32).reshape(scene.data.shape)
        scene.ct_valid=np.ones(scene.data.shape,bool); scene.ct_to_mri=np.eye(4)
        scene.raw_ct=scene.ct.copy(); scene.raw_ct_affine=scene.affine.copy()
        scene.contacts=[Contact('E1','E',scene.world([6,6,6]),(.2,.8,.4))]
        raw=np.arange(scene.data.size,dtype=np.float32).reshape(scene.data.shape)*.125
        pet=nib.Nifti1Image(raw,scene.affine); pet.header.set_xyzt_units('mm')
        pet.extra.update(value_units='BQML',input_kind='dicom',suv_conversion=False)
        with patch('brain_viewer.sequences.register_mri',return_value=(np.eye(4),{'review_status':'test'})) as register:
            updated=add_image(scene,pet,'PET','PET',self.root)
            add_image(scene,pet,'PET','PET again',self.root)
            self.assertEqual(register.call_count,1)
            self.assertEqual(register.call_args.kwargs,{'modality':'PT'})
        bone=nib.Nifti1Image(scene.raw_ct,scene.raw_ct_affine); bone.header.set_xyzt_units('mm')
        with patch('brain_viewer.registration.register_ct',return_value=(np.eye(4),{'review_status':'test'})):
            updated=add_image(updated,bone,'CT-bone','Bone CT',self.root)
        self.assertIs(updated.ct,scene.ct); self.assertIs(updated.contacts,scene.contacts)
        pet_layer,bone_layer=updated.extra_mris
        self.assertEqual((bone_layer.window,bone_layer.level),(2000.,500.))
        self.assertEqual(image_sources(updated)[1].units,'BQML')
        self.assertEqual(image_sources(updated)[2].units,'HU')
        state={'extra_styles':{pet_layer.uid:'heatmap'},'extra_palettes':{pet_layer.uid:'inferno'}}
        saved=save_scene(updated,self.root/'saved',state)
        restored=load_scene(saved/'scene.json')
        np.testing.assert_array_equal(restored.extra_mris[0].raw,raw)
        self.assertEqual(restored.extra_mris[0].quality['value_units'],'BQML')
        self.assertFalse(restored.extra_mris[0].quality['suv_conversion'])
        self.assertEqual(restored.view_state,state)
        np.testing.assert_array_equal(restored.contact_positions(),scene.contact_positions())

    def test_pet_nifti_sidecar_units_and_unknown_units(self):
        image=nib.Nifti1Image(np.ones((4,4,4),np.float32),np.eye(4)); image.header.set_xyzt_units('mm')
        path=self.root/'pet.nii.gz'; nib.save(image,path)
        self.assertEqual(read_volume(path,modality='PT').extra['value_units'],'')
        sidecar=self.root/'pet.json'; sidecar.write_text(json.dumps({'Modality':'PT','Units':'BQML'}))
        self.assertEqual(read_volume(path,modality='PT').extra['value_units'],'BQML')
        sidecar.write_text(json.dumps({'Modality':'CT'}))
        with self.assertRaises(InputError): read_volume(path,modality='PT')


class PETRegistrationTests(unittest.TestCase):
    def test_blurred_low_resolution_pet_recovers_known_rigid_pose(self):
        shape=(76,90,72); affine=np.diag([1.3,1.3,1.3,1.]); affine[:3,3]=[-49,-58,-46]
        xyz=nib.affines.apply_affine(affine,np.indices(shape).reshape(3,-1).T).T.reshape(3,*shape)
        x,y,z=xyz; radius=(x/42)**2+(y/50)**2+(z/39)**2
        t1=np.zeros(shape,np.float32); pet=np.zeros_like(t1)
        t1[radius<1]=220; pet[radius<1]=12000
        t1[radius<.8]=800; pet[radius<.8]=6500
        for a,b,c,rx,ry,rz,u,v in ((-15,-10,8,8,18,8,75,700),(12,4,12,7,15,6,100,500),
                (-6,-29,-22,15,11,12,450,9500),(20,25,-9,7,10,11,600,8200)):
            mask=((x-a)/rx)**2+((y-b)/ry)**2+((z-c)/rz)**2<1
            t1[mask]=u; pet[mask]=v
        pet=gaussian_filter(pet,1.5)
        vertices=np.array([[-36,-42,-32],[36,42,32],[-36,42,32],[36,-42,-32]],np.float32)
        mesh=Mesh(vertices,np.array([[0,1,2],[1,2,3]],np.int32))
        scene=Scene(t1,affine,affine.copy(),shape,{'pial_lh':mesh,'pial_rh':mesh})
        truth=rigid_delta([12,-5,7],[3,-4,5],[0,0,0])
        native=np.diag([2.,2.,2.5,1.]); native[:3,3]=[-66,-75,-62]
        raw=resample_from_to(nib.Nifti1Image(pet,affine),((68,78,52),truth@native),order=1).get_fdata(dtype=np.float32)
        estimated,quality=register_mri(scene,raw,native,modality='PT')
        points=np.array([[a,b,c] for a in (-22,0,22) for b in (-26,0,26) for c in (-22,0,22)])
        error=np.linalg.norm(nib.affines.apply_affine(estimated@np.linalg.inv(truth),points)-points,axis=1)
        p95=float(np.percentile(error,95)); print(f'PET phantom landmark p95 error: {p95:.3f} mm',flush=True)
        self.assertLess(p95,2.)
        self.assertIsNone(quality['landmark_accuracy_mm'])
