"""DICOM fixture contains synthetic intensities and no patient data."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from brain_viewer.imaging import InputError
from brain_viewer.volume_io import read_volume,dicom_series


def write_dicom(folder,uneven=False):
    data=np.arange(6*8*9,dtype=np.int16).reshape(6,8,9)-120
    image=sitk.GetImageFromArray(data)
    image.SetSpacing((.7,.8,1.3)); image.SetOrigin((17.,-21.,-8.))
    image.SetDirection((0.,-1.,0.,1.,0.,0.,0.,0.,1.))
    writer=sitk.ImageFileWriter(); writer.KeepOriginalImageUIDOn()
    for z in range(data.shape[0]):
        part=image[:,:,z]
        position=list(image.TransformIndexToPhysicalPoint((0,0,z)))
        if uneven and z==3: position[2]+=.3
        metadata={'0008|0060':'CT','0008|0016':'1.2.840.10008.5.1.4.1.1.2',
            '0010|0010':'SYNTHETIC','0010|0020':'UNIT_TEST',
            '0020|000d':'2.25.778811010','0020|000e':'2.25.778811011',
            '0020|0013':str(z+1),'0020|0032':'\\'.join(map(str,position)),
            '0020|0037':'0\\1\\0\\-1\\0\\0','0028|0030':'.8\\.7',
            '0018|0050':'1.3','0028|1052':'0','0028|1053':'1'}
        for key,value in metadata.items(): part.SetMetaData(key,value)
        writer.SetFileName(str(folder/f'{z:03d}.dcm')); writer.Execute(part)
    return image,data


class DicomInputTests(unittest.TestCase):
    def test_lps_to_ras_oblique_pixel_spacing_and_values(self):
        parent=Path(__file__).resolve().parents[1]/'private_reports'
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            original,data=write_dicom(Path(temp))
            series=dicom_series(temp)
            self.assertEqual(len(series),1)
            image=read_volume(series[0])
            np.testing.assert_array_equal(image.get_fdata(),data.transpose(2,1,0))
            native=np.array([3,5,4])
            expected=np.array(original.TransformIndexToPhysicalPoint(tuple(map(int,native))))*[-1,-1,1]
            np.testing.assert_allclose(nib.affines.apply_affine(image.affine,native),expected,atol=1e-5)
            self.assertEqual(image.extra['input_kind'],'dicom')

    def test_nonuniform_slice_positions_are_rejected(self):
        parent=Path(__file__).resolve().parents[1]/'private_reports'
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            write_dicom(Path(temp),uneven=True)
            with self.assertRaises(InputError): read_volume(temp,modality='CT')
