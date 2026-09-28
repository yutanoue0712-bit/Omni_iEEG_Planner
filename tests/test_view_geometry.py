import unittest
from types import SimpleNamespace
import numpy as np
from brain_viewer.segmentation import fill_contour
from brain_viewer.segmentation_sources import create_anatomy_roi
from brain_viewer.diffusion import TractBundle
from brain_viewer.diffusion_rendering import tract_segments, clip_segments_to_slab
from brain_viewer.trajectory_view import lead_frame, sample_reference


class ViewGeometryTests(unittest.TestCase):
    def test_closed_roi_anisotropic_axes_and_thickness(self):
        shape=(13,15,17);spacing=np.array([1.,2.,3.])
        for axis in (0,1,2):
            mask=np.zeros(shape,bool);other=[n for n in range(3) if n!=axis]
            vertices=np.full((4,3),6.)
            vertices[:,other]=[[2,2],[8,2],[8,8],[2,8]]
            fill_contour(mask,spacing,axis,vertices,thickness_mm=spacing[axis]*2)
            points=np.argwhere(mask)
            np.testing.assert_array_equal(np.unique(points[:,axis]),[5,6])
            self.assertTrue(mask[6,6,6]);self.assertFalse(mask[0,0,0])
            fill_contour(mask,spacing,axis,vertices,False,spacing[axis]*2)
            self.assertFalse(mask.any())

    def test_contour_click_and_collinear_do_not_modify(self):
        mask=np.zeros((9,9,9),bool)
        for vertices in ([[2,2,4]],[[2,2,4],[3,3,4],[6,6,4]]):
            fill_contour(mask,[1,1,1],2,vertices)
            self.assertFalse(mask.any())

    def test_slice_clipping_includes_crossings_without_false_bridges(self):
        bundle=TractBundle('tract_test','test','dti_test',
            np.array([[0,0,-10],[0,0,10],[1,1,8],[2,2,9],[3,0,0],[6,0,0]],float),
            np.array([0,2,4,6]))
        segments=tract_segments(bundle)
        self.assertEqual(len(segments),3)
        clipped,colors=clip_segments_to_slab(segments,2,0,2)
        self.assertEqual(len(clipped),2)
        np.testing.assert_allclose(clipped[0],[[0,0,-1],[0,0,1]])
        np.testing.assert_allclose(colors,[[0,0,1],[1,0,0]])
        reverse,_=clip_segments_to_slab(segments[:,::-1],2,0,2)
        np.testing.assert_allclose(reverse[:,::-1],clipped)

    def test_freesurfer_roi_union_is_independently_editable(self):
        labels=np.zeros((5,6,7),np.uint16);labels[1,1,1]=10;labels[3,3,3]=49
        nuclei=np.zeros_like(labels);nuclei[2,2,2]=8103
        scene=SimpleNamespace(label_volume=labels,label_names={10:'Left-Thalamus',49:'Right-Thalamus'},
            nuclei_display=nuclei,nuclei_names={8103:'Nucleus'},label_source='synthetic',
            nuclei_source='synthetic nuclei',spacing=np.ones(3))
        records=[{'source':'anatomy','label':10},{'source':'anatomy','label':49},{'source':'nuclei','label':8103}]
        segment=create_anatomy_roi(scene,records)
        self.assertEqual(segment.mask.sum(),3)
        self.assertEqual(len(segment.provenance['roi_labels']),3)
        segment.mask[:]=False
        self.assertEqual(labels[1,1,1],10);self.assertEqual(nuclei[2,2,2],8103)

    def test_oblique_trajectory_axis_and_sampling_are_in_reference_ras(self):
        origin=np.array([10.,-8.,3.]);axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
        contacts=[SimpleNamespace(position=origin+t*axis) for t in (0,5,10)]
        start,normal,u,v,length=lead_frame(contacts)
        np.testing.assert_allclose(start,origin);np.testing.assert_allclose(normal,axis)
        np.testing.assert_allclose(np.cross(u,v),axis);self.assertAlmostEqual(length,10)
        affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[7,-4,2]
        scene=SimpleNamespace(index=lambda pts:(pts-affine[:3,3])/np.diag(affine)[:3])
        raw=np.indices((9,9,9)).astype(float)[0]
        coordinates=np.array([[[11.,2.,10.],[13.,5.,14.]]])
        np.testing.assert_allclose(sample_reference(raw,scene,coordinates),[[2.,3.]])
